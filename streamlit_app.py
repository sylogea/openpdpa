import io
import os
import pathlib
import streamlit
import typing
import urllib.error
import urllib.request

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from generate import build_generate_node, make_prompt
from moderate import build_moderate_node, make_moderation_prompt
from retrieve import build_retrieve_node
from store import DISTANCE_SENTINEL, MODERATION_SENTINEL, init_store_from_existing, restore_qdrant_from_zip


ROOT = pathlib.Path(__file__).resolve().parent


class State(TypedDict):
	query: str
	context: str
	answer: str
	retrieved_count: int
	top_k: int
	moderation_ok: bool


def build_graph(vector_store, llm_main, llm_mod, top_k_default: int, prompt_model_name: str):
	def route_moderation(state: State) -> str:
		if state.get("moderation_ok"):
			return "ok"
		return "blocked"

	graph = StateGraph(State)
	graph.add_node("moderate", build_moderate_node(llm_mod, MODERATION_SENTINEL, make_moderation_prompt()))
	graph.add_node("retrieve", build_retrieve_node(vector_store, top_k_default))
	graph.add_node("generate", build_generate_node(llm_main, make_prompt(prompt_model_name), DISTANCE_SENTINEL))
	graph.add_edge(START, "moderate")
	graph.add_conditional_edges("moderate", route_moderation, {"ok": "retrieve", "blocked": END})
	graph.add_edge("retrieve", "generate")
	graph.add_edge("generate", END)
	return graph.compile()


def _to_raw_github_url(url: str) -> str:
	if "github.com" in url and "/blob/" in url:
		prefix, rest = url.split("github.com/", 1)
		user_repo, blob_part = rest.split("/blob/", 1)
		return f"https://raw.githubusercontent.com/{user_repo}/{blob_part}"
	return url


def _get_store_url() -> str:
	raw_url = os.environ.get("STORE_URL", "").strip()
	if not raw_url:
		raise RuntimeError("STORE_URL must be set as an environment variable or secret")
	return _to_raw_github_url(raw_url)


def _download_and_restore_store():
	url = _get_store_url()
	try:
		with urllib.request.urlopen(url) as response:
			data = response.read()
	except urllib.error.URLError as error:
		raise RuntimeError(f"Failed to download store from {url}: {error}") from error
	restore_qdrant_from_zip(io.BytesIO(data))


@streamlit.cache_resource(show_spinner="Loading PDPA knowledge base...")
def get_app_state():
	_download_and_restore_store()

	_, vector_store = init_store_from_existing()

	generate_model_name = os.environ.get("CHAT_MODEL", "")
	moderation_model_name = os.environ.get("MODERATION_MODEL", "")
	if not generate_model_name or not moderation_model_name:
		raise RuntimeError("CHAT_MODEL and MODERATION_MODEL must be set as environment variables or secrets")

	openrouter_api_key = os.environ.get("OPENROUTER_API_KEY", "")
	if not openrouter_api_key:
		raise RuntimeError("OPENROUTER_API_KEY must be set as an environment variable or secret")

	openrouter_base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

	llm_main = ChatOpenAI(
		model=generate_model_name,
		api_key=openrouter_api_key,
		base_url=openrouter_base_url,
		temperature=0
	)
	llm_mod = ChatOpenAI(
		model=moderation_model_name,
		api_key=openrouter_api_key,
		base_url=openrouter_base_url,
		temperature=0
	)

	prompt_model_name = os.environ.get("NEXT_PUBLIC_MODEL_NAME", "")
	model_label = os.environ.get("NEXT_PUBLIC_MODEL_NAME", "Assistant")
	top_k_default = int(os.environ.get("TOP_K", "5"))

	graph = build_graph(vector_store, llm_main, llm_mod, top_k_default, prompt_model_name)
	return graph, model_label, top_k_default


def _normalise_answer_text(text: str) -> str:
	if not text:
		return ""
	s = str(text)
	s = s.replace("<br />", "\n").replace("<br/>", "\n").replace("<br>", "\n")
	s = s.replace("&lt;br /&gt;", "\n").replace("&lt;br/&gt;", "\n").replace("&lt;br&gt;", "\n")
	return s


def main():
	streamlit.set_page_config(
		page_title="OpenPDPA",
		page_icon="🏛️"
	)

	streamlit.title("OpenPDPA")

	graph, model_label, top_k_default = get_app_state()

	if "messages" not in streamlit.session_state:
		streamlit.session_state.messages = []

	for message in streamlit.session_state.messages:
		role = message["role"]
		content = message["content"]
		with streamlit.chat_message(role):
			streamlit.markdown(content)

	prompt = streamlit.chat_input("Ask me a question about PDPA!")
	if not prompt:
		return

	streamlit.session_state.messages.append({"role": "user", "content": prompt})

	with streamlit.chat_message("user"):
		streamlit.markdown(prompt)

	with streamlit.chat_message("assistant"):
		with streamlit.spinner("Thinking..."):
			response = graph.invoke({
				"query": prompt,
				"context": "",
				"answer": "",
				"retrieved_count": 0,
				"top_k": top_k_default,
				"moderation_ok": False
			})
			answer_text = str(response.get("answer", "")).strip()
			answer_text = _normalise_answer_text(answer_text)
			if not answer_text:
				answer_text = "Sorry, I could not find any information on that."
			streamlit.markdown(answer_text)

	streamlit.session_state.messages.append({"role": "assistant", "content": answer_text})


if __name__ == "__main__":
	main()
