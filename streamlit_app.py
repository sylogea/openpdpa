import os
import pathlib
import streamlit
import typing

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from chat import build_generate_node, make_prompt
from moderate import build_moderate_node, make_moderation_prompt
from retrieve import build_retrieve_node
from store import DISTANCE_SENTINEL, MODERATION_SENTINEL, init_store


ROOT = pathlib.Path(__file__).resolve().parent


class State(TypedDict):
	query: str
	context: str
	answer: str
	retrieved_count: int
	max_output_tokens: int
	top_k: int
	moderation_ok: bool


def build_graph(vector_store, llm_main, llm_mod):
	def route_moderation(state: State) -> str:
		if state.get("moderation_ok"):
			return "ok"
		return "blocked"

	graph = StateGraph(State)
	graph.add_node("moderate", build_moderate_node(llm_mod, MODERATION_SENTINEL, make_moderation_prompt()))
	graph.add_node("retrieve", build_retrieve_node(vector_store, int(os.environ.get("TOP_K", "5"))))
	graph.add_node("generate", build_generate_node(llm_main, make_prompt(os.environ.get("NEXT_PUBLIC_MODEL_NAME", "")), DISTANCE_SENTINEL))
	graph.add_edge(START, "moderate")
	graph.add_conditional_edges("moderate", route_moderation, {"ok": "retrieve", "blocked": END})
	graph.add_edge("retrieve", "generate")
	graph.add_edge("generate", END)
	return graph.compile()


@streamlit.cache_resource(show_spinner="Loading PDPA knowledge base...")
def get_app_state():
	embeddings, vector_store = init_store()

	chat_model_name = os.environ.get("CHAT_MODEL", "")
	moderation_model_name = os.environ.get("MODERATION_MODEL", "")
	if not chat_model_name or not moderation_model_name:
		raise RuntimeError("CHAT_MODEL and MODERATION_MODEL must be set as environment variables or secrets")

	openrouter_api_key = os.environ.get("OPENROUTER_API_KEY", "")
	if not openrouter_api_key:
		raise RuntimeError("OPENROUTER_API_KEY must be set as an environment variable or secret")

	openrouter_base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

	llm_main = ChatOpenAI(
		model=chat_model_name,
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

	graph = build_graph(vector_store, llm_main, llm_mod)
	model_label = os.environ.get("NEXT_PUBLIC_MODEL_NAME", "Assistant")
	return graph, model_label


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

	graph, model_name = get_app_state()

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
				"top_k": int(os.environ.get("TOP_K", "5")),
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
