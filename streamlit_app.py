import os
import pathlib
import streamlit
import typing

from langchain_groq import ChatGroq
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
	llm_main = ChatGroq(model=chat_model_name, temperature=0)
	llm_mod = ChatGroq(model=moderation_model_name, temperature=0)
	graph = build_graph(vector_store, llm_main, llm_mod)
	model_label = os.environ.get("NEXT_PUBLIC_MODEL_NAME", "Assistant")
	return graph, model_label


def main():
	streamlit.set_page_config(
		page_title="PDPA helper",
		page_icon="📚",
		layout="wide"
	)
	streamlit.title("PDPA assistant for Singapore")
	streamlit.markdown("Ask questions about the Personal Data Protection Act. Answers are based only on the PDFs in this app.")
	graph, model_name = get_app_state()
	if "messages" not in streamlit.session_state:
		streamlit.session_state.messages = []
	for message in streamlit.session_state.messages:
		role = message["role"]
		content = message["content"]
		with streamlit.chat_message(role):
			streamlit.markdown(content)
	prompt = streamlit.chat_input("Ask a PDPA question")
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
				"max_output_tokens": 1024,
				"top_k": int(os.environ.get("TOP_K", "5")),
				"moderation_ok": False
			})
			answer_text = str(response.get("answer", "")).strip()
			if not answer_text:
				answer_text = "Sorry, I could not find any information on that."
			streamlit.markdown(answer_text)
	streamlit.session_state.messages.append({"role": "assistant", "content": answer_text})


if __name__ == "__main__":
	main()
