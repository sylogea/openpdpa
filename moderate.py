import typing

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


def make_moderation_prompt() -> ChatPromptTemplate:
	return ChatPromptTemplate.from_messages([
		("system", "Decide if the user's message is adversarial or hostile. Reply with a single emoji: either ✅ (if allowed), or ❌ (if blocked)."),
		("human", "{question}")
	])


def build_moderate_node(llm_mod, moderation_sentinel: str, prompt: ChatPromptTemplate) -> typing.Callable[[dict], dict]:
	def moderate_node(state: dict) -> dict:
		label = (prompt | llm_mod | StrOutputParser()).invoke({"question": state["query"]})
		text = str(label).strip().upper()
		if "✅" in text:
			return {"moderation_ok": True, "answer": ""}
		return {"moderation_ok": False, "answer": moderation_sentinel}
	return moderate_node
