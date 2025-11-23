import typing


def build_retrieve_node(vector_store, top_k_default: int) -> typing.Callable[[dict], dict]:
	def retrieve_node(state: dict) -> dict:
		top_k = max(1, int(state.get("top_k", top_k_default)))
		retriever = vector_store.as_retriever(
			search_type="similarity",
			search_kwargs={"k": top_k}
		)
		results = retriever.invoke(state["query"])
		if not results:
			return {"context": "", "retrieved_count": 0}
		chunks = []
		for index, doc in enumerate(results):
			source = doc.metadata.get("source")
			prefix = f"[{index + 1}] ({source})"
			chunks.append(f"{prefix} {doc.page_content}")
		context = "\n\n".join(chunks)
		return {"context": context, "retrieved_count": len(results)}
	return retrieve_node
