import typing


def build_retrieve_node(vector_store, top_k_default: int) -> typing.Callable[[dict], dict]:
	def retrieve_node(state: dict) -> dict:
		top_k = max(1, int(state.get("top_k", top_k_default)))
		results = vector_store.similarity_search(state["query"], k=top_k)
		if not results:
			return {"context": "", "retrieved_count": 0}
		chunks = []
		for result_index, result_doc in enumerate(results, start=1):
			source = result_doc.metadata.get("source")
			prefix = f"[{result_index}] ({source})"
			chunks.append(f"{prefix} {result_doc.page_content}")
		context = "\n\n".join(chunks)
		return {"context": context, "retrieved_count": len(results)}
	return retrieve_node
