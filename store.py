import hashlib
import os
import pathlib
import shutil
import uuid
import zipfile

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams


ROOT = pathlib.Path(__file__).resolve().parent

MODERATION_SENTINEL = """
Sorry, I cannot help with that.
"""

DISTANCE_SENTINEL = """
Sorry, I could not find any information on that.
"""

EMBEDDING_DEVICE = "cpu"
QDRANT_COLLECTION = "pdpa"
QDRANT_LOCATION = str(ROOT / "qdrant")

_qdrant_client = None


def compute_pdf_paths():
	pd_dir = pathlib.Path(QDRANT_LOCATION).resolve() / "collection" / QDRANT_COLLECTION
	if not pd_dir.exists():
		raise FileNotFoundError(f"Collection directory not found: {pd_dir}")
	paths = sorted(pd_dir.glob("*.pdf"))
	if not paths:
		raise FileNotFoundError(f"No PDFs found in {pd_dir}")
	return [str(path) for path in paths]


def read_pdf_texts(paths):
	results = []
	for path in paths:
		reader = PdfReader(path)
		pages = []
		for page in reader.pages:
			raw = page.extract_text() or ""
			if not raw:
				continue
			lines = [" ".join(line.split()) for line in raw.splitlines()]
			text = "\n".join(lines).strip()
			if text:
				pages.append(text)
		full_text = "\n\n".join(pages).strip()
		results.append((pathlib.Path(path).name, full_text))
	return results


def compute_corpus_hash(texts, embedding_model):
	h = hashlib.sha256()
	h.update(embedding_model.encode("utf-8"))
	for name, text in texts:
		h.update(name.encode("utf-8"))
		h.update(b"\x1f")
		h.update(text.encode("utf-8"))
	return h.hexdigest()


def _cache_dir():
	path = pathlib.Path(QDRANT_LOCATION).resolve() / "cache"
	path.mkdir(parents=True, exist_ok=True)
	return path


def _cache_file():
	return _cache_dir() / "pdf_sha256.txt"


def _get_qdrant_client():
	global _qdrant_client
	if _qdrant_client is None:
		path = pathlib.Path(QDRANT_LOCATION)
		if not path.is_absolute():
			path = (ROOT / path).resolve()
		path.mkdir(parents=True, exist_ok=True)
		_qdrant_client = QdrantClient(path=str(path))
	return _qdrant_client


def _reset_qdrant_client():
	global _qdrant_client
	_qdrant_client = None


def compute_stable_id(doc: Document) -> str:
	base = f"{doc.metadata.get('source') or ''}\x1f{doc.page_content}"
	return str(uuid.uuid5(uuid.NAMESPACE_URL, base))


def build_vector_store(docs, embeddings, recreate: bool):
	client = _get_qdrant_client()
	if recreate:
		if client.collection_exists(QDRANT_COLLECTION):
			client.delete_collection(collection_name=QDRANT_COLLECTION)
		dimension = len(embeddings.embed_query("dimension probe"))
		client.create_collection(
			collection_name=QDRANT_COLLECTION,
			vectors_config=VectorParams(size=dimension, distance=Distance.COSINE)
		)
	else:
		if not client.collection_exists(QDRANT_COLLECTION):
			dimension = len(embeddings.embed_query("dimension probe"))
			client.create_collection(
				collection_name=QDRANT_COLLECTION,
				vectors_config=VectorParams(size=dimension, distance=Distance.COSINE)
			)
	vector_store = QdrantVectorStore(
		client=client,
		collection_name=QDRANT_COLLECTION,
		embedding=embeddings
	)
	if docs:
		ids = [compute_stable_id(doc) for doc in docs]
		vector_store.add_documents(documents=docs, ids=ids)
	return vector_store


def _create_embeddings():
	embedding_model = os.environ.get("EMBEDDING_MODEL", "")
	if not embedding_model:
		raise RuntimeError("EMBEDDING_MODEL is not set")
	embeddings = HuggingFaceEmbeddings(
		model_name=embedding_model,
		model_kwargs={"device": EMBEDDING_DEVICE}
	)
	return embedding_model, embeddings


def init_store():
	embedding_model, embeddings = _create_embeddings()
	paths = compute_pdf_paths()
	texts = read_pdf_texts(paths)
	hash_value = compute_corpus_hash(texts, embedding_model)
	cache_path = _cache_file()
	previous_hash = cache_path.read_text(encoding="utf-8").strip() if cache_path.exists() else ""
	client = _get_qdrant_client()
	collection_exists = client.collection_exists(QDRANT_COLLECTION)
	rebuild = False
	if not collection_exists:
		rebuild = True
	elif hash_value != previous_hash:
		rebuild = True
	else:
		try:
			rebuild = client.count(collection_name=QDRANT_COLLECTION, exact=False).count == 0
		except Exception:
			rebuild = True
	if rebuild:
		documents = []
		for name, text in texts:
			if not text:
				continue
			documents.append(Document(page_content=text, metadata={"source": name}))
		if not documents:
			raise RuntimeError("No text extracted from PDFs")
		splitter = RecursiveCharacterTextSplitter()
		split_docs = splitter.split_documents(documents)
		vector_store = build_vector_store(split_docs, embeddings, recreate=True)
		cache_path.write_text(hash_value, encoding="utf-8")
		return embeddings, vector_store
	vector_store = build_vector_store(None, embeddings, recreate=False)
	return embeddings, vector_store


def init_store_from_pdfs():
	return init_store()


def init_store_from_existing():
	_, embeddings = _create_embeddings()
	client = _get_qdrant_client()
	if not client.collection_exists(QDRANT_COLLECTION):
		raise RuntimeError("Qdrant collection not found; run a PDF build first or upload a valid snapshot")
	vector_store = QdrantVectorStore(
		client=client,
		collection_name=QDRANT_COLLECTION,
		embedding=embeddings
	)
	return embeddings, vector_store


def restore_qdrant_from_zip(file_obj):
	target_dir = pathlib.Path(QDRANT_LOCATION).resolve()
	target_dir.mkdir(parents=True, exist_ok=True)
	for child in target_dir.iterdir():
		if child.is_dir():
			shutil.rmtree(child)
		else:
			child.unlink()
	with zipfile.ZipFile(file_obj) as archive:
		archive.extractall(target_dir)
	_reset_qdrant_client()
