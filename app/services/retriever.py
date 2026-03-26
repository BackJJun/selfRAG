import math

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DEFAULT_PDF_PATH,
    DEFAULT_TEXT_PATH,
    FILE_PATH,
    OPENAI_EMBEDDING_MODEL,
    TOP_K,
    logger,
)


def load_source_documents() -> list[Document]:
    candidate_paths = [FILE_PATH, DEFAULT_PDF_PATH, DEFAULT_TEXT_PATH]

    for path in candidate_paths:
        if not path.exists():
            logger.debug("Source path not found: %s", path)
            continue

        if path.suffix.lower() == ".pdf":
            logger.info("Loading PDF source: %s", path)
            return PyPDFLoader(str(path)).load()

        if path.suffix.lower() == ".txt":
            logger.info("Loading text source: %s", path)
            text = path.read_text(encoding="utf-8")
            return [Document(page_content=text, metadata={"source": str(path)})]

    searched = ", ".join(str(path) for path in candidate_paths)
    raise FileNotFoundError(f"No source document found. Checked: {searched}")


class LocalVectorRetriever:
    def __init__(self, documents: list[Document], embedding_model: OpenAIEmbeddings):
        self.embedding_model = embedding_model
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        self.chunks = splitter.split_documents(documents)
        chunk_texts = [chunk.page_content for chunk in self.chunks]
        logger.info("Embedding %d chunks for in-memory retriever", len(chunk_texts))
        self.chunk_embeddings = self.embedding_model.embed_documents(chunk_texts)
        logger.info("Prepared %d chunks for in-memory retrieval", len(self.chunks))

    @staticmethod
    def cosine_similarity(left: list[float], right: list[float]) -> float:
        dot = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return dot / (left_norm * right_norm)

    def invoke(self, query: str) -> list[Document]:
        logger.info("Retriever invoked | query=%r", query)
        query_embedding = self.embedding_model.embed_query(query)
        scored_chunks: list[tuple[float, Document]] = []

        for chunk, embedding in zip(self.chunks, self.chunk_embeddings):
            score = self.cosine_similarity(query_embedding, embedding)
            scored_chunks.append((score, chunk))

        scored_chunks.sort(key=lambda item: item[0], reverse=True)
        top_scored = scored_chunks[:TOP_K]
        logger.info(
            "Retriever finished | top_k=%d | scores=%s",
            TOP_K,
            [round(score, 4) for score, _ in top_scored],
        )
        return [chunk for _, chunk in top_scored]


def build_retriever() -> LocalVectorRetriever:
    documents = load_source_documents()
    embeddings_model = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)
    return LocalVectorRetriever(documents, embeddings_model)
