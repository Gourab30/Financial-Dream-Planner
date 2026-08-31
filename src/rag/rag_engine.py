"""
Fully local RAG (Retrieval-Augmented Generation) engine.

Design choice (documented): instead of calling a paid embeddings API, chunks
are embedded locally with scikit-learn's TfidfVectorizer, and the "vector
store" is simply the in-memory TF-IDF matrix. This satisfies the assignment
requirement (chunk -> embed -> store -> retrieve -> ground the answer) while
staying 100% local and free, matching rule "No paid API is required."
"""
import os
import glob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

KB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledge_base")

# Below this similarity score we treat the knowledge base as NOT containing the
# answer, and say so rather than letting an LLM invent one (hallucination control).
SIMILARITY_THRESHOLD = 0.12
# A single coincidentally-shared word (e.g. "capital" matching "capital-preservation")
# can otherwise produce a misleadingly high cosine score on a short query. Requiring
# at least MIN_SHARED_TERMS distinct vocabulary terms in common is a cheap, effective
# extra guard against that specific false-positive pattern.
MIN_SHARED_TERMS = 2
CHUNK_SIZE_CHARS = 400
CHUNK_OVERLAP_CHARS = 60


def _chunk_text(text: str, source: str) -> list:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    for para in paragraphs:
        if len(para) <= CHUNK_SIZE_CHARS:
            chunks.append({"text": para, "source": source})
        else:
            start = 0
            while start < len(para):
                end = start + CHUNK_SIZE_CHARS
                chunks.append({"text": para[start:end], "source": source})
                start = end - CHUNK_OVERLAP_CHARS
    return chunks


class RAGEngine:
    def __init__(self, kb_dir: str = KB_DIR):
        self.kb_dir = kb_dir
        self.chunks = []
        self._load_and_index()

    def _load_and_index(self):
        for path in sorted(glob.glob(os.path.join(self.kb_dir, "*.txt"))):
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            self.chunks.extend(_chunk_text(text, os.path.basename(path)))

        if not self.chunks:
            raise RuntimeError(f"No knowledge base documents found in {self.kb_dir}")

        corpus = [c["text"] for c in self.chunks]
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(corpus).toarray()

    def retrieve(self, query: str, top_k: int = 3) -> list:
        query_vec = self.vectorizer.transform([query]).toarray()[0]
        query_terms_present = query_vec > 0
        n_query_terms = int(query_terms_present.sum())

        scores = cosine_similarity([query_vec], self.matrix)[0]
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for i in ranked:
            if scores[i] < SIMILARITY_THRESHOLD:
                continue
            shared_terms = int(((self.matrix[i] > 0) & query_terms_present).sum())
            # Always require >= MIN_SHARED_TERMS distinct vocabulary words in common.
            # A single shared word (e.g. "capital" matching "capital-preservation", or
            # "today" matching "today's price") is not enough evidence that a short,
            # generic-sounding query is actually answerable from this knowledge base -
            # this is what stops the Agent from grounding "What is the capital of
            # France?" or "What's the weather today?" in unrelated finance chunks.
            if shared_terms < MIN_SHARED_TERMS or n_query_terms < MIN_SHARED_TERMS:
                continue
            results.append({
                "text": self.chunks[i]["text"],
                "source": self.chunks[i]["source"],
                "score": round(float(scores[i]), 4),
            })
        return results

    def answer(self, query: str, top_k: int = 3) -> dict:
        """Returns a grounded answer built ONLY from retrieved chunks, or an
        explicit 'not available' response if nothing clears the similarity bar."""
        retrieved = self.retrieve(query, top_k=top_k)
        if not retrieved:
            return {
                "answer": "This information is not available in the project knowledge base.",
                "grounded": False,
                "sources": [],
            }
        combined = " ".join(r["text"] for r in retrieved)
        return {
            "answer": combined,
            "grounded": True,
            "sources": [{"source": r["source"], "score": r["score"]} for r in retrieved],
        }
