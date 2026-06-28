"""Dense retrieval backend with sentence-transformers and FAISS."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Protocol

import faiss
import numpy as np

from resume_agent.retrieval.contracts import RetrievedEvidence
from resume_agent.retrieval.text import token_set
from resume_agent.schemas import Evidence, Requirement

INDEX_FILENAME = "index.faiss"
METADATA_FILENAME = "metadata.json"
DEFAULT_BGE_ZH_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："
DEFAULT_BGE_EN_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages:"


class EmbeddingEncoder(Protocol):
    """Encoder interface used by dense retrieval."""

    model_name: str

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        """Encode document texts into normalized dense vectors."""

    def encode_query(self, text: str) -> np.ndarray:
        """Encode a retrieval query into a normalized dense vector."""


class SentenceTransformerBGEEncoder:
    """Sentence-transformers encoder with BGE-aware query handling."""

    def __init__(
        self,
        model_name: str,
        query_instruction: str | None = None,
        device: str | None = None,
    ) -> None:
        if not model_name:
            raise ValueError("model_name is required for SentenceTransformerBGEEncoder")

        try:
            from sentence_transformers import SentenceTransformer
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "sentence-transformers is required for dense retrieval. "
                "Install it in the active environment before using dense or hybrid retrieval."
            ) from exc

        self.model_name = model_name
        self.query_instruction = query_instruction or _default_query_instruction(model_name)
        self._model = SentenceTransformer(model_name, device=device)

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype="float32")

        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return _ensure_2d_float32(embeddings)

    def encode_query(self, text: str) -> np.ndarray:
        query = text.strip()
        if not query:
            return np.empty((0, 0), dtype="float32")

        prompt = query if not self.query_instruction else f"{self.query_instruction}{query}"
        embedding = self._model.encode(
            [prompt],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return _ensure_2d_float32(embedding)


class DenseRetriever:
    """Dense retriever backed by normalized embeddings and FAISS IndexFlatIP."""

    def __init__(
        self,
        model_name: str,
        encoder: EmbeddingEncoder,
        *,
        evidence_pool: list[Evidence] | None = None,
        index: faiss.Index | None = None,
        embedding_dim: int | None = None,
        corpus_fingerprint: str | None = None,
    ) -> None:
        if not model_name:
            raise ValueError("model_name is required for DenseRetriever")

        self.model_name = model_name
        self.encoder = encoder
        self.evidence_pool = list(evidence_pool or [])
        self.index = index
        self.embedding_dim = embedding_dim
        self.corpus_fingerprint = corpus_fingerprint

    def build_index(self, evidence_pool: list[Evidence]) -> "DenseRetriever":
        """Build an in-memory FAISS index for the given evidence corpus."""

        self.evidence_pool = list(evidence_pool)
        self.corpus_fingerprint = corpus_fingerprint_for_evidence(self.evidence_pool)

        if not self.evidence_pool:
            self.index = None
            self.embedding_dim = None
            return self

        document_embeddings = self.encoder.encode_documents(
            [evidence.text for evidence in self.evidence_pool]
        )
        if document_embeddings.size == 0:
            self.index = None
            self.embedding_dim = None
            return self

        document_embeddings = _normalize_embeddings(document_embeddings)
        self.embedding_dim = int(document_embeddings.shape[1])
        index = faiss.IndexFlatIP(self.embedding_dim)
        index.add(document_embeddings)
        self.index = index
        return self

    def save(self, index_dir: str | Path) -> None:
        """Persist the FAISS index and metadata to disk."""

        if self.index is None or self.embedding_dim is None or self.corpus_fingerprint is None:
            raise ValueError("Cannot save a dense index before it has been built")

        target_dir = Path(index_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(target_dir / INDEX_FILENAME))
        metadata = {
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "corpus_fingerprint": self.corpus_fingerprint,
        }
        (target_dir / METADATA_FILENAME).write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(
        cls,
        index_dir: str | Path,
        evidence_pool: list[Evidence],
        encoder: EmbeddingEncoder,
        *,
        model_name: str,
    ) -> "DenseRetriever":
        """Load a persisted FAISS index and validate it against corpus and model metadata."""

        source_dir = Path(index_dir)
        metadata_path = source_dir / METADATA_FILENAME
        index_path = source_dir / INDEX_FILENAME
        if not metadata_path.exists() or not index_path.exists():
            raise FileNotFoundError(f"Dense index files not found in {source_dir}")

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected_fingerprint = corpus_fingerprint_for_evidence(evidence_pool)
        if metadata.get("model_name") != model_name:
            raise ValueError(
                f"Dense index model mismatch: expected {model_name}, found {metadata.get('model_name')}"
            )
        if metadata.get("corpus_fingerprint") != expected_fingerprint:
            raise ValueError("Dense index corpus fingerprint mismatch")

        index = faiss.read_index(str(index_path))
        return cls(
            model_name=model_name,
            encoder=encoder,
            evidence_pool=evidence_pool,
            index=index,
            embedding_dim=int(metadata["embedding_dim"]),
            corpus_fingerprint=expected_fingerprint,
        )

    def retrieve(self, requirement: Requirement, top_k: int) -> list[RetrievedEvidence]:
        """Retrieve top-k evidence items for a structured requirement."""

        return self.search_text(
            requirement.text,
            top_k=top_k,
            requirement_skill_refs=requirement.skill_refs,
        )

    def search_text(
        self,
        query_text: str,
        *,
        top_k: int,
        requirement_skill_refs: list[str] | None = None,
    ) -> list[RetrievedEvidence]:
        """Retrieve top-k evidence items for a raw query string."""

        query = query_text.strip()
        if not query or not self.evidence_pool or self.index is None or self.embedding_dim is None:
            return []

        query_embedding = self.encoder.encode_query(query)
        if query_embedding.size == 0:
            return []

        query_embedding = _normalize_embeddings(query_embedding)
        limit = min(max(top_k, 1), len(self.evidence_pool))
        distances, indices = self.index.search(query_embedding, limit)

        results: list[RetrievedEvidence] = []
        for score, evidence_index in zip(distances[0], indices[0], strict=False):
            if evidence_index < 0:
                continue
            evidence = self.evidence_pool[int(evidence_index)]
            results.append(
                _build_candidate(
                    query_text=query,
                    requirement_skill_refs=requirement_skill_refs or [],
                    evidence=evidence,
                    inner_product=float(score),
                )
            )
        return results

    @property
    def is_ready(self) -> bool:
        return self.index is not None and self.embedding_dim is not None


def load_or_build_dense_retriever(
    evidence_pool: list[Evidence],
    *,
    model_name: str,
    encoder: EmbeddingEncoder,
    index_dir: str | Path | None = None,
) -> DenseRetriever:
    """Load a compatible dense index if available, otherwise build it once."""

    if index_dir is not None:
        source_dir = Path(index_dir)
        if (source_dir / INDEX_FILENAME).exists() and (source_dir / METADATA_FILENAME).exists():
            return DenseRetriever.load(
                source_dir,
                evidence_pool,
                encoder,
                model_name=model_name,
            )

    retriever = DenseRetriever(model_name=model_name, encoder=encoder).build_index(evidence_pool)
    if index_dir is not None and retriever.is_ready:
        retriever.save(index_dir)
    return retriever


def corpus_fingerprint_for_evidence(evidence_pool: list[Evidence]) -> str:
    """Compute a stable fingerprint for an ordered evidence corpus."""

    payload = [
        {
            "id": evidence.id,
            "text": evidence.text,
            "source_ref": evidence.source_ref,
            "skill_refs": evidence.skill_refs,
        }
        for evidence in evidence_pool
    ]
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _build_candidate(
    *,
    query_text: str,
    requirement_skill_refs: list[str],
    evidence: Evidence,
    inner_product: float,
) -> RetrievedEvidence:
    matched_terms = sorted(token_set(query_text) & token_set(evidence.text))
    if requirement_skill_refs:
        matched_terms = sorted(set(matched_terms) | (set(requirement_skill_refs) & set(evidence.skill_refs)))

    normalized_score = _normalize_inner_product(inner_product)
    return RetrievedEvidence(
        evidence_id=evidence.id,
        score=round(normalized_score, 4),
        retrieval_score=round(normalized_score, 4),
        rerank_score=None,
        matched_terms=matched_terms,
        evidence=evidence,
    )


def _normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    array = _ensure_2d_float32(embeddings)
    if array.size == 0:
        return array
    faiss.normalize_L2(array)
    return array


def _ensure_2d_float32(embeddings: np.ndarray) -> np.ndarray:
    array = np.asarray(embeddings, dtype="float32")
    if array.ndim == 1:
        array = array.reshape(1, -1)
    return array


def _normalize_inner_product(inner_product: float) -> float:
    # With normalized vectors, inner product is cosine similarity in [-1, 1].
    # We map it monotonically into [0, 1] to preserve ranking while keeping the
    # current retrieval contract's non-negative score constraint.
    return max(0.0, min(1.0, (inner_product + 1.0) / 2.0))


def _default_query_instruction(model_name: str) -> str:
    lowered = model_name.lower()
    if "bge" in lowered and ("-zh" in lowered or "_zh" in lowered):
        return DEFAULT_BGE_ZH_QUERY_INSTRUCTION
    if "bge" in lowered:
        return DEFAULT_BGE_EN_QUERY_INSTRUCTION + " "
    return ""
