from pathlib import Path

import pytest

from resume_agent.retrieval.dense import (
    DenseRetriever,
    corpus_fingerprint_for_evidence,
    load_or_build_dense_retriever,
)
from resume_agent.retrieval.simple import retrieve_for_jd
from resume_agent.schemas import JD
from resume_agent.schemas import Evidence, Requirement


class FakeEncoder:
    def __init__(self, vectors: dict[str, list[float]], model_name: str = "fake-bge-zh") -> None:
        self.vectors = vectors
        self.model_name = model_name
        self.document_calls = 0
        self.query_calls = 0

    def encode_documents(self, texts: list[str]):
        import numpy as np

        self.document_calls += 1
        return np.array([self.vectors[text] for text in texts], dtype="float32")

    def encode_query(self, text: str):
        import numpy as np

        self.query_calls += 1
        return np.array([self.vectors[text]], dtype="float32")


def _sample_evidence_pool() -> list[Evidence]:
    return [
        Evidence(
            id="e1",
            text="Python FAISS retrieval project",
            source_type="project",
            source_ref="project:1",
            skill_refs=["python", "faiss"],
        ),
        Evidence(
            id="e2",
            text="Computer vision internship",
            source_type="experience",
            source_ref="experience:1",
            skill_refs=["cv"],
        ),
        Evidence(
            id="e3",
            text="RAG system with BM25 and ranking",
            source_type="project",
            source_ref="project:2",
            skill_refs=["bm25", "rag"],
        ),
    ]


def test_dense_retriever_build_save_and_load(tmp_path: Path):
    evidence_pool = _sample_evidence_pool()
    vectors = {
        "Python FAISS retrieval project": [1.0, 0.0, 0.0],
        "Computer vision internship": [0.0, 1.0, 0.0],
        "RAG system with BM25 and ranking": [0.5, 0.0, 0.5],
        "Python retrieval": [1.0, 0.0, 0.0],
    }
    encoder = FakeEncoder(vectors)

    retriever = DenseRetriever(model_name=encoder.model_name, encoder=encoder).build_index(evidence_pool)
    retriever.save(tmp_path)

    loaded = DenseRetriever.load(
        tmp_path,
        evidence_pool,
        encoder,
        model_name=encoder.model_name,
    )
    result = loaded.retrieve(Requirement(text="Python retrieval"), top_k=1)

    assert retriever.embedding_dim == 3
    assert loaded.embedding_dim == 3
    assert loaded.corpus_fingerprint == corpus_fingerprint_for_evidence(evidence_pool)
    assert result[0].evidence_id == "e1"


def test_dense_retriever_reuses_single_built_index(tmp_path: Path):
    evidence_pool = _sample_evidence_pool()
    vectors = {
        "Python FAISS retrieval project": [1.0, 0.0, 0.0],
        "Computer vision internship": [0.0, 1.0, 0.0],
        "RAG system with BM25 and ranking": [0.5, 0.0, 0.5],
    }
    encoder = FakeEncoder(vectors)

    retriever = load_or_build_dense_retriever(
        evidence_pool,
        model_name=encoder.model_name,
        encoder=encoder,
        index_dir=tmp_path,
    )

    assert retriever.is_ready
    assert encoder.document_calls == 1

    second_retriever = load_or_build_dense_retriever(
        evidence_pool,
        model_name=encoder.model_name,
        encoder=encoder,
        index_dir=tmp_path,
    )

    assert second_retriever.is_ready
    assert encoder.document_calls == 1


def test_retrieve_for_jd_reuses_dense_index_for_all_requirements():
    evidence_pool = _sample_evidence_pool()
    vectors = {
        "Python FAISS retrieval project": [1.0, 0.0, 0.0],
        "Computer vision internship": [0.0, 1.0, 0.0],
        "RAG system with BM25 and ranking": [0.5, 0.0, 0.5],
        "Python retrieval": [1.0, 0.0, 0.0],
        "BM25 ranking": [0.5, 0.0, 0.5],
    }
    encoder = FakeEncoder(vectors)
    jd = JD(
        title="AI Intern",
        requirements=[
            Requirement(text="Python retrieval"),
            Requirement(text="BM25 ranking"),
        ],
    )
    from resume_agent.schemas import Resume

    resume = Resume(candidate_name="Bruce", evidence_pool=evidence_pool)

    report = retrieve_for_jd(
        jd,
        resume,
        top_k=1,
        retrieval_method="dense",
        dense_encoder=encoder,
        dense_model_name=encoder.model_name,
    )

    assert len(report.results) == 2
    assert encoder.document_calls == 1
    assert encoder.query_calls == 2


def test_dense_retriever_top_k_returns_requested_count():
    evidence_pool = _sample_evidence_pool()
    vectors = {
        "Python FAISS retrieval project": [1.0, 0.0, 0.0],
        "Computer vision internship": [0.0, 1.0, 0.0],
        "RAG system with BM25 and ranking": [0.7, 0.0, 0.3],
        "retrieval systems": [1.0, 0.0, 0.0],
    }
    encoder = FakeEncoder(vectors)
    retriever = DenseRetriever(model_name=encoder.model_name, encoder=encoder).build_index(evidence_pool)

    result = retriever.search_text("retrieval systems", top_k=2)

    assert len(result) == 2


def test_dense_retriever_handles_empty_corpus_and_empty_query():
    encoder = FakeEncoder({}, model_name="fake-bge-zh")
    retriever = DenseRetriever(model_name=encoder.model_name, encoder=encoder).build_index([])

    assert retriever.search_text("anything", top_k=3) == []
    assert retriever.search_text("", top_k=3) == []


def test_dense_retriever_rejects_stale_index_metadata(tmp_path: Path):
    evidence_pool = _sample_evidence_pool()
    vectors = {
        "Python FAISS retrieval project": [1.0, 0.0, 0.0],
        "Computer vision internship": [0.0, 1.0, 0.0],
        "RAG system with BM25 and ranking": [0.5, 0.0, 0.5],
    }
    encoder = FakeEncoder(vectors)
    retriever = DenseRetriever(model_name=encoder.model_name, encoder=encoder).build_index(evidence_pool)
    retriever.save(tmp_path)

    changed_corpus = evidence_pool + [
        Evidence(
            id="e4",
            text="new evidence",
            source_type="project",
            source_ref="project:4",
            skill_refs=[],
        )
    ]

    with pytest.raises(ValueError, match="corpus fingerprint mismatch"):
        DenseRetriever.load(
            tmp_path,
            changed_corpus,
            encoder,
            model_name=encoder.model_name,
        )
