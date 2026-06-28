from resume_agent.retrieval.contracts import RetrievedEvidence
from resume_agent.retrieval.dense import DenseRetriever
from resume_agent.retrieval.hybrid import HybridRetriever, rrf_fuse_candidates
from resume_agent.schemas import Evidence, Requirement


class FakeEncoder:
    def __init__(self, vectors: dict[str, list[float]], model_name: str = "fake-bge-zh") -> None:
        self.vectors = vectors
        self.model_name = model_name

    def encode_documents(self, texts: list[str]):
        import numpy as np

        return np.array([self.vectors[text] for text in texts], dtype="float32")

    def encode_query(self, text: str):
        import numpy as np

        return np.array([self.vectors[text]], dtype="float32")


def test_rrf_fuse_candidates_deduplicates_and_scores():
    evidence = Evidence(
        id="e1",
        text="Python retrieval systems",
        source_type="project",
        source_ref="project:1",
        skill_refs=["python"],
    )
    other = Evidence(
        id="e2",
        text="Computer vision systems",
        source_type="project",
        source_ref="project:2",
        skill_refs=["cv"],
    )

    list_one = [
        RetrievedEvidence(evidence_id="e1", score=1.0, retrieval_score=1.0, matched_terms=["python"], evidence=evidence),
        RetrievedEvidence(evidence_id="e2", score=0.8, retrieval_score=0.8, matched_terms=["systems"], evidence=other),
    ]
    list_two = [
        RetrievedEvidence(evidence_id="e1", score=0.7, retrieval_score=0.7, matched_terms=["retrieval"], evidence=evidence),
    ]

    fused = rrf_fuse_candidates([list_one, list_two], rrf_k=60)

    assert len(fused) == 2
    assert fused[0].evidence_id == "e1"
    assert fused[0].matched_terms == ["python", "retrieval"]
    assert fused[0].retrieval_score == round((1 / 61) + (1 / 61), 6)


def test_hybrid_retriever_deduplicates_bm25_and_dense_candidates():
    evidence_pool = [
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
    ]
    vectors = {
        "Python FAISS retrieval project": [1.0, 0.0],
        "Computer vision internship": [0.0, 1.0],
        "Python retrieval": [1.0, 0.0],
    }
    encoder = FakeEncoder(vectors)
    dense_retriever = DenseRetriever(model_name=encoder.model_name, encoder=encoder).build_index(evidence_pool)
    hybrid = HybridRetriever(dense_retriever=dense_retriever, rrf_k=60)

    result = hybrid.retrieve(
        Requirement(text="Python retrieval", skill_refs=["python"]),
        evidence_pool,
        candidate_k=5,
    )

    assert len(result) == 2
    assert result[0].evidence_id == "e1"


def test_hybrid_retriever_handles_empty_corpus_and_empty_query():
    encoder = FakeEncoder({}, model_name="fake-bge-zh")
    dense_retriever = DenseRetriever(model_name=encoder.model_name, encoder=encoder).build_index([])
    hybrid = HybridRetriever(dense_retriever=dense_retriever, rrf_k=60)

    assert hybrid.retrieve(Requirement(text="retrieval"), [], candidate_k=5) == []
    assert hybrid.search_text("", [], candidate_k=5) == []
