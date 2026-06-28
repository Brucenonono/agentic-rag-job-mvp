from resume_agent.retrieval import retrieve_for_jd, retrieve_for_requirement
from resume_agent.schemas import Evidence, JD, Requirement, Resume


def test_retrieve_for_requirement_prefers_more_relevant_evidence():
    requirement = Requirement(
        text="Experience with Python and FAISS retrieval systems",
        category="hard_skill",
        importance="must",
        skill_refs=["python", "faiss"],
    )
    evidence_pool = [
        Evidence(
            id="project-1",
            text="Built a Python and FAISS prototype for document search.",
            source_type="project",
            source_ref="project:1",
            skill_refs=["python", "faiss"],
        ),
        Evidence(
            id="project-2",
            text="Prepared presentation slides for a campus event.",
            source_type="project",
            source_ref="project:2",
            skill_refs=[],
        ),
    ]

    result = retrieve_for_requirement(requirement, evidence_pool, top_k=1, candidate_k=2)

    assert len(result.retrieved) == 1
    assert result.candidate_count == 1
    assert result.retrieved[0].evidence_id == "project-1"
    assert "python" in result.retrieved[0].matched_terms
    assert result.retrieved[0].retrieval_score is not None
    assert result.retrieved[0].rerank_score is not None


def test_retrieve_for_jd_returns_one_result_per_requirement():
    jd = JD(
        title="AI Intern",
        requirements=[
            Requirement(text="Python experience", category="hard_skill", skill_refs=["python"]),
            Requirement(text="Clear communication", category="soft_skill"),
        ],
    )
    resume = Resume(
        candidate_name="Bruce",
        evidence_pool=[
            Evidence(
                id="exp-1",
                text="Built Python tools for experiment analysis.",
                source_type="experience",
                source_ref="experience:1",
                skill_refs=["python"],
            ),
            Evidence(
                id="exp-2",
                text="Presented results to cross-functional teammates.",
                source_type="experience",
                source_ref="experience:2",
                skill_refs=[],
            ),
        ],
    )

    report = retrieve_for_jd(jd, resume, top_k=2, candidate_k=4)

    assert report.jd_title == "AI Intern"
    assert report.candidate_name == "Bruce"
    assert report.retrieval_method == "lexical_overlap"
    assert report.rerank_method == "heuristic_rerank"
    assert len(report.results) == 2
    assert report.results[0].requirement_index == 0


def test_bm25_backend_returns_relevant_candidate():
    requirement = Requirement(
        text="Experience with Python and FAISS retrieval systems",
        category="hard_skill",
        importance="must",
        skill_refs=["python", "faiss"],
    )
    evidence_pool = [
        Evidence(
            id="project-1",
            text="Built a Python and FAISS prototype for document search.",
            source_type="project",
            source_ref="project:1",
            skill_refs=["python", "faiss"],
        ),
        Evidence(
            id="project-2",
            text="Prepared presentation slides for a campus event.",
            source_type="project",
            source_ref="project:2",
            skill_refs=[],
        ),
    ]

    result = retrieve_for_requirement(
        requirement,
        evidence_pool,
        top_k=1,
        candidate_k=2,
        use_rerank=False,
        retrieval_method="bm25",
    )

    assert result.retrieved[0].evidence_id == "project-1"
    assert result.retrieved[0].retrieval_score is not None


def test_rerank_prefers_more_focused_evidence_when_coarse_scores_tie():
    requirement = Requirement(
        text="retrieval systems",
        category="hard_skill",
        importance="must",
    )
    evidence_pool = [
        Evidence(
            id="project-2",
            text="Built retrieval systems for experiment search, dashboards, logging, and reports.",
            source_type="project",
            source_ref="project:2",
            skill_refs=[],
        ),
        Evidence(
            id="project-1",
            text="Built retrieval systems.",
            source_type="project",
            source_ref="project:1",
            skill_refs=[],
        ),
    ]

    without_rerank = retrieve_for_requirement(
        requirement,
        evidence_pool,
        top_k=2,
        candidate_k=2,
        use_rerank=False,
    )
    with_rerank = retrieve_for_requirement(
        requirement,
        evidence_pool,
        top_k=2,
        candidate_k=2,
        use_rerank=True,
    )

    assert without_rerank.retrieved[0].retrieval_score == without_rerank.retrieved[0].score
    assert without_rerank.retrieved[0].evidence_id == "project-2"
    assert with_rerank.retrieved[0].evidence_id == "project-1"
    assert with_rerank.retrieved[0].rerank_score == with_rerank.retrieved[0].score
