import pytest
from pydantic import ValidationError

from resume_agent.schemas import Evidence, JD, MatchDimension, MatchScore, Requirement, Resume, Skill


def test_skill_normalized_name_is_derived():
    skill = Skill(name="PyTorch Lightning", source="resume")

    assert skill.normalized_name == "pytorch lightning"


def test_jd_and_resume_models_accept_minimal_payloads():
    jd = JD(
        title="AI Intern",
        requirements=[
            Requirement(
                text="Experience with Python and deep learning",
                category="hard_skill",
                importance="must",
                skill_refs=["python", "deep learning"],
            )
        ],
    )
    resume = Resume(
        candidate_name="Bruce",
        skills=[Skill(name="Python", category="programming_language", source="resume")],
        evidence_pool=[
            Evidence(
                id="proj-1",
                text="Built a resume ranking model with Python.",
                source_type="project",
                source_ref="project:resume-ranker",
                skill_refs=["python"],
            )
        ],
    )

    assert jd.title == "AI Intern"
    assert resume.candidate_name == "Bruce"
    assert resume.evidence_pool[0].source_ref == "project:resume-ranker"


def test_match_score_rejects_duplicate_dimension_names():
    with pytest.raises(ValidationError):
        MatchScore(
            overall_score=0.8,
            dimension_scores=[
                MatchDimension(name="skills", score=0.8, reason="Good fit"),
                MatchDimension(name="skills", score=0.7, reason="Duplicate"),
            ],
        )
