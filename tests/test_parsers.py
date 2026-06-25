from pathlib import Path

from resume_agent.parsers import parse_jd_file, parse_jd_text, parse_resume_file, parse_resume_text


def test_parse_jd_text_extracts_requirements_and_skills():
    jd = parse_jd_text(
        "\n".join(
            [
                "AI Intern",
                "Company: Example",
                "",
                "Requirements:",
                "- Strong Python and machine learning background",
                "- Good communication skills",
            ]
        )
    )

    assert jd.title == "AI Intern"
    assert len(jd.requirements) == 2
    assert any(skill.normalized_name == "python" for skill in jd.hard_skills)
    assert "communication" in jd.soft_signals


def test_parse_resume_text_builds_evidence_pool():
    resume = parse_resume_text(
        "\n".join(
            [
                "Bruce",
                "",
                "Projects",
                "Resume Agent",
                "- Built a Python and FAISS prototype",
                "",
                "Skills",
                "Python, FAISS",
            ]
        )
    )

    assert resume.candidate_name == "Bruce"
    assert resume.projects[0].name == "Resume Agent"
    assert resume.evidence_pool[0].source_type == "project"
    assert any(skill.normalized_name == "faiss" for skill in resume.skills)


def test_parse_files_support_sample_txt_inputs():
    project_root = Path(__file__).resolve().parents[1]

    jd = parse_jd_file(project_root / "data" / "raw" / "jd" / "sample_jd.txt")
    resume = parse_resume_file(project_root / "data" / "raw" / "resume" / "sample_resume.txt")

    assert jd.company == "Example AI Lab"
    assert resume.candidate_name == "Bruce Lee"
    assert len(resume.evidence_pool) >= 2
