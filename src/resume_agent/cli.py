"""CLI entrypoint for the first runnable project flow."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import typer

from resume_agent.parsers import parse_jd_file, parse_resume_file
from resume_agent.retrieval import retrieve_for_jd

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command("parse-jd")
def parse_jd_command(
    input_path: Path = typer.Argument(..., exists=True, readable=True),
    output_path: Path = typer.Option(
        Path("data/processed/jd.normalized.json"),
        "--output",
        "-o",
        help="Where to write the normalized JD JSON.",
    ),
) -> None:
    """Parse one JD file into the normalized schema."""

    jd = parse_jd_file(input_path)
    _write_json(output_path, jd.model_dump_json(indent=2))
    typer.echo(f"Normalized JD written to {output_path}")


@app.command("parse-resume")
def parse_resume_command(
    input_path: Path = typer.Argument(..., exists=True, readable=True),
    output_path: Path = typer.Option(
        Path("data/processed/resume.normalized.json"),
        "--output",
        "-o",
        help="Where to write the normalized resume JSON.",
    ),
) -> None:
    """Parse one resume file into the normalized schema."""

    resume = parse_resume_file(input_path)
    _write_json(output_path, resume.model_dump_json(indent=2))
    typer.echo(f"Normalized resume written to {output_path}")


@app.command("parse-both")
def parse_both_command(
    jd_path: Path = typer.Option(..., "--jd", exists=True, readable=True, help="JD input file."),
    resume_path: Path = typer.Option(
        ..., "--resume", exists=True, readable=True, help="Resume input file."
    ),
    output_dir: Path = typer.Option(
        Path("data/processed"),
        "--output-dir",
        help="Directory for normalized artifacts.",
    ),
) -> None:
    """Parse one JD and one resume in a single command."""

    output_dir.mkdir(parents=True, exist_ok=True)
    jd = parse_jd_file(jd_path)
    resume = parse_resume_file(resume_path)
    _write_json(output_dir / "jd.normalized.json", jd.model_dump_json(indent=2))
    _write_json(output_dir / "resume.normalized.json", resume.model_dump_json(indent=2))
    typer.echo(f"Normalized artifacts written to {output_dir}")


@app.command("retrieve")
def retrieve_command(
    jd_path: Path = typer.Option(..., "--jd", exists=True, readable=True, help="JD input file."),
    resume_path: Path = typer.Option(
        ..., "--resume", exists=True, readable=True, help="Resume input file."
    ),
    method: Literal["lexical_overlap", "bm25", "dense", "hybrid"] = typer.Option(
        "lexical_overlap",
        "--method",
        help="Coarse retrieval backend to use before reranking.",
    ),
    dense_model_name: str | None = typer.Option(
        None,
        "--dense-model-name",
        help="Dense embedding model name. Required for dense or hybrid retrieval unless configured via env.",
    ),
    dense_index_dir: Path | None = typer.Option(
        None,
        "--dense-index-dir",
        help="Optional FAISS index directory to load or save for dense retrieval.",
    ),
    rrf_k: int = typer.Option(
        60,
        "--rrf-k",
        min=1,
        help="RRF constant used by the hybrid retriever.",
    ),
    top_k: int = typer.Option(3, "--top-k", min=1, help="Number of evidence candidates per JD requirement."),
    candidate_k: int = typer.Option(
        9,
        "--candidate-k",
        min=1,
        help="Number of coarse retrieval candidates to keep before reranking.",
    ),
    use_rerank: bool = typer.Option(
        True,
        "--rerank/--no-rerank",
        help="Whether to apply the rerank stage after coarse retrieval.",
    ),
    output_path: Path = typer.Option(
        Path("data/processed/retrieval.json"),
        "--output",
        "-o",
        help="Where to write the retrieval report JSON.",
    ),
) -> None:
    """Retrieve top-k resume evidence for each JD requirement."""

    jd = parse_jd_file(jd_path)
    resume = parse_resume_file(resume_path)
    retrieval_report = retrieve_for_jd(
        jd,
        resume,
        top_k=top_k,
        candidate_k=candidate_k,
        use_rerank=use_rerank,
        retrieval_method=method,
        dense_model_name=dense_model_name,
        dense_index_dir=dense_index_dir,
        hybrid_rrf_k=rrf_k,
    )
    _write_json(output_path, retrieval_report.model_dump_json(indent=2))
    typer.echo(f"Retrieval report written to {output_path}")


def _write_json(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload + "\n", encoding="utf-8")
