"""CLI entrypoint for the first runnable project flow."""

from __future__ import annotations

from pathlib import Path

import typer

from resume_agent.parsers import parse_jd_file, parse_resume_file

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


def _write_json(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload + "\n", encoding="utf-8")
