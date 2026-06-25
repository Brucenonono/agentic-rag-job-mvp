"""Root-level runner for the minimal CLI."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from resume_agent.cli import app


if __name__ == "__main__":
    app()
