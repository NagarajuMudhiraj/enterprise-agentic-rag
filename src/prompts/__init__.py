"""Versioned prompt templates.

Prompts live in `.md` files next to this module rather than inside Python so
they can be reviewed, diffed, and rolled back independently of code changes.
`PROMPTS_VERSION` is stamped into responses so an answer can be traced back to
the prompt that produced it.
"""

from functools import lru_cache
from pathlib import Path

PROMPT_DIR = Path(__file__).resolve().parent
PROMPTS_VERSION = "2026.11.1"


@lru_cache(maxsize=None)
def load_prompt(name: str) -> str:
    """Load a template by file stem. Cached; templates are immutable at runtime."""
    path = PROMPT_DIR / f"{name}.md"
    if not path.is_file():
        raise FileNotFoundError(f"Prompt template '{name}' is missing at {path}")
    return path.read_text(encoding="utf-8").strip()


def list_prompts() -> list:
    return sorted(p.stem for p in PROMPT_DIR.glob("*.md"))
