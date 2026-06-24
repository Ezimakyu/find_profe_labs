from __future__ import annotations

from pathlib import Path

from scraper.llm_client import LLMClient

QUERY_SYSTEM_PROMPT = """You answer questions about a lab corpus.

Rules:
- Base every claim on provided corpus text.
- Preserve specificity from source text.
- If uncertain, say so.
- Cite supporting URLs or file hints when available.
"""


def _load_corpus_text(lab_dir: Path, max_chars: int = 120000) -> str:
    pages_dir = lab_dir / "pages"
    chunks: list[str] = []
    total = 0
    for path in sorted(pages_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        block = f"\n---\nFile: {path.name}\n{text}\n"
        total += len(block)
        if total > max_chars:
            break
        chunks.append(block)
    return "".join(chunks)


def keyword_frequency(lab_dir: Path, keyword: str) -> int:
    corpus = _load_corpus_text(lab_dir, max_chars=300000).lower()
    return corpus.count(keyword.lower())


def ask_lab_question(llm: LLMClient, lab_dir: Path, question: str) -> str:
    corpus = _load_corpus_text(lab_dir)
    user_prompt = f"""
Question:
{question}

Corpus:
{corpus}
"""
    return llm.text_response(
        system_prompt=QUERY_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        max_output_tokens=2000,
    )

