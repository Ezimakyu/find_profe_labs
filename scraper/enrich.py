from __future__ import annotations

import re
from collections import deque
from pathlib import Path
from urllib.parse import urlparse

from pydantic import BaseModel, Field, field_validator

from scraper.config import Settings
from scraper.fetch import Fetcher
from scraper.links import extract_internal_links, registrable_host
from scraper.llm_client import LLMClient
from scraper.models import FacultyExtraction, LabCandidate, coerce_optional_str, coerce_str_list
from scraper.preprocess import clean_html_to_text


class LabResearchSummary(BaseModel):
    mission: str | None = None
    research_activities: list[str] = Field(default_factory=list)
    methods_and_systems: list[str] = Field(default_factory=list)
    application_domains: list[str] = Field(default_factory=list)
    representative_projects: list[str] = Field(default_factory=list)
    recruitment_notes: list[str] = Field(default_factory=list)

    @field_validator("mission", mode="before")
    @classmethod
    def _coerce_mission(cls, v):
        return coerce_optional_str(v)

    @field_validator(
        "research_activities",
        "methods_and_systems",
        "application_domains",
        "representative_projects",
        "recruitment_notes",
        mode="before",
    )
    @classmethod
    def _coerce_lists(cls, v):
        return coerce_str_list(v)


LAB_SUMMARY_SYSTEM_PROMPT = """You summarize what a research lab ACTUALLY does from its own website text.

Rules:
- Return JSON only, grounded strictly in the provided corpus.
- Focus on concrete research activities, methods/systems built, and application
  domains. Describe specific projects when named.
- DE-EMPHASIZE awards, honors, press, and rankings. Do not list them as activities.
- Prefer specificity over umbrella terms (say what problems/systems, not just "AI").
- recruitment_notes: include only explicit mentions of openings/joining/contact-for-positions.
"""


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
    return slug or "lab"


def _seed_urls_for_lab(lab: LabCandidate, extractions: list[FacultyExtraction]) -> list[str]:
    """Best starting points for the lab: canonical/lab URLs, personal sites, profiles."""
    lab_urls: list[str] = []
    personal_urls: list[str] = []
    profile_urls: list[str] = []
    if lab.canonical_url:
        lab_urls.append(lab.canonical_url)
    for extraction in extractions:
        matched = any(name.lower() == lab.lab_name.lower() for name in extraction.inferred_labs)
        if not matched:
            continue
        lab_urls.extend(str(item) for item in extraction.inferred_lab_urls if item)
        personal_urls.extend(extraction.personal_site_urls)
        profile_urls.append(extraction.professor_url)
    # Prefer dedicated lab/personal sites first; profiles are a last resort.
    ordered: list[str] = []
    for url in lab_urls + personal_urls + profile_urls:
        if url and url not in ordered:
            ordered.append(url)
    return ordered


def _recursive_crawl(
    settings: Settings,
    fetcher: Fetcher,
    seeds: list[str],
    pages_dir: Path,
    raw_dir: Path,
) -> tuple[list[dict], list[dict]]:
    """BFS crawl that stays within each seed's registrable domain.

    Returns ``(saved_pages, blocked_urls)``. Internal links are followed up to
    ``enrich_max_depth`` and ``max_enrich_pages`` to capture the lab's actual
    research content (projects/people/publications) rather than a single page.
    """
    seed_domains = {registrable_host(url) for url in seeds}
    queue: deque[tuple[str, int]] = deque((url, 0) for url in seeds)
    visited: set[str] = set()
    saved_pages: list[dict] = []
    blocked_urls: list[dict] = []

    while queue and len(saved_pages) < settings.max_enrich_pages:
        url, depth = queue.popleft()
        if url in visited:
            continue
        visited.add(url)

        result = fetcher.fetch_html(url)
        if not result.ok:
            blocked_urls.append(
                {
                    "url": url,
                    "reason": result.blocked_reason or "fetch_failed",
                    "status_code": result.status_code,
                }
            )
            continue

        text = clean_html_to_text(result.html)
        idx = len(saved_pages) + 1
        host = urlparse(url).netloc.replace(".", "_")
        page_path = pages_dir / f"{idx:03d}_{host}.txt"
        raw_path = raw_dir / f"{idx:03d}_{host}.html"
        page_path.write_text(text, encoding="utf-8")
        raw_path.write_text(result.html, encoding="utf-8")
        saved_pages.append(
            {
                "url": url,
                "path": str(page_path),
                "raw_html_path": str(raw_path),
                "chars": len(text),
                "depth": depth,
                "used_playwright": result.used_playwright,
            }
        )

        if depth >= settings.enrich_max_depth:
            continue
        for link in extract_internal_links(
            result.html,
            url,
            same_registrable_domain=True,
            skip_boilerplate=True,
            prioritize_research=True,
        ):
            if link not in visited and registrable_host(link) in seed_domains:
                queue.append((link, depth + 1))

    return saved_pages, blocked_urls


def _summarize_lab_research(llm: LLMClient, lab: LabCandidate, pages_dir: Path) -> dict:
    """LLM pass over the crawled corpus focused on research activities, not awards."""
    chunks: list[str] = []
    total = 0
    for path in sorted(pages_dir.glob("*.txt")):
        block = f"\n---\nFile: {path.name}\n{path.read_text(encoding='utf-8', errors='ignore')}\n"
        total += len(block)
        if total > 90000:
            break
        chunks.append(block)
    corpus = "".join(chunks)
    if not corpus.strip():
        return LabResearchSummary().model_dump()

    user_prompt = f"""
Lab name: {lab.lab_name}
Lab URL: {lab.canonical_url or "(unknown)"}

Corpus (the lab's own pages):
{corpus}

Return JSON with keys: mission, research_activities, methods_and_systems,
application_domains, representative_projects, recruitment_notes.
"""
    summary = llm.json_response(
        system_prompt=LAB_SUMMARY_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        schema=LabResearchSummary,
        max_output_tokens=2500,
    )
    return summary.model_dump()


def enrich_single_lab(
    settings: Settings,
    fetcher: Fetcher,
    lab: LabCandidate,
    extractions: list[FacultyExtraction],
    llm: LLMClient | None = None,
) -> dict:
    slug = _slugify(lab.lab_name)
    base_dir = settings.output_dir / "lab_test" / slug
    pages_dir = base_dir / "pages"
    raw_dir = base_dir / "raw_html"
    pages_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    seeds = _seed_urls_for_lab(lab, extractions)
    saved_pages, blocked_urls = _recursive_crawl(settings, fetcher, seeds, pages_dir, raw_dir)

    research_summary: dict | None = None
    if llm is not None and saved_pages:
        research_summary = _summarize_lab_research(llm, lab, pages_dir)

    return {
        "lab_name": lab.lab_name,
        "lab_url": lab.canonical_url,
        "slug": slug,
        "seed_urls": seeds,
        "research_summary": research_summary,
        "saved_pages": saved_pages,
        "blocked_urls": blocked_urls,
        "output_dir": str(base_dir),
    }
