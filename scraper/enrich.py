from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlparse

from scraper.config import Settings
from scraper.fetch import Fetcher
from scraper.models import FacultyExtraction, LabCandidate
from scraper.preprocess import clean_html_to_text


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
    return slug or "lab"


def _urls_for_lab(lab: LabCandidate, extractions: list[FacultyExtraction]) -> list[str]:
    urls = set()
    if lab.canonical_url:
        urls.add(lab.canonical_url)
    for extraction in extractions:
        matched = any(name.lower() == lab.lab_name.lower() for name in extraction.inferred_labs)
        if matched:
            urls.add(extraction.professor_url)
            urls.update(str(item) for item in extraction.inferred_lab_urls if item)
    return sorted(urls)


def enrich_single_lab(
    settings: Settings,
    fetcher: Fetcher,
    lab: LabCandidate,
    extractions: list[FacultyExtraction],
) -> dict:
    slug = _slugify(lab.lab_name)
    base_dir = settings.output_dir / "lab_test" / slug
    pages_dir = base_dir / "pages"
    raw_dir = base_dir / "raw_html"
    pages_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    urls = _urls_for_lab(lab, extractions)[: settings.max_enrich_pages]
    saved_pages: list[dict] = []
    blocked_urls: list[dict] = []

    for idx, url in enumerate(urls, start=1):
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
                "used_playwright": result.used_playwright,
            }
        )

    return {
        "lab_name": lab.lab_name,
        "lab_url": lab.canonical_url,
        "slug": slug,
        "saved_pages": saved_pages,
        "blocked_urls": blocked_urls,
        "output_dir": str(base_dir),
    }

