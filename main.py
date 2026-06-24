from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from scraper.config import load_settings
from scraper.discovery import discover_faculty
from scraper.enrich import enrich_single_lab
from scraper.extract_llm import extract_faculty_info
from scraper.fetch import Fetcher
from scraper.llm_client import LLMClient
from scraper.output import (
    build_lab_candidates,
    ensure_dirs,
    write_blocked_urls_csv,
    write_faculty_lab_links_csv,
    write_json,
    write_labs_csv,
)
from scraper.preprocess import clean_html_to_text
from scraper.query_llm import ask_lab_question, keyword_frequency


def run_pipeline(one_lab_name: str | None = None) -> None:
    settings = load_settings()
    ensure_dirs(settings.output_dir)
    fetcher = Fetcher(settings)
    llm = LLMClient(settings)

    faculty_records, blocked_urls = discover_faculty(settings, fetcher)
    extractions = []

    for faculty in faculty_records:
        result = fetcher.fetch_html(faculty.url)
        if not result.ok:
            blocked_urls.append(
                {
                    "url": faculty.url,
                    "reason": result.blocked_reason or "faculty_fetch_failed",
                    "status_code": result.status_code,
                }
            )
            continue
        cleaned = clean_html_to_text(result.html)
        if not cleaned.strip():
            blocked_urls.append(
                {
                    "url": faculty.url,
                    "reason": "empty_cleaned_text",
                    "status_code": result.status_code,
                }
            )
            continue
        extraction = extract_faculty_info(llm, professor_url=faculty.url, cleaned_text=cleaned)
        if not extraction.professor_name and faculty.name:
            extraction.professor_name = faculty.name
        extractions.append(extraction)

    labs, links = build_lab_candidates(faculty_records, extractions)
    write_labs_csv(settings.output_dir / "labs.csv", labs)
    write_faculty_lab_links_csv(settings.output_dir / "faculty_lab_links.csv", links)
    write_blocked_urls_csv(settings.output_dir / "blocked_urls.csv", blocked_urls)

    if not labs:
        print("No labs found from extraction.")
        return

    selected_lab = labs[0]
    if one_lab_name:
        for lab in labs:
            if lab.lab_name.lower() == one_lab_name.lower():
                selected_lab = lab
                break

    enriched = enrich_single_lab(settings, fetcher, selected_lab, extractions)
    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pipeline_stats": {
            "faculty_discovered": len(faculty_records),
            "faculty_extracted": len(extractions),
            "labs_found": len(labs),
            "blocked_urls": len(blocked_urls),
            "enriched_pages_saved": len(enriched["saved_pages"]),
        },
        "selected_lab": {
            "name": selected_lab.lab_name,
            "canonical_url": selected_lab.canonical_url,
        },
        "enrichment": enriched,
        "extractions": [item.model_dump() for item in extractions],
    }
    meta_path = Path(enriched["output_dir"]) / "metadata.json"
    write_json(meta_path, metadata)

    print(json.dumps(metadata["pipeline_stats"], indent=2))
    print(f"labs.csv: {settings.output_dir / 'labs.csv'}")
    print(f"faculty_lab_links.csv: {settings.output_dir / 'faculty_lab_links.csv'}")
    print(f"blocked_urls.csv: {settings.output_dir / 'blocked_urls.csv'}")
    print(f"lab metadata: {meta_path}")


def run_query(lab_slug: str, question: str, keyword: str | None = None, threshold: int | None = None) -> None:
    settings = load_settings()
    llm = LLMClient(settings)
    lab_dir = settings.output_dir / "lab_test" / lab_slug
    if not lab_dir.exists():
        raise FileNotFoundError(f"Lab directory not found: {lab_dir}")

    answer = ask_lab_question(llm, lab_dir=lab_dir, question=question)
    print("\n=== LLM Answer ===\n")
    print(answer)

    if keyword is not None:
        count = keyword_frequency(lab_dir=lab_dir, keyword=keyword)
        print("\n=== Keyword Frequency ===")
        print(f"'{keyword}' appears {count} times.")
        if threshold is not None:
            print(f"Meets threshold ({threshold}): {count >= threshold}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UIUC CS+ECE faculty/lab scraper")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run discovery, extraction, and one-lab enrichment.")
    run_parser.add_argument("--lab-name", type=str, default=None, help="Optional exact lab name to enrich.")

    query_parser = subparsers.add_parser("query", help="Query one enriched lab corpus with LLM.")
    query_parser.add_argument("--lab-slug", type=str, required=True, help="Lab folder slug under output/lab_test/")
    query_parser.add_argument("--question", type=str, required=True, help="Question to ask about this lab.")
    query_parser.add_argument("--keyword", type=str, default=None, help="Optional keyword for exact frequency count.")
    query_parser.add_argument("--threshold", type=int, default=None, help="Optional frequency threshold check.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.command == "run":
        run_pipeline(one_lab_name=args.lab_name)
    elif args.command == "query":
        run_query(
            lab_slug=args.lab_slug,
            question=args.question,
            keyword=args.keyword,
            threshold=args.threshold,
        )

