from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

from scraper.config import Settings, load_settings
from scraper.crawl import crawl_personal_site
from scraper.discovery import discover_faculty
from scraper.enrich import enrich_single_lab
from scraper.extract_llm import extract_faculty_info, select_homepages
from scraper.fetch import Fetcher
from scraper.links import extract_candidate_links, is_probably_personal_site, normalize_url
from scraper.llm_client import LLMClient, configure_llm_concurrency
from scraper.models import FacultyExtraction, FacultyRecord
from scraper.output import (
    build_lab_candidates,
    ensure_dirs,
    write_blocked_urls_csv,
    write_faculty_lab_links_csv,
    write_json,
    write_labs_csv,
    write_professors_csv,
)
from scraper.preprocess import clean_html_to_text
from scraper.query_llm import ask_lab_question, keyword_frequency
from scraper.retrieve import EmbeddingRetriever
from scraper.search import search_professor_homepage_candidates


def _discover_homepages(
    settings: Settings,
    llm: LLMClient,
    faculty: FacultyRecord,
    profile_candidates: list[dict[str, str]],
) -> list[str]:
    """Pick the professor's own site(s): profile links first, web search as fallback.

    The directory profile typically links a homepage; when it doesn't (e.g. David
    Forsyth), we web-search the name and let the LLM vet the top results.
    """
    profile_url_norm = normalize_url(faculty.url)
    filtered = [
        c
        for c in profile_candidates
        if is_probably_personal_site(c["url"]) and normalize_url(c["url"]) != profile_url_norm
    ]
    chosen = select_homepages(llm, faculty.name or "", filtered) if filtered else []

    if not chosen and settings.enable_web_search and faculty.name:
        results = search_professor_homepage_candidates(
            faculty.name, faculty.department, max_results=settings.web_search_results
        )
        web_candidates = [{"url": r.url, "text": r.title} for r in results]
        if web_candidates:
            chosen = select_homepages(llm, faculty.name, web_candidates)

    # De-dupe, drop the profile URL, and cap how many sites we crawl.
    out: list[str] = []
    for url in chosen:
        if normalize_url(url) == profile_url_norm:
            continue
        if url not in out:
            out.append(url)
    return out[: settings.max_personal_sites_per_faculty]


def _process_faculty(
    settings: Settings,
    fetcher: Fetcher,
    llm: LLMClient,
    faculty: FacultyRecord,
    retriever: EmbeddingRetriever | None = None,
) -> tuple[FacultyExtraction | None, list[dict]]:
    """Fetch a profile, discover the professor's own site, recursively crawl it,
    then extract specific research detail from the combined corpus in one pass.

    Returns the extraction (or ``None``) plus the blocked URLs encountered, so
    this can run safely on a worker thread without sharing mutable state.
    """
    blocked_urls: list[dict] = []
    result = fetcher.fetch_html(faculty.url)
    cleaned = clean_html_to_text(result.html) if result.ok else ""
    candidate_links = extract_candidate_links(result.html, faculty.url) if result.ok else []
    if not result.ok:
        blocked_urls.append(
            {
                "url": faculty.url,
                "reason": result.blocked_reason or "faculty_fetch_failed",
                "status_code": result.status_code,
            }
        )

    sources: list[tuple[str, str]] = []
    if cleaned.strip():
        sources.append((faculty.url, cleaned))

    personal: list[str] = []
    if settings.crawl_personal_sites:
        personal = _discover_homepages(settings, llm, faculty, candidate_links)
        if personal:
            sources.extend(crawl_personal_site(settings, fetcher, personal, blocked_urls))

    if not sources:
        if result.ok:
            blocked_urls.append(
                {"url": faculty.url, "reason": "empty_cleaned_text", "status_code": result.status_code}
            )
        return None, blocked_urls

    extraction = extract_faculty_info(
        llm,
        professor_url=faculty.url,
        sources=sources,
        candidate_links=candidate_links,
        settings=settings,
        retriever=retriever,
    )
    # Personal sites are the ones we actually discovered & crawled, not whatever
    # the extraction LLM echoed back.
    extraction.personal_site_urls = personal
    if not extraction.professor_name and faculty.name:
        extraction.professor_name = faculty.name
    return extraction, blocked_urls


def run_pipeline(one_lab_name: str | None = None) -> None:
    settings = load_settings()
    ensure_dirs(settings.output_dir)
    fetcher = Fetcher(settings)
    configure_llm_concurrency(settings.max_llm_concurrency)
    llm = LLMClient(settings)
    retriever = (
        EmbeddingRetriever(settings.embedding_model) if settings.use_embedding_retrieval else None
    )

    faculty_records, blocked_urls = discover_faculty(settings, fetcher)

    # Professors are independent and the work is almost entirely network I/O
    # (fetches, web search, LLM, crawl), so we fan out across worker threads.
    # Order is preserved to keep output deterministic.
    results: list[tuple[FacultyExtraction | None, list[dict]]] = [None] * len(faculty_records)  # type: ignore[list-item]
    workers = max(1, min(settings.max_workers, len(faculty_records) or 1))
    total = len(faculty_records)
    started = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_idx = {
            pool.submit(_process_faculty, settings, fetcher, llm, faculty, retriever): i
            for i, faculty in enumerate(faculty_records)
        }
        done = 0
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as exc:  # noqa: BLE001
                # One professor failing (e.g. an exhausted API rate limit or a
                # malformed page) must never sink the whole run; record nothing
                # for them and keep going.
                fac = faculty_records[idx]
                print(f"  ! {fac.name} failed: {exc}", file=sys.stderr, flush=True)
                results[idx] = (None, [{"url": fac.url, "reason": f"processing_error: {exc}"}])
            done += 1
            if done % 25 == 0 or done == total:
                print(
                    f"  processed {done}/{total} professors ({time.time() - started:.0f}s)",
                    file=sys.stderr,
                    flush=True,
                )

    extractions: list[FacultyExtraction] = []
    for extraction, faculty_blocked in results:
        blocked_urls.extend(faculty_blocked)
        if extraction is not None:
            extractions.append(extraction)

    labs, links = build_lab_candidates(faculty_records, extractions)
    write_labs_csv(settings.output_dir / "labs.csv", labs)
    write_faculty_lab_links_csv(settings.output_dir / "faculty_lab_links.csv", links)
    write_professors_csv(settings.output_dir / "professors.csv", faculty_records, extractions)
    write_blocked_urls_csv(settings.output_dir / "blocked_urls.csv", blocked_urls)

    # Per-lab deep capture is opt-in now; the focus is per-professor research.
    enriched: dict | None = None
    if settings.enable_lab_enrichment and labs:
        selected_lab = labs[0]
        if one_lab_name:
            for lab in labs:
                if lab.lab_name.lower() == one_lab_name.lower():
                    selected_lab = lab
                    break
        enriched = enrich_single_lab(settings, fetcher, selected_lab, extractions, llm=llm)

    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pipeline_stats": {
            "faculty_discovered": len(faculty_records),
            "faculty_extracted": len(extractions),
            "labs_found": len(labs),
            "blocked_urls": len(blocked_urls),
            "personal_sites_crawled": sum(len(e.personal_site_urls) for e in extractions),
            "professors_with_personal_site": sum(1 for e in extractions if e.personal_site_urls),
            "lab_enrichment_enabled": settings.enable_lab_enrichment,
            "enriched_pages_saved": len(enriched["saved_pages"]) if enriched else 0,
        },
        "selected_lab": (
            {"name": enriched["lab_name"], "canonical_url": enriched["lab_url"]} if enriched else None
        ),
        "enrichment": enriched,
        "extractions": [item.model_dump() for item in extractions],
    }
    meta_path = settings.output_dir / "metadata.json"
    write_json(meta_path, metadata)

    print(json.dumps(metadata["pipeline_stats"], indent=2))
    print(f"labs.csv: {settings.output_dir / 'labs.csv'}")
    print(f"faculty_lab_links.csv: {settings.output_dir / 'faculty_lab_links.csv'}")
    print(f"professors.csv: {settings.output_dir / 'professors.csv'}")
    print(f"blocked_urls.csv: {settings.output_dir / 'blocked_urls.csv'}")
    print(f"metadata: {meta_path}")


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

    run_parser = subparsers.add_parser("run", help="Run discovery + per-professor extraction.")
    run_parser.add_argument("--lab-name", type=str, default=None, help="Optional exact lab name to enrich.")
    run_parser.add_argument(
        "--enable-lab",
        action="store_true",
        help="Also run the recursive per-lab deep capture (off by default).",
    )

    query_parser = subparsers.add_parser("query", help="Query one enriched lab corpus with LLM.")
    query_parser.add_argument("--lab-slug", type=str, required=True, help="Lab folder slug under output/lab_test/")
    query_parser.add_argument("--question", type=str, required=True, help="Question to ask about this lab.")
    query_parser.add_argument("--keyword", type=str, default=None, help="Optional keyword for exact frequency count.")
    query_parser.add_argument("--threshold", type=int, default=None, help="Optional frequency threshold check.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.command == "run":
        if getattr(args, "enable_lab", False):
            os.environ["ENABLE_LAB_ENRICHMENT"] = "1"
        run_pipeline(one_lab_name=args.lab_name)
    elif args.command == "query":
        run_query(
            lab_slug=args.lab_slug,
            question=args.question,
            keyword=args.keyword,
            threshold=args.threshold,
        )

