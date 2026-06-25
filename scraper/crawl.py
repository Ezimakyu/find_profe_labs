from __future__ import annotations

from collections import deque

from scraper.config import Settings
from scraper.fetch import Fetcher
from scraper.links import (
    extract_internal_links,
    host_of,
    normalize_url,
    path_prefix_for,
)
from scraper.preprocess import clean_html_to_text


def crawl_personal_site(
    settings: Settings,
    fetcher: Fetcher,
    seeds: list[str],
    blocked_urls: list[dict],
    max_pages: int | None = None,
    max_depth: int | None = None,
) -> list[tuple[str, str]]:
    """BFS-crawl a professor's own site 1-2 layers deep.

    Stays on each seed's exact host (and, for shared ``/~netid/`` servers, within
    that user's directory) so we follow links like ``~daf/tracking.html`` while
    never wandering onto co-tenants. Research/CV pages are crawled first within
    the page budget. Returns ``[(url, cleaned_text)]`` for the relevant pages.
    """
    max_pages = settings.personal_site_max_pages if max_pages is None else max_pages
    max_depth = settings.personal_site_max_depth if max_depth is None else max_depth

    # We keep the *raw* URL (with its trailing slash) for fetching and relative
    # link resolution — stripping the slash from a directory URL like /~daf/
    # would make "tracking.html" resolve to /tracking.html. The normalized form
    # is used only for de-duplication in the visited set.
    seen_seeds: set[str] = set()
    queue: deque[tuple[str, int, str, str]] = deque()
    for seed in seeds:
        norm = normalize_url(seed)
        if norm in seen_seeds:
            continue
        seen_seeds.add(norm)
        queue.append((seed, 0, host_of(seed), path_prefix_for(seed)))

    visited: set[str] = set()
    sources: list[tuple[str, str]] = []

    while queue and len(sources) < max_pages:
        url, depth, host, prefix = queue.popleft()
        norm = normalize_url(url)
        if norm in visited:
            continue
        visited.add(norm)

        result = fetcher.fetch_html(url)
        if not result.ok:
            blocked_urls.append(
                {
                    "url": url,
                    "reason": result.blocked_reason or "personal_site_fetch_failed",
                    "status_code": result.status_code,
                }
            )
            continue

        text = clean_html_to_text(result.html)
        if text.strip():
            sources.append((url, text))

        if depth >= max_depth:
            continue
        for link in extract_internal_links(
            result.html,
            url,
            same_registrable_domain=False,
            same_host=True,
            path_prefix=prefix or None,
            skip_boilerplate=True,
            prioritize_research=True,
        ):
            if normalize_url(link) not in visited and host_of(link) == host:
                queue.append((link, depth + 1, host, prefix))

    return sources
