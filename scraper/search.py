from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from dataclasses import dataclass
from urllib.parse import urlparse

from scraper.links import is_probably_personal_site

# DuckDuckGo starts returning rate-limit errors when hit by many threads at
# once, so cap concurrent searches independently of the worker pool size. This
# lets us run a high fetch/LLM concurrency without tripping the search engine.
_SEARCH_SEMAPHORE = threading.Semaphore(6)

# A single ddgs call retries across several backends with its own timeout, so a
# rate-limited query can block a worker for a minute. Cap the wall-clock per
# search and bail fast, degrading to profile-only data instead of stalling.
_SEARCH_TIMEOUT_S = 8.0
_SEARCH_EXECUTOR = ThreadPoolExecutor(max_workers=6, thread_name_prefix="ddg")

# Circuit breaker: once DuckDuckGo starts rate-limiting it usually keeps doing
# so for the rest of the run, and burning the per-search timeout on every
# remaining professor adds minutes for nothing. After enough consecutive
# failures we stop searching entirely.
_FAILURE_THRESHOLD = 12
_breaker_lock = threading.Lock()
_consecutive_failures = 0
_search_disabled = False


def _record_search_outcome(ok: bool) -> None:
    global _consecutive_failures, _search_disabled
    with _breaker_lock:
        if ok:
            _consecutive_failures = 0
        else:
            _consecutive_failures += 1
            if _consecutive_failures >= _FAILURE_THRESHOLD:
                _search_disabled = True


def _search_is_disabled() -> bool:
    with _breaker_lock:
        return _search_disabled


def reset_search_breaker() -> None:
    """Reset the rate-limit circuit breaker (used by tests / between runs)."""
    global _consecutive_failures, _search_disabled
    with _breaker_lock:
        _consecutive_failures = 0
        _search_disabled = False


def _run_ddg(query: str, max_results: int) -> list[dict]:
    from ddgs import DDGS

    with _SEARCH_SEMAPHORE, DDGS(timeout=5) as ddgs:
        return list(ddgs.text(query, max_results=max_results))


# Result hosts that are never a professor's own homepage even though they pass
# the personal-site heuristic (aggregators, directories, encyclopedias, etc.).
# Note: we deliberately do NOT block illinois.edu wholesale, because personal
# homepages live on subdomains like slazebni.cs.illinois.edu; the directory
# pages are filtered by host/path checks instead.
_SEARCH_HOST_BLOCKLIST = (
    "wikipedia.org",
    "wikidata.org",
    "semanticscholar.org",
    "researchgate.net",
    "research.com",
    "academia.edu",
    "linkedin.com",
    "github.com",
    "scholar.google.com",
    "dblp.org",
    "mathgenealogy.org",
    "ratemyprofessors.com",
    "medium.com",
    "youtube.com",
    "twitter.com",
    "x.com",
    "facebook.com",
)

# Paths that mark an official directory profile (which we already scrape),
# regardless of which illinois.edu host serves them.
_DIRECTORY_PATH_HINTS = (
    "/about/people",
    "/about/directory",
)

_DEPARTMENT_WORDS = {
    "cs": "computer science",
    "ece": "electrical computer engineering",
}


@dataclass
class SearchResult:
    url: str
    title: str
    snippet: str


def _host(url: str) -> str:
    return urlparse(url).netloc.lower().split(":")[0]


def _blocked(url: str) -> bool:
    host = _host(url)
    if not host:
        return True
    if any(host == b or host.endswith(f".{b}") for b in _SEARCH_HOST_BLOCKLIST):
        return True
    path = urlparse(url).path.lower()
    return any(path.startswith(hint) for hint in _DIRECTORY_PATH_HINTS)


def web_search(query: str, max_results: int = 5) -> list[SearchResult]:
    """Run a DuckDuckGo text search; never raise (returns [] on any failure).

    DuckDuckGo needs no API key and is far less aggressive about blocking
    automated queries than Google, which makes it a practical engine for the
    handful of lookups we do per professor.
    """
    if _search_is_disabled():
        return []
    try:
        import ddgs  # noqa: F401
    except Exception:
        return []

    # Run the (blocking, retry-happy) ddgs call on a side thread so we can
    # enforce a hard wall-clock budget; a timed-out search counts as a failure
    # for the circuit breaker. The worker thread keeps running in the executor,
    # but it is bounded by ddgs's own internal timeout.
    future = _SEARCH_EXECUTOR.submit(_run_ddg, query, max_results)
    try:
        raw = future.result(timeout=_SEARCH_TIMEOUT_S)
    except FuturesTimeout:
        future.cancel()
        _record_search_outcome(False)
        return []
    except Exception:
        _record_search_outcome(False)
        return []
    _record_search_outcome(True)

    results: list[SearchResult] = []
    for item in raw:
        url = (item.get("href") or item.get("url") or "").strip()
        if not url:
            continue
        results.append(
            SearchResult(
                url=url,
                title=(item.get("title") or "").strip(),
                snippet=(item.get("body") or item.get("snippet") or "").strip(),
            )
        )
    return results


def search_professor_homepage_candidates(
    name: str,
    department: str | None = None,
    max_results: int = 5,
) -> list[SearchResult]:
    """Search for a professor's personal homepage and return vetted candidates.

    We query the name plus a UIUC qualifier, then drop directory/social/aggregator
    hosts up front so the LLM only has to choose among plausible personal sites.
    The directory profile (illinois.edu) is excluded because we already parse it.
    """
    if not name or not name.strip():
        return []
    dept_word = _DEPARTMENT_WORDS.get((department or "").lower(), "")
    query = " ".join(part for part in [name.strip(), dept_word, "uiuc"] if part)
    seen: set[str] = set()
    candidates: list[SearchResult] = []
    for result in web_search(query, max_results=max_results):
        if _blocked(result.url) or not is_probably_personal_site(result.url):
            continue
        key = result.url.rstrip("/")
        if key in seen:
            continue
        seen.add(key)
        candidates.append(result)
    return candidates
