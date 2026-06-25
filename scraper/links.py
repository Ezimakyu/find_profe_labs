from __future__ import annotations

from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

# Hosts that are part of the official directory structure, not a personal site.
DIRECTORY_HOSTS = (
    "cs.illinois.edu",
    "ece.illinois.edu",
    "engineering.illinois.edu",
    "grainger.illinois.edu",
)

# File extensions we never want to crawl.
NON_HTML_SUFFIXES = (
    ".pdf",
    ".doc",
    ".docx",
    ".ppt",
    ".pptx",
    ".xls",
    ".xlsx",
    ".zip",
    ".tar",
    ".gz",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".svg",
    ".mp4",
    ".mov",
    ".ics",
)

# Schemes / fragments we skip outright.
_SKIP_PREFIXES = ("mailto:", "tel:", "javascript:", "#")

# Generic site-chrome paths that rarely contain research content.
BOILERPLATE_PATH_HINTS = (
    "/alumni",
    "/corporate",
    "/giving",
    "/donate",
    "/news",
    "/events",
    "/calendar",
    "/contact",
    "/login",
    "/apply",
    "/admission",
    "/privacy",
    "/accessibility",
    "/search",
    "/tag/",
    "/category/",
    "/author/",
    "/feed",
    "/rss",
    "/newsletter",
    "/sitemap",
)

# Path/anchor keywords that signal a research-relevant page; used to order BFS.
RESEARCH_PATH_HINTS = (
    "research",
    "project",
    "publication",
    "/lab",
    "group",
    "people",
    "member",
    "team",
    "/work",
    "mission",
    "thrust",
    "topic",
    "paper",
    "about",
)


def _is_boilerplate(url: str) -> bool:
    path = urlparse(url).path.lower()
    return any(hint in path for hint in BOILERPLATE_PATH_HINTS)


def _research_score(url: str, anchor_text: str = "") -> int:
    blob = (urlparse(url).path.lower() + " " + anchor_text.lower())
    return sum(1 for hint in RESEARCH_PATH_HINTS if hint.strip("/") in blob)


# Multi-tenant hosting suffixes where each tenant is a distinct site, so the
# registrable domain must include the tenant label (e.g. user.github.io).
MULTI_TENANT_SUFFIXES = (
    "github.io",
    "gitlab.io",
    "netlify.app",
    "vercel.app",
    "pages.dev",
    "web.app",
    "firebaseapp.com",
    "wordpress.com",
    "weebly.com",
    "wixsite.com",
    "squarespace.com",
)


def _normalize(url: str) -> str:
    """Drop fragments and trailing slashes so we de-duplicate consistently."""
    parsed = urlparse(url)
    path = parsed.path.rstrip("/") or "/"
    normalized = parsed._replace(fragment="", path=path)
    return normalized.geturl()


# Public alias for callers outside this module (e.g. crawl seeds).
def normalize_url(url: str) -> str:
    return _normalize(url)


def _is_http(url: str) -> bool:
    return urlparse(url).scheme in {"http", "https"}


def _has_non_html_suffix(url: str) -> bool:
    path = urlparse(url).path.lower()
    return path.endswith(NON_HTML_SUFFIXES)


def registrable_host(url: str) -> str:
    """Best-effort registrable domain of the host (no PSL).

    Returns the last two labels, except for known multi-tenant hosting suffixes
    (``github.io`` etc.) where the tenant label is included so different tenants
    are treated as separate sites.
    """
    host = urlparse(url).netloc.lower().split(":")[0]
    parts = host.split(".")
    if len(parts) <= 2:
        return host
    for suffix in MULTI_TENANT_SUFFIXES:
        if host == suffix or host.endswith("." + suffix):
            labels = suffix.count(".") + 2
            return ".".join(parts[-labels:])
    return ".".join(parts[-2:])


def extract_candidate_links(html: str, base_url: str, limit: int = 60) -> list[dict[str, str]]:
    """Anchors that plausibly point to a personal or lab website.

    Returns ``[{"url", "text"}]`` so the LLM can decide which links are a
    professor's personal homepage or research-group site. We deliberately keep
    off-domain links and directory-external links; this is the raw candidate set.
    """
    soup = BeautifulSoup(html, "lxml")
    seen: set[str] = set()
    candidates: list[dict[str, str]] = []
    for anchor in soup.select("a[href]"):
        href = (anchor.get("href") or "").strip()
        if not href or href.startswith(_SKIP_PREFIXES):
            continue
        url = _normalize(urljoin(base_url, href))
        if not _is_http(url) or _has_non_html_suffix(url):
            continue
        if url in seen:
            continue
        seen.add(url)
        text = anchor.get_text(" ", strip=True)
        candidates.append({"url": url, "text": text[:160]})
        if len(candidates) >= limit:
            break
    return candidates


def is_probably_personal_site(url: str) -> bool:
    """Heuristic guard so we never crawl unrelated directory/social links."""
    host = urlparse(url).netloc.lower().split(":")[0]
    if not host:
        return False
    if host in DIRECTORY_HOSTS:
        return False
    social = (
        "twitter.com",
        "x.com",
        "linkedin.com",
        "facebook.com",
        "youtube.com",
        "instagram.com",
        "scholar.google.com",
        "orcid.org",
        "dblp.org",
        "researchgate.net",
        "github.com",
        "wikipedia.org",
        "amazon.com",
        "google.com",
    )
    if any(host == s or host.endswith(f".{s}") for s in social):
        return False
    return True


def extract_internal_links(
    html: str,
    base_url: str,
    same_registrable_domain: bool = True,
    limit: int = 40,
    skip_boilerplate: bool = False,
    prioritize_research: bool = False,
) -> list[str]:
    """Links on ``base_url`` that stay within the same site, for recursive crawl.

    With ``skip_boilerplate`` we drop generic site-chrome pages (alumni, news,
    contact, ...). With ``prioritize_research`` we order links so research/lab
    pages are crawled first within the page budget.
    """
    soup = BeautifulSoup(html, "lxml")
    base_reg = registrable_host(base_url)
    scored: list[tuple[int, str]] = []
    seen: set[str] = set()
    for anchor in soup.select("a[href]"):
        href = (anchor.get("href") or "").strip()
        if not href or href.startswith(_SKIP_PREFIXES):
            continue
        url = _normalize(urljoin(base_url, href))
        if not _is_http(url) or _has_non_html_suffix(url):
            continue
        if same_registrable_domain and registrable_host(url) != base_reg:
            continue
        if skip_boilerplate and _is_boilerplate(url):
            continue
        if url in seen:
            continue
        seen.add(url)
        score = _research_score(url, anchor.get_text(" ", strip=True)) if prioritize_research else 0
        scored.append((score, url))

    if prioritize_research:
        # Stable sort by descending research relevance (preserves discovery order
        # among equal scores).
        scored.sort(key=lambda pair: -pair[0])
    ordered = [url for _, url in scored]
    return ordered[:limit]
