from __future__ import annotations

from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from scraper.config import Settings
from scraper.fetch import Fetcher
from scraper.models import FacultyRecord


def _department_from_url(url: str) -> str:
    lowered = url.lower()
    if "cs.illinois.edu" in lowered:
        return "cs"
    if "ece.illinois.edu" in lowered:
        return "ece"
    return "unknown"


def _is_allowed_domain(url: str, allowed_domains: list[str]) -> bool:
    host = urlparse(url).netloc.lower()
    return any(host == domain or host.endswith(f".{domain}") for domain in allowed_domains)


def _looks_like_faculty_profile(url: str) -> bool:
    parsed = urlparse(url)
    lowered = parsed.path.lower().strip("/")

    exact_excludes = {
        "about/people/faculty",
        "about/people",
        "about/people/office-school-director",
        "about/people/school-leadership",
        "about/people/staff",
        "about/people/postdocs",
        "about/people/graduating-phd-students",
        "about/directory/faculty",
    }
    if lowered in exact_excludes:
        return False

    if lowered.startswith("about/people/all-faculty/"):
        tail = lowered.removeprefix("about/people/all-faculty/")
        if tail in {"", "department-faculty", "affiliate-faculty", "emeritus-faculty"}:
            return False
        return True

    strong_profile_hints = (
        "about/people/faculty/",
        "about/directory/faculty/",
        "about/people/all-faculty/",
        "people/faculty/",
        "faculty/",
    )
    if any(hint in lowered for hint in strong_profile_hints):
        return True

    profile_hints = (
        "/person/",
        "/profile/",
    )
    return any(hint in lowered for hint in profile_hints)


def discover_faculty(settings: Settings, fetcher: Fetcher) -> tuple[list[FacultyRecord], list[dict]]:
    found: dict[str, FacultyRecord] = {}
    blocked: list[dict] = []

    for seed_url in settings.cs_ece_seed_urls:
        result = fetcher.fetch_html(seed_url)
        if not result.ok:
            blocked.append(
                {
                    "url": seed_url,
                    "reason": result.blocked_reason or "seed_fetch_failed",
                    "status_code": result.status_code,
                }
            )
            continue

        soup = BeautifulSoup(result.html, "lxml")
        for anchor in soup.select("a[href]"):
            href = anchor.get("href", "").strip()
            if not href:
                continue
            url = urljoin(seed_url, href)
            if not _is_allowed_domain(url, settings.allowed_domains):
                continue
            if not _looks_like_faculty_profile(url):
                continue
            if url in found:
                continue
            found[url] = FacultyRecord(
                name=anchor.get_text(" ", strip=True) or None,
                url=url,
                department=_department_from_url(url),
            )
            if len(found) >= settings.max_faculty_pages:
                break

    return list(found.values()), blocked

