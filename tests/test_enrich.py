from scraper.config import Settings
from scraper.enrich import _recursive_crawl, _seed_urls_for_lab, _slugify
from scraper.fetch import FetchResult
from scraper.models import FacultyExtraction, LabCandidate


def _fast_settings(**overrides) -> Settings:
    base = dict(
        openai_api_key="dummy",
        crawl_delay_s=0.0,
        crawl_jitter_s=0.0,
        cooldown_s=0.0,
        max_cooldown_s=0.0,
        use_playwright_fallback=False,
        enrich_max_depth=2,
        max_enrich_pages=10,
    )
    base.update(overrides)
    return Settings(**base)


class FakeFetcher:
    def __init__(self, pages: dict[str, str]):
        self.pages = pages

    def fetch_html(self, url: str) -> FetchResult:
        if url in self.pages:
            return FetchResult(url=url, ok=True, html=self.pages[url], status_code=200)
        return FetchResult(url=url, ok=False, html="", status_code=404, blocked_reason="not_found")


def test_seed_urls_prefer_lab_then_personal_then_profile():
    lab = LabCandidate(lab_name="Cool Lab", canonical_url=None)
    extractions = [
        FacultyExtraction(
            professor_url="https://cs.illinois.edu/about/people/all-faculty/p",
            inferred_labs=["Cool Lab"],
            inferred_lab_urls=["https://cool-lab.cs.illinois.edu/"],
            personal_site_urls=["https://p.github.io/"],
        )
    ]
    seeds = _seed_urls_for_lab(lab, extractions)
    assert seeds[0] == "https://cool-lab.cs.illinois.edu/"
    assert "https://p.github.io/" in seeds
    assert seeds[-1] == "https://cs.illinois.edu/about/people/all-faculty/p"


def test_recursive_crawl_follows_internal_links(tmp_path):
    home = "https://cool-lab.illinois.edu/"
    pages = {
        home: '<a href="/research">Research</a><a href="/people">People</a>'
        + '<a href="https://external.com/x">ext</a>' + "body" * 100,
        "https://cool-lab.illinois.edu/research": "<a href='/projects'>Projects</a>" + "r" * 400,
        "https://cool-lab.illinois.edu/people": "p" * 400,
        "https://cool-lab.illinois.edu/projects": "proj" * 200,
    }
    pages_dir = tmp_path / "pages"
    raw_dir = tmp_path / "raw"
    pages_dir.mkdir()
    raw_dir.mkdir()
    saved, blocked = _recursive_crawl(_fast_settings(), FakeFetcher(pages), [home], pages_dir, raw_dir)
    saved_urls = {p["url"] for p in saved}
    # Home (depth 0), research/people (depth 1), projects (depth 2) all captured.
    assert home.rstrip("/") in {u.rstrip("/") for u in saved_urls}
    assert "https://cool-lab.illinois.edu/research" in saved_urls
    assert "https://cool-lab.illinois.edu/projects" in saved_urls
    # External domain never crawled.
    assert all("external.com" not in u for u in saved_urls)
    assert len(list(pages_dir.glob("*.txt"))) == len(saved)


def test_recursive_crawl_respects_max_depth(tmp_path):
    home = "https://cool-lab.illinois.edu/"
    pages = {
        home: "<a href='/a'>a</a>" + "x" * 400,
        "https://cool-lab.illinois.edu/a": "<a href='/b'>b</a>" + "y" * 400,
        "https://cool-lab.illinois.edu/b": "z" * 400,
    }
    pages_dir = tmp_path / "pages"
    raw_dir = tmp_path / "raw"
    pages_dir.mkdir()
    raw_dir.mkdir()
    saved, _ = _recursive_crawl(
        _fast_settings(enrich_max_depth=1), FakeFetcher(pages), [home], pages_dir, raw_dir
    )
    saved_urls = {u.rstrip("/") for u in (p["url"] for p in saved)}
    assert "https://cool-lab.illinois.edu/a" in saved_urls
    # /b is at depth 2, beyond max_depth=1, so it must not be crawled.
    assert "https://cool-lab.illinois.edu/b" not in saved_urls


def test_slugify():
    assert _slugify("Coordinated Science Laboratory") == "coordinated-science-laboratory"
    assert _slugify("") == "lab"
