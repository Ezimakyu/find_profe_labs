from scraper.config import Settings
from scraper.crawl import crawl_personal_site
from scraper.fetch import FetchResult


def _fast_settings(**overrides) -> Settings:
    base = dict(
        openai_api_key="dummy",
        crawl_delay_s=0.0,
        crawl_jitter_s=0.0,
        cooldown_s=0.0,
        max_cooldown_s=0.0,
        use_playwright_fallback=False,
        personal_site_max_pages=10,
        personal_site_max_depth=2,
    )
    base.update(overrides)
    return Settings(**base)


class FakeFetcher:
    def __init__(self, pages: dict[str, str]):
        self.pages = pages
        self.fetched: list[str] = []

    def fetch_html(self, url: str) -> FetchResult:
        self.fetched.append(url)
        if url in self.pages:
            return FetchResult(url=url, ok=True, html=self.pages[url], status_code=200)
        return FetchResult(url=url, ok=False, html="", status_code=404, blocked_reason="not_found")


def test_crawl_follows_links_and_stays_on_host():
    home = "http://luthuli.cs.uiuc.edu/~daf/"
    pages = {
        home: '<a href="tracking.html">Tracking</a><a href="animation.html">Animation</a>'
        + '<a href="http://other.com/x">ext</a><a href="/~someoneelse/">other</a>' + "body" * 80,
        "http://luthuli.cs.uiuc.edu/~daf/tracking.html": "tracking research " * 60,
        "http://luthuli.cs.uiuc.edu/~daf/animation.html": "animation research " * 60,
    }
    fetcher = FakeFetcher(pages)
    sources = crawl_personal_site(_fast_settings(), fetcher, [home], [])
    urls = {u for u, _ in sources}
    assert "http://luthuli.cs.uiuc.edu/~daf/tracking.html" in urls
    assert "http://luthuli.cs.uiuc.edu/~daf/animation.html" in urls
    # Never leaves the host, and never wanders to a co-tenant's /~user/ space.
    assert all("other.com" not in u for u in fetcher.fetched)
    assert all("someoneelse" not in u for u in fetcher.fetched)


def test_crawl_respects_page_budget():
    home = "https://prof.example.com/"
    pages = {home: "".join(f'<a href="/p{i}">p{i}</a>' for i in range(20)) + "body" * 80}
    for i in range(20):
        pages[f"https://prof.example.com/p{i}"] = f"content {i} " * 60
    sources = crawl_personal_site(_fast_settings(personal_site_max_pages=5), FakeFetcher(pages), [home], [])
    assert len(sources) == 5


def test_crawl_records_blocked():
    blocked: list[dict] = []
    sources = crawl_personal_site(
        _fast_settings(), FakeFetcher({}), ["https://dead.example.com/"], blocked
    )
    assert sources == []
    assert blocked and blocked[0]["url"] == "https://dead.example.com/"
