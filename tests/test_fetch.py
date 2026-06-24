import requests

from scraper.config import Settings
from scraper.fetch import Fetcher


def _fast_settings(**overrides) -> Settings:
    base = dict(
        openai_api_key="dummy",
        request_retries=3,
        crawl_delay_s=0.0,
        crawl_jitter_s=0.0,
        cooldown_s=0.0,
        max_cooldown_s=0.0,
        use_playwright_fallback=False,
    )
    base.update(overrides)
    return Settings(**base)


class _Resp:
    def __init__(self, status_code=200, text="x" * 500):
        self.status_code = status_code
        self.text = text

    @property
    def ok(self):
        return 200 <= self.status_code < 400


def test_blocked_reason_detects_status_and_body():
    fetcher = Fetcher(_fast_settings())
    assert fetcher._blocked_reason(403, "ok body" * 100) == "http_403"
    assert fetcher._blocked_reason(200, "Please complete the CAPTCHA" + "x" * 300).startswith("body_match")
    assert fetcher._blocked_reason(200, "short") == "body_too_short"
    assert fetcher._blocked_reason(200, "y" * 500) is None


def test_connection_error_retries_then_recovers(monkeypatch):
    fetcher = Fetcher(_fast_settings(request_retries=3))
    calls = {"n": 0}

    def fake_request(url):
        calls["n"] += 1
        if calls["n"] < 2:
            raise requests.ConnectionError("dropped")
        return _Resp(200, "y" * 800)

    monkeypatch.setattr(fetcher, "_request", fake_request)
    result = fetcher.fetch_html("https://example.edu/page")
    assert result.ok is True
    assert calls["n"] == 2


def test_connection_error_exhausts_and_reports(monkeypatch):
    fetcher = Fetcher(_fast_settings(request_retries=2))

    def always_fail(url):
        raise requests.ConnectionError("dropped")

    monkeypatch.setattr(fetcher, "_request", always_fail)
    result = fetcher.fetch_html("https://example.edu/page")
    assert result.ok is False
    assert result.blocked_reason == "request_error:ConnectionError"


def test_rate_limit_status_triggers_cooldown(monkeypatch):
    fetcher = Fetcher(_fast_settings(request_retries=1, cooldown_s=5.0))

    monkeypatch.setattr(fetcher, "_request", lambda url: _Resp(429, "rate limited" * 100))
    result = fetcher.fetch_html("https://example.edu/page")
    assert result.ok is False
    assert result.blocked_reason == "http_429"
    # A cooldown window should now be pending.
    assert fetcher._cooldown_until > 0
