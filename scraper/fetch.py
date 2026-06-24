from __future__ import annotations

import random
import re
import time
from dataclasses import dataclass

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from scraper.config import Settings

BLOCK_PATTERNS = (
    r"captcha",
    r"access denied",
    r"bot detection",
)

CLOUDFLARE_CHALLENGE_PATTERNS = (
    r"attention required",
    r"just a moment",
    r"cf-challenge",
    r"/cdn-cgi/challenge-platform",
)


@dataclass
class FetchResult:
    url: str
    ok: bool
    html: str
    status_code: int | None = None
    blocked_reason: str | None = None
    used_playwright: bool = False


class Fetcher:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.session = requests.Session()

    @retry(
        retry=retry_if_exception_type(requests.RequestException),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def _request(self, url: str) -> requests.Response:
        headers = {"User-Agent": random.choice(self.settings.user_agents)}
        return self.session.get(
            url,
            headers=headers,
            timeout=self.settings.request_timeout_s,
        )

    def fetch_html(self, url: str) -> FetchResult:
        time.sleep(self.settings.crawl_delay_s)
        try:
            response = self._request(url)
        except requests.RequestException as exc:
            return FetchResult(
                url=url,
                ok=False,
                html="",
                blocked_reason=f"request_error:{exc.__class__.__name__}",
            )

        text = response.text or ""
        blocked_reason = self._blocked_reason(response.status_code, text)
        if blocked_reason and self.settings.use_playwright_fallback:
            fallback = self._fetch_with_playwright(url)
            if fallback is not None and fallback.ok:
                return fallback
            if fallback is not None and fallback.blocked_reason:
                blocked_reason = f"{blocked_reason};fallback={fallback.blocked_reason}"

        return FetchResult(
            url=url,
            ok=response.ok and not blocked_reason,
            html=text,
            status_code=response.status_code,
            blocked_reason=blocked_reason,
        )

    def _blocked_reason(self, status_code: int, html: str) -> str | None:
        if status_code in {401, 403, 429}:
            return f"http_{status_code}"
        lowered = html.lower()
        for pattern in BLOCK_PATTERNS:
            if re.search(pattern, lowered):
                return f"body_match:{pattern}"
        # Avoid false positives: many legitimate sites mention Cloudflare scripts.
        if status_code in {503, 520, 521, 522, 523, 524}:
            for pattern in CLOUDFLARE_CHALLENGE_PATTERNS:
                if re.search(pattern, lowered):
                    return f"body_match:{pattern}"
        for pattern in CLOUDFLARE_CHALLENGE_PATTERNS:
            if re.search(pattern, lowered):
                return f"body_match:{pattern}"
        # Some JS-heavy pages may return almost no content.
        if len(lowered.strip()) < 200:
            return "body_too_short"
        return None

    def _fetch_with_playwright(self, url: str) -> FetchResult | None:
        try:
            from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
            from playwright.sync_api import sync_playwright
        except Exception:
            return None

        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                context = browser.new_context(
                    user_agent=random.choice(self.settings.user_agents),
                )
                page = context.new_page()
                page.goto(url, timeout=self.settings.request_timeout_s * 1000)
                page.wait_for_timeout(800)
                html = page.content()
                browser.close()
        except PlaywrightTimeoutError:
            return FetchResult(
                url=url,
                ok=False,
                html="",
                blocked_reason="playwright_timeout",
                used_playwright=True,
            )
        except Exception as exc:
            return FetchResult(
                url=url,
                ok=False,
                html="",
                blocked_reason=f"playwright_error:{exc.__class__.__name__}",
                used_playwright=True,
            )

        blocked_reason = self._blocked_reason(200, html)
        return FetchResult(
            url=url,
            ok=blocked_reason is None,
            html=html,
            status_code=200,
            blocked_reason=blocked_reason,
            used_playwright=True,
        )

