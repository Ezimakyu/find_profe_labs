from __future__ import annotations

import random
import re
import time
from dataclasses import dataclass

import requests

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

# Status codes that indicate rate limiting / bot blocking rather than a hard 404.
RATE_LIMIT_STATUS = {403, 429, 503}

# Exceptions that suggest the server is throttling or dropping our connections.
THROTTLE_EXCEPTIONS = (
    requests.ConnectionError,
    requests.Timeout,
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
        # Adaptive throttle state shared across requests in a run.
        self._cooldown_until = 0.0
        self._current_cooldown = settings.cooldown_s

    def _polite_sleep(self) -> None:
        """Base crawl delay plus jitter, and honor any active cooldown window."""
        delay = self.settings.crawl_delay_s + random.uniform(0.0, self.settings.crawl_jitter_s)
        time.sleep(delay)
        remaining = self._cooldown_until - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)

    def _enter_cooldown(self) -> None:
        """Back off globally after a throttling signal, growing each time."""
        self._cooldown_until = time.monotonic() + self._current_cooldown
        self._current_cooldown = min(
            self._current_cooldown * self.settings.cooldown_backoff,
            self.settings.max_cooldown_s,
        )

    def _reset_cooldown(self) -> None:
        self._current_cooldown = self.settings.cooldown_s

    def _request(self, url: str) -> requests.Response:
        headers = {
            "User-Agent": random.choice(self.settings.user_agents),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
        return self.session.get(url, headers=headers, timeout=self.settings.request_timeout_s)

    def fetch_html(self, url: str) -> FetchResult:
        """Fetch a URL with retry + adaptive cooldown to survive bot-blocking.

        On a connection error / rate-limit response we enter a growing global
        cooldown so subsequent requests slow down too, which is what actually
        gets us unblocked when a site starts dropping a burst of requests.
        """
        attempts = max(1, self.settings.request_retries)
        last_exc_name: str | None = None
        last_status: int | None = None
        last_blocked_reason: str | None = None

        for attempt in range(attempts):
            self._polite_sleep()
            try:
                response = self._request(url)
            except requests.RequestException as exc:
                last_exc_name = exc.__class__.__name__
                if isinstance(exc, THROTTLE_EXCEPTIONS):
                    self._enter_cooldown()
                else:
                    time.sleep(min(2.0 * (attempt + 1), 8.0))
                continue

            text = response.text or ""
            blocked_reason = self._blocked_reason(response.status_code, text)

            if response.status_code in RATE_LIMIT_STATUS:
                last_status = response.status_code
                last_blocked_reason = f"http_{response.status_code}"
                self._enter_cooldown()
                if self.settings.use_playwright_fallback:
                    fallback = self._fetch_with_playwright(url)
                    if fallback is not None and fallback.ok:
                        self._reset_cooldown()
                        return fallback
                continue

            if blocked_reason and self.settings.use_playwright_fallback:
                fallback = self._fetch_with_playwright(url)
                if fallback is not None and fallback.ok:
                    self._reset_cooldown()
                    return fallback
                if fallback is not None and fallback.blocked_reason:
                    blocked_reason = f"{blocked_reason};fallback={fallback.blocked_reason}"

            self._reset_cooldown()
            return FetchResult(
                url=url,
                ok=response.ok and not blocked_reason,
                html=text,
                status_code=response.status_code,
                blocked_reason=blocked_reason,
            )

        # All attempts exhausted: try Playwright once as a last resort, since a
        # real browser is more likely to slip past connection-level blocking.
        if self.settings.use_playwright_fallback:
            fallback = self._fetch_with_playwright(url)
            if fallback is not None and fallback.ok:
                self._reset_cooldown()
                return fallback

        reason = last_blocked_reason or (f"request_error:{last_exc_name}" if last_exc_name else "fetch_failed")
        return FetchResult(url=url, ok=False, html="", status_code=last_status, blocked_reason=reason)

    def _blocked_reason(self, status_code: int, html: str) -> str | None:
        if status_code in {401, 403, 429}:
            return f"http_{status_code}"
        lowered = html.lower()
        for pattern in BLOCK_PATTERNS:
            if re.search(pattern, lowered):
                return f"body_match:{pattern}"
        # Cloudflare challenge phrases ("just a moment", "attention required") also
        # appear on legitimate 200 pages, so only treat them as a block on the
        # error statuses Cloudflare actually uses for challenges.
        if status_code in {503, 520, 521, 522, 523, 524}:
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
