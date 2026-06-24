from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


@dataclass
class Settings:
    openai_api_key: str
    openai_model: str = "gpt-4o-mini"
    request_timeout_s: int = 20
    request_retries: int = 3
    crawl_delay_s: float = 1.0
    max_faculty_pages: int = 500
    max_enrich_pages: int = 30
    output_dir: Path = Path("output")
    use_playwright_fallback: bool = True
    user_agents: list[str] = field(
        default_factory=lambda: [
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
        ]
    )
    allowed_domains: list[str] = field(
        default_factory=lambda: [
            "illinois.edu",
            "cs.illinois.edu",
            "ece.illinois.edu",
            "engineering.illinois.edu",
        ]
    )
    cs_ece_seed_urls: list[str] = field(
        default_factory=lambda: [
            "https://cs.illinois.edu/about/people/faculty",
            "https://ece.illinois.edu/about/directory/faculty",
        ]
    )


def _as_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def load_settings() -> Settings:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required in .env")

    return Settings(
        openai_api_key=api_key,
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        request_timeout_s=int(os.getenv("REQUEST_TIMEOUT_S", "20")),
        request_retries=int(os.getenv("REQUEST_RETRIES", "3")),
        crawl_delay_s=float(os.getenv("CRAWL_DELAY_S", "1.0")),
        max_faculty_pages=int(os.getenv("MAX_FACULTY_PAGES", "500")),
        max_enrich_pages=int(os.getenv("MAX_ENRICH_PAGES", "30")),
        output_dir=Path(os.getenv("OUTPUT_DIR", "output")),
        use_playwright_fallback=_as_bool(
            os.getenv("USE_PLAYWRIGHT_FALLBACK"),
            True,
        ),
    )

