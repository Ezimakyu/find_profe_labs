from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


@dataclass
class Settings:
    openai_api_key: str
    openai_model: str = "gpt-4o-mini"
    # OpenAI-compatible base URL. Set this (e.g. http://localhost:11434/v1) to
    # route LLM calls at a local model such as Ollama-served qwen3:14b.
    openai_base_url: str | None = None
    request_timeout_s: int = 20
    request_retries: int = 4
    # Politeness defaults are deliberately light; the adaptive cooldown only
    # kicks in when a site actually throttles us, so the common case is fast.
    crawl_delay_s: float = 0.3
    crawl_jitter_s: float = 0.3
    cooldown_s: float = 15.0
    cooldown_backoff: float = 1.8
    max_cooldown_s: float = 120.0
    max_faculty_pages: int = 500
    # Per-lab deep capture (recursive lab crawl) is OFF by default now; the focus
    # is per-professor research detail. Flip ENABLE_LAB_ENRICHMENT=1 to re-enable.
    enable_lab_enrichment: bool = False
    max_enrich_pages: int = 30
    enrich_max_depth: int = 2
    crawl_personal_sites: bool = True
    max_personal_sites_per_faculty: int = 2
    # Recursive crawl of each professor's own site (1-2 layers) to reach pages
    # like ~daf/tracking.html where the real research detail lives.
    personal_site_max_pages: int = 5
    personal_site_max_depth: int = 2
    # Professors are processed concurrently (all the work is network I/O); the
    # Fetcher's shared cooldown still throttles everyone if a site pushes back.
    max_workers: int = 16
    # Chat calls in flight at once. Kept below max_workers because the OpenAI
    # account's tokens-per-minute cap, not CPU, is the real LLM bottleneck.
    max_llm_concurrency: int = 8
    # Cap response bodies so one giant page can't blow up memory under many
    # concurrent workers; non-HTML payloads are skipped entirely.
    max_response_bytes: int = 3_000_000
    # Web search to discover homepages the directory page doesn't link.
    enable_web_search: bool = True
    web_search_results: int = 5
    # Relevance filtering of crawled text before the LLM sees it. By default we
    # use a cheap lexical (keyword-density) ranker, which is fast on a low-core
    # box. Enable local embeddings (semantic) when pairing with a local LLM
    # whose small context window makes tight chunk selection worth the CPU cost.
    use_embedding_retrieval: bool = False
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_chunk_chars: int = 1200
    embedding_chunk_overlap: int = 150
    embedding_top_k: int = 14
    # Total characters of crawled text fed to the extraction LLM per professor.
    # ~4 chars/token, so this bounds input at ~2.5k tokens; with a TPM-limited
    # account, a tighter budget is what keeps the whole run from stalling on
    # 429s. The lexical/embedding ranker ensures these are the relevant chars.
    embedding_max_chars: int = 10000
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

    base_url = os.getenv("OPENAI_BASE_URL", "").strip() or None

    return Settings(
        openai_api_key=api_key,
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        openai_base_url=base_url,
        request_timeout_s=int(os.getenv("REQUEST_TIMEOUT_S", "20")),
        request_retries=int(os.getenv("REQUEST_RETRIES", "4")),
        crawl_delay_s=float(os.getenv("CRAWL_DELAY_S", "0.3")),
        crawl_jitter_s=float(os.getenv("CRAWL_JITTER_S", "0.3")),
        cooldown_s=float(os.getenv("COOLDOWN_S", "15.0")),
        cooldown_backoff=float(os.getenv("COOLDOWN_BACKOFF", "1.8")),
        max_cooldown_s=float(os.getenv("MAX_COOLDOWN_S", "120.0")),
        max_faculty_pages=int(os.getenv("MAX_FACULTY_PAGES", "500")),
        enable_lab_enrichment=_as_bool(os.getenv("ENABLE_LAB_ENRICHMENT"), False),
        max_enrich_pages=int(os.getenv("MAX_ENRICH_PAGES", "30")),
        enrich_max_depth=int(os.getenv("ENRICH_MAX_DEPTH", "2")),
        crawl_personal_sites=_as_bool(os.getenv("CRAWL_PERSONAL_SITES"), True),
        max_personal_sites_per_faculty=int(os.getenv("MAX_PERSONAL_SITES_PER_FACULTY", "2")),
        personal_site_max_pages=int(os.getenv("PERSONAL_SITE_MAX_PAGES", "5")),
        personal_site_max_depth=int(os.getenv("PERSONAL_SITE_MAX_DEPTH", "2")),
        max_workers=int(os.getenv("MAX_WORKERS", "16")),
        max_llm_concurrency=int(os.getenv("MAX_LLM_CONCURRENCY", "8")),
        max_response_bytes=int(os.getenv("MAX_RESPONSE_BYTES", "3000000")),
        enable_web_search=_as_bool(os.getenv("ENABLE_WEB_SEARCH"), True),
        web_search_results=int(os.getenv("WEB_SEARCH_RESULTS", "5")),
        use_embedding_retrieval=_as_bool(os.getenv("USE_EMBEDDING_RETRIEVAL"), False),
        embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
        embedding_chunk_chars=int(os.getenv("EMBEDDING_CHUNK_CHARS", "1200")),
        embedding_chunk_overlap=int(os.getenv("EMBEDDING_CHUNK_OVERLAP", "150")),
        embedding_top_k=int(os.getenv("EMBEDDING_TOP_K", "14")),
        embedding_max_chars=int(os.getenv("EMBEDDING_MAX_CHARS", "10000")),
        output_dir=Path(os.getenv("OUTPUT_DIR", "output")),
        use_playwright_fallback=_as_bool(
            os.getenv("USE_PLAYWRIGHT_FALLBACK"),
            True,
        ),
    )

