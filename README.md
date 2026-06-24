# UIUC CS+ECE Lab Scraper (LLM-first MVP)

This project discovers UIUC CS/ECE faculty pages, uses `gpt-4o-mini` to identify
each professor's **personal/lab website** and crawl it for deeper context,
extracts lab affiliations and specific research details, outputs a broad
`labs.csv` plus a per-professor `professors.csv`, and **recursively** saves raw +
cleaned content for one selected lab for testing.

## What it produces

- `output/labs.csv` - broad lab inventory, including a `professors` column listing all professors associated with each lab.
- `output/professors.csv` - one row per professor: multiple `source_urls`, discovered `personal_site_urls`, labs, topics, and a `research_description_detailed` column describing "what they actually do" so professors with overlapping interests can be told apart.
- `output/faculty_lab_links.csv` - faculty-to-lab linkage with evidence snippets.
- `output/blocked_urls.csv` - inaccessible pages (status and reason).
- `output/lab_test/<lab_slug>/raw_html/*.html` - raw page bodies (no HTTP headers).
- `output/lab_test/<lab_slug>/pages/*.txt` - cleaned text from recursively crawled lab pages.
- `output/lab_test/<lab_slug>/metadata.json` - one-lab enrichment metadata (seed URLs, crawled pages, and an LLM `research_summary` focused on research activities/mission rather than awards) + pipeline stats.

## How crawling works

1. Discover CS + ECE faculty profile pages from the directory seed pages.
2. For each professor, fetch the profile, then ask the LLM to pick the
   professor's personal homepage / research-group site from the page's outbound
   links. Those personal sites are fetched and folded into a multi-source
   extraction (profile + personal site), yielding deeper, multi-sourced records.
3. For the selected lab, perform a recursive (BFS) crawl of the lab's website
   that stays within the site's domain, skips generic site chrome
   (news/alumni/contact/...), and prioritizes research/projects/people pages.
4. Requests use a polite crawl delay with jitter plus an **adaptive cooldown**:
   on `ConnectionError`, timeouts, or rate-limit responses (403/429/503) the
   crawler backs off (growing each time) and retries, and falls back to
   Playwright for bot-blocked pages. This addresses the `ConnectionError` storms
   seen when visiting many faculty pages quickly.

## Setup

1. Create the Conda environment:
   - `conda create -y -n find_profe_labs python=3.11`
2. Activate it:
   - `conda activate find_profe_labs`
3. Install dependencies:
   - `pip install -r requirements.txt`
4. Install Playwright Chromium for JS-heavy fallback pages:
   - `python -m playwright install chromium`
5. Provide your API key, either via a `.env` file in the repo root or as an
   environment variable:
   - `OPENAI_API_KEY=...` (required)
   - `OPENAI_MODEL=gpt-4o-mini` (optional; default `gpt-4o-mini`)

### Tunable environment variables

| Var | Default | Purpose |
|-----|---------|---------|
| `MAX_FACULTY_PAGES` | `500` | Cap on faculty profiles processed. |
| `CRAWL_DELAY_S` / `CRAWL_JITTER_S` | `1.0` / `0.75` | Polite per-request delay + random jitter. |
| `COOLDOWN_S` / `COOLDOWN_BACKOFF` / `MAX_COOLDOWN_S` | `20` / `1.8` / `120` | Adaptive backoff after throttling signals. |
| `REQUEST_RETRIES` | `4` | Per-URL fetch attempts before giving up. |
| `CRAWL_PERSONAL_SITES` | `true` | Crawl LLM-identified personal/lab sites. |
| `MAX_PERSONAL_SITES_PER_FACULTY` | `2` | Personal sites crawled per professor. |
| `ENRICH_MAX_DEPTH` / `MAX_ENRICH_PAGES` | `2` / `30` | Recursive deep-capture crawl bounds. |
| `USE_PLAYWRIGHT_FALLBACK` | `true` | Use Playwright when a page looks bot-blocked. |

## Run pipeline

Run full discovery/extraction and enrich one lab (first discovered lab by default):

`python main.py run`

Choose a specific lab by exact name:

`python main.py run --lab-name "Coordinated Science Laboratory"`

## Query one enriched lab

After the run, get the lab slug from `output/lab_test/` and run:

`python main.py query --lab-slug <lab_slug> --question "What specific research does this lab do? Include examples."`

Optional exact keyword-frequency helper:

`python main.py query --lab-slug <lab_slug> --question "Summarize robotics work." --keyword robotics --threshold 3`

## Development / tests

Offline unit tests (no network or LLM calls) cover link extraction, fetch
retry/cooldown behavior, recursive crawl bounds, schema coercion, and CSV output:

```bash
pip install pytest
pytest -q
```

## Notes and limits

- Extraction is LLM-based and can infer implicit affiliations; verify with evidence snippets and source URLs.
- Raw page bodies are saved for auditability and future custom parsing.
- Some sites may block requests; blocked URLs are logged in `blocked_urls.csv`.
- This MVP intentionally focuses on CS/ECE-seeded discovery to avoid broad non-CS scraping.

