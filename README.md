# UIUC CS+ECE Lab Scraper (LLM-first MVP)

This project discovers UIUC CS/ECE faculty, finds each professor's **personal /
research-group website** (from their directory profile **and** via web search
when the directory has no usable link), recursively crawls that site 1-2 layers
to reach the pages where the real research detail lives, and uses an LLM to
extract **specific, distinct** research interests. It outputs a broad `labs.csv`
plus a per-professor `professors.csv`.

Professors are processed **concurrently** and only the most research-relevant
text is sent to the LLM (relevance filtering), so a full CS+ECE run is fast and
memory-flat. The LLM endpoint is OpenAI-compatible, so you can point it at a
**local model** (e.g. Ollama-served `qwen3:14b`) instead of the OpenAI API.

## What it produces

- `output/labs.csv` - broad lab inventory, including a `professors` column listing all professors associated with each lab.
- `output/professors.csv` - one row per professor: multiple `source_urls`, discovered `personal_site_urls`, labs, topics, and a `research_description_detailed` column describing "what they actually do" so professors with overlapping interests can be told apart.
- `output/faculty_lab_links.csv` - faculty-to-lab linkage with evidence snippets.
- `output/blocked_urls.csv` - inaccessible pages (status and reason).
- `output/lab_test/...` - per-lab recursive deep-capture artifacts, produced **only** when `ENABLE_LAB_ENRICHMENT=1`:
- `output/lab_test/<lab_slug>/raw_html/*.html` - raw page bodies (no HTTP headers).
- `output/lab_test/<lab_slug>/pages/*.txt` - cleaned text from recursively crawled lab pages.
- `output/lab_test/<lab_slug>/metadata.json` - one-lab enrichment metadata (seed URLs, crawled pages, and an LLM `research_summary` focused on research activities/mission rather than awards) + pipeline stats.

## How it works

1. **Discover** CS + ECE faculty from the directory seed pages. Cross-listed
   professors (the same person under both CS and ECE) are de-duplicated by name
   so each professor is processed once.
2. **Find the personal site.** Fetch the profile and let the LLM pick the
   professor's homepage from its outbound links. If the directory page has no
   usable link (e.g. David Forsyth, whose site `http://luthuli.cs.uiuc.edu/~daf/`
   isn't linked from the directory), run a **web search** (DuckDuckGo, no API
   key) for the professor and have the LLM vet the top ~5 results, rejecting the
   directory page we already scrape, social media, Google Scholar, DBLP, etc.
3. **Recursively crawl** the chosen site 1-2 layers deep, following
   research / project / topic / CV links (e.g. `~daf/tracking.html`) to reach the
   pages where research is actually described. Research interests are inferred
   from project and paper titles when not stated outright.
4. **Relevance-filter then extract.** Crawled HTML is cleaned (BeautifulSoup +
   boilerplate removal), split into chunks, and only the most research-relevant
   chunks are sent to the LLM (see *Relevance filtering* below). The LLM returns
   specific topics and a concrete `research_description_detailed`.
5. **Concurrency + robustness.** Professors are processed across a thread pool.
   A polite crawl delay with jitter plus an **adaptive cooldown** backs off on
   `ConnectionError`, timeouts, or rate-limit responses (403/429/503), with
   Playwright as a fallback for genuinely bot-blocked pages. Web search has its
   own hard timeout + circuit breaker, the LLM client honors `Retry-After` on
   429s, and a single professor failing never sinks the whole run.

> **Per-lab deep capture** (the recursive lab-site crawl under
> `output/lab_test/`) is **disabled by default** to keep runs focused on
> per-professor detail. Re-enable it with `ENABLE_LAB_ENRICHMENT=1`.

## Relevance filtering (which text the LLM sees)

Following the *website -> BeautifulSoup -> boilerplate removal -> rank ->
LLM-on-relevant-chunks* pattern, crawled text is chunked and ranked before the
LLM sees it, bounded by `EMBEDDING_MAX_CHARS`:

- **Lexical ranker (default).** A keyword-density score keeps research-bearing
  chunks and drops navigation/boilerplate. It is essentially free on CPU, which
  matters on low-core machines, and is the right choice with the cloud LLM.
- **Local embeddings (opt-in).** Set `USE_EMBEDDING_RETRIEVAL=1` to rank chunks
  by semantic similarity using a local `fastembed` model (no API calls). This is
  more precise but CPU-heavy, so it is best paired with a **local LLM** whose
  small context window makes tight chunk selection worthwhile.

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
   - `OPENAI_API_KEY=...` (required for the OpenAI path; use any placeholder for
     a local model)
   - `OPENAI_MODEL=gpt-4o-mini` (optional; default `gpt-4o-mini`)

### Tunable environment variables

| Var | Default | Purpose |
|-----|---------|---------|
| `OPENAI_BASE_URL` | _(unset)_ | OpenAI-compatible endpoint. Set to a local server (e.g. `http://localhost:11434/v1`) to use a local LLM. |
| `OPENAI_MODEL` | `gpt-4o-mini` | Chat model name (e.g. `qwen3:14b` for a local model). |
| `MAX_WORKERS` | `16` | Professors processed concurrently (work is network I/O bound). |
| `MAX_LLM_CONCURRENCY` | `8` | Chat calls in flight at once; kept below `MAX_WORKERS` so a tokens-per-minute cap isn't tripped. |
| `MAX_RESPONSE_BYTES` | `3000000` | Per-response body cap; non-HTML payloads are skipped. |
| `ENABLE_WEB_SEARCH` | `true` | Discover homepages via web search when the directory has no usable link. |
| `WEB_SEARCH_RESULTS` | `5` | Top results vetted by the LLM per professor. |
| `USE_EMBEDDING_RETRIEVAL` | `false` | Use local semantic embeddings instead of the lexical ranker (recommended only with a local LLM). |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | `fastembed` model used when embeddings are enabled. |
| `EMBEDDING_MAX_CHARS` | `10000` | Chars of crawled text sent to the LLM per professor (~2.5k tokens). |
| `MAX_FACULTY_PAGES` | `500` | Cap on faculty profiles processed. |
| `CRAWL_DELAY_S` / `CRAWL_JITTER_S` | `0.3` / `0.3` | Polite per-request delay + random jitter. |
| `COOLDOWN_S` / `COOLDOWN_BACKOFF` / `MAX_COOLDOWN_S` | `15` / `1.8` / `120` | Adaptive backoff after throttling signals. |
| `REQUEST_RETRIES` | `4` | Per-URL fetch attempts before giving up. |
| `CRAWL_PERSONAL_SITES` | `true` | Crawl the discovered personal/lab sites. |
| `MAX_PERSONAL_SITES_PER_FACULTY` | `2` | Personal sites crawled per professor. |
| `PERSONAL_SITE_MAX_PAGES` / `PERSONAL_SITE_MAX_DEPTH` | `5` / `2` | Recursive personal-site crawl bounds. |
| `ENABLE_LAB_ENRICHMENT` | `false` | Re-enable the per-lab recursive deep-capture test. |
| `ENRICH_MAX_DEPTH` / `MAX_ENRICH_PAGES` | `2` / `30` | Recursive deep-capture crawl bounds (when enabled). |
| `USE_PLAYWRIGHT_FALLBACK` | `true` | Use Playwright when a page looks bot-blocked. |

### Running a local LLM (e.g. `qwen3:14b` via Ollama)

The pipeline talks to any OpenAI-compatible chat endpoint, so a local model
removes the OpenAI tokens-per-minute ceiling (the main runtime limit for large
runs) at the cost of local GPU/CPU. Using [Ollama](https://ollama.com):

```bash
# 1. Install Ollama (see https://ollama.com/download), then pull the model.
#    qwen3:14b needs ~9 GB of RAM/VRAM; use a smaller tag (e.g. qwen3:4b) on
#    constrained machines.
ollama pull qwen3:14b

# 2. Ollama serves an OpenAI-compatible API at http://localhost:11434/v1.
#    Point the scraper at it:
export OPENAI_BASE_URL=http://localhost:11434/v1
export OPENAI_MODEL=qwen3:14b
export OPENAI_API_KEY=ollama          # any non-empty placeholder; Ollama ignores it

# 3. (Recommended with a local model) use semantic embeddings to keep the
#    prompt small, since local context windows are tighter:
export USE_EMBEDDING_RETRIEVAL=1

python main.py run
```

Any OpenAI-compatible server works the same way (vLLM, LM Studio, llama.cpp's
`server`, etc.) — just set `OPENAI_BASE_URL` and `OPENAI_MODEL` accordingly.

### Performance notes

- A full CS+ECE run (~410 de-duplicated professors, each with a web search,
  1-2 layer site crawl, and an extraction LLM call) completes in **~12-13 min**
  with flat memory (peaks well under 1 GB).
- With the hosted OpenAI API the binding constraint is the account's
  **tokens-per-minute (TPM) limit**, not CPU or network. `MAX_LLM_CONCURRENCY`
  and `EMBEDDING_MAX_CHARS` keep token usage smooth; the client also backs off
  and retries on 429s. Raising your TPM tier (or using a local LLM) is the way
  to go meaningfully faster.
- DuckDuckGo rate-limits sustained automated queries. When that happens, web
  search trips a circuit breaker and the run continues using directory/profile
  data only, so it never stalls.

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

