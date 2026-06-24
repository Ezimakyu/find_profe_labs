# UIUC CS+ECE Lab Scraper (LLM-first MVP)

This project discovers UIUC CS/ECE faculty pages, extracts likely lab affiliations and specific research details using `gpt-4o-mini`, outputs a broad `labs.csv`, and deeply saves raw + cleaned content for one selected lab for testing.

## What it produces

- `output/labs.csv` - broad lab inventory.
- `output/faculty_lab_links.csv` - faculty-to-lab linkage with evidence snippets.
- `output/blocked_urls.csv` - inaccessible pages (status and reason).
- `output/lab_test/<lab_slug>/raw_html/*.html` - raw page bodies (no HTTP headers).
- `output/lab_test/<lab_slug>/pages/*.txt` - cleaned text from raw pages.
- `output/lab_test/<lab_slug>/metadata.json` - one-lab enrichment metadata + pipeline stats.

## Setup

1. Create the Conda environment:
   - `conda create -y -n find_profe_labs python=3.11`
2. Activate it:
   - `conda activate find_profe_labs`
3. Install dependencies:
   - `pip install -r requirements.txt`
4. Install Playwright Chromium for JS-heavy fallback pages:
   - `python -m playwright install chromium`
5. Copy env template:
   - `cp .env.example .env`
6. Edit `.env` and set your API key:
   - `OPENAI_API_KEY=...`

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

## Notes and limits

- Extraction is LLM-based and can infer implicit affiliations; verify with evidence snippets and source URLs.
- Raw page bodies are saved for auditability and future custom parsing.
- Some sites may block requests; blocked URLs are logged in `blocked_urls.csv`.
- This MVP intentionally focuses on CS/ECE-seeded discovery to avoid broad non-CS scraping.

