from __future__ import annotations

from typing import List, Optional, Union

from pydantic import BaseModel, Field, HttpUrl, field_validator

from scraper.config import Settings
from scraper.llm_client import LLMClient
from scraper.models import Evidence, FacultyExtraction, coerce_optional_str, coerce_str_list
from scraper.retrieve import EmbeddingRetriever, build_research_corpus


class HomepageSelectionEnvelope(BaseModel):
    homepage_urls: List[str] = Field(default_factory=list)

    @field_validator("homepage_urls", mode="before")
    @classmethod
    def _coerce(cls, v):
        return coerce_str_list(v)


HOMEPAGE_SELECT_SYSTEM_PROMPT = """You identify a professor's own website from candidate links.

Choose ONLY URLs that are the professor's personal/academic homepage or their
research-group/lab website. INCLUDE personal domains and university subdomains
such as <netid>.cs.illinois.edu or publish.illinois.edu/<name>.

EXCLUDE:
- the university directory profile page (cs.illinois.edu / ece.illinois.edu /
  siebelschool.illinois.edu /about/people/... or /about/directory/...)
- social media, Google Scholar, DBLP, ORCID, ResearchGate, Academia.edu
- publisher/paper pages (IEEE, ACM, Springer, arXiv, hal.science, etc.)
- news articles, award/biography pages, Wikipedia, seminar announcements
- co-authors' or other people's sites

Return JSON {\"homepage_urls\": [...]} containing the chosen URLs exactly as given
(usually 0-2). If none qualify, return an empty list.
"""


def select_homepages(
    llm: LLMClient,
    professor_name: str,
    candidates: list[dict[str, str]],
) -> list[str]:
    """Ask the LLM which candidate links are the professor's own site.

    ``candidates`` is ``[{"url", "text"}]`` gathered from the profile's outbound
    links and/or a web search. Returns the subset judged to be personal/lab
    sites, preserving only URLs that were actually offered.
    """
    if not professor_name or not candidates:
        return []
    offered = {item["url"] for item in candidates}
    listing = "\n".join(f"- {item['url']} | {item.get('text', '')}" for item in candidates)
    user_prompt = f"""Professor: {professor_name}

Candidate links:
{listing}

Return JSON with key homepage_urls (array of chosen URLs from the list above).
"""
    try:
        payload = llm.json_response(
            system_prompt=HOMEPAGE_SELECT_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            schema=HomepageSelectionEnvelope,
            max_output_tokens=400,
        )
    except Exception:
        return []
    chosen: list[str] = []
    for url in payload.homepage_urls:
        url = url.strip()
        if url in offered and url not in chosen:
            chosen.append(url)
    return chosen


class FacultyExtractionEnvelope(BaseModel):
    professor_name: Optional[str] = None
    personal_site_urls: List[str] = Field(default_factory=list)
    inferred_labs: List[str] = Field(default_factory=list)
    inferred_lab_urls: List[Union[HttpUrl, str]] = Field(default_factory=list)
    research_topics_specific: List[str] = Field(default_factory=list)
    research_details_raw_notes: List[str] = Field(default_factory=list)
    research_description_detailed: Optional[str] = None
    availability_or_recruitment_mentions: List[str] = Field(default_factory=list)
    evidence: List[Evidence] = Field(default_factory=list)

    @field_validator("research_description_detailed", mode="before")
    @classmethod
    def _coerce_description(cls, v):
        return coerce_optional_str(v)

    @field_validator(
        "personal_site_urls",
        "inferred_labs",
        "inferred_lab_urls",
        "research_topics_specific",
        "research_details_raw_notes",
        "availability_or_recruitment_mentions",
        mode="before",
    )
    @classmethod
    def _coerce_lists(cls, v):
        return coerce_str_list(v)

    @field_validator("evidence", mode="before")
    @classmethod
    def _coerce_evidence(cls, v):
        if not isinstance(v, list):
            return []
        out = []
        for item in v:
            if isinstance(item, dict):
                out.append(
                    {
                        "source_url": str(item.get("source_url") or item.get("url") or ""),
                        "snippet": str(item.get("snippet") or item.get("text") or ""),
                    }
                )
            elif isinstance(item, str):
                out.append({"source_url": "", "snippet": item})
        return out


FACULTY_EXTRACT_SYSTEM_PROMPT = """You extract precise research/lab information from faculty webpages.

Rules:
- Return JSON only.
- Preserve specificity. Do not replace specific topics with generic umbrella terms.
- Infer likely lab/group affiliations even when implicit, but separate speculation from evidence in snippets.
- Keep recruitment/availability mentions as raw notes if present.
- Every major claim should have evidence with source_url and verbatim snippet.
- personal_site_urls: choose ONLY links from the provided candidate links that are
  the professor's own personal homepage or their research-group/lab website
  (e.g. a *.github.io page, a personal domain, publish.illinois.edu/<name>, or a
  named lab/group site). Do NOT include directory listings, social media,
  Google Scholar, DBLP, ORCID, or unrelated links.
- research_description_detailed: write 3-6 sentences describing what this
  professor ACTUALLY does day-to-day at a concrete, technical level (specific
  systems, methods, problems, applications) so they can be told apart from peers
  with the same umbrella area. Ground it in the provided text; do not invent.
  De-emphasize awards/honors and bio fluff.
"""


def _format_candidate_links(candidate_links: list[dict[str, str]]) -> str:
    if not candidate_links:
        return "(none)"
    lines = [f"- {item['url']} | {item.get('text', '')}" for item in candidate_links]
    return "\n".join(lines)


def _format_sources(sources: list[tuple[str, str]], budget: int = 45000) -> str:
    """Concatenate (url, text) source blocks within a character budget."""
    if not sources:
        return ""
    per_source = max(2000, budget // max(1, len(sources)))
    blocks: list[str] = []
    for url, text in sources:
        blocks.append(f"### SOURCE: {url}\n{text[:per_source]}")
    return "\n\n".join(blocks)


def extract_faculty_info(
    llm: LLMClient,
    professor_url: str,
    sources: list[tuple[str, str]],
    candidate_links: list[dict[str, str]] | None = None,
    settings: Settings | None = None,
    retriever: EmbeddingRetriever | None = None,
) -> FacultyExtraction:
    """Extract structured research info from one or more source pages.

    ``sources`` is a list of ``(url, cleaned_text)`` for the professor's profile
    page plus any crawled personal/lab pages. ``candidate_links`` are anchors
    pulled from the profile so the LLM can pick personal-site URLs. When
    ``settings`` is given, source text is reduced to the most research-relevant
    chunks via embedding retrieval before being sent to the LLM.
    """
    candidate_links = candidate_links or []
    if settings is not None:
        sources_text = build_research_corpus(settings, retriever, sources)
    else:
        sources_text = _format_sources(sources)
    user_prompt = f"""
Primary profile URL: {professor_url}

Candidate outbound links (choose personal/lab sites from here):
{_format_candidate_links(candidate_links)}

Source pages:
{sources_text}

Return JSON with keys:
- professor_name
- personal_site_urls (array of URLs chosen from candidate links; may be empty)
- inferred_labs (array of strings)
- inferred_lab_urls (array of URLs, may be empty)
- research_topics_specific (array of specific research topics)
- research_details_raw_notes (array of detailed points/examples of work)
- research_description_detailed (string: concrete description of what they actually do)
- availability_or_recruitment_mentions (array, raw mentions only)
- evidence (array of {{source_url, snippet}})
"""
    payload = llm.json_response(
        system_prompt=FACULTY_EXTRACT_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        schema=FacultyExtractionEnvelope,
        max_output_tokens=1500,
    )
    return FacultyExtraction(
        professor_name=payload.professor_name,
        professor_url=professor_url,
        source_urls=[url for url, _ in sources],
        personal_site_urls=payload.personal_site_urls,
        inferred_labs=payload.inferred_labs,
        inferred_lab_urls=[str(x) for x in payload.inferred_lab_urls],
        research_topics_specific=payload.research_topics_specific,
        research_details_raw_notes=payload.research_details_raw_notes,
        research_description_detailed=payload.research_description_detailed,
        availability_or_recruitment_mentions=payload.availability_or_recruitment_mentions,
        evidence=payload.evidence,  # pydantic coerces dict -> Evidence
    )
