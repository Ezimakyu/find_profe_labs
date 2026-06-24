from __future__ import annotations

from typing import List, Optional, Union

from pydantic import BaseModel, Field, HttpUrl

from scraper.llm_client import LLMClient
from scraper.models import Evidence
from scraper.models import FacultyExtraction


class FacultyExtractionEnvelope(BaseModel):
    professor_name: Optional[str] = None
    inferred_labs: List[str] = Field(default_factory=list)
    inferred_lab_urls: List[Union[HttpUrl, str]] = Field(default_factory=list)
    research_topics_specific: List[str] = Field(default_factory=list)
    research_details_raw_notes: List[str] = Field(default_factory=list)
    availability_or_recruitment_mentions: List[str] = Field(default_factory=list)
    evidence: List[Evidence] = Field(default_factory=list)


FACULTY_EXTRACT_SYSTEM_PROMPT = """You extract precise research/lab information from faculty webpages.

Rules:
- Return JSON only.
- Preserve specificity. Do not replace specific topics with generic umbrella terms.
- Infer likely lab/group affiliations even when implicit, but separate speculation from evidence in snippets.
- Keep recruitment/availability mentions as raw notes if present.
- Every major claim should have evidence with source_url and verbatim snippet.
"""


def extract_faculty_info(
    llm: LLMClient,
    professor_url: str,
    cleaned_text: str,
) -> FacultyExtraction:
    user_prompt = f"""
Source URL: {professor_url}

Page text:
{cleaned_text[:45000]}

Return JSON with keys:
- professor_name
- inferred_labs (array of strings)
- inferred_lab_urls (array of URLs, may be empty)
- research_topics_specific (array of specific research topics)
- research_details_raw_notes (array of detailed points/examples of work)
- availability_or_recruitment_mentions (array, raw mentions only)
- evidence (array of {{source_url, snippet}})
"""
    payload = llm.json_response(
        system_prompt=FACULTY_EXTRACT_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        schema=FacultyExtractionEnvelope,
        max_output_tokens=3500,
    )
    return FacultyExtraction(
        professor_name=payload.professor_name,
        professor_url=professor_url,
        inferred_labs=payload.inferred_labs,
        inferred_lab_urls=[str(x) for x in payload.inferred_lab_urls],
        research_topics_specific=payload.research_topics_specific,
        research_details_raw_notes=payload.research_details_raw_notes,
        availability_or_recruitment_mentions=payload.availability_or_recruitment_mentions,
        evidence=payload.evidence,  # pydantic coerces dict -> Evidence
    )

