from __future__ import annotations

from typing import Any, List, Literal, Optional, Union

from pydantic import BaseModel, Field, HttpUrl


def _stringify(value: Any) -> str:
    """Flatten a scalar or dict into a readable string."""
    if isinstance(value, dict):
        parts = [f"{k}: {v}" for k, v in value.items() if v not in (None, "")]
        return " | ".join(parts)
    return str(value)


def coerce_str_list(value: Any) -> list[str]:
    """Coerce assorted LLM shapes into a clean list[str].

    LLMs sometimes return a bare string, ``None``, or a list of objects where we
    asked for a list of strings. Normalize all of these so validation never
    crashes mid-run over hundreds of pages.
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, dict):
        return [_stringify(value)]
    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for item in value:
            text = _stringify(item).strip()
            if text:
                out.append(text)
        return out
    return [str(value)]


def coerce_optional_str(value: Any) -> Optional[str]:
    """Coerce a possibly list/dict value into an optional string."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return " ".join(_stringify(item) for item in value)
    return _stringify(value)


class Evidence(BaseModel):
    source_url: Union[HttpUrl, str]
    snippet: str


class FacultyExtraction(BaseModel):
    professor_name: Optional[str] = None
    professor_url: Union[HttpUrl, str]
    source_urls: List[str] = Field(default_factory=list)
    personal_site_urls: List[str] = Field(default_factory=list)
    inferred_labs: List[str] = Field(default_factory=list)
    inferred_lab_urls: List[Union[HttpUrl, str]] = Field(default_factory=list)
    research_topics_specific: List[str] = Field(default_factory=list)
    research_details_raw_notes: List[str] = Field(default_factory=list)
    research_description_detailed: Optional[str] = None
    availability_or_recruitment_mentions: List[str] = Field(default_factory=list)
    evidence: List[Evidence] = Field(default_factory=list)


class FacultyRecord(BaseModel):
    name: Optional[str] = None
    url: str
    department: Literal["cs", "ece", "unknown"] = "unknown"


class LabCandidate(BaseModel):
    lab_name: str
    canonical_url: Optional[str] = None
    source_faculty_count: int = 0
    professors: List[str] = Field(default_factory=list)
    departments: List[str] = Field(default_factory=list)
    llm_evidence_count: int = 0


class BlockedUrl(BaseModel):
    url: str
    reason: str
    status_code: Optional[int] = None

