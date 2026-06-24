from __future__ import annotations

from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field, HttpUrl


class Evidence(BaseModel):
    source_url: Union[HttpUrl, str]
    snippet: str


class FacultyExtraction(BaseModel):
    professor_name: Optional[str] = None
    professor_url: Union[HttpUrl, str]
    inferred_labs: List[str] = Field(default_factory=list)
    inferred_lab_urls: List[Union[HttpUrl, str]] = Field(default_factory=list)
    research_topics_specific: List[str] = Field(default_factory=list)
    research_details_raw_notes: List[str] = Field(default_factory=list)
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
    departments: List[str] = Field(default_factory=list)
    llm_evidence_count: int = 0


class BlockedUrl(BaseModel):
    url: str
    reason: str
    status_code: Optional[int] = None

