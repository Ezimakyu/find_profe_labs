from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from scraper.models import FacultyExtraction, FacultyRecord, LabCandidate


def ensure_dirs(base: Path) -> None:
    base.mkdir(parents=True, exist_ok=True)
    (base / "lab_test").mkdir(parents=True, exist_ok=True)


def _normalize_lab_name(name: str) -> str:
    return " ".join(name.split()).strip()


def build_lab_candidates(
    faculty_records: list[FacultyRecord],
    extractions: list[FacultyExtraction],
) -> tuple[list[LabCandidate], list[dict[str, str]]]:
    department_by_url = {item.url: item.department for item in faculty_records}
    bucket: dict[str, dict] = defaultdict(
        lambda: {
            "lab_name": "",
            "canonical_url": "",
            "faculty_urls": set(),
            "professors": set(),
            "departments": set(),
            "evidence_count": 0,
        }
    )
    links: list[dict[str, str]] = []

    for extraction in extractions:
        department = department_by_url.get(extraction.professor_url, "unknown")
        professor_label = extraction.professor_name or extraction.professor_url
        for idx, lab_name in enumerate(extraction.inferred_labs):
            clean_name = _normalize_lab_name(lab_name)
            if not clean_name:
                continue
            url = str(extraction.inferred_lab_urls[idx]) if idx < len(extraction.inferred_lab_urls) else ""
            key = url or clean_name.lower()
            bucket[key]["lab_name"] = clean_name
            bucket[key]["canonical_url"] = url
            bucket[key]["faculty_urls"].add(extraction.professor_url)
            bucket[key]["professors"].add(professor_label)
            bucket[key]["departments"].add(department)
            bucket[key]["evidence_count"] += len(extraction.evidence)
            links.append(
                {
                    "faculty_url": extraction.professor_url,
                    "faculty_name": extraction.professor_name or "",
                    "lab_name": clean_name,
                    "lab_url": url,
                    "evidence_snippet": extraction.evidence[0].snippet if extraction.evidence else "",
                }
            )

    labs = [
        LabCandidate(
            lab_name=item["lab_name"],
            canonical_url=item["canonical_url"] or None,
            source_faculty_count=len(item["faculty_urls"]),
            professors=sorted(item["professors"]),
            departments=sorted(item["departments"]),
            llm_evidence_count=item["evidence_count"],
        )
        for item in bucket.values()
    ]
    labs.sort(key=lambda x: (-x.source_faculty_count, x.lab_name.lower()))
    return labs, links


def write_labs_csv(path: Path, labs: list[LabCandidate]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "lab_name",
                "canonical_url",
                "source_faculty_count",
                "professors",
                "departments",
                "llm_evidence_count",
            ],
        )
        writer.writeheader()
        for lab in labs:
            writer.writerow(
                {
                    "lab_name": lab.lab_name,
                    "canonical_url": lab.canonical_url or "",
                    "source_faculty_count": lab.source_faculty_count,
                    "professors": "|".join(lab.professors),
                    "departments": "|".join(lab.departments),
                    "llm_evidence_count": lab.llm_evidence_count,
                }
            )


def write_faculty_lab_links_csv(path: Path, links: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["faculty_url", "faculty_name", "lab_name", "lab_url", "evidence_snippet"],
        )
        writer.writeheader()
        writer.writerows(links)


def write_professors_csv(path: Path, faculty_records: list[FacultyRecord], extractions: list[FacultyExtraction]) -> None:
    """Per-professor CSV with multiple source URLs and a detailed research description.

    The ``research_description_detailed`` column captures "what they actually do"
    so professors that share an umbrella area (e.g. "machine learning") can be
    told apart by concrete systems/methods/applications.
    """
    department_by_url = {item.url: item.department for item in faculty_records}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "professor_name",
                "department",
                "profile_url",
                "source_urls",
                "personal_site_urls",
                "labs",
                "research_topics",
                "research_description_detailed",
                "recruitment_mentions",
                "evidence_count",
            ],
        )
        writer.writeheader()
        for extraction in extractions:
            writer.writerow(
                {
                    "professor_name": extraction.professor_name or "",
                    "department": department_by_url.get(extraction.professor_url, "unknown"),
                    "profile_url": extraction.professor_url,
                    "source_urls": "|".join(extraction.source_urls),
                    "personal_site_urls": "|".join(extraction.personal_site_urls),
                    "labs": "|".join(extraction.inferred_labs),
                    "research_topics": "|".join(extraction.research_topics_specific),
                    "research_description_detailed": extraction.research_description_detailed or "",
                    "recruitment_mentions": "|".join(extraction.availability_or_recruitment_mentions),
                    "evidence_count": len(extraction.evidence),
                }
            )


def write_blocked_urls_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["url", "reason", "status_code"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "url": row.get("url", ""),
                    "reason": row.get("reason", ""),
                    "status_code": row.get("status_code", ""),
                }
            )


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)

