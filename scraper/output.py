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
        lambda: {"lab_name": "", "canonical_url": "", "faculty_urls": set(), "departments": set(), "evidence_count": 0}
    )
    links: list[dict[str, str]] = []

    for extraction in extractions:
        department = department_by_url.get(extraction.professor_url, "unknown")
        for idx, lab_name in enumerate(extraction.inferred_labs):
            clean_name = _normalize_lab_name(lab_name)
            if not clean_name:
                continue
            url = str(extraction.inferred_lab_urls[idx]) if idx < len(extraction.inferred_lab_urls) else ""
            key = url or clean_name.lower()
            bucket[key]["lab_name"] = clean_name
            bucket[key]["canonical_url"] = url
            bucket[key]["faculty_urls"].add(extraction.professor_url)
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

