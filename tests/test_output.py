import csv

from scraper.models import FacultyExtraction, FacultyRecord
from scraper.output import (
    build_lab_candidates,
    write_labs_csv,
    write_professors_csv,
)


def _records_and_extractions():
    records = [
        FacultyRecord(name="Alice Ng", url="https://cs.illinois.edu/about/people/all-faculty/alice", department="cs"),
        FacultyRecord(name="Bob Lee", url="https://cs.illinois.edu/about/people/all-faculty/bob", department="cs"),
    ]
    extractions = [
        FacultyExtraction(
            professor_name="Alice Ng",
            professor_url="https://cs.illinois.edu/about/people/all-faculty/alice",
            source_urls=[
                "https://cs.illinois.edu/about/people/all-faculty/alice",
                "https://alice.cs.illinois.edu/",
            ],
            personal_site_urls=["https://alice.cs.illinois.edu/"],
            inferred_labs=["Machine Learning Lab"],
            research_topics_specific=["graph neural networks"],
            research_description_detailed="Builds GNN training systems for molecular data.",
        ),
        FacultyExtraction(
            professor_name="Bob Lee",
            professor_url="https://cs.illinois.edu/about/people/all-faculty/bob",
            inferred_labs=["Machine Learning Lab"],
            research_topics_specific=["reinforcement learning"],
            research_description_detailed="Designs RL controllers for robotic manipulation.",
        ),
    ]
    return records, extractions


def test_build_lab_candidates_tracks_professors():
    records, extractions = _records_and_extractions()
    labs, links = build_lab_candidates(records, extractions)
    assert len(labs) == 1
    lab = labs[0]
    assert lab.source_faculty_count == 2
    assert set(lab.professors) == {"Alice Ng", "Bob Lee"}
    assert len(links) == 2


def test_write_labs_csv_has_professors_column(tmp_path):
    records, extractions = _records_and_extractions()
    labs, _ = build_lab_candidates(records, extractions)
    out = tmp_path / "labs.csv"
    write_labs_csv(out, labs)
    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    assert "professors" in rows[0]
    assert "Alice Ng" in rows[0]["professors"]
    assert "Bob Lee" in rows[0]["professors"]


def test_write_professors_csv_has_sources_and_description(tmp_path):
    records, extractions = _records_and_extractions()
    out = tmp_path / "professors.csv"
    write_professors_csv(out, records, extractions)
    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    assert len(rows) == 2
    alice = next(r for r in rows if r["professor_name"] == "Alice Ng")
    assert "alice.cs.illinois.edu" in alice["source_urls"]
    assert "alice.cs.illinois.edu" in alice["personal_site_urls"]
    assert "GNN" in alice["research_description_detailed"]
    assert alice["department"] == "cs"
