from scraper.discovery import _dedupe_by_name, _normalize_name
from scraper.models import FacultyRecord


def test_normalize_name_strips_titles_and_case():
    assert _normalize_name("Prof. David  Forsyth") == "david forsyth"
    assert _normalize_name("Dr. Nancy M. Amato") == "nancy m. amato"
    assert _normalize_name(None) == ""


def test_dedupe_collapses_cross_listed_professor():
    records = [
        FacultyRecord(name="David Forsyth", url="https://cs.illinois.edu/x/daf", department="cs"),
        FacultyRecord(name="David Forsyth", url="https://ece.illinois.edu/y/daf", department="ece"),
        FacultyRecord(name="Jane Doe", url="https://cs.illinois.edu/x/jane", department="cs"),
    ]
    out = _dedupe_by_name(records)
    assert [r.url for r in out] == [
        "https://cs.illinois.edu/x/daf",
        "https://cs.illinois.edu/x/jane",
    ]


def test_dedupe_keeps_unnamed_records():
    records = [
        FacultyRecord(name=None, url="https://cs.illinois.edu/a", department="cs"),
        FacultyRecord(name=None, url="https://cs.illinois.edu/b", department="cs"),
    ]
    assert len(_dedupe_by_name(records)) == 2
