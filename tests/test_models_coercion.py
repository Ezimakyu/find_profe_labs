from scraper.extract_llm import FacultyExtractionEnvelope
from scraper.models import coerce_optional_str, coerce_str_list


def test_coerce_str_list_handles_shapes():
    assert coerce_str_list(None) == []
    assert coerce_str_list("hello") == ["hello"]
    assert coerce_str_list("") == []
    assert coerce_str_list(["a", "b"]) == ["a", "b"]
    assert coerce_str_list([{"title": "LLVM", "description": "compiler"}]) == ["title: LLVM | description: compiler"]


def test_coerce_optional_str_handles_shapes():
    assert coerce_optional_str(None) is None
    assert coerce_optional_str("x") == "x"
    assert coerce_optional_str(["a", "b"]) == "a b"


def test_envelope_coerces_messy_llm_output():
    env = FacultyExtractionEnvelope.model_validate(
        {
            "professor_name": "Vikram Adve",
            "personal_site_urls": "https://vikram.example.edu/",
            "inferred_labs": ["LLVM Group"],
            "inferred_lab_urls": [{"url": "https://llvm.org"}],
            "research_topics_specific": [{"name": "compilers"}],
            "research_description_detailed": ["Builds", "compilers"],
            "availability_or_recruitment_mentions": "Email to join.",
            "evidence": ["a verbatim snippet", {"url": "https://x.edu", "text": "snip"}],
        }
    )
    assert env.personal_site_urls == ["https://vikram.example.edu/"]
    assert env.research_description_detailed == "Builds compilers"
    assert env.availability_or_recruitment_mentions == ["Email to join."]
    assert len(env.evidence) == 2
    assert env.evidence[0].snippet == "a verbatim snippet"
    assert env.evidence[1].source_url == "https://x.edu"
