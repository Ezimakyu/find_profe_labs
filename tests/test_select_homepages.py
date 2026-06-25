from scraper.extract_llm import HomepageSelectionEnvelope, select_homepages


class FakeLLM:
    def __init__(self, returned_urls):
        self.returned_urls = returned_urls
        self.calls = 0

    def json_response(self, system_prompt, user_prompt, schema, max_output_tokens=400):
        self.calls += 1
        return HomepageSelectionEnvelope(homepage_urls=self.returned_urls)


def test_select_homepages_keeps_only_offered_urls():
    candidates = [
        {"url": "http://luthuli.cs.uiuc.edu/~daf/", "text": "homepage"},
        {"url": "https://en.wikipedia.org/wiki/x", "text": "wiki"},
    ]
    # LLM hallucinates an extra URL that was never offered; it must be dropped.
    llm = FakeLLM(["http://luthuli.cs.uiuc.edu/~daf/", "https://made-up.example/"])
    chosen = select_homepages(llm, "David Forsyth", candidates)
    assert chosen == ["http://luthuli.cs.uiuc.edu/~daf/"]


def test_select_homepages_empty_inputs():
    assert select_homepages(FakeLLM([]), "", []) == []
    assert select_homepages(FakeLLM([]), "Someone", []) == []


def test_select_homepages_handles_llm_failure():
    class BoomLLM:
        def json_response(self, *a, **k):
            raise RuntimeError("boom")

    chosen = select_homepages(BoomLLM(), "Someone", [{"url": "https://x.io/", "text": "t"}])
    assert chosen == []
