from scraper import search
from scraper.search import SearchResult, _blocked, search_professor_homepage_candidates


def test_blocked_excludes_directory_and_aggregators():
    # Aggregators / social / encyclopedias.
    assert _blocked("https://en.wikipedia.org/wiki/David_Forsyth")
    assert _blocked("https://scholar.google.com/citations?user=x")
    assert _blocked("https://www.ratemyprofessors.com/professor/1")
    assert _blocked("https://research.com/u/david-forsyth")
    # Directory profile pages, by path, on any illinois host.
    assert _blocked("https://siebelschool.illinois.edu/about/people/faculty/daf")
    assert _blocked("https://grainger.illinois.edu/about/directory/faculty/slazebni")


def test_blocked_allows_personal_sites():
    # A shared ~user faculty server and a personal university subdomain.
    assert not _blocked("http://luthuli.cs.uiuc.edu/~daf/")
    assert not _blocked("http://slazebni.cs.illinois.edu/")
    assert not _blocked("https://someprof.github.io/")


def test_homepage_candidates_filter(monkeypatch):
    fake_results = [
        SearchResult("https://en.wikipedia.org/wiki/David_Forsyth", "wiki", ""),
        SearchResult("http://luthuli.cs.uiuc.edu/~daf/", "David Forsyth", ""),
        SearchResult("https://siebelschool.illinois.edu/about/people/faculty/daf", "dir", ""),
        SearchResult("https://hal.science/hal-01063327", "paper", ""),
    ]
    monkeypatch.setattr(search, "web_search", lambda q, max_results=5: fake_results)
    cands = search_professor_homepage_candidates("David Forsyth", "cs", max_results=5)
    urls = [c.url for c in cands]
    assert "http://luthuli.cs.uiuc.edu/~daf/" in urls
    # Wikipedia and the directory page are filtered out before the LLM sees them.
    assert all("wikipedia.org" not in u for u in urls)
    assert all("/about/people" not in u for u in urls)


def test_homepage_candidates_empty_name():
    assert search_professor_homepage_candidates("", "cs") == []


def test_search_circuit_breaker_trips_and_resets():
    search.reset_search_breaker()
    assert not search._search_is_disabled()
    for _ in range(search._FAILURE_THRESHOLD):
        search._record_search_outcome(False)
    assert search._search_is_disabled()
    # A success mid-streak clears the counter, so the breaker won't trip.
    search.reset_search_breaker()
    for _ in range(search._FAILURE_THRESHOLD - 1):
        search._record_search_outcome(False)
    search._record_search_outcome(True)
    search._record_search_outcome(False)
    assert not search._search_is_disabled()
    search.reset_search_breaker()


def test_web_search_returns_empty_when_breaker_open(monkeypatch):
    search.reset_search_breaker()
    for _ in range(search._FAILURE_THRESHOLD):
        search._record_search_outcome(False)

    def _boom(*a, **k):
        raise AssertionError("ddgs must not be called while the breaker is open")

    monkeypatch.setattr(search, "_run_ddg", _boom)
    assert search.web_search("anything") == []
    search.reset_search_breaker()
