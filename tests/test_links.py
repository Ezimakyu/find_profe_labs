from scraper.links import (
    extract_candidate_links,
    extract_internal_links,
    is_probably_personal_site,
    normalize_url,
    path_prefix_for,
    registrable_host,
)

PROFILE_HTML = """
<html><body>
  <a href="mailto:prof@illinois.edu">email</a>
  <a href="/about/people/all-faculty/other">Other Faculty</a>
  <a href="https://prof.cs.illinois.edu/">My homepage</a>
  <a href="https://example-lab.github.io/">Cool Lab</a>
  <a href="https://twitter.com/prof">twitter</a>
  <a href="cv.pdf">CV</a>
  <a href="#section">jump</a>
</body></html>
"""


def test_extract_candidate_links_filters_non_http_and_fragments():
    links = extract_candidate_links(PROFILE_HTML, "https://cs.illinois.edu/about/people/all-faculty/prof")
    urls = {item["url"] for item in links}
    assert "https://prof.cs.illinois.edu/" in urls
    assert "https://example-lab.github.io/" in urls
    # mailto, fragment and pdf are dropped
    assert all(not u.endswith(".pdf") for u in urls)
    assert all(not u.startswith("mailto:") for u in urls)
    assert all("#" not in u for u in urls)


def test_candidate_link_carries_anchor_text():
    links = extract_candidate_links(PROFILE_HTML, "https://cs.illinois.edu/x")
    by_url = {item["url"]: item["text"] for item in links}
    assert by_url["https://example-lab.github.io/"] == "Cool Lab"


def test_is_probably_personal_site_excludes_directory_and_social():
    assert is_probably_personal_site("https://prof.cs.illinois.edu/") is True
    assert is_probably_personal_site("https://example-lab.github.io/") is True
    assert is_probably_personal_site("https://cs.illinois.edu/about/people") is False
    assert is_probably_personal_site("https://twitter.com/prof") is False
    assert is_probably_personal_site("https://scholar.google.com/x") is False


def test_registrable_host():
    assert registrable_host("https://prof.cs.illinois.edu/page") == "illinois.edu"
    # Multi-tenant hosts include the tenant label so tenants are distinct sites.
    assert registrable_host("https://example-lab.github.io/x") == "example-lab.github.io"
    assert registrable_host("https://other-lab.github.io/y") == "other-lab.github.io"
    assert registrable_host("https://mysite.netlify.app/") == "mysite.netlify.app"


def test_normalize_url_strips_trailing_slash_and_fragment():
    assert normalize_url("https://lab.illinois.edu/research/") == "https://lab.illinois.edu/research"
    assert normalize_url("https://lab.illinois.edu/research#sec") == "https://lab.illinois.edu/research"
    assert normalize_url("https://lab.illinois.edu/") == "https://lab.illinois.edu/"


def test_extract_internal_links_skips_boilerplate_and_prioritizes_research():
    html = """
    <a href="/alumni">Alumni</a>
    <a href="/news">News</a>
    <a href="/contact">Contact</a>
    <a href="/research">Research Areas</a>
    <a href="/projects">Projects</a>
    <a href="/random-page">Random</a>
    """
    links = extract_internal_links(
        html,
        "https://lab.illinois.edu/",
        skip_boilerplate=True,
        prioritize_research=True,
    )
    assert all("/alumni" not in u and "/news" not in u and "/contact" not in u for u in links)
    # Research/projects pages should be ordered before the unrelated page.
    assert links.index("https://lab.illinois.edu/research") < links.index("https://lab.illinois.edu/random-page")
    assert "https://lab.illinois.edu/projects" in links


def test_extract_internal_links_same_domain_only():
    html = """
    <a href="/research">Research</a>
    <a href="https://lab.illinois.edu/people">People</a>
    <a href="https://other.com/x">External</a>
    """
    links = extract_internal_links(html, "https://lab.illinois.edu/")
    assert "https://lab.illinois.edu/research" in links
    assert "https://lab.illinois.edu/people" in links
    assert all("other.com" not in u for u in links)


def test_path_prefix_for_tilde_user():
    assert path_prefix_for("http://luthuli.cs.uiuc.edu/~daf/") == "/~daf/"
    assert path_prefix_for("http://luthuli.cs.uiuc.edu/~daf/tracking.html") == "/~daf/"
    # Ordinary sites get no path constraint.
    assert path_prefix_for("https://prof.example.com/research") == ""


def test_extract_internal_links_same_host_and_path_prefix():
    html = """
    <a href="tracking.html">Tracking</a>
    <a href="/~someoneelse/">Other person</a>
    <a href="https://prof.cs.illinois.edu/x">Different host</a>
    """
    base = "http://luthuli.cs.uiuc.edu/~daf/"
    links = extract_internal_links(
        html,
        base,
        same_registrable_domain=False,
        same_host=True,
        path_prefix="/~daf/",
    )
    assert "http://luthuli.cs.uiuc.edu/~daf/tracking.html" in links
    # A co-tenant on the same host (different /~user/) is excluded by path prefix.
    assert all("someoneelse" not in u for u in links)
    # A different host on the same registrable domain is excluded by same_host.
    assert all("prof.cs.illinois.edu" not in u for u in links)
