from __future__ import annotations

from bs4 import BeautifulSoup


def clean_html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")

    for tag_name in ("script", "style", "noscript", "svg", "iframe"):
        for node in soup.find_all(tag_name):
            node.decompose()

    # Remove common boilerplate blocks, but keep body content broad.
    for selector in (
        "nav",
        "footer",
        ".footer",
        ".site-footer",
        ".cookie",
        ".cookies",
        ".legal",
    ):
        for node in soup.select(selector):
            node.decompose()

    text = soup.get_text("\n", strip=True)
    lines = []
    for raw_line in text.splitlines():
        line = " ".join(raw_line.split())
        if not line:
            continue
        # Drop obvious navigation crumbs and tiny menu fragments.
        if len(line) < 3:
            continue
        lines.append(line)
    return "\n".join(lines)

