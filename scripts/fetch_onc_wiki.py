#!/usr/bin/env python3
import html
import re
import sys
from html.parser import HTMLParser
from urllib.request import Request, urlopen


class TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.in_li = False

    def handle_starttag(self, tag, attrs):
        if tag in ("p", "br", "tr", "h1", "h2", "h3", "h4", "h5", "h6"):
            self.parts.append("\n")
        elif tag == "li":
            self.parts.append("\n- ")
            self.in_li = True
        elif tag in ("th", "td"):
            self.parts.append("\t")

    def handle_endtag(self, tag):
        if tag == "li":
            self.in_li = False
        if tag == "tr":
            self.parts.append("\n")

    def handle_data(self, data):
        if data:
            self.parts.append(data)


def fetch(url, cookie=None):
    headers = {"User-Agent": "Mozilla/5.0"}
    if cookie:
        headers["Cookie"] = cookie
    req = Request(url, headers=headers)
    with urlopen(req) as resp:
        return resp.read().decode("utf-8", "replace")


def to_text(html_doc):
    parser = TextExtractor()
    parser.feed(html_doc)
    text = html.unescape("".join(parser.parts))
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def main():
    if len(sys.argv) < 2:
        print("Usage: fetch_onc_wiki.py URL [--raw]")
        sys.exit(1)

    url = sys.argv[1]
    raw = "--raw" in sys.argv[2:]

    # Optional: pass cookie via env var if the page requires login
    # export ONC_COOKIE="cookie1=value1; cookie2=value2"
    cookie = None

    html_doc = fetch(url, cookie=cookie)
    if raw:
        print(html_doc)
    else:
        print(to_text(html_doc))


if __name__ == "__main__":
    main()
