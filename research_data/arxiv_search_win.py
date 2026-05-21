"""Windows-compatible arXiv search script (no fcntl dependency).

Usage:
    python arxiv_search_win.py --query "Bosch Hale" --max_results 8 --output results.json
"""

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

_BASE_URL = "http://export.arxiv.org/api/query?"
_MIN_INTERVAL = 3.1  # seconds between requests (arXiv TOS: 1 req/3s)
_last_request_time = 0.0


def rate_limited_get(url: str) -> bytes:
    global _last_request_time
    now = time.monotonic()
    gap = _MIN_INTERVAL - (now - _last_request_time)
    if gap > 0:
        time.sleep(gap)
    req = urllib.request.Request(url, headers={"User-Agent": "SpinnyBallResearch/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = resp.read()
    _last_request_time = time.monotonic()
    return data


def strip_namespace(tag: str) -> str:
    if tag.startswith("{"):
        return tag.split("}", 1)[1]
    return tag


def search_arxiv(query: str, max_results: int = 10, sort_by: str = "relevance",
                 sort_order: str = "descending", start: int = 0) -> list:
    params = {
        "search_query": query,
        "start": start,
        "max_results": max_results,
        "sortBy": sort_by,
        "sortOrder": sort_order,
    }
    query_string = urllib.parse.urlencode(params, quote_via=urllib.parse.quote_plus)
    url = _BASE_URL + query_string
    xml_data = rate_limited_get(url)
    root = ET.fromstring(xml_data)
    results = []
    for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
        paper = {}
        authors = []
        for child in entry:
            tag = strip_namespace(child.tag)
            if tag == "id":
                if child.text:
                    paper["id"] = child.text.split("/abs/")[-1]
                    paper["abs_url"] = "https://arxiv.org/abs/" + paper["id"].split("v")[0]
            elif tag == "title":
                paper["title"] = child.text.replace("\n", " ").strip() if child.text else ""
            elif tag == "summary":
                paper["summary"] = child.text.replace("\n", " ").strip() if child.text else ""
            elif tag == "published":
                paper["published"] = child.text
            elif tag == "author":
                for name_node in child.findall("{http://www.w3.org/2005/Atom}name"):
                    authors.append(name_node.text)
            elif tag == "link":
                if child.get("title") == "pdf":
                    paper["pdf_url"] = child.get("href")
            elif tag == "primary_category":
                paper["primary_category"] = child.get("term")
            elif tag in {"doi", "journal_ref", "comment"}:
                paper[tag] = child.text
        paper["authors"] = authors
        results.append(paper)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--max_results", type=int, default=8)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sort_by", default="relevance")
    args = parser.parse_args()

    results = search_arxiv(args.query, args.max_results, sort_by=args.sort_by)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"query": args.query, "count": len(results), "papers": results}, f, indent=2)
    print(f"Saved {len(results)} results to {args.output}")


if __name__ == "__main__":
    main()
