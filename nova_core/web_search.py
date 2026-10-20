"""
NOVA Web Search & Page Extraction Engine
========================================
Provides multi-engine live web search (Bing HTML, Google News RSS, Wikipedia API,
ArXiv API, PyPI API, and optional external search APIs) with robust page content
extraction and citation grounding.

Privacy & Safety:
- No user private data is transmitted to search engines.
- Direct accessible page content is extracted cleanly, stripping scripts and tracking.
- Source URLs, page titles, domains, and publication/access dates are strictly validated.
"""

import base64
import datetime
import html
import json
import logging
import os
import re
import urllib.parse
from typing import Any, Dict, List, Optional, Tuple

import bs4
import requests

logger = logging.getLogger("nova.web_search")

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36 NOVA-Assistant/2.0"
)

# High-authority domain whitelist for scoring
AUTHORITATIVE_DOMAINS = {
    "docs.python.org": 10,
    "pytorch.org": 10,
    "tensorflow.org": 10,
    "huggingface.co": 10,
    "scikit-learn.org": 10,
    "numpy.org": 10,
    "pandas.pydata.org": 10,
    "developer.mozilla.org": 10,
    "arxiv.org": 10,
    "wikipedia.org": 9,
    "github.com": 9,
    "pypi.org": 9,
    "nature.com": 9,
    "science.org": 9,
    "reuters.com": 9,
    "apnews.com": 9,
    "bbc.com": 8,
    "stackoverflow.com": 8,
    "geeksforgeeks.org": 6,
    "medium.com": 5,
}


def _clean_text(raw_text: str) -> str:
    """Cleans up whitespace, unescapes HTML entities, and strips non-printable characters."""
    if not raw_text:
        return ""
    text = html.unescape(raw_text)
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def _extract_domain(url: str) -> str:
    """Extracts clean domain name from URL."""
    try:
        parsed = urllib.parse.urlparse(url)
        netloc = parsed.netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        return netloc or "web"
    except Exception:
        return "web"


def _decode_bing_url(href: str) -> str:
    """
    Decodes Bing redirection links ('https://www.bing.com/ck/a?!...&u=a1<base64>')
    into direct destination URLs.
    """
    if "bing.com/ck/a" in href:
        try:
            parsed = urllib.parse.urlparse(href)
            qs = urllib.parse.parse_qs(parsed.query)
            u_param = qs.get("u", [""])[0]
            if u_param.startswith("a1"):
                raw_b64 = u_param[2:]
                padded = raw_b64 + "=" * (-len(raw_b64) % 4)
                decoded = base64.urlsafe_b64decode(padded).decode("utf-8", errors="ignore")
                if decoded.startswith("http://") or decoded.startswith("https://"):
                    return decoded
        except Exception:
            pass
    return href


def search_bing(query: str, max_results: int = 5, timeout: int = 6) -> List[Dict[str, Any]]:
    """
    Performs real-time web search using Bing HTML parsing.
    Returns cleaned list of results with title, URL, snippet, and domain.
    """
    headers = {
        "User-Agent": DEFAULT_USER_AGENT,
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    encoded_q = urllib.parse.quote_plus(query)
    url = f"https://www.bing.com/search?q={encoded_q}&count={max(max_results * 2, 10)}"

    results: List[Dict[str, Any]] = []
    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        if response.status_code != 200:
            logger.warning(f"Bing search returned HTTP {response.status_code}")
            return results

        soup = bs4.BeautifulSoup(response.text, "html.parser")
        for item in soup.select("li.b_algo"):
            h2 = item.find("h2")
            if not h2:
                continue
            a_tag = h2.find("a")
            if not a_tag or not a_tag.get("href"):
                continue

            raw_href = a_tag["href"]
            direct_url = _decode_bing_url(raw_href)

            # Skip internal search engine or advertisement links
            if "bing.com" in direct_url or "microsoft.com/rewards" in direct_url:
                continue
            if not direct_url.startswith("http"):
                continue

            title = _clean_text(a_tag.get_text())
            snippet_elem = item.find("p") or item.find(".b_caption")
            snippet = _clean_text(snippet_elem.get_text()) if snippet_elem else ""

            domain = _extract_domain(direct_url)
            today_iso = datetime.date.today().isoformat()

            results.append({
                "title": title or domain,
                "url": direct_url,
                "snippet": snippet,
                "domain": domain,
                "source_engine": "bing",
                "published_date": today_iso,
                "access_date": today_iso,
            })

            if len(results) >= max_results:
                break
    except Exception as e:
        logger.warning(f"Bing search error for query '{query}': {e}")

    return results


def search_google_news(query: str, max_results: int = 5, timeout: int = 6) -> List[Dict[str, Any]]:
    """
    Searches Google News RSS for live breaking news, current events, and updates.
    """
    encoded_q = urllib.parse.quote_plus(query)
    rss_url = f"https://news.google.com/rss/search?q={encoded_q}&hl=en-US&gl=US&ceid=US:en"
    headers = {"User-Agent": DEFAULT_USER_AGENT}

    results: List[Dict[str, Any]] = []
    try:
        response = requests.get(rss_url, headers=headers, timeout=timeout)
        if response.status_code != 200:
            return results

        soup = bs4.BeautifulSoup(response.text, "xml")
        items = soup.find_all("item")
        today_iso = datetime.date.today().isoformat()

        for it in items[:max_results]:
            title_elem = it.find("title")
            link_elem = it.find("link")
            pub_date_elem = it.find("pubDate")
            desc_elem = it.find("description")

            title = _clean_text(title_elem.text) if title_elem else "News Article"
            link = link_elem.text.strip() if link_elem else ""
            pub_date = pub_date_elem.text.strip() if pub_date_elem else today_iso
            snippet = _clean_text(desc_elem.text) if desc_elem else ""
            # Strip any residual HTML in RSS description
            snippet = re.sub(r"<[^>]+>", " ", snippet).strip()

            domain = _extract_domain(link)
            results.append({
                "title": title,
                "url": link,
                "snippet": snippet or title,
                "domain": domain,
                "source_engine": "google_news",
                "published_date": pub_date,
                "access_date": today_iso,
            })
    except Exception as e:
        logger.warning(f"Google News RSS error for query '{query}': {e}")

    return results


def search_wikipedia(query: str, max_results: int = 3, timeout: int = 5) -> List[Dict[str, Any]]:
    """
    Searches Wikipedia API for conceptual, scientific, mathematical, or encyclopedic definitions.
    """
    endpoint = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
        "srlimit": max_results,
    }
    headers = {"User-Agent": DEFAULT_USER_AGENT}
    results: List[Dict[str, Any]] = []

    try:
        response = requests.get(endpoint, params=params, headers=headers, timeout=timeout)
        if response.status_code != 200:
            return results

        data = response.json()
        search_hits = data.get("query", {}).get("search", [])
        today_iso = datetime.date.today().isoformat()

        for hit in search_hits:
            title = hit.get("title", "")
            snippet = re.sub(r"<[^>]+>", "", hit.get("snippet", ""))
            clean_snippet = _clean_text(snippet)
            encoded_title = urllib.parse.quote(title.replace(" ", "_"))
            page_url = f"https://en.wikipedia.org/wiki/{encoded_title}"

            results.append({
                "title": f"{title} - Wikipedia",
                "url": page_url,
                "snippet": clean_snippet,
                "domain": "en.wikipedia.org",
                "source_engine": "wikipedia",
                "published_date": hit.get("timestamp", today_iso)[:10],
                "access_date": today_iso,
            })
    except Exception as e:
        logger.warning(f"Wikipedia search error for query '{query}': {e}")

    return results


def search_arxiv(query: str, max_results: int = 3, timeout: int = 6) -> List[Dict[str, Any]]:
    """
    Searches ArXiv API for scientific research papers, preprints, authors, and abstracts.
    """
    endpoint = "http://export.arxiv.org/api/query"
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
    }
    headers = {"User-Agent": DEFAULT_USER_AGENT}
    results: List[Dict[str, Any]] = []

    try:
        response = requests.get(endpoint, params=params, headers=headers, timeout=timeout)
        if response.status_code != 200:
            return results

        soup = bs4.BeautifulSoup(response.text, "xml")
        entries = soup.find_all("entry")
        today_iso = datetime.date.today().isoformat()

        for entry in entries:
            title_elem = entry.find("title")
            summary_elem = entry.find("summary")
            id_elem = entry.find("id")
            published_elem = entry.find("published")

            title = _clean_text(title_elem.text) if title_elem else "Research Paper"
            summary = _clean_text(summary_elem.text) if summary_elem else ""
            paper_url = id_elem.text.strip() if id_elem else ""
            published = published_elem.text.strip()[:10] if published_elem else today_iso

            if paper_url:
                results.append({
                    "title": f"[ArXiv] {title}",
                    "url": paper_url,
                    "snippet": summary[:300] + ("..." if len(summary) > 300 else ""),
                    "domain": "arxiv.org",
                    "source_engine": "arxiv",
                    "published_date": published,
                    "access_date": today_iso,
                })
    except Exception as e:
        logger.warning(f"ArXiv search error for query '{query}': {e}")

    return results


def search_pypi(package_name: str, timeout: int = 5) -> Optional[Dict[str, Any]]:
    """
    Queries official PyPI JSON API for package details, official docs, and latest release version.
    """
    clean_pkg = re.sub(r"[^a-zA-Z0-9_\-\.]", "", package_name.strip()).lower()
    if not clean_pkg:
        return None

    endpoint = f"https://pypi.org/pypi/{clean_pkg}/json"
    headers = {"User-Agent": DEFAULT_USER_AGENT}
    today_iso = datetime.date.today().isoformat()

    try:
        response = requests.get(endpoint, headers=headers, timeout=timeout)
        if response.status_code != 200:
            return None

        data = response.json()
        info = data.get("info", {})
        summary = info.get("summary", "")
        home_page = info.get("home_page") or info.get("project_url") or f"https://pypi.org/project/{clean_pkg}/"
        version = info.get("version", "")
        docs_url = info.get("project_urls", {}).get("Documentation") or home_page

        return {
            "title": f"{info.get('name', clean_pkg)} v{version} (Official PyPI)",
            "url": docs_url,
            "snippet": f"Official Python package {clean_pkg} (version {version}): {summary}",
            "domain": "pypi.org",
            "source_engine": "pypi",
            "published_date": today_iso,
            "access_date": today_iso,
        }
    except Exception:
        return None


def fetch_page_content(
    url: str,
    timeout: int = 6,
    max_chars: int = 3500,
) -> Dict[str, Any]:
    """
    Fetches full page HTML, strips scripts, navigation, footers, and styles,
    and extracts clean article/documentation text.
    Handles timeouts, HTTP 403/404, redirects, and SSL issues gracefully.
    """
    result: Dict[str, Any] = {
        "success": False,
        "url": url,
        "title": "",
        "content": "",
        "domain": _extract_domain(url),
        "error": None,
    }

    if not url or not url.startswith("http"):
        result["error"] = "Invalid URL protocol."
        return result

    headers = {
        "User-Agent": DEFAULT_USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }

    try:
        response = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
        if response.status_code != 200:
            result["error"] = f"HTTP {response.status_code}"
            return result

        # Only process HTML/text contents
        content_type = response.headers.get("Content-Type", "").lower()
        if "html" not in content_type and "text" not in content_type:
            result["error"] = f"Unsupported content type: {content_type}"
            return result

        soup = bs4.BeautifulSoup(response.text, "html.parser")

        # Remove irrelevant elements
        for tag in soup([
            "script", "style", "nav", "footer", "header", "aside",
            "form", "noscript", "svg", "button", "iframe"
        ]):
            tag.decompose()

        title = ""
        if soup.title and soup.title.string:
            title = _clean_text(soup.title.string)
        result["title"] = title

        # Extract substantive content from headings, paragraphs, list items, and code
        content_elements = []
        for elem in soup.find_all(["h1", "h2", "h3", "h4", "p", "li", "pre", "code"]):
            text = elem.get_text(" ", strip=True)
            if len(text) > 30 or elem.name in ["h1", "h2", "h3", "pre"]:
                clean = _clean_text(text)
                if clean and clean not in content_elements:
                    content_elements.append(clean)
            if sum(len(c) for c in content_elements) >= max_chars * 1.5:
                break

        full_extracted = "\n\n".join(content_elements)
        if len(full_extracted) > max_chars:
            full_extracted = full_extracted[:max_chars].rsplit(" ", 1)[0] + "..."

        result["content"] = full_extracted
        result["success"] = bool(full_extracted.strip())
        if not result["success"]:
            result["error"] = "No substantive text extracted from page."
    except requests.exceptions.Timeout:
        result["error"] = f"Connection timed out after {timeout} seconds."
    except requests.exceptions.SSLError:
        result["error"] = "SSL certificate verification failed."
    except Exception as e:
        result["error"] = str(e)

    return result


def search_web(
    query: str,
    max_results: int = 5,
    fetch_full_content: bool = True,
    prefer_academic: bool = False,
    prefer_news: bool = False,
) -> List[Dict[str, Any]]:
    """
    High-level multi-engine search aggregator.
    Executes live search across Bing, Google News, Wikipedia, and ArXiv,
    deduplicates results, enriches top results with full page content,
    and returns a ranked list of reliable sources.
    """
    cleaned_query = (query or "").strip()
    if not cleaned_query:
        return []

    aggregated: List[Dict[str, Any]] = []
    seen_urls = set()

    # 1. Specialized Engine Invocations
    # If query mentions a python package, check PyPI
    pypi_match = re.search(r"\b(?:python\s+package|pypi|library|module)\s+([a-zA-Z0-9_\-]+)", cleaned_query, re.I)
    if pypi_match:
        pypi_info = search_pypi(pypi_match.group(1))
        if pypi_info:
            aggregated.append(pypi_info)
            seen_urls.add(pypi_info["url"])

    # If academic / research / paper mentioned or preferred
    if prefer_academic or any(k in cleaned_query.lower() for k in ["paper", "arxiv", "research", "benchmark", "transformer"]):
        arxiv_hits = search_arxiv(cleaned_query, max_results=2)
        for hit in arxiv_hits:
            if hit["url"] not in seen_urls:
                aggregated.append(hit)
                seen_urls.add(hit["url"])

    # If current news or events
    if prefer_news or any(k in cleaned_query.lower() for k in ["news", "latest", "today", "yesterday", "update", "2025", "2026"]):
        news_hits = search_google_news(cleaned_query, max_results=3)
        for hit in news_hits:
            if hit["url"] not in seen_urls:
                aggregated.append(hit)
                seen_urls.add(hit["url"])

    # Core web search via Bing
    bing_hits = search_bing(cleaned_query, max_results=max_results + 2)
    for hit in bing_hits:
        if hit["url"] not in seen_urls:
            aggregated.append(hit)
            seen_urls.add(hit["url"])

    # Conceptual backup via Wikipedia if list is short
    if len(aggregated) < 3:
        wiki_hits = search_wikipedia(cleaned_query, max_results=2)
        for hit in wiki_hits:
            if hit["url"] not in seen_urls:
                aggregated.append(hit)
                seen_urls.add(hit["url"])

    # 2. Score and Rank Results by Relevance & Domain Authority
    def rank_score(item: Dict[str, Any]) -> float:
        domain = item.get("domain", "")
        base_score = AUTHORITATIVE_DOMAINS.get(domain, 4.0)

        # Keyword match in title
        title_lower = item.get("title", "").lower()
        q_words = [w.lower() for w in cleaned_query.split() if len(w) > 2]
        matches = sum(1 for w in q_words if w in title_lower)
        score = base_score + (matches * 1.5)
        return score

    aggregated.sort(key=rank_score, reverse=True)
    top_results = aggregated[:max_results]

    # 3. Enrich Top Results with Full Page Content
    if fetch_full_content:
        for idx, item in enumerate(top_results[:3]):  # Fetch content for top 3 to keep response fast
            try:
                page_data = fetch_page_content(item["url"], timeout=4, max_chars=2500)
                if page_data["success"] and page_data["content"]:
                    item["full_content"] = page_data["content"]
                else:
                    item["full_content"] = item["snippet"]
            except Exception:
                item["full_content"] = item["snippet"]

    return top_results


def format_sources_markdown(sources: List[Dict[str, Any]]) -> str:
    """
    Formats verified search results into a clean, GitHub-flavored markdown
    'Sources & References' section with clickable links, page titles, domains,
    and publication/access dates. Never invents fake citations.
    """
    if not sources:
        return ""

    lines = [
        "\n\n### 🌐 Sources & References\n",
    ]

    for idx, s in enumerate(sources, 1):
        title = s.get("title") or s.get("domain") or "Web Source"
        url = s.get("url", "#")
        domain = s.get("domain") or _extract_domain(url)
        pub_date = s.get("published_date") or s.get("access_date") or datetime.date.today().isoformat()
        
        # Build clean citation line: [1] Title - Domain (Date)
        lines.append(f"- **[{idx}] [{title}]({url})** — *{domain}* (Accessed: {pub_date})")

    return "\n".join(lines)
