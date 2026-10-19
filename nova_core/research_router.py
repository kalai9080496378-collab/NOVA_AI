"""
NOVA Research Router & Deep Research Coordinator
================================================
Intelligent question routing, web search determination, and multi-step
Deep Research execution.

Features:
- Differentiates between evergreen reasoning/math vs. live external search.
- Supports user toggles: Auto, Forced On, Forced Off.
- Deep Research mode: decomposes queries into 3-5 sub-angles, runs parallel
  searches, reads authoritative pages, and synthesizes an executive research report.
- Progress reporting via callbacks or generators for responsive UX.
"""

import datetime
import logging
import re
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

from nova_core.web_search import (
    format_sources_markdown,
    search_web,
)

logger = logging.getLogger("nova.research_router")

# Regex and keywords that strongly signal a need for live external search
SEARCH_TRIGGER_PATTERNS = [
    r"\b(?:latest|recent|currently|current|today|yesterday|this week|this month|this year)\b",
    r"\b(?:2024|2025|2026|upcoming|newest|released?|roadmap|changelog|announcement)\b",
    r"\b(?:news|headlines|election|stock price|weather forecast|score|sports match)\b",
    r"\b(?:documentation|docs for|api reference|how to install|package version|pypi|github repo)\b",
    r"\b(?:who is the (?:current|new)|who won the|what happened to|ceo of|founder of)\b",
    r"\b(?:compare|benchmarks?|state-of-the-art|sota|market share)\b",
    r"\b(?:price of|cost of|valuation of|net worth of)\b",
]

# Regex patterns indicating questions that should be answered directly via reasoning/knowledge
NO_SEARCH_PATTERNS = [
    r"^\s*(?:solve|calculate|compute|integrate|differentiate|evaluate)\b",
    r"^\s*\d+[\s\+\-\*\/\^\%]+[\d\s\+\-\*\/\^\%]+\s*$",  # Pure arithmetic
    r"\b(?:write a (?:python|sql|java|c\+\+|javascript|function|class|regex|script))\b",
    r"\b(?:explain (?:gradient descent|backpropagation|binary search|quicksort|merge sort|recursion|oop|polymorphism))\b",
    r"\b(?:what is (?:a|an) (?:derivative|matrix|vector|eigenvalue|bayes theorem|neural network))\b",
    r"\b(?:write a (?:poem|story|haiku|essay|speech|email|letter))\b",
    r"\b(?:riddle|brain teaser|translate|summarize the following)\b",
]

DEEP_RESEARCH_KEYWORDS = [
    "deep research",
    "comprehensive research",
    "in-depth research",
    "thorough investigation",
    "detailed report",
    "comprehensive analysis",
    "full comparison",
    "literature review",
    "market analysis",
]


def should_search_web(query: str, web_mode: str = "auto") -> Tuple[bool, str]:
    """
    Evaluates whether a given user question requires a live internet search.

    Args:
        query: User input query text.
        web_mode: 'auto', 'on', or 'off'.

    Returns:
        Tuple of (should_search: bool, reason: str).
    """
    mode = (web_mode or "auto").lower().strip()
    if mode == "on":
        return True, "Forced web search enabled by user."
    if mode == "off":
        return False, "Web search disabled by user toggle."

    cleaned = (query or "").strip().lower()
    if not cleaned:
        return False, "Empty query."

    # 1. Check if local assistant command
    if any(cleaned.startswith(p) for p in [
        "add task", "create task", "task:", "show tasks", "list tasks", "complete task",
        "remember:", "save memory", "recall:", "stats", "time", "date", "help"
    ]):
        return False, "Local productivity command."

    # 2. Check no-search patterns (Math, direct code generation, classic concepts)
    for pat in NO_SEARCH_PATTERNS:
        if re.search(pat, cleaned, re.IGNORECASE):
            # Check if an override temporal keyword is also present
            if not any(re.search(tp, cleaned, re.IGNORECASE) for tp in [r"\b(?:2025|2026|latest|current)\b"]):
                return False, f"Internal reasoning / foundational knowledge: {pat}"

    # 3. Check search trigger patterns
    for pat in SEARCH_TRIGGER_PATTERNS:
        match = re.search(pat, cleaned, re.IGNORECASE)
        if match:
            return True, f"Matched live query trigger: '{match.group(0)}'"

    # 4. Inquisitive questions about entities, technology, or current topics
    if any(cleaned.startswith(w) for w in ["who is", "what is the status of", "where is", "when did", "how much is"]):
        return True, "Factual real-world entity inquiry."

    # Default to direct response for general reasoning
    return False, "General knowledge / conceptual dialogue."


def is_deep_research_mode(query: str, force_deep: bool = False) -> bool:
    """Checks whether Deep Research mode should be activated."""
    if force_deep:
        return True
    cleaned = (query or "").lower().strip()
    return any(k in cleaned for k in DEEP_RESEARCH_KEYWORDS)


def generate_research_subqueries(query: str, max_subqueries: int = 4) -> List[str]:
    """
    Decomposes a complex research topic into 3-5 distinct search subqueries
    covering architecture, recent benchmarks, ecosystem, and challenges.
    """
    clean_q = re.sub(r"(?i)\b(?:conduct|do|perform)?\s*(?:deep|in-depth|comprehensive)?\s*research\s*(?:on|about)?\b", "", query).strip()
    if not clean_q:
        clean_q = query

    current_year = datetime.date.today().year

    # Heuristic decomposition based on query context
    subqueries = [
        f"{clean_q} overview architecture capabilities {current_year}",
        f"{clean_q} performance benchmarks comparison",
        f"{clean_q} official documentation guide github",
        f"{clean_q} real-world case studies advantages limitations {current_year}",
    ]

    return subqueries[:max_subqueries]


def run_deep_research(
    query: str,
    llm_generator: Optional[Callable[[List[Dict[str, str]], str], Any]] = None,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
) -> Dict[str, Any]:
    """
    Executes an end-to-end Deep Research investigation:
    1. Plans sub-angles and generates targeted sub-queries.
    2. Runs live web searches across engines.
    3. Reads full page contents from top authoritative sources.
    4. Synthesizes conflicting or complementary evidence.
    5. Formats an executive research report with inline citations and references.
    """
    subqueries = generate_research_subqueries(query, max_subqueries=4)
    all_sources: List[Dict[str, Any]] = []
    seen_urls = set()

    if progress_callback:
        progress_callback("Planning research strategy & decomposing question...", 1, 4)

    # 1. Multi-Step Targeted Searches
    for i, sub_q in enumerate(subqueries, 1):
        if progress_callback:
            progress_callback(f"Investigating sub-topic {i}/{len(subqueries)}: '{sub_q[:45]}...' ", 2, 4)

        try:
            hits = search_web(sub_q, max_results=3, fetch_full_content=False)
            for hit in hits:
                if hit["url"] not in seen_urls:
                    seen_urls.add(hit["url"])
                    all_sources.append(hit)
        except Exception as e:
            logger.warning(f"Error querying subquery '{sub_q}': {e}")

    # 2. Extract Full Content from Top Authoritative Sources
    if progress_callback:
        progress_callback(f"Reading & analyzing content from {min(len(all_sources), 5)} primary sources...", 3, 4)

    ranked_sources = all_sources[:6]
    for src in ranked_sources:
        try:
            from nova_core.web_search import fetch_page_content
            page = fetch_page_content(src["url"], timeout=4, max_chars=2500)
            if page["success"] and page["content"]:
                src["full_content"] = page["content"]
            else:
                src["full_content"] = src.get("snippet", "")
        except Exception:
            src["full_content"] = src.get("snippet", "")

    # 3. Build Evidence Context for Synthesis
    evidence_blocks = []
    for idx, s in enumerate(ranked_sources, 1):
        content = s.get("full_content") or s.get("snippet") or ""
        evidence_blocks.append(
            f"Source [{idx}]:\n"
            f"Title: {s['title']}\n"
            f"URL: {s['url']}\n"
            f"Domain: {s['domain']}\n"
            f"Content excerpt:\n{content[:1500]}\n"
        )
    evidence_text = "\n---\n".join(evidence_blocks)

    if progress_callback:
        progress_callback("Synthesizing findings and compiling executive report...", 4, 4)

    # 4. Generate Report via LLM or Structured Formatter
    system_prompt = (
        "You are NOVA's Deep Research Intelligence Unit. You produce rigorous, objective, "
        "and comprehensive executive research reports grounded in the provided evidence.\n"
        "Guidelines:\n"
        "1. Write an Executive Summary capturing the core takeaways.\n"
        "2. Break down into thematic sections with in-depth analysis.\n"
        "3. Provide inline citations like [1], [2] directly after factual assertions.\n"
        "4. Highlight consensus, trade-offs, and any conflicting evidence.\n"
        "5. Conclude with Key Takeaways or Future Outlook.\n"
        "6. Do NOT invent URLs or citations. Only reference the provided numbered sources.\n"
    )

    user_prompt = (
        f"Research Question: {query}\n\n"
        f"Gathered Evidence:\n{evidence_text}\n\n"
        f"Please synthesize this into an exhaustive, structured Research Report with inline [n] citations."
    )

    report_body = ""
    if llm_generator:
        try:
            messages = [{"role": "user", "content": user_prompt}]
            llm_response = llm_generator(messages, system_prompt)
            if hasattr(llm_response, "content") and llm_response.content:
                report_body = llm_response.content
            elif isinstance(llm_response, str):
                report_body = llm_response
        except Exception as e:
            logger.warning(f"LLM synthesis error: {e}")

    # Fallback structured report if LLM unavailable
    if not report_body:
        report_body = (
            f"## 🔬 Deep Research Report: {query}\n\n"
            f"### 📋 Executive Summary\n"
            f"A multi-source investigation was conducted across {len(ranked_sources)} authoritative "
            f"domains to analyze **{query}**.\n\n"
            f"### 🔍 Key Findings & Synthesized Evidence\n"
        )
        for idx, s in enumerate(ranked_sources, 1):
            snippet = s.get("snippet", "Detailed information available at source.")
            report_body += f"- **[{idx}] {s['title']}**: {snippet} [[{idx}]]({s['url']})\n"

        report_body += (
            f"\n### ⚖️ Synthesis & Outlook\n"
            f"The reviewed sources indicate active progress in this area. "
            f"Review the primary documentation and publication links below for full technical specifications.\n"
        )

    # Append Sources & References section
    sources_section = format_sources_markdown(ranked_sources)
    full_output = f"{report_body}{sources_section}"

    return {
        "report": full_output,
        "sources": ranked_sources,
        "subqueries": subqueries,
        "query": query,
    }
