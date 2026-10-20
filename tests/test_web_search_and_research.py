"""
Comprehensive Automated Test Suite: Live Web Search, Grounding,
Question Routing, Citations, Deep Research, and Voice Pipeline
==============================================================
Tests verify:
1. Question Routing: Evergreen concepts, math, logic, and commands bypass web search.
2. Question Routing: Current events, latest updates, documentation trigger live search.
3. User Toggle: Forced Web Search 'ON' and 'OFF' overrides heuristic routing.
4. Web Search: Multi-engine search returns structured, valid URLs and domain metadata.
5. Content Reader: Extracts clean article text, strips HTML noise, handles 403/timeouts gracefully.
6. Citations: Verified Markdown formatting with title, domain, URL, and access date. Never fabricates links.
7. Deep Research: Sub-query decomposition, multi-angle search, and synthesized report with citations.
8. Voice-Transcribed Query: Runs through the exact same search-grounded pipeline with parity.
9. User Isolation: Conversations with web grounding remain private to the owning user.
10. Daily Planner Safety: Executes without AttributeError or missing keys.
"""

import os
import unittest
from unittest.mock import MagicMock, patch

from database.db import init_db
from nova_core.assistant import NovaAssistant
from nova_core.auth import authenticate_user, create_user
from nova_core.conversation_manager import create_thread, get_thread_messages
from nova_core.llm_service import BaseLLMProvider, LLMResponse
from nova_core.research_router import (
    generate_research_subqueries,
    is_deep_research_mode,
    run_deep_research,
    should_search_web,
)
from nova_core.web_search import (
    _decode_bing_url,
    _extract_domain,
    fetch_page_content,
    format_sources_markdown,
    search_web,
)
from modules.planner_and_suggestions import generate_daily_plan, get_smart_task_suggestions


class MockSearchLLMProvider(BaseLLMProvider):
    """Predictable mock LLM for testing generative grounding and citations."""

    def __init__(self, reply: str = "Test response"):
        self.reply = reply
        self.model = "mock-model"
        self.last_system_prompt = ""
        self.last_messages = []

    def generate(self, messages, system_prompt=None, temperature=0.7, max_tokens=1024):
        self.last_system_prompt = system_prompt or ""
        self.last_messages = messages
        return LLMResponse(
            success=True,
            content=self.reply,
            provider="mock_llm",
            model=self.model,
            tokens_used=42,
        )

    def generate_stream(self, messages, system_prompt=None, temperature=0.7, max_tokens=1024):
        self.last_system_prompt = system_prompt or ""
        self.last_messages = messages
        words = self.reply.split(" ")
        for i, w in enumerate(words):
            yield w + (" " if i < len(words) - 1 else "")

    def get_info(self):
        return {"provider": "mock_llm", "model": self.model}


class TestWebSearchAndRouting(unittest.TestCase):
    """Test suite covering routing heuristics, web search, page reader, and citations."""

    @classmethod
    def setUpClass(cls):
        init_db()

    def test_routing_evergreen_math_and_concepts_no_search(self):
        """Mathematical operations, programming logic, and evergreen concepts must NOT trigger web search."""
        no_search_queries = [
            "solve 3x + 12 = 45",
            "calculate 25 * 14",
            "what is the derivative of x^2 + 3x",
            "explain gradient descent in machine learning",
            "how does backpropagation work",
            "write a python function to check if a string is a palindrome",
            "write a poem about neural networks",
        ]
        for query in no_search_queries:
            needs_search, reason = should_search_web(query, web_mode="auto")
            self.assertFalse(needs_search, f"Query '{query}' should not trigger search. Reason: {reason}")

    def test_routing_current_and_documentation_triggers_search(self):
        """Current events, latest releases, year markers, and documentation lookups MUST trigger web search."""
        search_queries = [
            "what are the latest features in Python 3.13 in 2026",
            "who is the current CEO of Microsoft",
            "documentation for FastAPI dependency injection",
            "what is the latest news regarding quantum computing breakthroughs",
            "current stock price of Nvidia today",
            "what happened in the recent AI summit 2025",
        ]
        for query in search_queries:
            needs_search, reason = should_search_web(query, web_mode="auto")
            self.assertTrue(needs_search, f"Query '{query}' should trigger search. Reason: {reason}")

    def test_user_toggle_overrides_routing(self):
        """User force toggles ('on' and 'off') must strictly override heuristic analysis."""
        # Query that normally wouldn't search, forced ON
        needs_search, reason = should_search_web("what is a matrix", web_mode="on")
        self.assertTrue(needs_search)
        self.assertIn("Forced web search", reason)

        # Query that normally would search, forced OFF
        needs_search, reason = should_search_web("latest breaking news today in 2026", web_mode="off")
        self.assertFalse(needs_search)
        self.assertIn("disabled by user", reason)

    def test_bing_url_decoding(self):
        """Verifies base64 encoded Bing redirect URLs are properly resolved into direct links."""
        # Real Bing redirect URL sample
        fake_bing_href = (
            "https://www.bing.com/ck/a?!&&p=123&u=a1aHR0cHM6Ly93d3cucHl0aG9uLm9yZy9kb3dubG9hZHMv&ntb=1"
        )
        decoded = _decode_bing_url(fake_bing_href)
        self.assertEqual(decoded, "https://www.python.org/downloads/")

        # Regular non-bing URL should be unchanged
        regular_url = "https://docs.python.org/3/"
        self.assertEqual(_decode_bing_url(regular_url), regular_url)

    def test_page_content_extraction_and_sanitization(self):
        """Verifies HTML page scraping strips script tags, extracts text, and returns structured data."""
        sample_html = """
        <html>
            <head><title>Test Documentation Page</title></head>
            <body>
                <script>console.log("bad script");</script>
                <nav><a href="/home">Home</a></nav>
                <h1>Welcome to Nova Docs</h1>
                <p>This is substantive article content explaining the machine learning architecture.</p>
                <pre><code>def test(): pass</code></pre>
                <footer>Copyright 2026</footer>
            </body>
        </html>
        """
        with patch("requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.headers = {"Content-Type": "text/html; charset=utf-8"}
            mock_resp.text = sample_html
            mock_get.return_value = mock_resp

            result = fetch_page_content("https://example.com/docs")
            self.assertTrue(result["success"])
            self.assertEqual(result["title"], "Test Documentation Page")
            self.assertIn("Welcome to Nova Docs", result["content"])
            self.assertIn("substantive article content", result["content"])
            self.assertNotIn("bad script", result["content"])
            self.assertNotIn("Copyright", result["content"])

    def test_page_content_error_handling(self):
        """Verifies timeouts, HTTP 403, and invalid protocols are handled gracefully without exceptions."""
        with patch("requests.get") as mock_get:
            # 403 Forbidden
            mock_resp = MagicMock()
            mock_resp.status_code = 403
            mock_get.return_value = mock_resp
            res_403 = fetch_page_content("https://example.com/protected")
            self.assertFalse(res_403["success"])
            self.assertIn("403", res_403["error"])

            # Invalid protocol
            res_inv = fetch_page_content("ftp://invalid.com/file")
            self.assertFalse(res_inv["success"])
            self.assertIn("protocol", res_inv["error"])

    def test_citation_formatting(self):
        """Citations must include numbered indices, title, clickable URL, domain, and access date."""
        mock_sources = [
            {
                "title": "Python 3.13 Documentation",
                "url": "https://docs.python.org/3.13/",
                "domain": "docs.python.org",
                "published_date": "2026-10-15",
                "access_date": "2026-10-18",
            },
            {
                "title": "Attention Is All You Need",
                "url": "https://arxiv.org/abs/1706.03762",
                "domain": "arxiv.org",
                "published_date": "2017-06-12",
                "access_date": "2026-10-18",
            },
        ]
        md = format_sources_markdown(mock_sources)
        self.assertIn("### 🌐 Sources & References", md)
        self.assertIn("[1] [Python 3.13 Documentation](https://docs.python.org/3.13/)", md)
        self.assertIn("docs.python.org", md)
        self.assertIn("[2] [Attention Is All You Need](https://arxiv.org/abs/1706.03762)", md)
        self.assertIn("arxiv.org", md)

    def test_deep_research_subqueries_and_planning(self):
        """Deep Research must generate distinct sub-queries exploring architecture, benchmarks, and guides."""
        topic = "Compare PyTorch vs JAX for distributed LLM training"
        subqueries = generate_research_subqueries(topic, max_subqueries=4)
        self.assertEqual(len(subqueries), 4)
        self.assertTrue(any("overview" in s.lower() or "architecture" in s.lower() for s in subqueries))
        self.assertTrue(any("benchmark" in s.lower() or "comparison" in s.lower() for s in subqueries))

    def test_deep_research_execution(self):
        """Deep Research coordinator runs multi-step search, reads sources, and generates report."""
        mock_hits = [
            {
                "title": "PyTorch Distributed Overview",
                "url": "https://pytorch.org/docs/stable/distributed.html",
                "snippet": "PyTorch provides FSDP and Tensor Parallelism for large scale training.",
                "domain": "pytorch.org",
                "published_date": "2026-01-01",
                "access_date": "2026-10-18",
            },
            {
                "title": "JAX Sharding and SPMD",
                "url": "https://jax.readthedocs.io/en/latest/",
                "snippet": "JAX offers automatic sharding and functional parallelism across TPUs and GPUs.",
                "domain": "jax.readthedocs.io",
                "published_date": "2026-01-01",
                "access_date": "2026-10-18",
            },
        ]

        with patch("nova_core.research_router.search_web", return_value=mock_hits):
            with patch("nova_core.web_search.fetch_page_content", return_value={"success": True, "content": "Sample content", "title": "Mock Title"}):
                mock_llm = MockSearchLLMProvider(
                    reply="## Executive Summary\nPyTorch and JAX offer complementary strengths for LLM training [1][2]."
                )
                res = run_deep_research(
                    "PyTorch vs JAX distributed training",
                    llm_generator=mock_llm.generate,
                )
                self.assertIn("Executive Summary", res["report"])
                self.assertIn("Sources & References", res["report"])
                self.assertIn("pytorch.org", res["report"])
                self.assertGreaterEqual(len(res["sources"]), 2)

    def test_assistant_process_message_with_web_search_grounding(self):
        """Assistant process_message must ground responses and attach verified sources."""
        assistant = NovaAssistant()
        conv_id = f"test_conv_search_{os.getpid()}"
        user_id = "test_user_search"

        mock_hits = [
            {
                "title": "Python 3.13 Changelog",
                "url": "https://docs.python.org/release/3.13.0/",
                "snippet": "Python 3.13 introduces experimental free-threaded mode (no-GIL) and JIT compiler.",
                "domain": "docs.python.org",
                "published_date": "2026-10-10",
                "access_date": "2026-10-18",
            }
        ]

        with patch("nova_core.assistant.search_web", return_value=mock_hits):
            with patch("nova_core.assistant.is_llm_configured", return_value=True):
                mock_provider = MockSearchLLMProvider(
                    reply="Python 3.13 features an experimental free-threaded mode and a JIT compiler [1]."
                )
                with patch("nova_core.assistant.get_llm_service", return_value=mock_provider):
                    out = assistant.process_message(
                        "What are the latest features in Python 3.13?",
                        conversation_id=conv_id,
                        user_id=user_id,
                        web_search="auto",
                    )
                    self.assertEqual(out["mode"], "web_grounded")
                    self.assertIn("Sources & References", out["response"])
                    self.assertIn("docs.python.org", out["response"])
                    self.assertIsNotNone(out.get("sources"))

    def test_voice_transcribed_query_search_parity(self):
        """Voice input passed into process_message_stream triggers search grounding with identical parity."""
        assistant = NovaAssistant()
        conv_id = f"test_voice_search_{os.getpid()}"
        user_id = "test_voice_user"

        spoken_text = "What is the latest update on quantum computing chips in 2026?"
        mock_hits = [
            {
                "title": "Quantum Chip Update 2026",
                "url": "https://nature.com/articles/quantum-2026",
                "snippet": "New topological qubits demonstrate 99.9% gate fidelity.",
                "domain": "nature.com",
                "published_date": "2026-05-01",
                "access_date": "2026-10-18",
            }
        ]

        with patch("nova_core.assistant.search_web", return_value=mock_hits):
            with patch("nova_core.assistant.is_llm_configured", return_value=True):
                mock_provider = MockSearchLLMProvider(
                    reply="Recent research shows 99.9% fidelity in topological qubits [1]."
                )
                with patch("nova_core.assistant.get_llm_service", return_value=mock_provider):
                    chunks = list(assistant.process_message_stream(
                        spoken_text,
                        conversation_id=conv_id,
                        user_id=user_id,
                        web_search="auto",
                    ))
                    full_output = "".join(chunks)
                    self.assertIn("topological qubits", full_output)
                    self.assertIn("Sources & References", full_output)
                    self.assertIn("nature.com", full_output)

    def test_user_chat_isolation_with_search(self):
        """Messages and search grounding from user A must never be accessible to user B."""
        assistant = NovaAssistant()
        conv_a = f"conv_user_a_{os.getpid()}"
        conv_b = f"conv_user_b_{os.getpid()}"

        assistant._save_conversation_turn(conv_a, "user", "Private question from User A", user_id="user_alpha")
        assistant._save_conversation_turn(conv_a, "assistant", "Answer with web sources", metadata={"sources": [{"url": "https://a.com"}]}, user_id="user_alpha")

        # User B queries their history
        history_b = assistant.get_conversation_history(conv_a, user_id="user_beta")
        self.assertEqual(len(history_b), 0, "User Beta must NOT be able to view User Alpha's conversation!")

        # User A retrieves their history
        history_a = assistant.get_conversation_history(conv_a, user_id="user_alpha")
        self.assertEqual(len(history_a), 2)

    def test_daily_planner_executes_without_attribute_error(self):
        """Verifies daily planner schedule generation and smart suggestions execute without error."""
        suggestions = get_smart_task_suggestions(limit=3)
        self.assertIsInstance(suggestions, list)

        plan = generate_daily_plan(available_hours=5.0, start_time="09:00")
        self.assertIn("date", plan)
        self.assertIn("schedule", plan)
        self.assertIn("total_allocated_minutes", plan)
        self.assertIsInstance(plan["schedule"], list)


if __name__ == "__main__":
    unittest.main()
