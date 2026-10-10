import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import check_citations
import research
import tools


class RetryTests(unittest.TestCase):
    def test_retry_uses_backoff_and_stops_after_success(self):
        calls = []

        def operation():
            calls.append(1)
            if len(calls) < 3:
                raise tools.RetryableError("temporary")
            return "ok"

        with patch("tools.random.uniform", return_value=0.25), patch("tools.time.sleep") as sleep:
            self.assertEqual(tools.with_retry(operation, attempts=4, base=1, cap=10), "ok")
        self.assertEqual(len(calls), 3)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [1.25, 2.25])

    def test_retry_after_is_capped_and_no_sleep_after_last_failure(self):
        with patch("tools.time.sleep") as sleep:
            with self.assertRaises(tools.RetryableError):
                tools.with_retry(lambda: (_ for _ in ()).throw(tools.RetryableError("busy", 99)),
                                 attempts=2, cap=5)
        sleep.assert_called_once_with(5.0)


class CitationTests(unittest.TestCase):
    def test_valid_grouped_citations_ignore_code_and_links(self):
        report = """# Survey

Claim [1, 2] and another [3]. Ignore `example [99]` and [4](https://example.test).

## References
[1] A. arxiv. https://arxiv.org/abs/1 (2025-01-01)
[2] B. hf-search. https://huggingface.co/papers/2 (2025-01-02)
[3] C. web. https://example.org/c (2025-01-03)
"""
        sources = [
            {"n": 1, "url": "https://arxiv.org/abs/1"},
            {"n": 2, "url": "https://huggingface.co/papers/2"},
            {"n": 3, "url": "https://example.org/c"},
        ]
        self.assertEqual(check_citations.check(report, sources), [])

    def test_reports_duplicate_urls_and_wrong_reference_url(self):
        report = "Claim [1][2].\n\n## References\n[1] A. https://wrong.test\n[2] B. https://a.test\n"
        sources = [{"n": 1, "url": "https://a.test"}, {"n": 2, "url": "https://a.test"}]
        problems = check_citations.check(report, sources)
        self.assertTrue(any("duplicate source url" in item for item in problems))
        self.assertTrue(any("does not match" in item for item in problems))


class ResearchTests(unittest.TestCase):
    def test_slugify_is_safe_and_bounded(self):
        self.assertEqual(research.slugify("../../Survey: World Models!?"), "survey-world-models")
        self.assertEqual(research.slugify("..."), "topic")
        self.assertLessEqual(len(research.slugify("a" * 100)), 60)

    def test_summarize_counts_calls_and_tokens(self):
        messages = [{
            "tool_calls": [{"name": "task"}, {"name": "web_search"}],
            "usage_metadata": {"input_tokens": 10, "output_tokens": 4},
        }, {
            "tool_calls": [{"function": {"name": "task"}}],
            "response_metadata": {"token_usage": {"prompt_tokens": 3, "completion_tokens": 2}},
        }]
        summary = research.summarize(messages, 1.26, "model-x")
        self.assertEqual(summary["subagent_calls"], 2)
        self.assertEqual(summary["tool_calls"], {"task": 2, "web_search": 1})
        self.assertEqual(summary["tokens"], {"input": 13, "output": 6})
        self.assertEqual(summary["elapsed_s"], 1.3)

    def test_save_outputs_validates_then_writes_all_artifacts(self):
        source_data = [{"n": 1, "url": "https://arxiv.org/abs/1", "source": "arxiv"}]
        files = {
            research.REPORT_PATH: b"# Report\n\nClaim [1]\n\n## References\n[1] A. https://arxiv.org/abs/1\n",
            research.SOURCES_PATH: json.dumps(source_data).encode(),
        }

        class Backend:
            def download_files(self, paths):
                return [SimpleNamespace(path=path, content=files.get(path)) for path in paths]

        with tempfile.TemporaryDirectory() as directory:
            path = research.save_outputs(Backend(), "A topic", [], 2.0, "model-x", Path(directory))
            self.assertTrue(path.exists())
            self.assertTrue((Path(directory) / "a-topic.sources.json").exists())
            meta = json.loads((Path(directory) / "a-topic.meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["n_sources"], 1)
            self.assertEqual(meta["source_families"], ["arxiv"])


if __name__ == "__main__":
    unittest.main()
