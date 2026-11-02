import importlib.util
import pathlib
import unittest

SPEC = importlib.util.spec_from_file_location(
    "usage_report", pathlib.Path(__file__).resolve().parents[1] / "usage_report.py"
)
usage_report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(usage_report)

SEARCH = (
    '2026-11-02T01:00:00.000001Z  INFO graphrag_usage: tool call principal="openclaw.clawd" '
    'tool="search_notes" outcome="ok" elapsed_ms=42.5 mode="hybrid" graph="on" scope="notes" '
    "limit=3 query_chars=18 query_terms=3 hits=3 top_score=0.82\n"
)
EMPTY = SEARCH.replace("hits=3 top_score=0.82", "hits=0").replace("42.5", "10.0")
CONTEXT = (
    '\x1b[2m2026-11-02T01:05:00.000001Z\x1b[0m \x1b[32m INFO\x1b[0m graphrag_usage: tool call '
    'principal="openclaw.clawd" tool="build_context" outcome="ok" elapsed_ms=120.0 graph="on" '
    "hits=4 top_score=0.02 graph_considered=7 graph_selected=1 total_tokens=1013\n"
)
FAILED = (
    '2026-11-02T01:06:00.000001Z  INFO graphrag_usage: tool call principal="shiva-importer" '
    'tool="capture_note" outcome="busy" elapsed_ms=1.0\n'
)


class UsageReportTest(unittest.TestCase):
    def test_parses_plain_and_ansi_lines_and_ignores_other_logs(self):
        records = [usage_report.parse_line(line) for line in (SEARCH, CONTEXT, "2026 INFO other: x\n")]
        self.assertEqual(records[0]["hits"], 3.0)
        self.assertEqual(records[0]["mode"], "hybrid")
        self.assertEqual(records[1]["graph_selected"], 1.0)
        self.assertIsNone(records[2])

    def test_summarizes_empty_rate_errors_and_graph_contribution(self):
        records = [usage_report.parse_line(line) for line in (SEARCH, EMPTY, CONTEXT, FAILED)]
        by_tool = {(e["principal"], e["tool"]): e for e in usage_report.summarize(records)}
        search = by_tool[("openclaw.clawd", "search_notes")]
        self.assertEqual((search["calls"], search["empty_results"], search["empty_rate"]), (2, 1, 0.5))
        self.assertEqual(search["options"], {"hybrid/on/notes": 2})
        self.assertEqual(by_tool[("openclaw.clawd", "build_context")]["graph_contributed_rate"], 1.0)
        self.assertEqual(by_tool[("shiva-importer", "capture_note")]["errors"], {"busy": 1})
        self.assertIn("50%", usage_report.render(list(by_tool.values())))


if __name__ == "__main__":
    unittest.main()
