import contextlib
import datetime as dt
import importlib.util
import io
import json
import pathlib
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location(
    "usage_report", pathlib.Path(__file__).resolve().parents[1] / "usage_report.py"
)
usage_report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(usage_report)

SEARCH = (
    '2026-11-02T01:00:00.000001Z  INFO graphrag_usage principal="openclaw.clawd" '
    'tool="search_notes" outcome="ok" elapsed_ms=42.5 mode="hybrid" graph="on" scope="notes" '
    "limit=3 query_chars=18 query_terms=3 hits=3 top_score=0.82\n"
)
# Verbatim shape of a production line from fam (graphrag serve, 2026-11-02).
PRODUCTION = (
    '2026-11-02T01:24:18.069390Z  INFO serve_inner: graphrag_usage principal="openclaw-clawd-daily" '
    'tool="search_notes" outcome="ok" elapsed_ms=22.1 mode="keyword" graph="off" scope="notes" '
    "limit=5 query_chars=18 query_terms=3 hits=5 top_score=12.272516250610352\n"
)
EMPTY = SEARCH.replace("hits=3 top_score=0.82", "hits=0").replace("42.5", "10.0")
CONTEXT = (
    '\x1b[2m2026-11-02T01:05:00.000001Z\x1b[0m \x1b[32m INFO\x1b[0m graphrag_usage: graphrag_usage '
    'principal="openclaw.clawd" tool="build_context" outcome="ok" elapsed_ms=120.0 graph="on" '
    "hits=4 top_score=0.02 graph_considered=7 graph_selected=1 total_tokens=1013\n"
)
FAILED = (
    '2026-11-02T01:06:00.000001Z  INFO graphrag_usage principal="shiva-importer" '
    'tool="capture_note" outcome="busy" elapsed_ms=1.0\n'
)


def usage_line(ts, principal, tool, outcome="ok", extra="elapsed_ms=1.0"):
    return f'{ts}  INFO serve_inner: graphrag_usage principal="{principal}" tool="{tool}" outcome="{outcome}" {extra}\n'


def at(text):
    return usage_report.parse_timestamp(text)


def run(lines, *argv, now=None):
    """Run main() on a temporary log and return (exit code, stdout)."""
    with tempfile.TemporaryDirectory() as directory:
        log = pathlib.Path(directory) / "service.stderr.log"
        log.write_text("".join(lines))
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = usage_report.main([str(log), *argv], now=now)
    return code, out.getvalue()


class UsageReportTest(unittest.TestCase):
    def test_parses_plain_and_ansi_lines_and_ignores_other_logs(self):
        records = [usage_report.parse_line(line) for line in (SEARCH, CONTEXT, "2026 INFO other: x\n", "2026  INFO tool call principal=\"x\" tool=\"y\"\n")]
        self.assertEqual(records[0]["hits"], 3.0)
        self.assertEqual(records[0]["mode"], "hybrid")
        self.assertEqual(records[1]["graph_selected"], 1.0)
        self.assertIsNone(records[2])
        self.assertIsNone(records[3])

    def test_parses_the_production_line_with_its_span_prefix(self):
        record = usage_report.parse_line(PRODUCTION)
        self.assertEqual(record["principal"], "openclaw-clawd-daily")
        self.assertEqual((record["hits"], record["mode"]), (5.0, "keyword"))
        self.assertEqual(record["at"], dt.datetime(2026, 11, 2, 1, 24, 18, 69390, tzinfo=dt.timezone.utc))

    def test_optional_source_cap_is_numeric_and_reported(self):
        record = usage_report.parse_line(CONTEXT.rstrip() + " max_chunks_per_source=1\n")
        self.assertEqual(record["max_chunks_per_source"], 1.0)
        report = usage_report.build_report([record, usage_report.parse_line(CONTEXT)])
        self.assertEqual(report["groups"][0]["max_chunks_per_source"], {"1": 1})
        self.assertIn("max_chunks_per_source {'1': 1}", usage_report.render(report))

    def test_phase_durations_and_status_are_additive_and_optional(self):
        row = usage_report.parse_line(SEARCH.rstrip() +
            ' graph_status="timed_out" embedding_ms=1.2 lexical_ms=0.0 graph_entity_matching_ms=2000.1\n')
        self.assertEqual(row["graph_status"], "timed_out")
        self.assertEqual(row["graph_entity_matching_ms"], 2000.1)
        self.assertNotIn("graph_note_hydration_ms", row)
        summary = usage_report.summarize([row, usage_report.parse_line(SEARCH)])[0]
        self.assertEqual(summary["graph_status"], {"timed_out": 1})
        self.assertEqual(
            summary["phase_ms"]["lexical_ms"],
            {"n": 1, "low_sample": True, "min": 0.0, "median": 0.0, "max": 0.0},
        )
        self.assertEqual(summary["calls"], 2)

    def test_summarizes_empty_rate_errors_and_graph_annotation(self):
        records = [usage_report.parse_line(line) for line in (SEARCH, EMPTY, CONTEXT, FAILED)]
        report = usage_report.build_report(records)
        by_tool = {(e["principal"], e["tool"]): e for e in report["groups"]}
        search = by_tool[("openclaw.clawd", "search_notes")]
        self.assertEqual((search["calls"], search["empty_results"], search["empty_rate"]), (2, 1, 0.5))
        self.assertEqual([s["shape"] for s in search["shapes"]], ["hybrid/on/notes limit=3"])
        context = by_tool[("openclaw.clawd", "build_context")]
        self.assertEqual((context["graph_annotated"], context["graph_annotated_rate"]), (1, 1.0))
        self.assertNotIn("graph_contributed", context)
        self.assertEqual(by_tool[("shiva-importer", "capture_note")]["errors"], {"busy": 1})
        text = usage_report.render(report)
        self.assertIn("50%", text)
        self.assertIn("graph-annotated contexts 1 of 1", text)
        self.assertIn("does not show added support", text)
        self.assertNotIn("graph contributed", text)
        self.assertIn("A principal identifies a credential, not a workload", text)

    def test_timestamps_parse_zulu_offsets_and_any_fraction_as_utc_instants(self):
        expected = dt.datetime(2026, 11, 2, 15, 0, 0, 123456, tzinfo=dt.timezone.utc)
        for text in (
            "2026-11-02T15:00:00.123456Z",
            "2026-11-02T15:00:00.123456789Z",
            "2026-11-02T08:00:00.123456-07:00",
            "2026-11-02T16:30:00.123456+0130",
        ):
            self.assertEqual(at(text), expected, text)
        self.assertEqual(at("2026-11-02T15:00:00.5Z").microsecond, 500000)
        for text in ("2026-11-02T15:00:00", "2026-11-02 15:00:00Z", "2026-13-02T15:00:00Z", "yesterday"):
            with self.assertRaises(ValueError, msg=text):
                at(text)

    def test_bounds_accept_a_utc_date_or_a_timestamp(self):
        self.assertEqual(usage_report.parse_bound("2026-11-02"), at("2026-11-02T00:00:00Z"))
        self.assertEqual(usage_report.parse_bound("2026-11-02T08:00:00-07:00"), at("2026-11-02T15:00:00Z"))
        for text in ("2026-11-02T15:00:00", "2026-02-30", "7d"):
            with self.assertRaises(usage_report.argparse.ArgumentTypeError, msg=text):
                usage_report.parse_bound(text)

    def test_interval_is_half_open_and_compares_instants_not_strings(self):
        lines = [
            usage_line("2026-11-02T14:59:59.999999Z", "p", "service_status"),
            usage_line("2026-11-02T15:00:00.000000Z", "p", "service_status"),
            # 08:30 at -07:00 is 15:30Z: inside by instant, outside by string order.
            usage_line("2026-11-02T08:30:00.000000-07:00", "p", "service_status"),
            usage_line("2026-11-09T14:59:59.999999Z", "p", "service_status"),
            usage_line("2026-11-09T15:00:00.000000Z", "p", "service_status"),
        ]
        records, skipped = usage_report.read_records(lines)
        report = usage_report.build_report(records, skipped, at("2026-11-02T15:00:00Z"), at("2026-11-09T15:00:00Z"))
        window = report["window"]
        self.assertEqual(window["records"], 3)
        self.assertEqual(window["first_record"], "2026-11-02T15:00:00Z")
        self.assertEqual(window["last_record"], "2026-11-09T14:59:59.999999Z")
        self.assertEqual(window["log_first_record"], "2026-11-02T14:59:59.999999Z")
        self.assertEqual(window["log_last_record"], "2026-11-09T15:00:00Z")
        self.assertFalse(window["starts_before_log"])
        self.assertEqual(report["groups"][0]["days"], {"2026-11-02": 2, "2026-11-09": 1})

    def test_date_only_since_keeps_calendar_day_selection(self):
        lines = [
            usage_line("2026-11-01T23:59:59.999999Z", "p", "service_status"),
            usage_line("2026-11-02T00:00:00.000000Z", "p", "service_status"),
            usage_line("2026-11-05T12:00:00.000000Z", "p", "service_status"),
        ]
        code, out = run(lines, "--since", "2026-11-02", "--json")
        report = json.loads(out)
        self.assertEqual(code, 0)
        self.assertEqual(report["schema_version"], 2)
        self.assertEqual(report["window"]["requested_since"], "2026-11-02T00:00:00Z")
        self.assertIsNone(report["window"]["requested_until"])
        self.assertEqual(report["window"]["records"], 2)
        self.assertEqual([g["calls"] for g in report["groups"]], [2])

    def test_last_is_a_rolling_interval_ending_at_until_or_now(self):
        lines = [
            usage_line("2026-11-02T01:41:38.000000Z", "p", "search_notes", extra="elapsed_ms=83812.0"),
            usage_line("2026-11-02T15:00:00.000000Z", "p", "search_notes", extra="elapsed_ms=900.0"),
            usage_line("2026-11-09T14:00:00.000000Z", "p", "search_notes", extra="elapsed_ms=1000.0"),
            usage_line("2026-11-09T15:00:00.000000Z", "p", "search_notes", extra="elapsed_ms=1.0"),
        ]
        now = at("2026-11-09T15:00:00Z")
        _, out = run(lines, "--last", "7d", "--json", now=now)
        window = json.loads(out)["window"]
        self.assertEqual((window["requested_since"], window["requested_until"]), ("2026-11-02T15:00:00Z", "2026-11-09T15:00:00Z"))
        self.assertEqual(window["records"], 2)
        _, out = run(lines, "--last", "36h", "--until", "2026-11-03T13:41:38Z", "--json", now=now)
        window = json.loads(out)["window"]
        self.assertEqual(window["requested_since"], "2026-11-02T01:41:38Z")
        self.assertEqual(window["records"], 2)

    def test_conflicting_or_invalid_interval_arguments_exit_2(self):
        line = [usage_line("2026-11-02T15:00:00.000000Z", "p", "service_status")]
        for argv in (
            ("--last", "7d", "--since", "2026-11-02"),
            ("--last", "7"),
            ("--last", "0d"),
            ("--since", "2026-11-02T15:00:00"),
            ("--since", "2026-11-09", "--until", "2026-11-02"),
            ("--since", "2026-11-02", "--until", "2026-11-02"),
        ):
            with self.assertRaises(SystemExit, msg=argv) as raised, contextlib.redirect_stderr(io.StringIO()):
                run(line, *argv)
            self.assertEqual(raised.exception.code, 2, argv)

    def test_reports_coverage_and_counts_unparseable_usage_lines(self):
        lines = [
            usage_line("2026-11-02T01:24:18.069390Z", "p", "service_status"),
            usage_line("not-a-timestamp", "p", "service_status"),
            '2026-11-02T01:25:00.000000Z  INFO serve_inner: graphrag_usage principal="p" outcome="ok"\n',
            "2026-11-02T01:26:00.000000Z  INFO serve_inner: unrelated message\n",
        ]
        records, skipped = usage_report.read_records(lines)
        self.assertEqual((len(records), skipped), (1, 2))
        report = usage_report.build_report(records, skipped, since=at("2026-11-01T00:00:00Z"))
        self.assertTrue(report["window"]["starts_before_log"])
        self.assertEqual(report["window"]["skipped_usage_lines"], 2)
        text = usage_report.render(report)
        self.assertIn("skipped 2 unparseable usage lines", text)
        self.assertIn("requested start precedes the first usage record", text)
        self.assertIn("first 2026-11-02T01:24:18.069390Z", text)

    def test_empty_interval_still_states_what_was_requested(self):
        code, out = run([usage_line("2026-11-02T15:00:00.000000Z", "p", "service_status")], "--since", "2026-12-01")
        self.assertEqual(code, 0)
        self.assertIn("2026-12-01T00:00:00Z to end of log", out)
        self.assertIn("Included 0 records", out)
        self.assertIn("No graphrag_usage records in this interval.", out)

    def test_low_samples_report_min_median_max_instead_of_percentiles(self):
        def searches(count):
            return [
                usage_report.parse_line(usage_line(
                    f"2026-11-03T00:00:{index:02d}.000000Z", "p", "search_notes",
                    extra=f'elapsed_ms={index + 1}.0 mode="hybrid" graph="on" scope="notes" limit=3 hits=3 top_score=0.016'))
                for index in range(count)
            ]
        low = usage_report.summarize(searches(usage_report.MIN_PERCENTILE_SAMPLES - 1))[0]
        self.assertEqual(low["latency_ms"], {"n": 19, "low_sample": True, "min": 1.0, "median": 10.0, "max": 19.0})
        self.assertNotIn("p95", low["latency_ms"])
        enough = usage_report.build_report(searches(usage_report.MIN_PERCENTILE_SAMPLES))
        self.assertEqual(enough["groups"][0]["latency_ms"], {"n": 20, "low_sample": False, "p50": 11.0, "p95": 19.0})
        self.assertIn("n=20 p50 11.0 ms, p95 19.0 ms", usage_report.render(enough))
        text = usage_report.render(usage_report.build_report(searches(3)))
        self.assertIn("n=3 (low sample) min 1.0 ms, median 2.0 ms, max 3.0 ms", text)
        self.assertNotIn("p95", text)

    def test_retrieval_calls_break_down_by_shape_without_pooling_score_scales(self):
        def search(second, mode, graph, limit, score, extra=""):
            return usage_report.parse_line(usage_line(
                f"2026-11-03T00:00:{second:02d}.000000Z", "shared", "build_context",
                extra=f'elapsed_ms={second}.0 mode="{mode}" graph="{graph}" scope="notes" limit={limit} hits=2 top_score={score}{extra}'))
        records = [
            search(1, "hybrid", "on", 3, 0.0164),
            search(2, "hybrid", "on", 3, 0.0160),
            search(3, "hybrid", "off", 2, 0.0150, " max_chunks_per_source=1"),
            search(4, "keyword", "off", 2, 12.27),
        ]
        entry = usage_report.build_report(records)["groups"][0]
        self.assertEqual(entry["workload"], "retrieval")
        self.assertNotIn("top_score_median", entry)
        self.assertNotIn("options", entry)
        shapes = {s["shape"]: s for s in entry["shapes"]}
        self.assertEqual(sorted(shapes), [
            "hybrid/off/notes limit=2 cap=1", "hybrid/on/notes limit=3", "keyword/off/notes limit=2",
        ])
        self.assertEqual(shapes["hybrid/on/notes limit=3"]["calls"], 2)
        self.assertEqual(shapes["hybrid/on/notes limit=3"]["top_score_median"], 0.0162)
        self.assertEqual(shapes["keyword/off/notes limit=2"]["top_score_median"], 12.27)
        self.assertEqual(shapes["hybrid/on/notes limit=3"]["latency_ms"]["n"], 2)
        text = usage_report.render(usage_report.build_report(records))
        self.assertIn("shape hybrid/on/notes limit=3: 2 ok", text)
        for label in ("evaluation", "conversational", "probe traffic"):
            self.assertNotIn(label, text)

    def test_tools_group_under_workload_classes_and_unknown_tools_are_kept(self):
        tools = ["service_status", "brand_new_tool", "decide_proposal", "upload_source", "get_record", "search_notes"]
        records = [
            usage_report.parse_line(usage_line(f"2026-11-03T00:00:0{index}.000000Z", "p", tool))
            for index, tool in enumerate(tools)
        ]
        report = usage_report.build_report(records)
        self.assertEqual(
            [(g["workload"], g["tool"]) for g in report["groups"]],
            [("retrieval", "search_notes"), ("record_read", "get_record"), ("ingest_sync", "upload_source"),
             ("graph_review", "decide_proposal"), ("status", "service_status"), ("other", "brand_new_tool")],
        )
        text = usage_report.render(report)
        self.assertLess(text.index("[retrieval]"), text.index("[ingest_sync]"))
        self.assertIn("[other]", text)

    def test_outcomes_separate_expected_probes_role_denials_and_errors(self):
        lines = [
            # Probe then the same principal's upload 4.9 s later: expected.
            usage_line("2026-11-03T04:03:00.000000Z", "sync", "get_source", "not_found"),
            usage_line("2026-11-03T04:03:04.900000Z", "sync", "upload_source"),
            # Probe with an upload just outside the correlation window: error.
            usage_line("2026-11-03T05:03:00.000000Z", "sync", "get_source", "not_found"),
            usage_line("2026-11-03T05:03:05.000001Z", "sync", "upload_source"),
            # Two probes, one upload: only one is vouched for.
            usage_line("2026-11-03T06:03:00.000000Z", "sync", "get_source", "not_found"),
            usage_line("2026-11-03T06:03:01.000000Z", "sync", "get_source", "not_found"),
            usage_line("2026-11-03T06:03:02.000000Z", "sync", "upload_source"),
            # Another principal's upload does not vouch for this probe.
            usage_line("2026-11-03T07:03:00.000000Z", "reader", "get_source", "not_found"),
            usage_line("2026-11-03T07:03:01.000000Z", "sync", "upload_source"),
            # A failed upload vouches for nothing; not_found on other tools is an error.
            usage_line("2026-11-03T08:03:00.000000Z", "sync", "get_source", "not_found"),
            usage_line("2026-11-03T08:03:01.000000Z", "sync", "upload_source", "busy"),
            usage_line("2026-11-03T09:00:00.000000Z", "sync", "get_note", "not_found"),
            usage_line("2026-11-03T09:00:01.000000Z", "sync", "upload_source"),
            usage_line("2026-11-03T10:00:00.000000Z", "gardener", "list_proposals", "forbidden"),
            usage_line("2026-11-03T10:00:01.000000Z", "gardener", "unknown", "protocol_error"),
        ]
        records, _ = usage_report.read_records(lines)
        report = usage_report.build_report(records)
        by_tool = {(e["principal"], e["tool"]): e for e in report["groups"]}
        probe = by_tool[("sync", "get_source")]
        self.assertEqual((probe["calls"], probe["ok"], probe["expected_probe"]), (5, 0, 2))
        self.assertEqual(probe["errors"], {"not_found": 3})
        self.assertEqual(by_tool[("reader", "get_source")]["expected_probe"], 0)
        self.assertEqual(by_tool[("reader", "get_source")]["errors"], {"not_found": 1})
        self.assertEqual(by_tool[("sync", "get_note")]["errors"], {"not_found": 1})
        self.assertEqual(by_tool[("sync", "upload_source")]["errors"], {"busy": 1})
        denied = by_tool[("gardener", "list_proposals")]
        self.assertEqual((denied["role_denied"], denied["errors"]), (1, {}))
        self.assertEqual(by_tool[("gardener", "unknown")]["errors"], {"protocol_error": 1})
        text = usage_report.render(report)
        self.assertIn("sync · get_source: 5 calls, 0 ok, 2 expected pre-upload probes (timing correlation), errors {'not_found': 3}", text)
        self.assertIn("gardener · list_proposals: 1 calls, 0 ok, 1 role denials", text)

    def test_probe_is_correlated_with_an_upload_beyond_the_interval_end(self):
        lines = [
            usage_line("2026-11-03T04:02:59.000000Z", "sync", "get_source", "not_found"),
            usage_line("2026-11-03T04:03:01.000000Z", "sync", "upload_source"),
        ]
        records, _ = usage_report.read_records(lines)
        report = usage_report.build_report(records, until=at("2026-11-03T04:03:00Z"))
        self.assertEqual(report["window"]["records"], 1)
        self.assertEqual((report["groups"][0]["expected_probe"], report["groups"][0]["errors"]), (1, {}))

    def test_principal_filter_and_json_envelope(self):
        lines = [SEARCH, PRODUCTION, FAILED]
        code, out = run(lines, "--principal", "openclaw.clawd", "--json")
        report = json.loads(out)
        self.assertEqual(code, 0)
        self.assertEqual(sorted(report), ["groups", "schema_version", "window"])
        self.assertEqual(report["window"]["principal"], "openclaw.clawd")
        self.assertEqual(report["window"]["records"], 1)
        # Coverage of the supplied logs is reported regardless of the filter.
        self.assertEqual(report["window"]["log_last_record"], "2026-11-02T01:24:18.069390Z")
        self.assertEqual([g["principal"] for g in report["groups"]], ["openclaw.clawd"])
        self.assertNotIn("at", json.dumps(report["groups"][0]["days"]))


if __name__ == "__main__":
    unittest.main()
