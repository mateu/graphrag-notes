import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock


SPEC = importlib.util.spec_from_file_location("retrieval_evaluation", Path(__file__).resolve().parents[1] / "evaluate-retrieval.py")
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)


def case():
    return {"name": "fictional-atlas", "split": "holdout", "category": "paraphrase",
            "answerability": "answerable", "judgment": "fictional fixture",
            "eval": {"schema_version": 2, "query": "When does Atlas launch?", "k": 5, "limit": 5,
                     "relevance": [{"id": "note:a", "grade": 3}, {"id": "note:b", "grade": 1},
                                   {"id": "note:c", "grade": 0}]}}


def record(identifier="note:a"):
    return {"id": identifier, "revision": "revision-1", "title": "Atlas launch",
            "content": "short preview", "provenance": {"instance_id": "fictional", "source": {"uri": "mcp://fictional"}}}


def inspect(value):
    return {**value, "content": "short preview followed by complete fictional content"}


class RetrievalEvaluationTests(unittest.TestCase):
    def test_report_hashes_the_inputs_loaded_before_retrieval(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            root.chmod(0o700)
            suite, fixture, output = (root / value for value in ("suite.json", "fixture.json", "report.json"))
            original = json.dumps({"schema_version": 1, "metadata": {}, "cases": [case()]}).encode()
            suite.write_bytes(original)
            fixture.write_text(json.dumps({"rankings": {}, "inspections": {}}))
            def replace_files(loaded, search, inspect):
                self.assertEqual(loaded["cases"][0]["name"], "fictional-atlas")
                suite.write_text("replaced suite")
                return {"metadata": {}, "cases": []}
            fake_runner = root / "runner.py"
            fake_runner.write_text("replacement code")
            with mock.patch.object(evaluation, "evaluate", side_effect=replace_files), \
                    mock.patch.object(evaluation, "__file__", str(fake_runner)), mock.patch("sys.stdout"):
                code = evaluation.main(["--suite", str(suite), "--recorded", str(fixture), "--output", str(output)])
            report = json.loads(output.read_text())
            self.assertEqual(code, 0)
            self.assertEqual(report["suite_sha256"], hashlib.sha256(original).hexdigest())
            self.assertEqual(report["runner_sha256"], evaluation.RUNNER_SHA256)
            self.assertNotEqual(report["runner_sha256"], hashlib.sha256(fake_runner.read_bytes()).hexdigest())

    def test_unknown_hits_do_not_become_irrelevant_or_judged_precision(self):
        result = evaluation.metrics(case(), [record(), record("note:unknown")])
        self.assertEqual(result["unjudged_hits"], 1)
        self.assertEqual(result["recall_of_judged_positives"], 0.5)
        self.assertEqual(result["top_k_useful_lower_bound"], 0.2)
        self.assertEqual(result["top_k_useful_upper_bound"], 0.4)
        self.assertIsNone(result["precision_at_k"])
        self.assertIsNone(result["ndcg_at_k"])

    def test_grade_zero_is_an_explicit_false_positive(self):
        result = evaluation.metrics(case(), [record("note:c"), record()])
        self.assertEqual(result["false_positive_count"], 1)
        self.assertEqual(result["reciprocal_rank_of_judged_positives"], 0.5)
        self.assertEqual(result["precision_at_k"], 0.2)
        self.assertLess(result["ndcg_at_k"], 1)

    def test_empty_answerable_case_is_a_miss(self):
        result = evaluation.metrics(case(), [])
        self.assertEqual(result["recall_of_judged_positives"], 0)
        self.assertEqual(result["reciprocal_rank_of_judged_positives"], 0)
        self.assertEqual(result["ndcg_at_k"], 0)

    def test_reviewed_no_answer_case_counts_returned_false_positives(self):
        item = case()
        item["answerability"] = "unanswerable"
        item["eval"]["relevance"] = []
        result = evaluation.metrics(item, [record()])
        self.assertFalse(result["negative_empty"])
        self.assertEqual(result["false_positive_count"], 1)
        self.assertIsNone(result["recall_of_judged_positives"])
        self.assertTrue(evaluation.metrics(item, [])["negative_empty"])

    def test_unreviewed_answerability_does_not_score_provisional_positives(self):
        item = case()
        item["answerability"] = "unjudged"
        result = evaluation.metrics(item, [record()])
        self.assertIsNone(result["recall_of_judged_positives"])
        self.assertIsNone(result["reciprocal_rank_of_judged_positives"])
        self.assertEqual(result["unjudged_hits"], 1)

    def test_duplicate_ranked_ids_are_rejected(self):
        with self.assertRaises(evaluation.EvaluationError):
            evaluation.metrics(case(), [record(), record()])

    def test_reciprocal_rank_of_judged_positives_is_only_a_lower_bound(self):
        result = evaluation.metrics(case(), [record("note:unknown"), record("note:c"),
                                             record("note:x"), record("note:y"), record()])
        self.assertEqual(result["reciprocal_rank_of_judged_positives"], 0.2)
        self.assertNotIn("reciprocal_rank", result)
        self.assertEqual(result["unjudged_hits"], 3)

    def test_legacy_expected_ids_and_graded_relevance_share_eval_semantics(self):
        item = case()["eval"]
        item["expected_ids"] = ["note:a"]
        self.assertEqual(evaluation.judgments(item)["note:a"], 3)
        item["relevance"].append({"id": "note:a", "grade": 2})
        with self.assertRaises(evaluation.EvaluationError):
            evaluation.judgments(item)

    def test_stale_or_foreign_inspection_is_never_verified(self):
        for key, value in (("id", "note:other"), ("revision", "new-revision"),
                           ("provenance", {"instance_id": "foreign"})):
            with self.subTest(key=key), self.assertRaises(evaluation.EvaluationError):
                evaluation.verify_readback(record(), {**inspect(record()), key: value})
        self.assertEqual(evaluation.verify_readback(record(), inspect(record()))["id"], "note:a")

    def test_cli_envelopes_require_authoritative_success(self):
        value = {"schema_version": 1, "command": "search_notes", "success": True, "errors": [],
                 "data": {"schema_version": 1, "error": None, "data": {"records": []}}}
        self.assertEqual(evaluation.cli_data(value, "search_notes"), {"records": []})
        for changed in ("schema_version", "success", "errors"):
            bad = copy.deepcopy(value)
            bad[changed] = None
            with self.assertRaises(evaluation.EvaluationError):
                evaluation.cli_data(bad, "search_notes")
        for nested in (False, True):
            bad = copy.deepcopy(value)
            envelope = bad["data"] if nested else bad
            envelope["schema_version"] = True
            with self.assertRaises(evaluation.EvaluationError):
                evaluation.cli_data(bad, "search_notes")

    def test_protocol_json_refuses_duplicate_keys_and_nonfinite_values(self):
        for raw in ('{"schema_version": 99, "schema_version": 1}', '{"metadata": {"value": NaN}}'):
            with self.subTest(raw=raw), self.assertRaises(evaluation.EvaluationError):
                evaluation.strict_json(raw)

    def test_query_and_filters_are_literal_arguments_not_shell_text(self):
        item = case()
        query = "--odd $(touch forbidden) `echo nope`"
        item["eval"].update(query=query, source_uri="file:///fictional/a b.md", since_days=30)
        value = {"schema_version": 1, "command": "search_notes", "success": True, "errors": [],
                 "data": {"schema_version": 1, "error": None, "data": {"records": []}}}
        client = evaluation.RemoteCli("fictional-binary", "http://127.0.0.1:3000/mcp", "TEST_TOKEN", 10)
        with mock.patch.object(subprocess, "run", return_value=mock.Mock(returncode=0, stdout=json.dumps(value))) as run:
            self.assertEqual(client.search(item, "keyword-off"), [])
            argv = run.call_args.args[0]
            self.assertEqual(argv[-2:], ["--", query])
            self.assertEqual(argv.count("--format"), 1)
            self.assertIn("--source-uri=file:///fictional/a b.md", argv)
            self.assertNotIn("shell", run.call_args.kwargs)

    def test_option_like_source_uri_is_an_attached_literal_value(self):
        item = case()
        item["eval"]["source_uri"] = "-missing-uri"
        client = evaluation.RemoteCli("fixture", "http://127.0.0.1:3000/mcp", "TEST_TOKEN", 10)
        with mock.patch.object(client, "call", return_value={"records": []}) as call:
            client.search(item, "keyword-off")
        self.assertIn("--source-uri=-missing-uri", call.call_args.args[0])
        self.assertNotIn("-missing-uri", call.call_args.args[0])

    def test_eval_ids_are_trimmed_and_unicode_lowercased_before_merging(self):
        item = case()
        item["eval"].update(expected_ids=[" NOTE:ÉCOLE ", "note:école"],
                            relevance=[{"id": " Note:École ", "grade": 3}])
        result = evaluation.metrics(item, [record("NOTE:ÉCOLE")])
        self.assertEqual(result["known_positive_total"], 1)
        self.assertEqual(result["reciprocal_rank_of_judged_positives"], 1)
        self.assertEqual(result["recall_of_judged_positives"], 1)

    def test_no_answer_false_positives_count_results_beyond_k(self):
        item = case()
        item["answerability"] = "unanswerable"
        item["eval"].update(k=5, limit=20, relevance=[])
        result = evaluation.metrics(item, [record("note:" + str(i)) for i in range(20)])
        self.assertEqual(result["retrieved"], 5)
        self.assertEqual(result["false_positive_count"], 20)

    def test_failure_stays_in_requested_policy_without_fallback(self):
        calls = []
        def failed(item, policy):
            calls.append(policy)
            raise evaluation.EvaluationError("provider unavailable")
        report = evaluation.evaluate({"metadata": {}, "cases": [case()]}, failed, inspect)
        self.assertEqual(calls, list(evaluation.POLICIES))
        self.assertTrue(all(row["status"] == "failed" for row in report["cases"]))

    def test_recorded_missing_ranking_or_readback_retains_complete_report(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            root.chmod(0o700)
            suite = root / "suite.json"
            fixture = root / "fixture.json"
            output = root / "report.json"
            suite.write_text(json.dumps({"schema_version": 1, "metadata": {}, "cases": [case()]}))
            fixture.write_text(json.dumps({"rankings": {"fictional-atlas": {
                "keyword-off": [record()], "hybrid-off": [], "hybrid-auto": []}}, "inspections": {}}))
            with mock.patch("sys.stdout"):
                code = evaluation.main(["--suite", str(suite), "--recorded", str(fixture), "--output", str(output)])
            report = json.loads(output.read_text())
            self.assertEqual(code, 1)
            self.assertEqual(len(report["cases"]), 4)
            self.assertEqual([row["status"] for row in report["cases"]], ["failed", "ok", "ok", "failed"])

    def test_summary_cannot_leak_private_labels_metadata_or_content(self):
        report = evaluation.evaluate({"metadata": {"private": "SECRET"}, "cases": [case()]},
                                     lambda item, policy: [record()], inspect)
        summary = json.dumps(evaluation.aggregate(report))
        for private in ("SECRET", "fictional-atlas", "note:a", "When does Atlas", "short preview", "mcp://fictional"):
            self.assertNotIn(private, summary)
        self.assertEqual(len(evaluation.aggregate(report)["groups"]), 8)

    def test_private_reports_refuse_overwrite_and_symlink_directory(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            root.chmod(0o700)
            path = root / "private-report.json"
            evaluation.write_new(path, {"fictional": True})
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            with self.assertRaises(FileExistsError):
                evaluation.write_new(path, {})
            link = root / "link"
            link.symlink_to(root, target_is_directory=True)
            with self.assertRaises(evaluation.EvaluationError):
                evaluation.private_directory(link)

    def test_formulation_comparison_has_separate_scored_denominators(self):
        question, terms = case(), case()
        terms.update(name="fictional-terms", formulation="terms")
        report = evaluation.evaluate({"metadata": {}, "cases": [question, terms]},
                                     lambda item, policy: [] if item is question else [record()], inspect)
        group = next(g for g in evaluation.aggregate(report)["groups"]
                     if g["split"] == "holdout" and g["policy"] == "keyword-off")
        self.assertEqual(group["formulation_metrics"]["question"]["reciprocal_rank_of_judged_positives"],
                         {"mean": 0, "cases_scored": 1})
        self.assertEqual(group["formulation_metrics"]["terms"]["reciprocal_rank_of_judged_positives"],
                         {"mean": 1, "cases_scored": 1})
        self.assertIsNone(group["formulation_metrics"]["title"]["recall_of_judged_positives"]["mean"])

    def test_summary_separates_reviewed_no_answer_false_positives(self):
        negative = case()
        negative.update(name="fictional-absent", answerability="unanswerable")
        negative["eval"]["relevance"] = []
        report = evaluation.evaluate({"metadata": {}, "cases": [case(), negative]},
                                     lambda item, policy: [record()], inspect)
        group = next(g for g in evaluation.aggregate(report)["groups"]
                     if g["split"] == "holdout" and g["policy"] == "hybrid-off")
        self.assertEqual(group["reviewed_no_answer"],
                         {"cases": 1, "empty_results": 0, "returned_false_positives": 1})
        self.assertEqual(group["metrics"]["false_positive_count"], {"mean": 0.5, "cases_scored": 2})

    def test_suite_rejects_typos_and_conflicting_answerability(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "suite.json"
            value = {"schema_version": 1, "metadata": {}, "cases": [case()]}
            path.write_text(json.dumps(value))
            self.assertEqual(len(evaluation.load_suite(path)["cases"]), 1)
            value["cases"][0]["eval"]["source_url"] = "typo"
            path.write_text(json.dumps(value))
            with self.assertRaises(evaluation.EvaluationError):
                evaluation.load_suite(path)
            del value["cases"][0]["eval"]["source_url"]
            value["cases"][0]["answerability"] = "unanswerable"
            path.write_text(json.dumps(value))
            with self.assertRaises(evaluation.EvaluationError):
                evaluation.load_suite(path)


if __name__ == "__main__":
    unittest.main()
