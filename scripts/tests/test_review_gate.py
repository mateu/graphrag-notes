import copy
import datetime as dt
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('review_gate', ROOT / 'review-gate.py')
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


class ReviewGateTests(unittest.TestCase):
    def setUp(self):
        self.head = '16132a90b8d36048907d51ee4d39c24ddc5ebbc9'
        self.now = dt.datetime(2026, 11, 3, tzinfo=dt.timezone.utc)
        self.comments = json.loads((ROOT / 'tests/fixtures/review-gate/pr188.json').read_text())

    def evaluate(self, comments=None, **kwargs):
        values = dict(head=self.head, draft=False, since=self.now - dt.timedelta(minutes=21),
                      comments=self.comments if comments is None else comments,
                      threads=[], now=self.now, resolved_receipts={'16132a90b8': self.head})
        values.update(kwargs)
        return gate.evaluate(**values)[0]

    def attestation(self, author='mateu', sha=None, verdict='clean'):
        return {'user': {'login': author}, 'body': '<!-- local-review:v1 ' + json.dumps(
            dict(commit_sha=sha or self.head, verdict=verdict, reviewer='independent-session')) + ' -->'}

    def test_recorded_summary_and_receipts(self):
        self.assertEqual(self.evaluate([self.comments[0]]), 'success')
        self.assertEqual(self.evaluate([self.comments[2]]), 'success')
        self.assertEqual(self.evaluate([self.comments[1]]), 'pending')

    def test_receipt_requires_unambiguous_resolution(self):
        self.assertEqual(self.evaluate([self.comments[2]], resolved_receipts={}), 'pending')
        self.assertEqual(self.evaluate([self.comments[2]], head='16132a90b8' + 'a' * 30), 'pending')

    def test_draft_and_new_head(self):
        self.assertEqual(self.evaluate(draft=True), 'pending')
        self.assertEqual(self.evaluate(head='a' * 40), 'pending')

    def test_forged_author(self):
        for comment in self.comments:
            forged = copy.deepcopy(comment)
            forged['user']['login'] = 'contributor'
            self.assertEqual(self.evaluate([forged]), 'pending')

    def test_completed_findings_and_incomplete(self):
        comment = copy.deepcopy(self.comments[0])
        comment['body'] = comment['body'].replace('"has_unresolved_findings":false', '"has_unresolved_findings":true')
        self.assertEqual(self.evaluate([comment]), 'failure')
        self.assertEqual(self.evaluate([comment, self.comments[2]]), 'failure')
        comment['body'] = comment['body'].replace('"status":"completed"', '"status":"in_progress"')
        self.assertEqual(self.evaluate([comment]), 'pending')

    def test_fallback_timeout_and_identity(self):
        self.assertEqual(self.evaluate([self.attestation()]), 'success')
        self.assertEqual(self.evaluate([self.attestation()], since=self.now), 'pending')
        self.assertEqual(self.evaluate([self.attestation('contributor')]), 'pending')
        self.assertEqual(self.evaluate([self.attestation(sha='b' * 40)]), 'pending')
        self.assertEqual(self.evaluate([self.attestation(verdict='findings')]), 'failure')
        self.assertEqual(self.evaluate([]), 'pending')

    def test_codex_precedes_fallback(self):
        self.assertEqual(self.evaluate(self.comments + [self.attestation(verdict='findings')]), 'success')

    def test_unresolved_head_threads_override_clean(self):
        thread = dict(isResolved=False, comments={'nodes': [{'originalCommit': {'oid': self.head}}]})
        self.assertEqual(self.evaluate(threads=[thread]), 'failure')
        self.assertEqual(self.evaluate([self.attestation()], threads=[thread]), 'failure')
        thread['isResolved'] = True
        self.assertEqual(self.evaluate(threads=[thread]), 'success')
        thread['isResolved'] = False
        thread['comments']['nodes'][0]['originalCommit']['oid'] = 'a' * 40
        self.assertEqual(self.evaluate(threads=[thread]), 'success')

    def test_malformed_markers_fail_closed(self):
        for body in ['<!-- codex-review-state:v1 nope -->', '<!-- codex-review-state:v1 [] -->']:
            self.assertEqual(self.evaluate([dict(user={'login': gate.BOT}, body=body)]), 'pending')
