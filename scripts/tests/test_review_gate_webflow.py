"""Real recorded GitHub API evidence and fail-closed identity mutations."""
import copy
import json
import unittest
from test_review_gate import gate, ROOT


class WebFlowTests(unittest.TestCase):
    def test_real_and_negative_evidence(self):
        for number in (222, 223, 224):
            data = json.loads((ROOT / f'tests/fixtures/review-gate/dependabot{number}.json').read_text())
            def evaluate(d):
                return gate.bot_exemption(d['pr']['head']['sha'], d['pr'], d['commits'], d['files'])
            with self.subTest(number=number, case='recorded'):
                ok, reason = evaluate(data)
                self.assertTrue(ok, reason)
                self.assertIn('github-signed web-flow', reason)
            def invalid_signature(d):
                d['commits'][0]['commit']['verification']['verified'] = False
            def invalid_reason(d):
                d['commits'][0]['commit']['verification']['reason'] = 'unknown_key'
            def missing_signature(d):
                del d['commits'][0]['commit']['verification']
            def wrong_author(d):
                d['commits'][0]['author']['login'] = 'human'
            def wrong_author_email(d):
                d['commits'][0]['commit']['author']['email'] = 'human@example.com'
            def human(d):
                wrong_author(d)
                d['pr']['user']['login'] = 'human'
            def wrong_branch(d):
                d['pr']['head']['ref'] = 'human/update'
            def wrong_email(d):
                d['commits'][0]['commit']['committer']['email'] = 'human@example.com'
            def human_committer(d):
                d['commits'][0]['committer']['login'] = 'human'
            def mixed(d):
                bad = copy.deepcopy(d['commits'][0])
                bad['committer']['login'] = 'human'
                d['commits'].insert(0, bad)
                d['pr']['commits'] += 1
            for mutation in (invalid_signature, invalid_reason, missing_signature, wrong_author,
                             wrong_author_email, human, wrong_branch, wrong_email, human_committer, mixed):
                d = copy.deepcopy(data)
                mutation(d)
                with self.subTest(number=number, case=mutation.__name__):
                    self.assertFalse(evaluate(d)[0])
