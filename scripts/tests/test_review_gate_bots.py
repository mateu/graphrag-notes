"""Recorded API evidence plus explicitly synthetic bot-only mutations."""
import copy
import datetime as dt
import json
from test_review_gate import gate, ROOT
import unittest


class BotTests(unittest.TestCase):
    def setUp(self):
        self.recorded = json.loads((ROOT / 'tests/fixtures/review-gate/dependabot224.json').read_text())
        self.data = copy.deepcopy(self.recorded)
        # Synthetic bot-committed variants retain coverage of the bot path.
        # Real Dependabot uses verified web-flow; synthetic
        # bot-only variants below deliberately replace that attributed identity.
        for c in self.data['commits']:
            c['committer'] = c['author']
            c['commit']['committer'] = c['commit']['author']
        self.head = self.data['pr']['head']['sha']

    def exemption(self, config=None):
        return gate.bot_exemption(self.head, self.data['pr'], self.data['commits'], self.data['files'], config)[0]

    def test_recorded_web_flow_is_exempt(self):
        d = self.recorded
        self.assertTrue(gate.bot_exemption(self.head, d['pr'], d['commits'], d['files'])[0])

    def test_synthetic_dependabot_and_renovate(self):
        self.data['files'].append({'filename': 'package.json', 'patch': '-  "qs": "6.16.0"\n+  "qs": "6.17.1"'})
        self.data['pr']['changed_files'] += 1
        self.assertTrue(self.exemption())
        self.data['pr']['user']['login'] = 'renovate[bot]'
        self.data['pr']['head']['ref'] = 'renovate/qs'
        for c in self.data['commits']:
            for role in ('author', 'committer'):
                c[role]['login'] = 'renovate[bot]'
                c['commit'][role]['email'] = 'bot@renovateapp.com'
        self.data['files'] = [f for f in self.data['files'] if f['filename'].endswith('package-lock.json')]
        self.data['pr']['changed_files'] = len(self.data['files'])
        self.assertTrue(self.exemption())

    def test_identity_and_mixed_commits(self):
        for role in ('author', 'committer'):
            for identity in (None, {'login': 'human'}):
                self.setUp()
                self.data['commits'][0][role] = identity
                self.assertFalse(self.exemption())
        self.setUp()
        self.data['commits'][0]['commit']['committer']['email'] = 'human@example.com'
        self.assertFalse(self.exemption())

    def test_paths_rename_major_unknown_and_spoof(self):
        for key in ('filename', 'previous_filename'):
            self.setUp()
            self.data['files'][0][key] = '.github/workflows/evil.yml'
            self.assertFalse(self.exemption())
        for title in ('bump x from 1.0.0 to 2.0.0', 'dependency update'):
            self.setUp()
            self.data['pr']['title'] = title
            self.assertFalse(self.exemption())
        self.setUp()
        self.data['pr']['user']['login'] = 'human'
        self.data['pr']['body'] = 'renovate[bot] patch update'
        self.data['pr']['labels'] = [{'name': 'patch'}]
        self.assertFalse(self.exemption())

    def test_freshness_and_completeness(self):
        for key in ('commits', 'changed_files'):
            self.setUp()
            self.data['pr'][key] += 1
            self.assertFalse(self.exemption())
        self.setUp()
        self.head = 'a' * 40
        self.assertFalse(self.exemption())

    def test_config_override(self):
        config = copy.deepcopy(gate.BOT_CONFIG)
        config['bots'] = {}
        self.assertFalse(self.exemption(config))
        config = copy.deepcopy(gate.BOT_CONFIG)
        config['paths'] = []
        self.assertFalse(self.exemption(config))

    def test_diff_major_and_missing(self):
        for patch in (None, '-foo = "1"\n+foo = "2"', '-name = "foo"\n+name = "bar"', '-version = "1.0.0"\n+version = "2.0.0"'):
            self.setUp()
            self.data['files'][0]['patch'] = patch
            self.assertFalse(self.exemption())

    def test_exempt_does_not_override_findings_or_draft(self):
        now = dt.datetime.now(dt.timezone.utc)
        def result(comments=(), threads=()):
            return gate.evaluate_pr(self.head, self.data['pr'], self.data['commits'], self.data['files'], self.data['pr']['draft'], now, comments, threads, now)[0]
        self.assertEqual(result(), 'success')
        body = '<!-- codex-review-state:v1 ' + json.dumps({'code_review': {'commit_sha': self.head, 'status': 'completed', 'has_unresolved_findings': True}}) + ' -->'
        self.assertEqual(result([{'user': {'login': gate.BOT}, 'body': body}]), 'failure')
        self.assertEqual(result(threads=[{'isResolved': False, 'comments': {'nodes': [{'commit': {'oid': self.head}}]}}]), 'failure')
        self.data['pr']['draft'] = True
        self.assertEqual(result(), 'pending')
