"""Offline contract tests: no credentials or network needed."""
import contextlib
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / 'evaluate-graph-context.py'
spec = importlib.util.spec_from_file_location('evaluate_graph_context', SCRIPT)
eval_graph = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eval_graph)
Q = {'id': 'Q1', 'question': 'Fictional question?', 'seed': ['seed.md'],
     'support': ['support.md'], 'support_chunks': {'support.md': ['note:support']},
     'allowed_relations': ['depends_on']}
MAP = {'aaa': 'seed.md', 'bbb': 'support.md'}
IDS = {'aaa': 'note:seed', 'bbb': 'note:support'}
FIXTURES = Path(__file__).parent / 'fixtures'


def chunk(key, graph=None, chunk_id=None):
    return {'id': chunk_id or IDS.get(key, 'note:' + key), 'source_uri': 'mcp://upload/' + key,
            'score': 0.2, 'graph': graph, 'snippet': 'PRIVATE NOTE CONTENT'}


def edge(kind='depends_on', hops=1, from_id='note:seed', to_id='note:support'):
    return {'hops': hops, 'seed_note_id': 'note:seed', 'query_entities': ['PRIVATE ENTITY'], 'path':
            [{'edge_id': kind + ':123', 'edge_type': kind, 'direction': 'outbound', 'confidence': 1.0,
              'from_id': from_id, 'to_id': to_id, 'reason': 'PRIVATE REASON'}]}


class FakeClient:
    protocol = '2025-11-25'
    calls = []
    tokens = []

    def __init__(self, endpoint, token, deadline):
        self.tokens.append(token)

    def initialize(self):
        pass

    def call(self, name, args):
        self.calls.append((name, args))
        return {'chunks': [chunk('aaa')] + ([chunk('bbb', edge())] if args['graph'] == 'on' else []),
                'diagnostics': {'graph_candidates_selected': int(args['graph'] == 'on'),
                                'unexpected_text': 'PRIVATE DIAGNOSTIC'},
                'rendered_context': 'PRIVATE RENDERED'}


class GraphEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        FakeClient.calls = []
        FakeClient.tokens = []

    def report(self, q=None, client=None):
        return eval_graph.evaluate({'questions': [q or Q]}, MAP, client or FakeClient('', '', 0),
                                   {'scope': 'notes'}, ['off', 'on', 'on', 'off'])[0][0]

    def test_source_mapping_registry_and_simple(self):
        self.assertEqual(eval_graph.source_map(MAP), MAP)
        for parts in ({'0': {'id': 'source:aaa'}}, [{'id': 'source:aaa'}]):
            self.assertEqual(eval_graph.source_map({'files': {'seed.md': {'parts': parts}}}), {'aaa': 'seed.md'})
        row = eval_graph.round_result({'chunks': [chunk('aaa'), chunk('unknown')]}, 'off', 0, MAP, Q)
        self.assertEqual(row['returned_paths'], ['seed.md', 'unmapped'])
        self.assertEqual(row['expected_found'], {'seed.md': True, 'support.md': False})

    def test_expected_paths_stability_and_privacy(self):
        row = self.report()
        summary = row['summary']
        self.assertTrue(summary['off_finds_seed'])
        self.assertFalse(summary['off_finds_support'])
        self.assertTrue(summary['on_finds_support'])
        self.assertFalse(summary['on_same_paths_as_off'])
        self.assertTrue(summary['on_has_edge_path'])
        self.assertEqual(summary['on_graph_selected'], 2)
        self.assertTrue(summary['stable_on'])
        self.assertTrue(summary['stable_off'])
        self.assertNotIn('PRIVATE', json.dumps(row))

    def test_three_state_pass_and_typed_requirements(self):
        self.assertEqual(self.report()['summary']['pass'], 'needs_judgement')
        self.assertEqual(self.report()['summary']['fail_reasons'], [])
        q = dict(Q, answer_differs_materially='yes')
        self.assertEqual(self.report(q)['summary']['pass'], 'pass')
        self.assertEqual(self.report(dict(Q, answer_differs_materially='no'))['summary']['pass'], 'fail')
        self.assertFalse(eval_graph.accepted_path(edge(hops=0), Q['allowed_relations'], 'note:support'))
        self.assertFalse(eval_graph.accepted_path(edge('related_to'), Q['allowed_relations'], 'note:support'))
        self.assertTrue(eval_graph.accepted_path(edge('related_to'), ['related_to'], 'note:support'))
        self.assertFalse(eval_graph.accepted_path({'hops': 1, 'path': []}, ['depends_on'], 'note:support'))
        self.assertFalse(eval_graph.accepted_path({'hops': 2, 'path': edge()['path']}, ['depends_on'], 'note:support'))

    def test_accepted_path_is_a_contiguous_chain_ending_at_the_chunk(self):
        allowed = ['depends_on']
        self.assertTrue(eval_graph.accepted_path(edge(), allowed, 'note:support'))
        # The terminal must be the returned chunk, and the chain must start at the graph seed.
        for chunk_id in ('note:seed', 'note:other', None, ''):
            self.assertFalse(eval_graph.accepted_path(edge(), allowed, chunk_id))
        self.assertFalse(eval_graph.accepted_path(edge(from_id='note:elsewhere'), allowed, 'note:support'))
        two = edge(hops=2)
        two['path'].append(dict(two['path'][0], edge_id='depends_on:456', from_id='note:gap', to_id='note:end'))
        self.assertFalse(eval_graph.accepted_path(two, allowed, 'note:end'))
        two['path'][1]['from_id'] = 'note:support'
        self.assertTrue(eval_graph.accepted_path(two, allowed, 'note:end'))
        # Directions are the service's traversal labels, optionally narrowed per question.
        self.assertTrue(eval_graph.accepted_path(edge(), allowed, 'note:support', ['outbound']))
        self.assertFalse(eval_graph.accepted_path(edge(), allowed, 'note:support', ['inbound']))
        for key, value in (('direction', 'outgoing'), ('edge_id', 'related_to:123'), ('edge_id', ''),
                           ('to_id', 'note:seed')):
            bad = edge()
            bad['path'][0][key] = value
            self.assertFalse(eval_graph.accepted_path(bad, allowed, bad['path'][0]['to_id']), key)
        for key in ('edge_id', 'edge_type', 'direction', 'from_id', 'to_id'):
            bad = edge()
            del bad['path'][0][key]
            self.assertFalse(eval_graph.accepted_path(bad, allowed, 'note:support'), key)
        for key, value in (('hops', True), ('seed_note_id', None), ('path', None)):
            bad = edge()
            bad[key] = value
            self.assertFalse(eval_graph.accepted_path(bad, allowed, 'note:support'), key)

    def test_support_missing_from_one_on_round_cannot_pass(self):
        unrelated = edge(from_id='note:unrelated-a', to_id='note:unrelated-b')

        class Flaky(FakeClient):
            def call(self, name, args):
                data = super().call(name, args)
                if args['graph'] == 'on' and sum(a['graph'] == 'on' for _, a in self.calls) == 2:
                    data['chunks'] = [chunk('aaa'), chunk('ccc', unrelated)]
                return data
        summary = self.report(dict(Q, answer_differs_materially='yes'), Flaky('', '', 0))['summary']
        self.assertFalse(summary['stable_on'])
        self.assertEqual(summary['pass'], 'fail')
        self.assertEqual(summary['on_support_by_round'], [True, False])
        self.assertFalse(summary['on_returns_support_via_allowed_path'])
        self.assertEqual(summary['fail_reasons'], ['inconsistent_support', 'support_missing'])

    def test_fictional_path_negative_fixtures(self):
        fixture = json.loads((FIXTURES / 'graph-eval-path-negatives.json').read_text())
        self.assertNotIn('snippet', json.dumps(fixture))
        controls = {'valid_control', 'valid_control_unknown_materiality', 'valid_two_hop_inbound_control'}
        self.assertTrue(controls <= {case['name'] for case in fixture['cases']})
        for case in fixture['cases']:
            with self.subTest(case=case['name']):
                q = dict(fixture['question'], **case.get('question_overrides', {}))
                eval_graph.load_suite({'questions': [q]})
                off = case.get('off', fixture['off'])
                rounds = [eval_graph.round_result(data, policy, 0, fixture['source_map'], q) for policy, data in
                          [('off', off)] + [('on', on) for on in case['on']] + [('off', off)]]
                summary = eval_graph.summarize(q, rounds)
                self.assertEqual(summary['pass'], case['expected_pass'])
                self.assertEqual(summary['fail_reasons'], case['expected_reasons'])
                self.assertEqual(case['name'] in controls, not case['expected_reasons'])
                for key, value in case.get('expected_flags', {}).items():
                    self.assertEqual(summary[key], value, key)

    def test_projection_keeps_opaque_chunk_ids(self):
        row = eval_graph.round_result({'chunks': [chunk('aaa'), chunk('bbb', edge()), chunk('ccc', chunk_id=7)]},
                                      'on', 0, MAP, Q)
        self.assertEqual(row['returned_ids'], ['note:seed', 'note:support', None])
        self.assertEqual(row['chunk_graph'][1]['seed_note_id'], 'note:seed')
        self.assertEqual(row['chunk_graph'][1]['path'][0]['to_id'], 'note:support')
        self.assertNotIn('PRIVATE', json.dumps(row))
        moved = eval_graph.round_result({'chunks': [chunk('aaa'), chunk('bbb', edge(), 'note:moved'),
                                                    chunk('ccc', chunk_id=7)]}, 'on', 0, MAP, Q)
        self.assertEqual(moved['returned_paths'], row['returned_paths'])
        self.assertNotEqual(moved['response_sha256'], row['response_sha256'])

    def test_missing_ids_rounds_or_support_chunks_cannot_pass(self):
        q = dict(Q, answer_differs_materially='yes')
        rounds = self.report(q)['rounds']
        self.assertEqual(eval_graph.summarize(q, rounds)['pass'], 'pass')
        legacy = copy.deepcopy(rounds)
        for r in legacy:
            del r['returned_ids']
        misaligned = copy.deepcopy(rounds)
        misaligned[1]['chunk_graph'] = misaligned[1]['chunk_graph'][:1]
        for bad, reasons in [(legacy, ['missing_fields']), (misaligned, ['inconsistent_support', 'missing_fields']),
                             (rounds[:2], ['insufficient_rounds']),
                             ([r for r in rounds if r['policy'] == 'off'], ['insufficient_rounds'])]:
            summary = eval_graph.summarize(q, bad)
            self.assertEqual((summary['pass'], summary['fail_reasons']), ('fail', reasons))
        unspecified = {k: v for k, v in q.items() if k != 'support_chunks'}
        self.assertEqual(eval_graph.summarize(unspecified, rounds)['fail_reasons'], ['support_chunk_unspecified'])

    def test_every_support_path_needs_its_own_delivered_chunk(self):
        q = dict(Q, support=['support.md', 'second.md'], answer_differs_materially='yes',
                 support_chunks={'support.md': ['note:support'], 'second.md': ['note:second']})
        rounds = eval_graph.evaluate({'questions': [q]}, dict(MAP, ddd='second.md'), FakeClient('', '', 0),
                                     {'scope': 'notes'}, ['off', 'on', 'on', 'off'])[0][0]['rounds']
        summary = eval_graph.summarize(q, rounds)
        self.assertEqual((summary['pass'], summary['fail_reasons']), ('fail', ['support_missing']))
        self.assertEqual(summary['on_support_by_round'], [False, False])
        # A delivered chunk does not vouch for a support path without its own chunk identity.
        partial = dict(q, support_chunks={'support.md': ['note:support']})
        summary = eval_graph.summarize(partial, rounds)
        self.assertEqual((summary['pass'], summary['fail_reasons']), ('fail', ['support_chunk_unspecified']))
        self.assertEqual(summary['on_support_by_round'], [False, False])

    def test_suite_support_chunks_and_directions_are_validated(self):
        eval_graph.load_suite({'questions': [dict(Q, allowed_directions=['outbound'])]})
        # Suites written before chunk identities still load; they just cannot pass.
        eval_graph.load_suite({'questions': [{k: v for k, v in Q.items() if k != 'support_chunks'}]})
        for bad in ({'support_chunks': {'other.md': ['note:x']}}, {'support_chunks': {'support.md': []}},
                    {'support_chunks': {'support.md': 'note:support'}}, {'support_chunks': {'support.md': ['']}},
                    {'support_chunks': {'support.md': ['note:x', 'note:x']}}, {'support_chunks': ['note:x']},
                    {'allowed_directions': []}, {'allowed_directions': ['sideways']},
                    {'allowed_directions': 'outbound'}):
            with self.assertRaises(eval_graph.EvaluationError, msg=str(bad)):
                eval_graph.load_suite({'questions': [dict(Q, **bad)]})

    def test_committed_baselines_keep_their_fail_verdicts(self):
        for name in ('graph-eval-20261102.json', 'graph-eval-20261103.json'):
            report = json.loads((FIXTURES / name).read_text())
            self.assertEqual(report['schema_version'], 1)
            for row in report['questions']:
                q = {'seed': row['seed'], 'support': row['support'],
                     'allowed_relations': row.get('allowed_relations', [])}
                summary = eval_graph.summarize(q, row['rounds'])
                self.assertEqual(summary['pass'], 'fail')
                self.assertEqual(row['summary'].get('pass', 'fail'), 'fail')
                # v1 rounds predate chunk identities, so they are never re-graded as passes.
                self.assertIn('missing_fields', summary['fail_reasons'])
                self.assertIn('support_chunk_unspecified', summary['fail_reasons'])

    def test_support_path_must_belong_to_support_chunk(self):
        row = self.report(dict(Q, answer_differs_materially='yes'))
        for r in row['rounds']:
            if r['policy'] == 'on':
                r['chunk_graph'] = [edge(), None]
        self.assertEqual(eval_graph.summarize(Q, row['rounds'])['pass'], 'fail')
        # Off already returning support defeats the first suite condition.
        for r in row['rounds']:
            r['returned_paths'] = ['seed.md', 'support.md']
        self.assertTrue(eval_graph.summarize(Q, row['rounds'])['on_same_paths_as_off'])
        self.assertEqual(eval_graph.summarize(Q, row['rounds'])['pass'], 'fail')

    def test_call_failure_never_passes_or_leaks_exception(self):
        class Bad(FakeClient):
            def call(self, name, args):
                raise eval_graph.mcp.ImportFailure('PRIVATE TOKEN')
        row = self.report(client=Bad('', '', 0))
        self.assertEqual(row['summary']['pass'], 'fail')
        self.assertEqual(row['summary']['fail_reasons'], ['call_error'])
        self.assertNotIn('PRIVATE', json.dumps(row))

    def test_baseline_diff_fixture(self):
        baseline = json.loads((Path(__file__).parent / 'fixtures/graph-eval-20261102.json').read_text())
        current = copy.deepcopy(baseline)
        for q in current['questions']:
            q['summary']['pass'] = 'fail'
        diff = eval_graph.compare(current, baseline)
        self.assertEqual(diff.count('pass=fail -> fail'), 8)
        self.assertEqual(diff.count('gained=[] lost=[]'), 16)
        current['questions'][0]['rounds'][1]['returned_paths'] = ['new.md']
        current['questions'][0]['rounds'][1]['elapsed_ms'] += 20
        current['questions'][0]['summary'].update(pass_='ignored', stable_on=False)
        current['questions'][0]['summary']['pass'] = 'pass'
        diff = eval_graph.compare(current, baseline)
        self.assertIn('gained=["new.md"]', diff)
        self.assertIn('elapsed_delta_ms=+10.0', diff)
        self.assertIn('"stable_on": [true, false]', diff)
        self.assertIn('pass=fail -> pass', diff)

    def test_seed_revision_diff_fixtures(self):
        fixtures = Path(__file__).parent / 'fixtures'
        old = json.loads((fixtures / 'graph-eval-20261102.json').read_text())
        new = json.loads((fixtures / 'graph-eval-20261103.json').read_text())
        lines = eval_graph.compare(new, old).splitlines()
        expected = {
            'Q2': (['memory/2026-08-01-0701.md', 'memory/2026-10-07-pr142-review.md'],
                   ['memory/2026-07-25-1234.md', 'memory/2026-07-31-2051.md']),
            'Q7': (['memory/2026-10-07-pr142-review.md', 'memory/2026-10-08-codex-cli-version.md'],
                   ['memory/2026-10-15-graphrag-ingestion-failure.md', 'memory/2026-10-27-write-conflict-retry.md']),
        }
        flips = {key: [False, True] for key in
                 ('off_finds_seed', 'off_finds_support', 'on_finds_seed', 'on_finds_support')}
        self.assertEqual(len(lines), 8)
        self.assertEqual(new['protocol'], old['protocol'])
        for row, line in zip(new['questions'], lines):
            qid = row['id']
            gained, lost = expected.get(qid, ([], []))
            for policy in ('off', 'on'):
                self.assertIn('%s gained=%s lost=%s' %
                              (policy, json.dumps(gained), json.dumps(lost)), line)
            self.assertIn('flags_flipped=' + json.dumps(flips if qid in expected else {}, sort_keys=True), line)
            self.assertIn('pass=fail -> fail', line)
            self.assertTrue(all(r['error'] is None for r in row['rounds']))
            if qid in expected:
                self.assertEqual(row['seed'], ['memory/2026-10-07-pr142-review.md'])
                self.assertTrue(row['summary']['off_finds_seed'])
                # Same-file support already found off cannot establish graph lift.
                self.assertFalse(row['summary']['off_lacks_support'])

    def argv(self, *extra, question=Q):
        (self.root / 'suite.json').write_text(json.dumps({'questions': [question]}))
        (self.root / 'map.json').write_text(json.dumps(MAP))
        return ['--suite', str(self.root / 'suite.json'), '--source-map', str(self.root / 'map.json'),
                '--endpoint', 'http://127.0.0.1:33157/mcp', '--output', str(self.root / 'report.json'),
                '--credential-env', 'TEST_GRAPH_TOKEN'] + list(extra)

    def run_main(self, argv, env=None):
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch.dict(os.environ, env or {}, clear=True), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            result = eval_graph.main(argv, FakeClient)
        return result, stdout.getvalue(), stderr.getvalue()

    def test_output_mode_refuse_overwrite_private_dir_and_cap(self):
        private = self.root / 'private'
        args = self.argv('--private-dir', str(private), '--max-chunks-per-source', '1')
        result, stdout, _ = self.run_main(args, {'TEST_GRAPH_TOKEN': 'SECRET'})
        self.assertEqual(result, 0)
        self.assertEqual(FakeClient.tokens, ['SECRET'])
        self.assertEqual(stat.S_IMODE((self.root / 'report.json').stat().st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(private.stat().st_mode), 0o700)
        self.assertEqual(stat.S_IMODE((private / 'private-responses.json').stat().st_mode), 0o600)
        self.assertIn('PRIVATE', (private / 'private-responses.json').read_text())
        self.assertNotIn('PRIVATE', (self.root / 'report.json').read_text() + stdout)
        self.assertNotIn('SECRET', (self.root / 'report.json').read_text() + stdout)
        report = json.loads((self.root / 'report.json').read_text())
        self.assertEqual(report['schema_version'], 2)
        row = report['questions'][0]
        self.assertEqual((row['support_chunks'], row['allowed_directions']),
                         (Q['support_chunks'], ['outbound', 'inbound']))
        self.assertEqual([r['returned_ids'] for r in row['rounds']],
                         [['note:seed'], ['note:seed', 'note:support'], ['note:seed', 'note:support'], ['note:seed']])
        self.assertIn('Q1: needs_judgement on==off=False graph_selected=2\n', stdout)
        self.assertTrue(all(name == 'build_context' and args['max_chunks_per_source'] == 1 for name, args in FakeClient.calls))
        self.assertEqual(self.run_main(args, {'TEST_GRAPH_TOKEN': 'SECRET'})[0], 2)
        self.assertEqual(len(FakeClient.calls), 4)

    def test_summary_output_names_fail_reasons(self):
        legacy = {k: v for k, v in Q.items() if k != 'support_chunks'}
        result, stdout, _ = self.run_main(self.argv(question=legacy), {'TEST_GRAPH_TOKEN': 'SECRET'})
        self.assertEqual(result, 0)
        self.assertIn('Q1: fail on==off=False graph_selected=2 reasons=support_chunk_unspecified\n', stdout)
        self.assertIn('0/1 pass', stdout)

    def test_bad_permissions_credentials_and_endpoint_before_network(self):
        private = self.root / 'private'
        private.mkdir(mode=0o755)
        for extra, env in [([], {}), (['--credential-env', 'invalid-name'], {'TEST_GRAPH_TOKEN': 'secret'}),
                           (['--private-dir', str(private)], {'TEST_GRAPH_TOKEN': 'secret'}),
                           (['--endpoint', 'http://example.com/mcp'], {'TEST_GRAPH_TOKEN': 'secret'}),
                           (['--endpoint', 'http://secret@localhost/mcp'], {'TEST_GRAPH_TOKEN': 'secret'})]:
            self.assertEqual(self.run_main(self.argv(*extra), env)[0], 2)
        self.assertEqual(FakeClient.calls, [])


if __name__ == '__main__':
    unittest.main()
