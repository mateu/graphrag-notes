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
        self.assertEqual(report['schema_version'], 3)
        self.assertEqual(report['questions'][0]['summary']['span_check'], 'unchecked')
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


TERMS = ['Launch Window', 'tuesday 09:00']
EVIDENCE = {'note:support': {'required_terms': TERMS}}
ANSWER = 'The LAUNCH WINDOW is Tuesday 09:00, after the freeze.'
ELSEWHERE = 'An unrelated passage of the same chunk about the launch window.'


def rendered(snippet=ANSWER, start=120, end=320, truncated=True, **extra):
    """The support chunk as build_context renders it: one clipped span."""
    value = dict(chunk('bbb', edge()), snippet=snippet, selected_span_start=start,
                 selected_span_end=end, truncated=truncated)
    value.update(extra)
    return value


class SpanClient(FakeClient):
    """Graph-on returns `on` (one support chunk per on round); graph-off returns `off` snippets."""
    on = [rendered(), rendered()]
    off = ['Seed premise only.']

    def __init__(self, endpoint='', token='', deadline=0):
        super().__init__(endpoint, token, deadline)
        self.on_calls = 0

    def call(self, name, args):
        self.calls.append((name, args))
        if args['graph'] == 'off':
            return {'chunks': [dict(chunk(key), snippet=text) for key, text in zip(('aaa', 'ccc'), self.off)],
                    'diagnostics': {}}
        support = self.on[self.on_calls % len(self.on)]
        self.on_calls += 1
        return {'chunks': [dict(chunk('aaa'), snippet='Seed premise only.'), support],
                'diagnostics': {'graph_candidates_selected': 1}}


def span_client(on=None, off=None):
    return type('Client', (SpanClient,), {'on': on or SpanClient.on, 'off': off or SpanClient.off})()


class SpanEvidenceTests(unittest.TestCase):
    # Shared fixtures only; inheriting the class would run its tests twice.
    setUp, report = GraphEvaluationTests.setUp, GraphEvaluationTests.report
    argv, run_main = GraphEvaluationTests.argv, GraphEvaluationTests.run_main

    def graded(self, on=None, off=None, **question):
        q = dict(Q, support_evidence=EVIDENCE)
        q.update(question)
        eval_graph.load_suite({'questions': [q]})
        return self.report(q, span_client(on, off))

    def test_covered_span_stays_needs_judgement_and_never_passes_by_itself(self):
        row = self.graded()
        summary = row['summary']
        self.assertEqual((summary['pass'], summary['fail_reasons']), ('needs_judgement', []))
        self.assertEqual((summary['span_check'], summary['span_unchecked_chunks']), ('covered', 0))
        self.assertEqual(summary['on_span_by_round'],
                         [{'checked': 1, 'covered': 1, 'terms_found': 2, 'terms_required': 2}] * 2)
        self.assertIs(summary['off_answer_terms_found'], False)
        on = [r for r in row['rounds'] if r['policy'] == 'on'][0]
        self.assertEqual(on['returned_spans'], [{'start': None, 'end': None, 'truncated': None},
                                                {'start': 120, 'end': 320, 'truncated': True}])
        self.assertEqual(on['span_evidence'], {'note:support': {
            'status': 'covered', 'terms_required': 2, 'context_terms_found': 2, 'terms_found': 2}})
        self.assertEqual(row['support_evidence'], {'note:support': {'required_terms': 2, 'answer_span': None}})
        self.assertEqual(self.graded(answer_differs_materially='yes')['summary']['pass'], 'pass')

    def test_delivered_chunk_with_the_answer_outside_its_span_fails(self):
        # The V2-E07 shape: chunk ID delivered over a valid path, rendered span elsewhere.
        summary = self.graded(on=[rendered(ELSEWHERE, 514, 854)], answer_differs_materially='yes')['summary']
        self.assertEqual((summary['pass'], summary['fail_reasons']), ('fail', ['support_span_missing']))
        self.assertEqual(summary['span_check'], 'missing')
        # ID delivery is still reported as such; only the rendered text is missing.
        self.assertTrue(summary['on_returns_support_via_allowed_path'])
        self.assertEqual(summary['on_support_by_round'], [True, True])
        self.assertEqual(summary['on_span_by_round'][0],
                         {'checked': 1, 'covered': 0, 'terms_found': 1, 'terms_required': 2})

    def test_terms_match_case_folded_nfc_substrings(self):
        decomposed = 'Café OPENS at 9'
        evidence = {'note:support': {'required_terms': ['café opens']}}
        self.assertEqual(self.graded(on=[rendered(decomposed)], support_evidence=evidence)['summary']['span_check'],
                         'covered')

    def test_answer_span_needs_no_text_and_untruncated_chunks_cover_any_range(self):
        evidence = {'note:support': {'answer_span': [150, 300]}}
        def check(**fields):
            return self.graded(on=[rendered(ELSEWHERE, **fields)], support_evidence=evidence)['summary']
        self.assertEqual(check()['span_check'], 'covered')
        self.assertEqual(check(start=120, end=300)['span_check'], 'covered')
        for start, end in ((151, 320), (120, 299), (514, 854)):
            self.assertEqual(check(start=start, end=end)['fail_reasons'], ['support_span_missing'])
        self.assertEqual(check(start=None, end=None, truncated=False)['span_check'], 'covered')
        self.assertIsNone(check()['off_answer_terms_found'])
        both = {'note:support': {'required_terms': TERMS, 'answer_span': [150, 300]}}
        self.assertEqual(self.graded(support_evidence=both)['summary']['span_check'], 'covered')
        self.assertEqual(self.graded(on=[rendered(start=514, end=854)], support_evidence=both)['summary']['fail_reasons'],
                         ['support_span_missing'])

    def test_missing_span_fields_fail_as_missing_fields(self):
        span = {'note:support': {'answer_span': [150, 300]}}
        for fields in ({'start': None}, {'end': None}, {'truncated': None}, {'start': '120'}, {'start': True}):
            summary = self.graded(on=[rendered(**fields)], support_evidence=span)['summary']
            self.assertEqual((summary['pass'], summary['fail_reasons']), ('fail', ['missing_fields']), fields)
        summary = self.graded(on=[rendered(snippet=None)])['summary']
        self.assertEqual((summary['pass'], summary['fail_reasons']), ('fail', ['missing_fields']))

    def test_rounds_that_disagree_fail_and_show_each_round(self):
        summary = self.graded(on=[rendered(), rendered(ELSEWHERE, 514, 854)])['summary']
        self.assertEqual((summary['pass'], summary['fail_reasons']), ('fail', ['support_span_missing']))
        self.assertEqual([r['covered'] for r in summary['on_span_by_round']], [1, 0])
        self.assertEqual([r['terms_found'] for r in summary['on_span_by_round']], [2, 1])
        self.assertFalse(summary['stable_on'])

    def test_id_level_reasons_keep_priority_and_skip_the_span_check(self):
        no_path = dict(rendered(ELSEWHERE), graph=None)
        summary = self.graded(on=[no_path])['summary']
        self.assertEqual(summary['fail_reasons'], ['invalid_path'])
        self.assertEqual(summary['span_check'], 'not_delivered')
        self.assertEqual(summary['on_span_by_round'][0]['checked'], 0)

    def test_suites_without_evidence_keep_their_verdicts_and_report_unchecked(self):
        plain = self.report(Q, span_client(on=[rendered(ELSEWHERE, 514, 854)]))['summary']
        self.assertEqual((plain['pass'], plain['fail_reasons']), ('needs_judgement', []))
        self.assertEqual((plain['span_check'], plain['span_unchecked_chunks']), ('unchecked', 1))
        self.assertIsNone(plain['off_answer_terms_found'])
        self.assertEqual(plain['on_span_by_round'], [{'checked': 0, 'covered': 0, 'terms_found': 0, 'terms_required': 0}] * 2)
        # Evidence for one of two support chunks: the other is counted, not failed.
        two = dict(Q, support_chunks={'support.md': ['note:support', 'note:other']}, support_evidence=EVIDENCE)
        summary = self.report(two, span_client())['summary']
        self.assertEqual(summary['span_unchecked_chunks'], 1)
        self.assertNotIn('support_span_missing', summary['fail_reasons'])

    def test_off_answer_terms_flag_is_recorded_without_changing_the_verdict(self):
        # The V2-Q4c shape: another chunk already states the answer with the graph off.
        summary = self.graded(off=['Seed premise only.', 'Launch window: Tuesday 09:00 sharp.'])['summary']
        self.assertIs(summary['off_answer_terms_found'], True)
        self.assertEqual((summary['pass'], summary['fail_reasons']), ('needs_judgement', []))
        # Terms spread over two chunks also set it, which is why it is only a flag.
        summary = self.graded(off=['The launch window moved.', 'Standup is Tuesday 09:00.'])['summary']
        self.assertIs(summary['off_answer_terms_found'], True)
        self.assertIs(self.graded(off=['The launch window moved.'])['summary']['off_answer_terms_found'], False)

    def test_support_evidence_is_validated(self):
        ok = [{'note:support': {'answer_span': [0, 1]}}, {'note:support': {'required_terms': ['a'], 'answer_span': [3, 9]}}, {}]
        bad = [[], {'note:other': {'required_terms': ['a']}}, {'note:support': {}}, {'note:support': ['a']},
               {'note:support': {'terms': ['a']}}, {'note:support': {'required_terms': []}},
               {'note:support': {'required_terms': 'a'}}, {'note:support': {'required_terms': ['a', ' ']}},
               {'note:support': {'required_terms': ['Dup', 'dup ']}}, {'note:support': {'required_terms': [1]}},
               {'note:support': {'answer_span': [5, 5]}}, {'note:support': {'answer_span': [9, 3]}},
               {'note:support': {'answer_span': [-1, 3]}}, {'note:support': {'answer_span': [1, 2, 3]}},
               {'note:support': {'answer_span': [1.0, 3]}}, {'note:support': {'answer_span': [True, 3]}},
               {'note:support': {'answer_span': None}}, {'note:support': {'required_terms': None}}]
        for evidence in ok:
            eval_graph.load_suite({'questions': [dict(Q, support_evidence=evidence)]})
        for evidence in bad:
            with self.assertRaises(eval_graph.EvaluationError, msg=str(evidence)):
                eval_graph.load_suite({'questions': [dict(Q, support_evidence=evidence)]})

    def test_terms_snippets_and_questions_leak_nowhere_but_the_private_directory(self):
        question = dict(Q, question='SENTINEL-QUESTION text?', support_evidence={
            'note:support': {'required_terms': ['SENTINEL-TERM-FOUND', 'SENTINEL-TERM-ABSENT']}})
        client = type('Client', (SpanClient,), {
            'on': [rendered('SENTINEL-SNIPPET with sentinel-term-found only.')], 'off': ['SENTINEL-SNIPPET off.']})
        private = self.root / 'private'
        stdout, stderr = io.StringIO(), io.StringIO()
        argv = self.argv('--private-dir', str(private), '--json', '--summary', question=question)
        with patch.dict(os.environ, {'TEST_GRAPH_TOKEN': 'SECRET'}, clear=True), \
                contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            self.assertEqual(eval_graph.main(argv, client), 0)
        public = (self.root / 'report.json').read_text() + stdout.getvalue() + stderr.getvalue()
        self.assertNotIn('SENTINEL', public.upper())
        self.assertIn('Q1: fail on==off=False graph_selected=2 span=missing reasons=support_span_missing\n',
                      stdout.getvalue())
        self.assertIn('SENTINEL-SNIPPET', (private / 'private-responses.json').read_text())
        self.assertTrue(all(name == 'build_context' for name, _ in FakeClient.calls))
        self.assertEqual(len(FakeClient.calls), 4)
        # A rejected suite names neither the field value nor the term.
        bad = dict(question, support_evidence={'note:support': {'required_terms': ['SENTINEL-TERM', '']}})
        (self.root / 'report.json').unlink()
        result, out, err = self.run_main(self.argv(question=bad), {'TEST_GRAPH_TOKEN': 'SECRET'})
        self.assertEqual(result, 2)
        self.assertNotIn('SENTINEL', (out + err).upper())

    def test_compare_treats_older_baselines_as_unchecked(self):
        current = {'questions': [self.graded(on=[rendered(ELSEWHERE, 514, 854)])]}
        schema2 = copy.deepcopy(current)
        for key in ('span_check', 'on_span_by_round', 'span_unchecked_chunks', 'off_answer_terms_found'):
            del schema2['questions'][0]['summary'][key]
        schema2['questions'][0]['summary']['pass'] = 'needs_judgement'
        for r in schema2['questions'][0]['rounds']:
            del r['returned_spans'], r['span_evidence']
        line = eval_graph.compare(current, schema2)
        self.assertIn('pass=needs_judgement -> fail', line)
        self.assertIn('span_check=unchecked -> missing', line)
        self.assertIn('span_check=unchecked -> unchecked',
                      eval_graph.compare({'questions': [self.report()]}, schema2))


if __name__ == '__main__':
    unittest.main()
