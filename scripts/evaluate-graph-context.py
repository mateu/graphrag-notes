#!/usr/bin/env python3
"""Read-only fixed-question build_context graph off/on evaluation (Python 3.9+)."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import time
import unicodedata
from urllib.parse import urlparse

# Explicit local import also works under python3 -I in an isolated allowlist copy.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import openclaw_memory_refresh as mcp


class EvaluationError(Exception):
    pass


# v2 adds per-round `returned_ids` (opaque chunk record IDs) so path terminals
# can be checked against the chunk that actually carried the graph evidence.
# v3 adds per-round `returned_spans` and `span_evidence`: build_context renders
# one clipped span per chunk, so a returned chunk ID does not show that the
# answer passage was rendered.
REPORT_SCHEMA_VERSION = 3
EVIDENCE_FIELDS = ('required_terms', 'answer_span')
# Traversal labels emitted by the service's graph path steps.
DIRECTIONS = ('outbound', 'inbound')
STEP_FIELDS = ('edge_id', 'edge_type', 'direction', 'from_id', 'to_id')


def require(condition, message):
    if not condition:
        raise EvaluationError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def private_directory(path):
    path = Path(path)
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = path.lstat()
    require(stat.S_ISDIR(info.st_mode) and info.st_uid == os.getuid() and
            stat.S_IMODE(info.st_mode) == 0o700, 'private directory must be owned and mode 0700')
    return path


def write_new(path, value):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def source_map(value):
    if 'files' not in value:
        require(all(isinstance(k, str) and isinstance(v, str) for k, v in value.items()),
                'invalid source map')
        return {k.removeprefix('source:'): v for k, v in value.items()}
    result = {}
    for path, record in value['files'].items():
        parts = record.get('parts') or {}
        for part in (parts.values() if isinstance(parts, dict) else parts):
            sid = part.get('id', '')
            if sid.startswith('source:'):
                result[sid.split(':', 1)[1]] = path
    return result


def load_suite(value):
    require(isinstance(value, dict) and isinstance(value.get('questions'), list) and value['questions'],
            'suite requires questions')
    seen = set()
    for q in value['questions']:
        require(isinstance(q.get('id'), str) and q['id'] and q['id'] not in seen, 'invalid question id')
        seen.add(q['id'])
        require(isinstance(q.get('question'), str) and q['question'].strip(), 'missing question')
        for key in ('seed', 'support', 'allowed_relations'):
            require(isinstance(q.get(key), list) and q[key] and
                    all(isinstance(x, str) and x for x in q[key]), 'missing question paths/relations')
        require(q.get('answer_differs_materially', 'unknown') in ('yes', 'no', 'unknown'),
                'invalid human judgement')
        # Optional per support path: the answer-bearing chunk IDs every graph-on round must deliver.
        # Suites without them still run, but such questions cannot pass.
        chunks = q.get('support_chunks', {})
        require(isinstance(chunks, dict) and set(chunks) <= set(q['support']) and
                all(isinstance(ids, list) and ids and all(isinstance(x, str) and x for x in ids) and
                    len(set(ids)) == len(ids) for ids in chunks.values()),
                'invalid support chunks')
        directions = q.get('allowed_directions', list(DIRECTIONS))
        require(isinstance(directions, list) and directions and all(d in DIRECTIONS for d in directions),
                'invalid allowed directions')
        # Optional per support chunk: what its rendered span must contain. Terms are private.
        evidence = q.get('support_evidence', {})
        known = {c for ids in chunks.values() for c in ids}
        require(isinstance(evidence, dict) and set(evidence) <= known, 'invalid support evidence')
        for spec in evidence.values():
            require(isinstance(spec, dict) and spec and set(spec) <= set(EVIDENCE_FIELDS),
                    'invalid support evidence')
            terms = spec.get('required_terms')
            if 'required_terms' in spec:
                require(isinstance(terms, list) and terms and
                        all(isinstance(t, str) and normalized(t) for t in terms) and
                        len({normalized(t) for t in terms}) == len(terms), 'invalid support evidence')
            span = spec.get('answer_span')
            if 'answer_span' in spec:
                require(isinstance(span, list) and len(span) == 2 and all(type(x) is int for x in span) and
                        0 <= span[0] < span[1], 'invalid support evidence')
    return value


def normalized(text):
    """NFC, case-folded and stripped, for exact substring checks only."""
    return unicodedata.normalize('NFC', text).casefold().strip()


def span_fields(chunk):
    """The numeric rendered-span offsets the service returned for one chunk; never its text."""
    def offset(key):
        return chunk[key] if type(chunk.get(key)) is int and chunk[key] >= 0 else None
    truncated = chunk.get('truncated')
    return {'start': offset('selected_span_start'), 'end': offset('selected_span_end'),
            'truncated': truncated if isinstance(truncated, bool) else None}


def span_status(chunk, spec):
    """Whether one rendered chunk carries its evidence, with found/required term counts."""
    statuses, found, terms = [], None, spec.get('required_terms')
    if terms:
        snippet = chunk.get('snippet')
        if isinstance(snippet, str):
            rendered = normalized(snippet)
            found = sum(normalized(t) in rendered for t in terms)
            statuses.append('covered' if found == len(terms) else 'missing')
        else:
            statuses.append('missing_fields')
    if 'answer_span' in spec:
        span, (start, end) = span_fields(chunk), spec['answer_span']
        if span['truncated'] is False:
            # An untruncated chunk renders its whole content, so it covers any range.
            statuses.append('covered')
        elif span['start'] is None or span['end'] is None or span['truncated'] is None:
            statuses.append('missing_fields')
        else:
            statuses.append('covered' if span['start'] <= start and end <= span['end'] else 'missing')
    status = next((s for s in ('missing_fields', 'missing') if s in statuses), 'covered')
    return status, found


def span_evidence(chunks, question):
    """Check suite evidence against the rendered response in memory; keep statuses and counts only."""
    result = {}
    context = normalized('\n'.join(ch['snippet'] for ch in chunks if isinstance(ch.get('snippet'), str)))
    for chunk_id, spec in (question.get('support_evidence') or {}).items():
        entry, terms = {'status': 'not_returned'}, spec.get('required_terms')
        if terms:
            entry['terms_required'] = len(terms)
            # Across every rendered chunk: shows an answer already present without the support chunk.
            entry['context_terms_found'] = sum(normalized(t) in context for t in terms)
        outcomes = [span_status(ch, spec) for ch in chunks if ch.get('id') == chunk_id]
        if outcomes:
            order = ('covered', 'missing', 'missing_fields')
            entry['status'], found = min(outcomes, key=lambda item: order.index(item[0]))
            if found is not None:
                entry['terms_found'] = found
        result[chunk_id] = entry
    return result


def graph_fields(graph):
    if not isinstance(graph, dict):
        return None
    # Never retain query_entities or arbitrary provenance/content fields.
    fields = {k: graph[k] for k in ('hops', 'seed_note_id', 'decay', 'score') if k in graph}
    fields['path'] = [{k: step[k] for k in
                       ('edge_id', 'edge_type', 'direction', 'confidence', 'from_id', 'to_id') if k in step}
                      for step in graph.get('path', [])]
    return fields


def round_result(data, policy, elapsed, mapping, question):
    chunks = data.get('chunks')
    require(isinstance(chunks, list), 'missing chunks')
    paths = [mapping.get((ch.get('source_uri') or '').rsplit('/', 1)[-1], 'unmapped') for ch in chunks]
    # Allowlist only numerical diagnostics and the known token counting label.
    diagnostics = {k: v for k, v in (data.get('diagnostics') or {}).items()
                   if type(v) in (int, float, bool) or
                   (k == 'token_count_mode' and v in ('estimated', 'exact'))}
    row = {'policy': policy, 'elapsed_ms': elapsed, 'error': None, 'returned_paths': paths,
           'returned_ids': [ch.get('id') if isinstance(ch.get('id'), str) and ch.get('id') else None
                            for ch in chunks],
           'scores': [ch.get('score') for ch in chunks],
           'chunk_graph': [graph_fields(ch.get('graph')) for ch in chunks],
           'returned_spans': [span_fields(ch) for ch in chunks],
           'span_evidence': span_evidence(chunks, question),
           'diagnostics': diagnostics, 'total_tokens': data.get('total_tokens'),
           'expected_found': {p: p in paths for p in question['seed'] + question['support']}}
    # Hash the sanitized response projection, excluding timing and policy.
    row['response_sha256'] = digest({k: v for k, v in row.items() if k not in ('elapsed_ms', 'policy')})
    return row


def accepted_path(graph, allowed, chunk_id, directions=DIRECTIONS):
    """True only for a contiguous accepted path from the graph seed to `chunk_id`.

    Every step needs all of STEP_FIELDS, an allowed type whose edge ID belongs to
    that relation table, a permitted direction, and must start where the previous
    step ended without revisiting a node.
    """
    if not isinstance(graph, dict) or not isinstance(chunk_id, str) or not chunk_id:
        return False
    hops, path, node = graph.get('hops'), graph.get('path'), graph.get('seed_note_id')
    if (type(hops) is not int or hops < 1 or not isinstance(path, list) or len(path) != hops or
            not isinstance(node, str) or not node):
        return False
    visited = {node}
    for step in path:
        if not isinstance(step, dict) or not all(isinstance(step.get(k), str) and step[k] for k in STEP_FIELDS):
            return False
        if (step['edge_type'] not in allowed or not step['edge_id'].startswith(step['edge_type'] + ':') or
                step['direction'] not in directions or step['from_id'] != node or step['to_id'] in visited):
            return False
        node = step['to_id']
        visited.add(node)
    return node == chunk_id


def support_status(row, path, chunk_id, question):
    """How one successful graph-on round returned one expected answer-bearing chunk."""
    ids, paths, graphs = row.get('returned_ids'), row.get('returned_paths'), row.get('chunk_graph')
    if not (isinstance(ids, list) and isinstance(paths, list) and isinstance(graphs, list) and
            len(ids) == len(paths) == len(graphs)):
        return 'missing_fields'
    hits = [i for i, value in enumerate(ids) if value == chunk_id]
    allowed = question.get('allowed_relations', [])
    directions = question.get('allowed_directions', DIRECTIONS)
    if any(paths[i] == path and accepted_path(graphs[i], allowed, chunk_id, directions) for i in hits):
        return 'delivered'
    if hits:
        return 'invalid_path' if any(paths[i] == path for i in hits) else 'support_source_mismatch'
    if any(p == path and not (isinstance(i, str) and i) for p, i in zip(paths, ids)):
        return 'missing_fields'
    return 'unrelated_chunk' if path in paths else 'support_missing'


def summarize(question, rounds):
    off = [r for r in rounds if r['policy'] == 'off']
    on = [r for r in rounds if r['policy'] == 'on']
    def found(rows, key):
        return all(any(r.get('expected_found', {}).get(p, False) for r in rows) for p in question[key])
    def paths(rows):
        return set(p for r in rows for p in r['returned_paths'])
    summary = {'off_finds_seed': found(off, 'seed'), 'on_finds_seed': found(on, 'seed'),
               'off_finds_support': found(off, 'support'), 'on_finds_support': found(on, 'support'),
               'on_same_paths_as_off': paths(off) == paths(on),
               'on_has_edge_path': any(g and g.get('hops', 0) >= 1 and g.get('path') for r in on for g in r['chunk_graph']),
               'on_graph_selected': sum(r['diagnostics'].get('graph_candidates_selected', 0) for r in on),
               'stable_off': len({r['response_sha256'] for r in off}) == 1,
               'stable_on': len({r['response_sha256'] for r in on}) == 1}
    expected = question.get('support_chunks') or {}
    targets = [(p, c) for p in question['support'] for c in expected.get(p, [])]
    unspecified = [p for p in question['support'] if not expected.get(p)]
    reasons = set()
    if any(r['error'] for r in rounds):
        reasons.add('call_error')
    if len(off) < 2 or len(on) < 2:
        reasons.add('insufficient_rounds')
    if unspecified or not targets:
        reasons.add('support_chunk_unspecified')
    if any(not isinstance(r.get('returned_ids'), list) for r in off):
        reasons.add('missing_fields')
    off_ids = {i for r in off for i in (r.get('returned_ids') or [])}
    # Every successful on repetition must deliver every expected chunk on its own.
    # A delivered chunk with suite evidence must also carry it in its rendered span.
    evidence = question.get('support_evidence') or {}
    delivered, span_rounds = [], []
    for r in on:
        by_target = {} if r['error'] else {(p, c): support_status(r, p, c, question) for p, c in targets}
        statuses = set(by_target.values())
        reasons.update(statuses - {'delivered'})
        delivered.append(not r['error'] and not unspecified and bool(targets) and statuses == {'delivered'})
        counts = {'checked': 0, 'covered': 0, 'terms_found': 0, 'terms_required': 0}
        for chunk_id in sorted({c for (_, c), status in by_target.items()
                                if status == 'delivered' and c in evidence}):
            entry = (r.get('span_evidence') or {}).get(chunk_id) or {}
            counts['checked'] += 1
            counts['covered'] += entry.get('status') == 'covered'
            counts['terms_found'] += entry.get('terms_found', 0)
            counts['terms_required'] += entry.get('terms_required', 0)
            if entry.get('status') != 'covered':
                reasons.add('support_span_missing' if entry.get('status') == 'missing' else 'missing_fields')
        span_rounds.append(counts)
    successful = [d for r, d in zip(on, delivered) if not r['error']]
    if any(successful) and not all(successful):
        reasons.add('inconsistent_support')
    # No support may already be present off, by file or by expected chunk.
    conditions = {'off_lacks_support': not any(p in paths(off) for p in question['support']) and
                  not any(c in off_ids for _, c in targets),
                  'on_returns_support_via_allowed_path': bool(on) and all(delivered)}
    if not conditions['off_lacks_support']:
        reasons.add('off_has_support')
    judgement = question.get('answer_differs_materially', 'unknown')
    if judgement == 'no':
        reasons.add('human_rejected')
    summary.update(conditions)
    summary['on_support_by_round'] = delivered
    checked = sum(c['checked'] for c in span_rounds)
    summary['on_span_by_round'] = span_rounds
    summary['span_unchecked_chunks'] = sum(c not in evidence for _, c in targets)
    # `unchecked`: no evidence in the suite. `not_delivered`: evidence, but no delivered chunk to check.
    summary['span_check'] = ('unchecked' if not any(c in evidence for _, c in targets)
                             else 'not_delivered' if not checked
                             else 'covered' if checked == sum(c['covered'] for c in span_rounds)
                             else 'missing')
    # A flag, not a fail reason: every required term already rendered in every graph-off round,
    # from any chunk. Terms spread over unrelated chunks can set it, so a person still decides.
    with_terms = sorted({c for _, c in targets if (evidence.get(c) or {}).get('required_terms')})
    off_ok = [r for r in off if not r['error']]
    def off_terms(row, chunk_id):
        entry = (row.get('span_evidence') or {}).get(chunk_id) or {}
        return entry.get('terms_required') and entry.get('context_terms_found') == entry['terms_required']
    summary['off_answer_terms_found'] = (bool(all(off_terms(r, c) for r in off_ok for c in with_terms))
                                         if with_terms and off_ok else None)
    summary['answer_differs_materially'] = judgement
    summary['fail_reasons'] = sorted(reasons)
    summary['pass'] = ('fail' if reasons or not all(conditions.values())
                       else 'needs_judgement' if judgement == 'unknown' else 'pass')
    # Normalize any(...) above to a boolean even when a path is an empty list.
    summary['on_has_edge_path'] = bool(summary['on_has_edge_path'])
    return summary


def evaluate(suite, mapping, client, params, policies):
    public, private = [], []
    for q in suite['questions']:
        rounds, raw = [], []
        for policy in policies:
            started = time.monotonic()
            try:
                data = client.call('build_context', dict(params, query=q['question'], graph=policy))
                row = round_result(data, policy, round((time.monotonic() - started) * 1000), mapping, q)
                raw.append({'policy': policy, 'response': data})
            except (mcp.ImportFailure, EvaluationError, ValueError, TypeError, KeyError):
                # Exception strings can echo queries/credentials; never export them.
                row = {'policy': policy, 'elapsed_ms': round((time.monotonic() - started) * 1000),
                       'error': 'context_failed', 'returned_paths': [], 'returned_ids': [], 'scores': [],
                       'chunk_graph': [], 'returned_spans': [], 'span_evidence': {},
                       'diagnostics': {}, 'expected_found': {},
                       'response_sha256': digest({'error': 'context_failed'})}
            rounds.append(row)
        public.append({'id': q['id'], 'seed': q['seed'], 'support': q['support'],
                       'support_chunks': q.get('support_chunks', {}),
                       # Counts and byte offsets only; the terms stay in the private suite.
                       'support_evidence': {c: {'required_terms': len(spec.get('required_terms', [])),
                                                'answer_span': spec.get('answer_span')}
                                            for c, spec in (q.get('support_evidence') or {}).items()},
                       'allowed_relations': q['allowed_relations'],
                       'allowed_directions': q.get('allowed_directions', list(DIRECTIONS)),
                       'coverage': {p: p in set(mapping.values()) for p in q['seed'] + q['support']},
                       'rounds': rounds, 'summary': summarize(q, rounds)})
        private.append({'id': q['id'], 'rounds': raw})
    return public, private


def compare(report, baseline):
    old = {q['id']: q for q in baseline['questions']}
    lines = []
    for q in report['questions']:
        before = old.get(q['id'])
        if before is None:
            lines.append(q['id'] + ': new question')
            continue
        changes = []
        for policy in ('off', 'on'):
            def paths(item):
                return {p for r in item['rounds'] if r['policy'] == policy for p in r['returned_paths']}
            gained, lost = sorted(paths(q) - paths(before)), sorted(paths(before) - paths(q))
            def elapsed(item):
                rs = [r['elapsed_ms'] for r in item['rounds'] if r['policy'] == policy]
                return sum(rs) / len(rs) if rs else 0
            changes.append('%s gained=%s lost=%s elapsed_delta_ms=%+.1f' %
                           (policy, json.dumps(gained), json.dumps(lost), elapsed(q) - elapsed(before)))
        flips = {k: [before['summary'][k], v] for k, v in q['summary'].items()
                 if isinstance(v, bool) and k in before['summary'] and before['summary'][k] != v}
        # Legacy baseline predates tri-state scoring; it has no accepted paths, hence fails.
        old_pass = before['summary'].get('pass', 'fail' if not before['summary'].get('on_has_edge_path') else 'needs_judgement')
        changes.append('flags_flipped=' + json.dumps(flips, sort_keys=True))
        changes.append('pass=%s -> %s' % (old_pass, q['summary']['pass']))
        # Schema 1 and 2 baselines did not check rendered spans.
        changes.append('span_check=%s -> %s' % (before['summary'].get('span_check', 'unchecked'),
                                                 q['summary'].get('span_check', 'unchecked')))
        lines.append(q['id'] + ': ' + '; '.join(changes))
    for qid in sorted(set(old) - {q['id'] for q in report['questions']}):
        lines.append(qid + ': removed question')
    return '\n'.join(lines)


def main(argv=None, client_factory=mcp.Client):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('suite', 'source-map', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--private-dir', type=Path)
    parser.add_argument('--endpoint', required=True)
    parser.add_argument('--credential-env', default='GRAPHRAG_TOKEN')
    parser.add_argument('--policies', default='off,on,on,off')
    parser.add_argument('--scope', choices=('notes', 'messages', 'all'), default='notes')
    for name, default, limit in [('max-chunks', 2, 200), ('max-total-tokens', 1000, 32768),
                                 ('max-chunk-tokens', 400, 8192), ('max-chunks-per-source', None, 200)]:
        parser.add_argument('--' + name, type=int, default=default)
    parser.add_argument('--compare', type=Path)
    parser.add_argument('--json', action='store_true', help='Print sanitized report JSON')
    parser.add_argument('--summary', action='store_true', help='Print summary (default)')
    for name in ('binary-sha256', 'source-revision', 'embedding-identity'):
        parser.add_argument('--' + name)
    for name in ('source-count', 'accepted-edge-count'):
        parser.add_argument('--' + name, type=int)
    args = parser.parse_args(argv)
    try:
        suite = load_suite(json.loads(args.suite.read_text()))
        mapping = source_map(json.loads(args.source_map.read_text()))
        policies = args.policies.split(',')
        require(set(policies) == {'off', 'on'} and all(policies.count(p) >= 2 for p in ('off', 'on')),
                'policies require at least two off and two on rounds')
        params = {'scope': args.scope}
        for key, limit in [('max_chunks', 200), ('max_total_tokens', 32768), ('max_chunk_tokens', 8192),
                           ('max_chunks_per_source', 200)]:
            value = getattr(args, key)
            require(value is None or 1 <= value <= limit, 'invalid budget')
            if value is not None:
                params[key] = value
        require(re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', args.credential_env) is not None,
                'invalid credential environment name')
        token = os.environ.get(args.credential_env)
        require(bool(token), 'credential environment variable missing')
        # Validate independently even with a test client; the shared client enforces the same rule.
        endpoint = urlparse(args.endpoint)
        require(endpoint.scheme == 'http' and endpoint.hostname in ('127.0.0.1', 'localhost', '::1') and
                not endpoint.username and not endpoint.password and endpoint.path == '/mcp' and
                not endpoint.query and not endpoint.fragment, 'use loopback/tunnel HTTP /mcp endpoint')
        baseline = json.loads(args.compare.read_text()) if args.compare else None
        private = private_directory(args.private_dir) if args.private_dir else None
        private_path = private / 'private-responses.json' if private else None
        require(private_path is None or not private_path.exists(), 'private output exists')
        require(private_path is None or args.output.resolve() != private_path.resolve(), 'outputs must differ')
        # Reserve public output before network calls; O_EXCL also rejects symlinks.
        fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            started = datetime.now(timezone.utc).isoformat()
            client = client_factory(args.endpoint, token, time.monotonic() + 3600)
            client.initialize()
            questions, raw = evaluate(suite, mapping, client, params, policies)
            metadata = {k: getattr(args, k) for k in ('binary_sha256', 'source_revision', 'embedding_identity',
                        'source_count', 'accepted_edge_count') if getattr(args, k) is not None}
            metadata.update(endpoint_host=endpoint.hostname, started_at=started,
                            completed_at=datetime.now(timezone.utc).isoformat(), source_map_size=len(mapping),
                            mcp_protocol=getattr(client, 'protocol', '2025-11-25'),
                            suite_sha256=hashlib.sha256(args.suite.read_bytes()).hexdigest(),
                            runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
            report = {'schema_version': REPORT_SCHEMA_VERSION, 'read_only': True, 'metadata': metadata,
                      'protocol': dict(params, policies=policies), 'questions': questions}
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write('\n')
        if private_path:
            write_new(private_path, {'questions': raw})
        if args.json:
            print(json.dumps(report, indent=2))
        if args.summary or not args.json:
            for q in questions:
                reasons = q['summary']['fail_reasons']
                span = q['summary']['span_check']
                print('%s: %s on==off=%s graph_selected=%d%s%s' %
                      (q['id'], q['summary']['pass'], q['summary']['on_same_paths_as_off'],
                       q['summary']['on_graph_selected'], '' if span == 'unchecked' else ' span=' + span,
                       ' reasons=' + ','.join(reasons) if reasons else ''))
            print('%d/%d pass' % (sum(q['summary']['pass'] == 'pass' for q in questions), len(questions)))
        if baseline:
            print(compare(report, baseline))
        return 1 if any(r['error'] for q in questions for r in q['rounds']) else 0
    except (EvaluationError, mcp.ImportFailure, OSError, ValueError, TypeError, KeyError):
        print('Evaluation failed; check input schema, permissions, credential environment, and connection.', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
