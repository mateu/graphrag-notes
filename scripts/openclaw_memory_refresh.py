#!/usr/bin/env python3
"""Source-aware indexed OpenClaw memory refresh; standard library, no local corpus.

Only memory_index_sources source=memory and approved Markdown originals are read.
A pinned SQLite inventory must agree with exact original bytes before admission.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import sqlite3
import stat
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request

MAX_UPLOAD = 65536
SAFE_CODES = {'busy', 'service_unavailable', 'service_unreachable', 'provider_unavailable', 'compatibility', 'validation', 'invalid_input', 'forbidden', 'unauthorized', 'not_found', 'revision_conflict', 'conflict', 'internal', 'cancelled', 'interrupted', 'worker_interrupted'}

class ImportFailure(Exception):
    def __init__(self, code, retryable=False):
        self.code, self.retryable = code, retryable
        super().__init__(code)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')


def sha(value):
    return hashlib.sha256(value).hexdigest()


def private_directory(path, create=False):
    if create:
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ImportFailure('unsafe_directory')


def read_private(path):
    private_directory(path.parent)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_uid != os.getuid() or before.st_mode & 0o077 or before.st_nlink != 1:
            raise ImportFailure('unsafe_file')
        value = stream.read()
        after = os.fstat(stream.fileno())
        if (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise ImportFailure('changed_file')
        return value


def split_utf8(raw, maximum):
    """Exact partition, preferring newline boundaries; never normalize or drop bytes."""
    raw.decode('utf-8', errors='strict')
    pieces, offset = [], 0
    while offset < len(raw):
        cut = min(offset + maximum, len(raw))
        if cut < len(raw):
            newline = raw.rfind(b'\n', offset + maximum // 2, cut)
            if newline >= 0:
                cut = newline + 1
            while cut > offset:
                try:
                    raw[offset:cut].decode('utf-8', errors='strict')
                    break
                except UnicodeDecodeError as error:
                    cut = offset + error.start
        if cut <= offset:
            raise ImportFailure('split_failed')
        piece = raw[offset:cut]
        if not piece.decode('utf-8').strip():
            if pieces and len(pieces[-1]) + len(piece) <= MAX_UPLOAD:
                pieces[-1] += piece
            else:
                raise ImportFailure('whitespace_only_piece')
        else:
            pieces.append(piece)
        offset = cut
    if not pieces or b''.join(pieces) != raw or any(len(p) > MAX_UPLOAD for p in pieces):
        raise ImportFailure('empty_or_invalid_source')
    return pieces


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


class Client:
    def __init__(self, server, token, deadline, timeout=15):
        parsed = urllib.parse.urlparse(server)
        if (parsed.scheme != 'http' or parsed.hostname not in ('127.0.0.1', 'localhost', '::1') or parsed.username or parsed.password or
                parsed.path != '/mcp' or parsed.query or parsed.fragment):
            raise ImportFailure('unsafe_server')
        self.server, self.deadline, self.timeout = server, deadline, timeout
        self.protocol, self.counter = '2025-11-25', 0
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
        self.token = token

    def remaining(self):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise ImportFailure('deadline')
        return remaining

    def request(self, method, params):
        self.counter += 1
        request = urllib.request.Request(self.server, data=canonical({'jsonrpc': '2.0', 'id': self.counter, 'method': method, 'params': params}),
                  headers={'Content-Type': 'application/json', 'Accept': 'application/json, text/event-stream',
                           'Authorization': 'Bearer ' + self.token, 'MCP-Protocol-Version': self.protocol})
        try:
            with self.opener.open(request, timeout=min(self.timeout, self.remaining())) as response:
                body = response.read(1048577)
            if len(body) > 1048576:
                raise ImportFailure('response_too_large')
            value = json.loads(body)
        except urllib.error.HTTPError as error:
            code = 'unauthorized' if error.code == 401 else 'forbidden' if error.code == 403 else 'http_error'
            retryable = error.code in (429, 500, 502, 503, 504)
            error.close()
            raise ImportFailure(code, retryable) from None
        except (urllib.error.URLError, TimeoutError, OSError):
            raise ImportFailure('transport', True) from None
        except (ValueError, UnicodeError):
            raise ImportFailure('malformed_response') from None
        if value.get('id') != self.counter or value.get('jsonrpc') != '2.0' or value.get('error') or 'result' not in value:
            raise ImportFailure('protocol_error')
        return value['result']

    def initialize(self):
        result = self.request('initialize', {'protocolVersion': self.protocol, 'capabilities': {},
                    'clientInfo': {'name': 'private-openclaw-memory-importer', 'version': '1'}})
        if result.get('protocolVersion') != self.protocol:
            raise ImportFailure('protocol_version')

    def call(self, name, arguments):
        result = self.request('tools/call', {'name': name, 'arguments': arguments})
        envelope = result.get('structuredContent', {})
        if envelope.get('schema_version') != 1:
            raise ImportFailure('incompatible_envelope')
        error = envelope.get('error')
        if error:
            code = error.get('code') if error.get('code') in SAFE_CODES else 'remote_error'
            raise ImportFailure(code, error.get('retryable') is True)
        if result.get('isError') or not isinstance(envelope.get('data'), dict):
            raise ImportFailure('invalid_result')
        return envelope['data']


def retry_call(client, name, arguments, attempts=3):
    for attempt in range(attempts):
        try:
            return client.call(name, arguments)
        except ImportFailure as error:
            if not error.retryable or attempt + 1 == attempts:
                raise
            time.sleep(min(2 ** attempt, client.remaining()))


def valid_id(value, kind):
    return isinstance(value, str) and re.fullmatch(kind + r':[0-9a-f]{64}', value) is not None


def check_admission(task, value):
    if (value.get('request_id') != task['payload']['request_id'] or not valid_id(value.get('job_id'), 'processing_job') or
            not valid_id(value.get('source_id'), 'source') or value.get('source_uri') != 'mcp://upload/' + value['source_id'][7:] or
            not isinstance(value.get('replayed'), bool)):
        raise ImportFailure('invalid_admission')


def check_job(entry, job, principal):
    if (job.get('id') != entry['admission']['job_id'] or job.get('source_id') != entry['admission']['source_id'] or
            job.get('instance_id') != principal or job.get('job_type') != 'remote_upload' or
            job.get('status') not in ('queued', 'running', 'completed', 'failed', 'cancelled', 'interrupted')):
        raise ImportFailure('job_identity_or_state')



def verified_policy(source, payload, expected=None):
    policy = {'extract_entities': source.get('extract_entities'),
              'processing_policy_sha256': source.get('processing_policy_sha256')}
    retained = (policy['extract_entities'] is True and source.get('processing_policy_current') is False
                and source.get('ingestion_policy_current') is True and source.get('extraction_policy_current') is False)
    if (policy['extract_entities'] is not payload['extract_entities']
            or source.get('ingestion_policy_current') is not True
            or (policy['extract_entities'] is False and source.get('extraction_policy_current') is not None)
            or (policy['extract_entities'] is True and source.get('extraction_policy_current') is not True and not retained)
            or (source.get('processing_policy_current') is not True
                and not (retained and expected and expected.get('retained_extraction_drift') is True))
            or not isinstance(policy['processing_policy_sha256'], str)
            or not re.fullmatch(r'[0-9a-f]{64}', policy['processing_policy_sha256'])
            or (expected is not None and any(policy[name] != expected[name] for name in policy))):
        raise ImportFailure('committed_processing_policy_mismatch')
    policy['retained_extraction_drift'] = retained
    return policy


def verify_completed(client, task, entry, job, principal):
    result = job.get('result') or {}
    generation = job.get('generation')
    notes = result.get('note_ids')
    if (job.get('phase') != 'completed' or job.get('error_code') is not None or job.get('failed') != 0 or
            not isinstance(generation, int) or isinstance(generation, bool) or generation < 1 or
            not isinstance(notes, list) or not notes or len(set(notes)) != len(notes) or
            not all(valid_id(n, 'note') for n in notes) or job.get('completed') != len(notes) or
            job.get('total') != len(notes) or result.get('generation') != generation or
            result.get('source_id') != entry['admission']['source_id'] or
            result.get('source_uri') != entry['admission']['source_uri'] or result.get('extracted') is not task['payload']['extract_entities']):
        raise ImportFailure('incomplete_job_result')
    source = retry_call(client, 'get_source', {'id': entry['admission']['source_id']})
    payload = task['payload']
    if (source.get('id') != entry['admission']['source_id'] or source.get('uri') != entry['admission']['source_uri'] or
            source.get('instance_id') != principal or source.get('document_key') != payload['document_key'] or
            source.get('content') != payload['content'] or source.get('title') != payload['title'] or
            source.get('provenance') != payload['provenance'] or source.get('generation') != generation or
            source.get('successful_generation') != generation or source.get('status') != 'ready'):
        raise ImportFailure('committed_source_mismatch')
    policy = verified_policy(source, payload, task.get('expected_policy'))
    entry['source_verified'] = {'id': source['id'], 'revision': source['revision'], 'generation': generation,
                                'content_sha256': sha(source['content'].encode('utf-8')), 'provenance_matches': True, 'processing_policy': policy}
    return policy


def now():
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def safe_name(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,63}', value):
        raise ImportFailure('invalid_collection_identity')
    return value


def memory_path(value):
    if (not isinstance(value, str) or not value or len(value.encode('utf-8')) > 1024
            or any(ord(ch) < 32 for ch in value) or '\\' in value
            or PurePosixPath(value).is_absolute() or str(PurePosixPath(value)) != value
            or '..' in PurePosixPath(value).parts
            or not (value in ('MEMORY.md', 'USER.md') or
                    (value.startswith('memory/') and value.endswith('.md')))):
        raise ImportFailure('source_outside_approved_scope')
    return value


def fingerprint(metadata):
    return (metadata.st_dev, metadata.st_ino, metadata.st_size,
            metadata.st_mtime_ns, metadata.st_ctime_ns)


def read_original(workspace, relative):
    """Refuse symlinks under the explicit workspace and detect mid-read edits."""
    directory = os.open(workspace.resolve(), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    descriptor = None
    try:
        pieces = PurePosixPath(memory_path(relative)).parts
        for part in pieces[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = child
        descriptor = os.open(pieces[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > 64 * 1024 * 1024:
            raise ImportFailure('invalid_original_file')
        with os.fdopen(descriptor, 'rb') as stream:
            descriptor = None
            raw = stream.read(64 * 1024 * 1024 + 1)
            after = os.fstat(stream.fileno())
        current = os.stat(pieces[-1], dir_fd=directory, follow_symlinks=False)
        if fingerprint(before) != fingerprint(after) or fingerprint(after) != fingerprint(current):
            raise ImportFailure('original_changed_during_snapshot')
        raw.decode('utf-8', errors='strict')
        if not raw.decode('utf-8').strip() or b'\0' in raw:
            raise ImportFailure('empty_or_invalid_source')
        return raw
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(directory)


def snapshot(database, workspace, after_inventory=None):
    """One read-only SQLite snapshot; files must exactly match its SHA256/size.

    SQLite and a filesystem cannot have a common atomic snapshot. Checking the
    index digest against each immutable captured byte string detects divergence
    rather than claiming that newer unindexed content belongs to this snapshot.
    """
    database = database.resolve(strict=True)
    before = database.stat()
    if not stat.S_ISREG(before.st_mode):
        raise ImportFailure('invalid_source_database')
    connection = sqlite3.connect(database.as_uri() + '?mode=ro', uri=True)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute('PRAGMA query_only=ON')
        connection.execute('BEGIN')
        columns = {row[1] for row in connection.execute('PRAGMA table_info(memory_index_sources)')}
        if not {'id', 'path', 'source', 'hash', 'mtime', 'size'} <= columns:
            raise ImportFailure('unsupported_memory_index_schema')
        rows = [dict(row) for row in connection.execute(
            "SELECT id,path,source,hash,mtime,size FROM memory_index_sources WHERE source='memory' ORDER BY path")]
        if len(rows) > 100000 or len({row['path'] for row in rows}) != len(rows):
            raise ImportFailure('invalid_index_inventory')
        for row in rows:
            memory_path(row['path'])
        if after_inventory:
            after_inventory(connection)
        documents, failures = [], []
        for row in rows:
            try:
                raw = read_original(workspace, row['path'])
                if sha(raw) != row['hash'] or len(raw) != row['size']:
                    raise ImportFailure('indexed_original_diverged')
                documents.append({'path': row['path'], 'raw': raw})
            except (OSError, UnicodeError, ImportFailure) as error:
                failures.append({'document_id': sha(row['path'].encode()),
                                 'error_code': error.code if isinstance(error, ImportFailure) else 'original_unreadable'})
        # The inventory is still pinned even if another connection committed.
        again = [dict(row) for row in connection.execute(
            "SELECT id,path,source,hash,mtime,size FROM memory_index_sources WHERE source='memory' ORDER BY path")]
        if again != rows or (database.stat().st_dev, database.stat().st_ino) != (before.st_dev, before.st_ino):
            raise ImportFailure('source_database_replaced')
        return documents, failures, len(rows)
    finally:
        connection.rollback()
        connection.close()


def payloads(documents, config, attempt, source_policies=None):
    tasks = {}
    for document in documents:
        path, raw = document['path'], document['raw']
        identity = (config['host'], config['agent'], path)
        base = 'openclaw-memory/' + config['host'] + '/' + config['agent'] + '/' + path
        if len(base) > 230 or len(base.encode('utf-8')) > 480:
            base = 'openclaw-memory/' + config['host'] + '/' + config['agent'] + '/path-sha256-' + sha(canonical(identity))
        pieces = split_utf8(raw, config['part_bytes']) if len(raw) > MAX_UPLOAD else [raw]
        filename = PurePosixPath(path).name
        title = ''.join(ch if ch.isalnum() or ch in ' ._()-' else '_' for ch in filename)[:240]
        uri = 'openclaw://' + config['host'] + '/' + config['agent'] + '/' + urllib.parse.quote(path, safe='/')
        if len(uri) > 2048:
            raise ImportFailure('provenance_uri_too_long')
        for index, piece in enumerate(pieces, 1):
            key = base + f'/part-{index:04d}'
            payload = {'document_key': key, 'content': piece.decode('utf-8'),
                       'title': title if len(pieces) == 1 else f'{title} [part {index}/{len(pieces)}]',
                       'extract_entities': (source_policies or {}).get(key, {}).get('extract_entities', config['extract_entities']),
                       'provenance': {'uri': uri,
                                      'label': 'OpenClaw indexed memory',
                                      'metadata': {'host': config['host'], 'agent': config['agent'], 'source_path': path,
                                                   'original_sha256': sha(raw), 'part': str(index), 'parts': str(len(pieces)),
                                                   'collection_id': config['collection_id']}}}
            digest = sha(canonical(payload))
            # A→B→A is a new update, not replay of the historical first A job.
            payload['request_id'] = 'ocmem-' + sha(canonical([attempt, digest]))
            tasks[key] = {'payload': payload, 'payload_hash': digest, 'entry': {}}
    return tasks


def atomic_json(path, value):
    private_directory(path.parent)
    if path.is_symlink():
        raise ImportFailure('unsafe_state_file')
    descriptor, temporary = tempfile.mkstemp(prefix='.refresh-', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(canonical(value) + b'\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)


def endpoint_hash(server):
    parsed = urllib.parse.urlsplit(server)
    host = parsed.hostname or ''
    if ':' in host:
        host = '[' + host + ']'
    if parsed.port is not None and not (parsed.scheme == 'http' and parsed.port == 80):
        host += ':' + str(parsed.port)
    canonical_url = urllib.parse.urlunsplit((parsed.scheme.lower(), host.lower(), '/mcp' if parsed.path in ('', '/') else parsed.path, '', ''))
    return sha(canonical_url.encode())


def evidence(state):
    config = state['config']
    pending = state.get('pending') or {}
    return {'schema_version': 1, 'collection_id': config['collection_id'],
            'endpoint_sha256': endpoint_hash(config['server']), 'instance_id': config['instance_id'],
            'retained_extraction_policy_parts': sum(state.get('source_policies', {}).get(k, {}).get('retained_extraction_drift') is True for k in state['documents']),
            'status': state['status'], 'last_attempt_at': state.get('last_attempt_at'),
            'last_success_at': state.get('last_success_at'), 'counts': state.get('counts', {}),
            'pending_parts': sum(not task.get('verified_this_attempt', False) for task in pending.get('tasks', {}).values()),
            'error_code': state.get('error_code'),
            'retry': {'action': 'resume' if pending else 'reconcile' if state.get('reconciliation') else 'refresh',
                      'plan_sha256': state.get('reconciliation', {}).get('plan_sha256')}}


def report(state):
    # Keep the freshness evidence contract unchanged. Operator output also
    # distinguishes collection membership from actual server generations.
    return dict(evidence(state), counts_scope='collection_parts', service_actions=state.get('service_actions', {}))


def new_state(config):
    return {'schema_version': 1, 'config': config, 'status': 'unknown', 'documents': {},
            'last_attempt_at': None, 'last_success_at': None,
            'counts': dict.fromkeys(('created', 'changed', 'unchanged', 'failed', 'missing'), 0)}


def prepare(state, documents, failures, attempt=None):
    if state.get('reconciliation'):
        raise ImportFailure('pending_reconciliation_requires_same_plan')
    if state.get('pending'):
        raise ImportFailure('pending_attempt_requires_resume')
    attempt = attempt or secrets.token_hex(16)
    tasks = payloads(documents, state['config'], attempt, state.get('source_policies'))
    registry = state['documents']
    counts = dict.fromkeys(('created', 'changed', 'unchanged', 'failed', 'missing'), 0)
    for key, task in tasks.items():
        previous = registry.get(key)
        action = 'created' if not previous else 'unchanged' if previous['payload_hash'] == task['payload_hash'] else 'changed'
        task['action'] = action
        task['verified_this_attempt'] = False
        counts[action] += 1
        if action == 'unchanged':
            task['entry'] = json.loads(json.dumps(previous['entry']))
    counts['failed'] = len(failures)
    # A failed read is not a removed document: block the whole snapshot rather
    # than create a destructive missing plan from an incomplete inventory.
    if failures:
        state.update(status='failed', counts=counts, last_attempt_at=now(), error_code='snapshot_incomplete')
        return None
    missing = sorted(set(registry) - set(tasks))
    counts['missing'] = len(missing)
    state.update(status='running', counts=counts, last_attempt_at=now(), error_code=None,
                 service_actions=dict.fromkeys(('created', 'updated', 'unchanged'), 0),
                 pending={'tasks': tasks, 'desired_keys': sorted(tasks), 'missing_keys': missing, 'attempt': attempt})
    return state['pending']


def source_matches(client, task, entry, principal, expected_policy=None, adopt_existing_policy=False):
    source = retry_call(client, 'get_source', {'id': entry['admission']['source_id']})
    payload = task['payload']
    if (source.get('id') != entry['admission']['source_id'] or source.get('instance_id') != principal
            or source.get('document_key') != payload['document_key'] or source.get('content') != payload['content']
            or source.get('title') != payload['title'] or source.get('provenance') != payload['provenance']
            or source.get('status') != 'ready' or source.get('generation') != source.get('successful_generation')
            or source.get('generation') != entry['source_verified']['generation']):
        raise ImportFailure('committed_source_mismatch')
    expected_policy = expected_policy or entry['source_verified'].get('processing_policy')
    if expected_policy is None and source['revision'] != entry['source_verified']['revision']:
        raise ImportFailure('registered_processing_policy_unknown')
    if (adopt_existing_policy and expected_policy is not None
            and source['revision'] == entry['source_verified']['revision']):
        # Explicit adoption can retain a previously verified graph snapshot
        # after only the owner's extraction policy has changed.
        expected_policy = dict(expected_policy, retained_extraction_drift=True)
    policy = verified_policy(source, payload, expected_policy)
    entry['source_verified']['processing_policy'] = policy
    entry['source_verified']['revision'] = source['revision']
    return policy


def verify_principal(client, principal):
    """Check authenticated caller before admission, not only source ownership.

    Read permission may expose another owner's sources. Source inspection alone
    therefore cannot prove that a new upload will use the registered principal.
    """
    try:
        report = retry_call(client, 'service_status', {})
    except ImportFailure as error:
        if error.code in ('invalid_input', 'validation', 'not_found'):
            raise ImportFailure('service_status_required') from None
        raise
    if (report.get('schema_version') != 1 or report.get('read_only') is not True
            or report.get('inference_probed') is not False):
        raise ImportFailure('incompatible_service_status')
    if report.get('instance_id') != principal:
        raise ImportFailure('authenticated_instance_mismatch')


def review_existing_policies(client, state, save, args):
    try:
        _review_existing_policies(client, state, save, args)
    except ImportFailure as error:
        key = state['pending'].get('reviewing_key')
        if key in state['pending']['tasks']:
            state['pending']['tasks'][key]['entry']['error_code'] = error.code
            state['counts']['failed'] = sum(bool(t['entry'].get('error_code')) for t in state['pending']['tasks'].values())
        state.update(status='failed', error_code=error.code)
        save()
        raise


def _review_existing_policies(client, state, save, args):
    """Inspect every prospective admission before the first upload.

    Legacy enriched parts require reviewed policy adoption. A server-side
    preserve guard makes metadata registration fail rather than replace a
    graph-bearing generation if source contents/policies change after review.
    """
    pending = state['pending']
    if pending.get('policy_review_complete'):
        return
    policies = state.setdefault('source_policies', {})
    principal = state['config']['instance_id']
    for key, task in pending['tasks'].items():
        if task.get('policy_reviewed') or task['entry'].get('admission') or task['action'] == 'unchanged':
            continue
        pending['reviewing_key'] = key
        try:
            source = retry_call(client, 'get_source', {'document_key': key})
        except ImportFailure as error:
            if error.code == 'not_found':
                if key in state['documents']:
                    raise ImportFailure('committed_source_mismatch') from None
                payload = task['payload']
                payload['create_only'] = True
                payload['request_id'] = 'ocmem-' + sha(canonical([pending['attempt'], task['payload_hash'], False, True]))
                task['policy_reviewed'] = True
                continue
            if error.code in ('invalid_input', 'validation'):
                raise ImportFailure('source_policy_service_required') from None
            raise
        payload = task['payload']
        retired = source.get('retired') is True
        retired_registration = state.get('retired_sources', {}).get(key)
        if retired and (not retired_registration or retired_registration['id'] != source.get('id')):
            raise ImportFailure('retired_source_requires_registered_cleanup')
        if (not valid_id(source.get('id'), 'source') or source.get('instance_id') != principal
                or source.get('document_key') != key or source.get('uri') != 'mcp://upload/' + source['id'][7:]
                or source.get('status') != 'ready'
                or (not retired and source.get('generation') != source.get('successful_generation'))
                or (retired and (source.get('successful_generation') != 0 or source.get('content_hash') is not None
                                  or not isinstance(source.get('generation'), int) or source['generation'] < 1))):
            raise ImportFailure('existing_source_scope_mismatch')
        if not isinstance(source.get('revision'), str) or not re.fullmatch(r'[0-9a-f]{64}', source['revision']):
            raise ImportFailure('existing_source_revision_invalid')
        payload['expected_source_revision'] = source['revision']
        policy_hash = source.get('processing_policy_sha256')
        retained_drift = (source.get('extract_entities') is True and source.get('processing_policy_current') is False
                          and source.get('ingestion_policy_current') is True and source.get('extraction_policy_current') is False)
        if (not isinstance(source.get('extract_entities'), bool)
                or source.get('ingestion_policy_current') is not True
                or (source.get('extract_entities') is False and source.get('extraction_policy_current') is not None)
                or (source.get('extract_entities') is True and source.get('extraction_policy_current') is not True and not retained_drift)
                or (source.get('processing_policy_current') is not True and not retained_drift)
                or not isinstance(policy_hash, str) or not re.fullmatch(r'[0-9a-f]{64}', policy_hash)):
            raise ImportFailure('existing_processing_policy_mismatch')
        prior_policy = policies.get(key)
        if prior_policy and any(prior_policy[name] != value for name, value in
                                {'extract_entities': source['extract_entities'], 'processing_policy_sha256': policy_hash}.items()):
            raise ImportFailure('registered_processing_policy_mismatch')
        if retained_drift:
            if retired or source.get('content') != payload['content'] or source.get('title') != payload['title']:
                raise ImportFailure('retained_extraction_policy_requires_owner_reprocessing')
            if not (getattr(args, 'adopt_existing_policy', False) or (prior_policy and prior_policy.get('retained_extraction_drift'))):
                raise ImportFailure('existing_extraction_policy_requires_adoption')
        # Existing sources must belong to this original source, regardless of
        # extraction policy. Compare edited registered parts with their last
        # verified provenance, not their newly indexed original hash.
        existing_provenance = json.loads(json.dumps(source.get('provenance')))
        expected_provenance = json.loads(json.dumps(state['documents'].get(key, retired_registration or {}).get('provenance', payload['provenance'])))
        if not isinstance(existing_provenance, dict) or not isinstance(existing_provenance.get('metadata'), dict):
            raise ImportFailure('existing_source_provenance_mismatch')
        existing_provenance['metadata'].pop('collection_id', None)
        expected_provenance['metadata'].pop('collection_id', None)
        if (key not in state['documents'] and not retired and source.get('content') != payload['content']
                and source['extract_entities'] == payload['extract_entities']):
            # An indexed original may have changed since a legacy import.
            # Its content hash/split count are version facts; host, agent,
            # URI, original path and part identity still must agree exactly.
            for provenance in (existing_provenance, expected_provenance):
                for version_key in ('original_sha256', 'parts'):
                    provenance['metadata'].pop(version_key, None)
        if existing_provenance != expected_provenance:
            raise ImportFailure('existing_source_provenance_mismatch')
        if key not in state['documents'] and source['extract_entities'] != payload['extract_entities']:
            if not getattr(args, 'adopt_existing_policy', False):
                raise ImportFailure('existing_extraction_policy_requires_adoption')
            if source.get('content') != payload['content'] or source.get('title') != payload['title']:
                raise ImportFailure('existing_adoption_source_mismatch')
            payload['extract_entities'] = source['extract_entities']
        policies[key] = {'extract_entities': source['extract_entities'], 'processing_policy_sha256': policy_hash,
                         'retained_extraction_drift': retained_drift}
        task['expected_policy'] = policies[key]
        # The guard belongs to this initial admission only; semantic content
        # identities exclude it so a later unchanged run requires no upload.
        if (key not in state['documents'] or retained_drift) and not retired and source.get('content') == payload['content'] and source.get('title') == payload['title']:
            payload['preserve_unchanged'] = True
        semantic = {k: v for k, v in payload.items() if k not in ('request_id', 'preserve_unchanged', 'create_only', 'expected_source_revision')}
        task['payload_hash'] = sha(canonical(semantic))
        payload['request_id'] = 'ocmem-' + sha(canonical([pending['attempt'], task['payload_hash'], bool(payload.get('preserve_unchanged')), bool(payload.get('create_only')), payload.get('expected_source_revision')]))
        task['policy_reviewed'] = True
    pending['policy_review_complete'] = True
    pending.pop('reviewing_key', None)
    save()


def run_refresh(client, state, save, args):
    tasks = state['pending']['tasks']
    principal = state['config']['instance_id']
    checked = set()
    waiting_since = {}
    while len(checked) < len(tasks):
        client.remaining()
        active = 0
        for key, task in tasks.items():
            if key in checked:
                continue
            entry = task['entry']
            try:
                if task['action'] == 'unchanged':
                    policy = source_matches(client, task, entry, principal, state.get('source_policies', {}).get(key), getattr(args, 'adopt_existing_policy', False))
                    state.setdefault('source_policies', {})[key] = policy
                    state['documents'][key]['entry'] = json.loads(json.dumps(entry))
                    task['verified_this_attempt'] = True
                    entry.pop('error_code', None)
                    checked.add(key)
                    # Repeating a read after interruption is safe. Save this
                    # evidence once at completion (or with a later failure),
                    # avoiding rewriting all pinned Markdown per unchanged part.
                    continue
                if not entry.get('admission'):
                    if active >= args.max_inflight:
                        continue
                    entry['admission'] = retry_call(client, 'upload_source', task['payload'])
                    check_admission(task, entry['admission'])
                    save()  # Exact payload was already pinned before admission.
                waiting_since.setdefault(key, time.monotonic())
                job = retry_call(client, 'get_job', {'id': entry['admission']['job_id']})
                check_job(entry, job, principal)
                entry['job'] = job
                save()
                if job['status'] == 'completed':
                    policy = verify_completed(client, task, entry, job, principal)
                    state.setdefault('source_policies', {})[key] = policy
                    action = job.get('result', {}).get('action')
                    if action in ('created', 'updated', 'unchanged'):
                        task['service_action'] = action
                    state['service_actions'] = {name: sum(t.get('service_action') == name for t in tasks.values())
                                                for name in ('created', 'updated', 'unchanged')}
                    state['documents'][key] = {'payload_hash': task['payload_hash'], 'entry': json.loads(json.dumps(entry)),
                                               'document_key': key, 'provenance': task['payload']['provenance']}
                    state.get('retired_sources', {}).pop(key, None)
                    checked.add(key)
                    task['verified_this_attempt'] = True
                    entry.pop('error_code', None)
                    save()
                elif job['status'] in ('failed', 'interrupted'):
                    recoverable = job['status'] == 'interrupted' or job.get('error_code') in (
                        'provider_unavailable', 'service_unreachable', 'internal', 'worker_interrupted', 'interrupted')
                    if not recoverable or not args.resume_jobs or entry.get('resume_attempts', 0) >= args.max_resumes:
                        raise ImportFailure('job_requires_explicit_resume' if recoverable else 'job_failed_nonretryable')
                    entry['resume_attempts'] = entry.get('resume_attempts', 0) + 1
                    save()
                    try:
                        retry = client.call('resume_job', {'id': entry['admission']['job_id']})
                        check_job(entry, retry, principal)
                    except ImportFailure as error:
                        if not error.retryable:
                            raise
                        retry = retry_call(client, 'get_job', {'id': entry['admission']['job_id']})
                        check_job(entry, retry, principal)
                        if retry['status'] not in ('queued', 'running', 'completed'):
                            raise ImportFailure('resume_outcome_uncertain')
                    waiting_since[key] = time.monotonic()
                    active += 1
                elif job['status'] == 'cancelled':
                    raise ImportFailure('job_cancelled')
                else:
                    if time.monotonic() - waiting_since[key] > args.job_timeout_seconds:
                        raise ImportFailure('job_deadline')
                    active += 1
            except ImportFailure as error:
                task['verified_this_attempt'] = False
                entry['error_code'] = error.code
                state['error_code'] = error.code
                state['counts']['failed'] = sum(bool(t['entry'].get('error_code')) for t in tasks.values())
                state['status'] = 'partial' if checked else 'failed'
                save()
                raise
        if len(checked) < len(tasks):
            time.sleep(min(args.poll_seconds, client.remaining()))
    state['counts']['failed'] = 0
    state['missing_keys'] = state['pending']['missing_keys']
    state['desired_keys'] = state['pending']['desired_keys']
    state.pop('pending')
    state.update(status='complete', last_success_at=now(), error_code=None)
    save()


def reconciliation_preview(client, state):
    if state.get('pending') or state['status'] != 'complete':
        raise ImportFailure('complete_refresh_required_before_reconciliation')
    targets = []
    for key in state.get('missing_keys', []):
        registered = state['documents'][key]
        source = retry_call(client, 'get_source', {'id': registered['entry']['admission']['source_id']})
        if (source.get('id') != registered['entry']['admission']['source_id']
                or source.get('uri') != registered['entry']['admission']['source_uri']
                or source.get('instance_id') != state['config']['instance_id']
                or source.get('document_key') != key or source.get('provenance') != registered['provenance']
                or source.get('provenance', {}).get('metadata', {}).get('collection_id') != state['config']['collection_id']
                or source.get('status') != 'ready' or source.get('generation') != source.get('successful_generation')
                or source.get('generation') != registered['entry']['source_verified']['generation']
                or sha(source.get('content', '').encode()) != registered['entry']['source_verified']['content_sha256']):
            raise ImportFailure('reconciliation_scope_mismatch')
        classification = 'obsolete_part' if any(desired.rsplit('/part-', 1)[0] == key.rsplit('/part-', 1)[0]
                                               for desired in state.get('desired_keys', [])) else 'removed_original'
        targets.append({'id': source['id'], 'revision': source['revision'], 'document_key': key,
                        'source_path': registered['provenance']['metadata']['source_path'], 'reason': classification,
                        'collection_id': state['config']['collection_id']})
    return {'schema_version': 1, 'collection_id': state['config']['collection_id'],
            'targets': targets, 'plan_sha256': sha(canonical(targets))}


def reconcile(client, state, plan_hash, save):
    plan = state.get('reconciliation')
    if plan is None:
        plan = reconciliation_preview(client, state)
        if plan_hash != plan['plan_sha256']:
            raise ImportFailure('reviewed_reconciliation_plan_changed')
        state['reconciliation'] = plan
        save()
    elif plan_hash != plan['plan_sha256']:
        raise ImportFailure('reviewed_reconciliation_plan_changed')
    for target in plan['targets']:
        if target.get('deleted'):
            continue
        request = {key: target[key] for key in ('id', 'revision', 'collection_id')}
        request.update(request_id='ocmem-delete-' + sha(canonical([plan_hash, request])), confirmed=True)
        result = retry_call(client, 'delete_uploaded_source', request)
        outcome = result.get('outcome', {})
        if (result.get('request_id') != request['request_id'] or not isinstance(result.get('replayed'), bool)
                or outcome.get('id') != request['id'] or outcome.get('operation') != 'delete_source'
                or outcome.get('previous_revision') != request['revision'] or outcome.get('status') != 'deleted'
                or outcome.get('actor') != 'mcp:' + state['config']['instance_id']):
            raise ImportFailure('invalid_deletion_receipt')
        target['deleted'] = True
        registered = state['documents'].pop(target['document_key'], None)
        if registered:
            state.setdefault('retired_sources', {})[target['document_key']] = {
                'id': target['id'], 'provenance': registered['provenance']}
        state['missing_keys'].remove(target['document_key'])
        save()
    state['counts']['missing'] = len(state['missing_keys'])
    state.pop('reconciliation', None)
    state.update(status='complete', error_code=None)
    save()


@contextmanager
def collection_lock(path):
    private_directory(path, create=True)
    descriptor = os.open(path / 'refresh.lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid() or metadata.st_mode & 0o077 or metadata.st_nlink != 1:
            raise ImportFailure('unsafe_lock')
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise ImportFailure('refresh_already_running') from None
        yield
    finally:
        os.close(descriptor)


def parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', type=Path, help='Indexed memory SQLite, never a conversation-summary database.')
    parser.add_argument('--workspace', type=Path, help='OpenClaw workspace containing indexed Markdown originals.')
    parser.add_argument('--state-dir', type=Path, required=True, help='Private registered collection state, retained for recovery.')
    parser.add_argument('--collection-id', default='clawd-main-memory')
    parser.add_argument('--source-host', default='clawd')
    parser.add_argument('--source-agent', default='main')
    parser.add_argument('--instance-id', default='shiva-importer', help='Expected authenticated importer principal; verified against jobs and sources.')
    parser.add_argument('--server', default='http://127.0.0.1:31057/mcp')
    parser.add_argument('--credential-env', default='GRAPHRAG_TOKEN')
    parser.add_argument('--part-bytes', type=int, default=49152)
    parser.add_argument('--extract-entities', action='store_true', help='Explicit opt-in; default performs embeddings only.')
    parser.add_argument('--adopt-existing-policy', action='store_true', help='Explicitly retain an existing compatible extraction policy only when exact owner/source/content matches; new parts keep collection default.')
    parser.add_argument('--dry-run', action='store_true', help='Snapshot and plan only; no credentials, network, or state writes.')
    parser.add_argument('--status', action='store_true', help='Read redacted local refresh evidence; no SQLite/network/provider use.')
    parser.add_argument('--resume', action='store_true', help='Reuse the exact saved snapshot/payloads without reading the source again.')
    parser.add_argument('--resume-jobs', action='store_true', help='Explicitly resume owned recoverable failed/interrupted jobs.')
    parser.add_argument('--max-resumes', type=int, default=1)
    parser.add_argument('--max-inflight', type=int, default=4)
    parser.add_argument('--poll-seconds', type=float, default=2)
    parser.add_argument('--deadline-seconds', type=int, default=14400)
    parser.add_argument('--job-timeout-seconds', type=int, default=1200)
    parser.add_argument('--reconcile', action='store_true', help='Preview missing originals/obsolete parts against server revisions.')
    parser.add_argument('--yes', action='store_true', help='Explicitly apply the reviewed --plan-sha256 cleanup.')
    parser.add_argument('--plan-sha256', help='Exact hash printed by --reconcile preview.')
    parser.add_argument('--format', choices=('human', 'json'), default='human')
    return parser


def emit(value, format):
    if format == 'json':
        print(json.dumps(value, sort_keys=True))
        return
    print('Memory refresh: ' + value['status'])
    if value.get('counts'):
        label = 'Collection parts: ' if value.get('counts_scope') == 'collection_parts' else ''
        print(label + ', '.join(f'{key}: {count}' for key, count in value['counts'].items()))
    if value.get('service_actions'):
        print('Server source generations: ' + ', '.join(f'{key}: {count}' for key, count in value['service_actions'].items()))
    if value.get('retained_extraction_policy_parts'):
        print('Retained older extraction-policy parts: ' + str(value['retained_extraction_policy_parts']) + '; graph policy needs explicit owner reprocessing. Collection status describes indexed original/vector evidence separately.')
    if value.get('last_success_at'):
        print('Last successful refresh: ' + value['last_success_at'])
    if value.get('plan_sha256'):
        print('Review cleanup count: ' + str(value['missing_parts']))
        print('Plan SHA256: ' + value['plan_sha256'])
    if value.get('error_code'):
        print('Reason: ' + value['error_code'])
    if value.get('retry', {}).get('action') == 'resume':
        print('Retry: rerun with the same --state-dir and --resume; add --resume-jobs only after repairing a failed provider.')
    elif value.get('retry', {}).get('action') == 'reconcile':
        print('Retry: rerun with the same --state-dir --reconcile --yes --plan-sha256 ' + value['retry']['plan_sha256'])


def main(argv=None):
    args = parser().parse_args(argv)
    state = None
    save = None
    started = time.monotonic()
    try:
        if (sum((args.status, args.resume, args.reconcile)) > 1 or (args.dry_run and (args.resume or args.reconcile))
                or args.yes != bool(args.plan_sha256) or (args.yes and not args.reconcile)
                or not 4096 <= args.part_bytes <= 57344 or not 1 <= args.max_inflight <= 8
                or not 0.1 <= args.poll_seconds <= 30 or not 1 <= args.max_resumes <= 5
                or not 1 <= args.deadline_seconds <= 86400 or not 1 <= args.job_timeout_seconds <= 86400
                or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', args.credential_env)):
            raise ImportFailure('invalid_arguments')
        config = {'collection_id': safe_name(args.collection_id), 'host': safe_name(args.source_host),
                  'agent': safe_name(args.source_agent), 'instance_id': safe_name(args.instance_id),
                  'server': args.server, 'part_bytes': args.part_bytes, 'extract_entities': args.extract_entities}
        # Validate URL before touching credentials or state. Only loopback over
        # a private tunnel is supported by this adapter, matching daily setup.
        Client(args.server, '', started + args.deadline_seconds)
        state_path = args.state_dir / 'collection.json'
        if state_path.exists() or state_path.is_symlink():
            state = json.loads(read_private(state_path))
            if state.get('schema_version') != 1 or not isinstance(state.get('config'), dict):
                raise ImportFailure('incompatible_collection_state')
            if not (args.status or args.resume or args.reconcile) and state['config'] != config:
                raise ImportFailure('registered_collection_mismatch')
        else:
            state = new_state(config)
        if args.status:
            emit(report(state), args.format)
            return 0
        if args.dry_run:
            if not args.database or not args.workspace:
                raise ImportFailure('database_and_workspace_required')
            documents, failures, count = snapshot(args.database, args.workspace)
            prepare(state, documents, failures, 'dry-run')
            value = evidence(state)
            value.update(status='dry_run_complete' if not failures else 'failed', indexed_documents=count,
                         network_calls=0, credentials_read=False)
            emit(value, args.format)
            return 0 if not failures else 1
        with collection_lock(args.state_dir):
            # Re-read under the collection lock to avoid a plan built from
            # another process's outdated registry or pending input.
            if state_path.exists():
                state = json.loads(read_private(state_path))
                if not (args.resume or args.reconcile) and state['config'] != config:
                    raise ImportFailure('registered_collection_mismatch')
            def persist():
                atomic_json(state_path, state)
                atomic_json(args.state_dir / 'freshness.json', evidence(state))
            save = persist
            if args.resume:
                if not state.get('pending'):
                    raise ImportFailure('no_pending_attempt')
                state.update(status='running', error_code=None)
            elif not args.reconcile:
                if not args.database or not args.workspace:
                    raise ImportFailure('database_and_workspace_required')
                state['last_attempt_at'] = now()
                save()
                documents, failures, _ = snapshot(args.database, args.workspace)
                if prepare(state, documents, failures) is None:
                    save()
                    emit(report(state), args.format)
                    return 1
            save()  # Persist exact inputs before any network/model request.
            token = os.environ.get(args.credential_env, '')
            if not token or any(ch.isspace() for ch in token):
                raise ImportFailure('credential_unavailable')
            client = Client(state['config']['server'], token, started + args.deadline_seconds)
            client.initialize()
            verify_principal(client, state['config']['instance_id'])
            if args.reconcile:
                if args.yes:
                    reconcile(client, state, args.plan_sha256, save)
                    emit(dict(evidence(state), status='reconciliation_complete'), args.format)
                else:
                    plan = state.get('reconciliation') or reconciliation_preview(client, state)
                    atomic_json(args.state_dir / 'cleanup-preview.json', plan)
                    emit({'status': 'reconciliation_preview', 'missing_parts': len(plan['targets']),
                          'removed_original_parts': sum(t['reason'] == 'removed_original' for t in plan['targets']),
                          'obsolete_parts': sum(t['reason'] == 'obsolete_part' for t in plan['targets']),
                          'plan_sha256': plan['plan_sha256']}, args.format)
            else:
                review_existing_policies(client, state, save, args)
                run_refresh(client, state, save, args)
                emit(report(state), args.format)
            return 0
    except (Exception, KeyboardInterrupt) as error:
        code = error.code if isinstance(error, ImportFailure) else 'paused' if isinstance(error, KeyboardInterrupt) else 'invalid_input_or_unexpected_failure'
        if state is not None and save:
            state['status'] = 'paused' if code == 'paused' else 'partial' if state.get('pending') and any(
                t.get('verified_this_attempt', False) for t in state['pending']['tasks'].values()) else 'failed'
            state['error_code'] = code
            try:
                save()
            except Exception:
                code = 'state_write_failed'
        value = evidence(state) if state else {'status': 'failed'}
        value['error_code'] = code
        emit(value, args.format)
        return 130 if code == 'paused' else 1


if __name__ == '__main__':
    raise SystemExit(main())
