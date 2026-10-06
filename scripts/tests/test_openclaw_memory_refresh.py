"""Fictional SQLite + MCP doubles, no real OpenClaw/provider/corpus access."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('memory_refresh', Path(__file__).parents[1] / 'openclaw_memory_refresh.py')
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


class FakeService:
    def __init__(self):
        self.receipts, self.jobs, self.sources, self.deletions = {}, {}, {}, {}
        self.calls = []
        self.embedding_calls = self.extraction_calls = 0
        self.lost_ack = self.lost_delete_ack = False
        self.fail_key = None
        self.fail_code = 'provider_unavailable'
        self.wrong_owner = self.bad_source = self.retain_retired_source = False
        self.extraction_policy_drift = False
        self.authenticated_instance = 'fixture-importer'
        self.manual_notes = ['synthetic unrelated manual note']

    def remaining(self):
        return 120

    def initialize(self):
        pass

    def call(self, name, value):
        self.calls.append((name, json.loads(json.dumps(value))))
        if name == 'service_status':
            return {'schema_version': 1, 'read_only': True, 'inference_probed': False,
                    'instance_id': self.authenticated_instance}
        if name == 'upload_source':
            request = value['request_id']
            if request in self.receipts:
                self.assert_payload = self.jobs[self.receipts[request]['job_id']]['payload']
                if self.assert_payload != value:
                    raise m.ImportFailure('revision_conflict')
                return dict(self.receipts[request], replayed=True)
            source_id = 'source:' + m.sha(value['document_key'].encode())
            job_id = 'processing_job:' + m.sha(request.encode())
            admission = {'request_id': request, 'source_id': source_id, 'source_uri': 'mcp://upload/' + source_id[7:],
                         'job_id': job_id, 'replayed': False}
            generation = self.sources.get(source_id, {}).get('generation', 0) + 1
            self.receipts[request] = admission
            self.jobs[job_id] = {'payload': json.loads(json.dumps(value)), 'admission': admission,
                                 'generation': generation, 'failed': value['document_key'] == self.fail_key}
            if self.lost_ack:
                self.lost_ack = False
                raise m.ImportFailure('transport', True)
            return admission
        if name in ('get_job', 'resume_job'):
            job = self.jobs[value['id']]
            payload, admission = job['payload'], job['admission']
            if name == 'resume_job':
                job['failed'] = False
            if not job['failed'] and not job.get('published'):
                old = self.sources.get(admission['source_id'], {})
                if (payload.get('create_only') and old) or (payload.get('expected_source_revision') and old.get('revision') != payload['expected_source_revision']):
                    job['failed'] = True
                    self.fail_code = 'conflict'
                    return self.call(name, value)
                job['action'] = 'unchanged' if old.get('content') == payload['content'] and old.get('extract_entities') == payload['extract_entities'] else 'updated' if old else 'created'
                if old.get('content') != payload['content']:
                    self.embedding_calls += 1
                    self.extraction_calls += payload['extract_entities']
                self.sources[admission['source_id']] = {
                    'id': admission['source_id'], 'uri': admission['source_uri'], 'instance_id': 'fixture-importer',
                    'document_key': payload['document_key'], 'content': payload['content'], 'title': payload['title'],
                    'provenance': payload['provenance'], 'status': 'ready', 'generation': job['generation'],
                    'successful_generation': job['generation'], 'revision': m.sha(m.canonical(payload)),
                    'extract_entities': payload['extract_entities'], 'processing_policy_sha256': 'a' * 64,
                    'processing_policy_current': True}
                job['published'] = True
            complete = not job['failed']
            return {'id': value['id'], 'source_id': admission['source_id'], 'instance_id': 'foreign' if self.wrong_owner else 'fixture-importer',
                    'job_type': 'remote_upload', 'status': 'completed' if complete else 'failed',
                    'phase': 'completed' if complete else 'preparing', 'generation': job['generation'],
                    'completed': int(complete), 'total': 1, 'failed': 0, 'error_code': None if complete else self.fail_code,
                    'result': {'source_id': admission['source_id'], 'source_uri': admission['source_uri'],
                               'generation': job['generation'], 'note_ids': ['note:' + m.sha(value['id'].encode())],
                               'extracted': payload['extract_entities'], 'action': job.get('action')} if complete else None}
        if name == 'get_source':
            if 'document_key' in value:
                value = {'id': 'source:' + m.sha(value['document_key'].encode())}
            if value['id'] not in self.sources:
                raise m.ImportFailure('not_found')
            result = json.loads(json.dumps(self.sources[value['id']]))
            result.setdefault('ingestion_policy_current', True)
            result.setdefault('extraction_policy_current', True if result['extract_entities'] else None)
            if self.extraction_policy_drift and result['extract_entities']:
                result.update(processing_policy_current=False, ingestion_policy_current=True,
                              extraction_policy_current=False)
            if self.bad_source:
                result['content'] = 'foreign bytes'
            return result
        if name == 'delete_uploaded_source':
            if value['request_id'] in self.deletions:
                return dict(self.deletions[value['request_id']], replayed=True)
            source = self.sources[value['id']]
            if source['revision'] != value['revision'] or source['provenance']['metadata']['collection_id'] != value['collection_id']:
                raise m.ImportFailure('revision_conflict')
            if self.retain_retired_source:
                source.update(retired=True, successful_generation=0, content_hash=None,
                              revision=m.sha(m.canonical([source['revision'], 'retired'])))
            else:
                del self.sources[value['id']]
            result = {'request_id': value['request_id'], 'replayed': False, 'outcome': {
                'id': value['id'], 'operation': 'delete_source', 'previous_revision': value['revision'],
                'status': 'deleted', 'actor': 'mcp:fixture-importer'}}
            self.deletions[value['request_id']] = result
            if self.lost_delete_ack:
                self.lost_delete_ack = False
                raise m.ImportFailure('transport', True)
            return result
        raise AssertionError(name)


class RefreshTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.root.chmod(0o700)
        self.workspace = self.root / 'workspace'
        (self.workspace / 'memory').mkdir(parents=True)
        self.database = self.root / 'indexed.sqlite'
        with contextlib.closing(sqlite3.connect(self.database)) as connection, connection:
            connection.execute('PRAGMA journal_mode=WAL')
            connection.execute('CREATE TABLE memory_index_sources (id INTEGER PRIMARY KEY,path TEXT,source TEXT,hash TEXT,mtime REAL,size INTEGER)')
            connection.execute('CREATE TABLE conversation_summaries (content TEXT)')
            connection.execute("INSERT INTO conversation_summaries VALUES ('DO NOT IMPORT conversation summary')")
        self.write('memory/atlas.md', b'# Atlas\n\nThe fictional launch is Monday.')
        self.config = {'collection_id': 'fixture-memory', 'host': 'clawd', 'agent': 'main',
                       'instance_id': 'fixture-importer', 'server': 'http://127.0.0.1:31057/mcp',
                       'part_bytes': 49152, 'extract_entities': False}
        self.state = m.new_state(self.config)
        self.client = FakeService()
        self.args = types.SimpleNamespace(max_inflight=2, poll_seconds=0, job_timeout_seconds=1200, resume_jobs=False, max_resumes=1)
        self.saved = []

    def tearDown(self):
        self.temp.cleanup()

    def save(self):
        self.saved.append(json.loads(json.dumps(self.state)))

    def write(self, path, raw, source='memory'):
        (self.workspace / path).write_bytes(raw)
        with contextlib.closing(sqlite3.connect(self.database)) as connection, connection:
            connection.execute('DELETE FROM memory_index_sources WHERE path=?', (path,))
            connection.execute('INSERT INTO memory_index_sources(path,source,hash,mtime,size) VALUES (?,?,?,?,?)',
                               (path, source, m.sha(raw), 1.0, len(raw)))

    def prepare(self):
        documents, failures, _ = m.snapshot(self.database, self.workspace)
        return m.prepare(self.state, documents, failures)

    def run_refresh(self):
        with patch.object(m.time, 'sleep', lambda _: None):
            m.review_existing_policies(self.client, self.state, self.save, self.args)
            m.run_refresh(self.client, self.state, self.save, self.args)

    def test_reviewed_legacy_extraction_policy_is_retained_and_changed_upload_uses_it(self):
        self.config['extract_entities'] = True
        self.state = m.new_state(self.config)
        self.prepare(); self.run_refresh()
        source = next(iter(self.client.sources.values()))
        source['provenance']['metadata'].pop('collection_id')
        self.config['extract_entities'] = False
        self.state = m.new_state(self.config)
        self.prepare()
        before_uploads = len(self.client.receipts)
        with self.assertRaisesRegex(m.ImportFailure, 'existing_extraction_policy_requires_adoption'):
            self.run_refresh()
        self.assertEqual(len(self.client.receipts), before_uploads)
        self.assertEqual(self.state['counts']['failed'], 1)
        self.args.adopt_existing_policy = True
        self.run_refresh()
        registered = next(iter(self.state['documents'].values()))
        self.assertTrue(self.state['source_policies'][source['document_key']]['extract_entities'])
        self.assertTrue(registered['entry']['admission'])
        adopted_payload = self.client.jobs[registered['entry']['admission']['job_id']]['payload']
        self.assertTrue(adopted_payload['extract_entities'])
        self.assertTrue(adopted_payload['preserve_unchanged'])
        uploads = len(self.client.receipts)
        self.prepare(); self.run_refresh()
        self.assertEqual(self.state['counts']['unchanged'], 1)
        self.assertEqual(len(self.client.receipts), uploads)
        self.write('memory/atlas.md', b'# Atlas\n\nReviewed edited launch')
        self.prepare(); self.run_refresh()
        latest = list(self.client.jobs.values())[-1]['payload']
        self.assertTrue(latest['extract_entities'])
        self.assertNotIn('preserve_unchanged', latest)

    def test_reviewed_older_graph_policy_keeps_exact_evidence_and_blocks_changed_content(self):
        self.config['extract_entities'] = True
        self.state = m.new_state(self.config)
        self.prepare(); self.run_refresh()
        source = next(iter(self.client.sources.values()))
        source['provenance']['metadata'].pop('collection_id')
        self.client.extraction_policy_drift = True
        self.config['extract_entities'] = False
        self.state = m.new_state(self.config)
        self.prepare()
        before = (self.client.embedding_calls, self.client.extraction_calls)
        uploads = len(self.client.receipts)
        with self.assertRaisesRegex(m.ImportFailure, 'existing_extraction_policy_requires_adoption'):
            self.run_refresh()
        self.assertEqual(len(self.client.receipts), uploads)
        self.args.adopt_existing_policy = True
        self.run_refresh()
        self.assertEqual(m.evidence(self.state)['retained_extraction_policy_parts'], 1)
        self.assertEqual(self.state['status'], 'complete')
        self.assertEqual((self.client.embedding_calls, self.client.extraction_calls), before)

        uploads = len(self.client.receipts)
        self.prepare(); self.run_refresh()
        self.assertEqual(len(self.client.receipts), uploads)
        self.assertEqual(self.state['counts']['unchanged'], 1)
        self.assertEqual(m.evidence(self.state)['retained_extraction_policy_parts'], 1)
        self.write('memory/atlas.md', b'# Atlas\n\nNew indexed graph-bearing body')
        self.prepare()
        with self.assertRaisesRegex(m.ImportFailure, 'retained_extraction_policy_requires_owner_reprocessing'):
            self.run_refresh()
        self.assertEqual(len(self.client.receipts), uploads)
        self.assertEqual((self.client.embedding_calls, self.client.extraction_calls), before)

    def test_registered_graph_drift_requires_explicit_adoption_and_zero_uploads(self):
        self.config['extract_entities'] = True
        self.state = m.new_state(self.config)
        self.prepare(); self.run_refresh()
        self.client.extraction_policy_drift = True
        self.prepare()
        uploads = len(self.client.receipts)
        with self.assertRaisesRegex(m.ImportFailure, 'committed_processing_policy_mismatch'):
            self.run_refresh()
        self.args.adopt_existing_policy = True
        self.run_refresh()
        self.assertEqual(len(self.client.receipts), uploads)
        self.assertEqual(self.state['status'], 'complete')
        self.assertEqual(m.evidence(self.state)['retained_extraction_policy_parts'], 1)

    def test_installed_shim_help_creates_no_sibling_bytecode(self):
        scripts = self.root / 'installed' / 'scripts'
        scripts.mkdir(parents=True)
        for name in ('refresh-openclaw-memory.py', 'openclaw_memory_refresh.py'):
            (scripts / name).write_bytes((Path(__file__).parents[1] / name).read_bytes())
        environment = {k: v for k, v in os.environ.items() if k not in ('PYTHONDONTWRITEBYTECODE', 'PYTHONPYCACHEPREFIX')}
        completed = subprocess.run([sys.executable, str(scripts / 'refresh-openclaw-memory.py'), '--help'],
                                   capture_output=True, env=environment, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(list(scripts.rglob('__pycache__')), [])

    def test_legacy_adoption_rejects_changed_bytes_owner_and_processing_before_upload(self):
        self.prepare(); self.run_refresh()
        original = json.loads(json.dumps(next(iter(self.client.sources.values()))))
        original['extract_entities'] = True
        original['provenance']['metadata'].pop('collection_id')
        self.args.adopt_existing_policy = True
        for field, value, code in [('content', 'changed since review', 'existing_adoption_source_mismatch'),
                                   ('instance_id', 'foreign', 'existing_source_scope_mismatch'),
                                   ('processing_policy_current', False, 'existing_processing_policy_mismatch')]:
            self.state = m.new_state(self.config)
            source = json.loads(json.dumps(original)); source[field] = value
            self.client.sources[source['id']] = source
            self.prepare(); uploads = len(self.client.receipts)
            with self.assertRaisesRegex(m.ImportFailure, code):
                self.run_refresh()
            self.assertEqual(len(self.client.receipts), uploads)

    def test_missing_policy_lookup_is_create_only_and_fences_late_source(self):
        self.prepare()
        m.review_existing_policies(self.client, self.state, self.save, self.args)
        task = next(iter(self.state['pending']['tasks'].values()))
        self.assertTrue(task['payload']['create_only'])
        late = dict(task['payload'], request_id='older-pending-upload', extract_entities=True)
        late.pop('create_only')
        admission = self.client.call('upload_source', late)
        self.client.call('get_job', {'id': admission['job_id']})
        original = json.loads(json.dumps(self.client.sources[admission['source_id']]))
        with self.assertRaisesRegex(m.ImportFailure, 'job_failed_nonretryable'):
            m.run_refresh(self.client, self.state, self.save, self.args)
        self.assertEqual(self.client.sources[admission['source_id']], original)
        self.assertEqual(self.state['counts']['failed'], 1)

    def test_changed_source_revision_fences_concurrent_origin_and_policy_change(self):
        self.prepare(); self.run_refresh()
        source = next(iter(self.client.sources.values()))
        source['provenance']['metadata'].pop('collection_id')
        self.state = m.new_state(self.config)
        self.write('memory/atlas.md', b'# Atlas\n\nNew indexed version after legacy import.')
        self.prepare()
        m.review_existing_policies(self.client, self.state, self.save, self.args)
        task = next(iter(self.state['pending']['tasks'].values()))
        self.assertEqual(task['payload']['expected_source_revision'], source['revision'])
        late = json.loads(json.dumps(task['payload']))
        late.update(request_id='older-pending-edited', extract_entities=True)
        late.pop('expected_source_revision')
        late['provenance']['metadata']['host'] = 'foreign'
        admission = self.client.call('upload_source', late)
        self.client.call('get_job', {'id': admission['job_id']})
        foreign = json.loads(json.dumps(self.client.sources[admission['source_id']]))
        with self.assertRaisesRegex(m.ImportFailure, 'job_failed_nonretryable'):
            m.run_refresh(self.client, self.state, self.save, self.args)
        self.assertEqual(self.client.sources[admission['source_id']], foreign)
        self.assertEqual(self.state['counts']['failed'], 1)

    def test_reviewed_retained_retired_source_can_reappear_with_new_content(self):
        self.client.retain_retired_source = True
        self.prepare(); self.run_refresh()
        original = json.loads(json.dumps(next(iter(self.client.sources.values()))))
        (self.workspace / 'memory/atlas.md').unlink()
        with contextlib.closing(sqlite3.connect(self.database)) as connection, connection:
            connection.execute('DELETE FROM memory_index_sources')
        self.prepare(); self.run_refresh()
        plan = m.reconciliation_preview(self.client, self.state)
        m.reconcile(self.client, self.state, plan['plan_sha256'], self.save)
        retired = self.client.sources[original['id']]
        self.assertTrue(retired['retired'])
        self.assertNotEqual(retired['revision'], original['revision'])
        self.assertEqual(self.state['documents'], {})
        self.write('memory/atlas.md', b'# Atlas\n\nNew indexed body after reviewed cleanup.')
        self.prepare(); self.run_refresh()
        latest = list(self.client.jobs.values())[-1]['payload']
        self.assertNotIn('preserve_unchanged', latest)
        self.assertNotIn('create_only', latest)
        self.assertIn('expected_source_revision', latest)
        self.assertEqual(self.state['status'], 'complete')
        self.assertEqual(self.state['retired_sources'], {})
        self.assertGreater(self.client.sources[original['id']]['generation'], original['generation'])
        self.assertEqual(self.client.manual_notes, ['synthetic unrelated manual note'])

    def test_retired_residual_requires_this_collections_verified_cleanup(self):
        self.prepare(); self.run_refresh()
        source = next(iter(self.client.sources.values()))
        source.update(retired=True, successful_generation=0, content_hash=None)
        self.state = m.new_state(self.config)
        self.prepare(); before = len(self.client.receipts)
        with self.assertRaisesRegex(m.ImportFailure, 'retired_source_requires_registered_cleanup'):
            self.run_refresh()
        self.assertEqual(len(self.client.receipts), before)

    def test_unchanged_processing_drift_fails_without_upload_and_recovers(self):
        self.prepare(); self.run_refresh()
        original = json.loads(json.dumps(next(iter(self.client.sources.values()))))
        last_success = self.state['last_success_at']
        uploads = len(self.client.receipts)
        for field, value in [('processing_policy_current', False), ('processing_policy_sha256', 'b' * 64),
                             ('extract_entities', True)]:
            source = json.loads(json.dumps(original)); source[field] = value
            self.client.sources[source['id']] = source
            self.prepare()
            with self.assertRaisesRegex(m.ImportFailure, 'committed_processing_policy_mismatch'):
                self.run_refresh()
            self.assertEqual(self.state['counts']['failed'], 1)
            self.assertEqual(self.state['last_success_at'], last_success)
            self.assertEqual(len(self.client.receipts), uploads)
            self.client.sources[original['id']] = original
            self.run_refresh()
            self.assertEqual(self.state['status'], 'complete')
            last_success = self.state['last_success_at']
        self.assertEqual(len(self.client.receipts), uploads)

    def test_legacy_policy_evidence_requires_exact_previously_verified_revision(self):
        self.prepare(); self.run_refresh()
        self.state.pop('source_policies')
        for registered in self.state['documents'].values():
            registered['entry']['source_verified'].pop('processing_policy', None)
        self.prepare(); self.run_refresh()
        self.assertTrue(self.state['source_policies'])
        self.state.pop('source_policies')
        for registered in self.state['documents'].values():
            registered['entry']['source_verified'].pop('processing_policy', None)
        source = next(iter(self.client.sources.values())); source['revision'] = 'e' * 64
        self.prepare()
        with self.assertRaisesRegex(m.ImportFailure, 'registered_processing_policy_unknown'):
            self.run_refresh()

    def test_operator_counts_distinguish_collection_registration_and_service_actions(self):
        self.prepare(); self.run_refresh()
        self.assertEqual(m.report(self.state)['service_actions']['created'], 1)
        self.assertNotIn('service_actions', m.evidence(self.state))
        self.state = m.new_state(self.config)
        self.prepare(); self.run_refresh()
        self.assertEqual(self.state['counts']['created'], 1)
        self.assertEqual(m.report(self.state)['service_actions'], {'created': 0, 'updated': 0, 'unchanged': 1})
        self.prepare(); self.run_refresh()
        self.assertEqual(self.state['counts']['unchanged'], 1)
        self.assertEqual(sum(m.report(self.state)['service_actions'].values()), 0)

    def test_matching_policy_still_rejects_wrong_original_identity(self):
        self.prepare(); self.run_refresh()
        original = json.loads(json.dumps(next(iter(self.client.sources.values()))))
        for field in ('host', 'agent', 'source_path', 'part'):
            self.state = m.new_state(self.config)
            source = json.loads(json.dumps(original))
            source['provenance']['metadata'][field] = 'foreign'
            self.client.sources[source['id']] = source
            self.prepare(); uploads = len(self.client.receipts)
            with self.assertRaisesRegex(m.ImportFailure, 'existing_source_provenance_mismatch'):
                self.run_refresh()
            self.assertEqual(len(self.client.receipts), uploads)

    def test_legacy_default_policy_can_refresh_new_indexed_version_of_same_original(self):
        self.prepare(); self.run_refresh()
        source = next(iter(self.client.sources.values()))
        source['provenance']['metadata'].pop('collection_id')
        self.state = m.new_state(self.config)
        self.write('memory/atlas.md', b'# Atlas\n\nNewer indexed default-policy version.')
        self.prepare(); self.run_refresh()
        self.assertEqual(self.state['status'], 'complete')
        latest = list(self.client.jobs.values())[-1]['payload']
        self.assertNotIn('preserve_unchanged', latest)
        self.assertNotIn('create_only', latest)

    def test_snapshot_is_read_only_consistent_and_excludes_summaries(self):
        self.write('memory/session.md', b'Excluded indexed session', 'sessions')
        before = m.sha(self.database.read_bytes())
        def during(connection):
            with self.assertRaises(sqlite3.OperationalError):
                connection.execute('DELETE FROM memory_index_sources')
            with contextlib.closing(sqlite3.connect(self.database)) as writer, writer:
                writer.execute("INSERT INTO memory_index_sources(path,source,hash,mtime,size) VALUES ('memory/new.md','memory','unseen',1,1)")
        documents, failures, count = m.snapshot(self.database, self.workspace, during)
        self.assertEqual(count, 1)
        self.assertEqual(failures, [])
        self.assertEqual([d['path'] for d in documents], ['memory/atlas.md'])
        self.assertNotIn(b'conversation summary', b''.join(d['raw'] for d in documents))
        # With no external writer, reads change neither DB nor original bytes.
        before = m.sha(self.database.read_bytes())
        m.snapshot(self.database, self.workspace)
        self.assertEqual(m.sha(self.database.read_bytes()), before)

    def test_unindexed_edits_fail_without_upload_and_preserve_last_success(self):
        self.prepare(); self.run_refresh()
        last_success = self.state['last_success_at']
        (self.workspace / 'memory/atlas.md').write_bytes(b'Changed but not indexed')
        self.assertIsNone(self.prepare())
        self.assertEqual(self.state['status'], 'failed')
        self.assertEqual(self.state['counts']['failed'], 1)
        self.assertEqual(self.state['last_success_at'], last_success)
        self.assertEqual(len(self.client.sources), 1)

    def test_unchanged_refresh_has_no_duplicate_upload_or_inference(self):
        for index in range(10):
            self.write(f'memory/fiction-{index}.md', f'# Fiction {index}\nExact unchanged text.'.encode())
        self.prepare(); self.run_refresh()
        first = list(self.state['documents'])
        before_saves = len(self.saved)
        self.prepare(); self.run_refresh()
        # Read-only revalidation checkpoints at completion, rather than
        # repeatedly rewriting the full pinned corpus for every part.
        self.assertLessEqual(len(self.saved) - before_saves, 2)
        self.assertEqual(self.state['counts']['unchanged'], 11)
        self.assertEqual(len(self.client.receipts), 11)
        self.assertEqual(self.client.embedding_calls, 11)
        self.assertEqual(self.client.extraction_calls, 0)
        self.assertEqual(list(self.state['documents']), first)

    def test_failed_unchanged_revalidation_is_failed_pending_evidence(self):
        self.prepare(); self.run_refresh()
        last_success = self.state['last_success_at']
        prior = json.loads(json.dumps(self.state['documents']))
        self.prepare()
        self.client.bad_source = True
        with self.assertRaisesRegex(m.ImportFailure, 'committed_source_mismatch'):
            self.run_refresh()
        value = m.evidence(self.state)
        self.assertEqual(value['status'], 'failed')
        self.assertEqual(value['counts']['failed'], 1)
        self.assertEqual(value['pending_parts'], 1)
        self.assertEqual(value['last_success_at'], last_success)
        self.assertEqual(self.state['documents'], prior)
        self.client.bad_source = False
        self.run_refresh()
        self.assertEqual(m.evidence(self.state)['pending_parts'], 0)
        self.assertEqual(self.state['counts']['failed'], 0)
        self.assertEqual(len(self.client.receipts), 1)

    def test_content_aba_gets_new_request_with_stable_document_identity(self):
        for raw in (b'# Atlas\n\nA', b'# Atlas\n\nB', b'# Atlas\n\nA'):
            self.write('memory/atlas.md', raw)
            self.prepare(); self.run_refresh()
        self.assertEqual(len(self.client.receipts), 3)
        self.assertEqual(len(self.client.sources), 1)
        self.assertEqual(next(iter(self.client.sources.values()))['content'], '# Atlas\n\nA')

    def test_exact_unicode_parts_resize_and_explicit_cleanup_preserve_manual_notes(self):
        original = ('\ufeff# Synthetic\r\n' + 'Exact Unicode 😀 text\r\n' * 6000).encode()
        self.write('memory/atlas.md', original)
        self.prepare()
        tasks = self.state['pending']['tasks']
        self.assertEqual(b''.join(t['payload']['content'].encode() for t in tasks.values()), original)
        self.assertTrue(all(len(t['payload']['content'].encode()) <= 65536 for t in tasks.values()))
        first_key = next(iter(tasks))
        self.run_refresh()
        self.write('memory/atlas.md', b'# Atlas\n\nShort again')
        self.prepare(); self.run_refresh()
        self.assertIn(first_key, self.state['documents'])
        missing = self.state['counts']['missing']
        self.assertGreater(missing, 0)
        before = len(self.client.sources)
        preview = m.reconciliation_preview(self.client, self.state)
        self.assertEqual(len(self.client.sources), before)
        with self.assertRaisesRegex(m.ImportFailure, 'reviewed_reconciliation_plan_changed'):
            m.reconcile(self.client, self.state, '0' * 64, self.save)
        self.client.lost_delete_ack = True
        with patch.object(m.time, 'sleep', lambda _: None):
            m.reconcile(self.client, self.state, preview['plan_sha256'], self.save)
        self.assertEqual(len(self.client.sources), 1)
        self.assertEqual(len(self.client.deletions), missing)
        self.assertEqual(self.client.manual_notes, ['synthetic unrelated manual note'])

    def test_lost_admission_and_interrupted_upload_resume_same_exact_payload(self):
        self.prepare()
        m.review_existing_policies(self.client, self.state, self.save, self.args)
        pinned = json.loads(json.dumps(self.state['pending']))
        self.client.lost_ack = True
        self.run_refresh()
        uploads = [value for name, value in self.client.calls if name == 'upload_source']
        self.assertEqual(uploads[0], uploads[1])
        self.assertEqual(uploads[0], next(iter(pinned['tasks'].values()))['payload'])
        self.assertEqual(len(self.client.receipts), 1)
        # An interruption after durable admission can recover from saved state.
        self.write('memory/atlas.md', b'# Atlas\n\nNew version')
        self.prepare()
        self.client.fail_key = next(iter(self.state['pending']['tasks']))
        with self.assertRaisesRegex(m.ImportFailure, 'job_requires_explicit_resume'):
            self.run_refresh()
        self.assertEqual(next(iter(self.client.sources.values()))['content'], uploads[0]['content'])
        with self.assertRaisesRegex(m.ImportFailure, 'pending_attempt_requires_resume'):
            self.prepare()
        self.args.resume_jobs = True
        self.run_refresh()
        self.assertEqual(next(iter(self.client.sources.values()))['content'], '# Atlas\n\nNew version')
        self.assertEqual(len(self.client.receipts), 2)

    def test_failed_part_keeps_old_generation_and_requires_matching_owner(self):
        self.prepare(); self.run_refresh()
        self.write('memory/atlas.md', b'# Atlas\n\nEdited')
        self.prepare()
        self.client.wrong_owner = True
        with self.assertRaisesRegex(m.ImportFailure, 'job_identity_or_state'):
            self.run_refresh()
        self.assertNotEqual(self.state['status'], 'complete')
        self.client.wrong_owner = False
        self.client.bad_source = True
        with self.assertRaisesRegex(m.ImportFailure, 'committed_source_mismatch'):
            self.run_refresh()
        self.assertNotEqual(self.state['status'], 'complete')

    def test_removed_original_and_foreign_collection_preview_is_scoped(self):
        self.prepare(); self.run_refresh()
        with contextlib.closing(sqlite3.connect(self.database)) as connection, connection:
            connection.execute('DELETE FROM memory_index_sources')
        self.prepare(); self.run_refresh()
        self.assertEqual(self.state['counts']['missing'], 1)
        source = next(iter(self.client.sources.values()))
        source['provenance']['metadata']['collection_id'] = 'foreign'
        with self.assertRaisesRegex(m.ImportFailure, 'reconciliation_scope_mismatch'):
            m.reconciliation_preview(self.client, self.state)
        self.assertEqual(len(self.client.sources), 1)

    def test_dry_run_and_status_do_not_create_state_read_credentials_or_network(self):
        args = ['--database', str(self.database), '--workspace', str(self.workspace), '--state-dir', str(self.root / 'state'),
                '--instance-id', 'fixture-importer', '--dry-run', '--format', 'json']
        with patch.object(m.Client, 'initialize', side_effect=AssertionError('network')), \
                patch.dict(os.environ, {}, clear=True), contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(m.main(args), 0)
        value = json.loads(output.getvalue())
        self.assertEqual(value['network_calls'], 0)
        self.assertFalse(value['credentials_read'])
        self.assertFalse((self.root / 'state').exists())
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(m.main(['--state-dir', str(self.root / 'state'), '--status', '--format', 'json']), 0)
        self.assertEqual(json.loads(output.getvalue())['status'], 'unknown')

    def test_uncertain_cleanup_has_exact_reconciliation_retry_evidence(self):
        self.prepare(); self.run_refresh()
        with contextlib.closing(sqlite3.connect(self.database)) as connection, connection:
            connection.execute('DELETE FROM memory_index_sources')
        self.prepare(); self.run_refresh()
        preview = m.reconciliation_preview(self.client, self.state)
        original_call = self.client.call
        def fail_delete(name, arguments):
            if name == 'delete_uploaded_source':
                raise m.ImportFailure('transport')
            return original_call(name, arguments)
        with patch.object(self.client, 'call', side_effect=fail_delete):
            with self.assertRaises(m.ImportFailure):
                m.reconcile(self.client, self.state, preview['plan_sha256'], self.save)
        value = m.evidence(self.state)
        self.assertEqual(value['retry']['action'], 'reconcile')
        self.assertEqual(value['retry']['plan_sha256'], preview['plan_sha256'])
        with contextlib.redirect_stdout(io.StringIO()) as output:
            m.emit(value, 'human')
        self.assertIn('--reconcile --yes --plan-sha256 ' + preview['plan_sha256'], output.getvalue())
        with self.assertRaisesRegex(m.ImportFailure, 'pending_reconciliation_requires_same_plan'):
            self.prepare()
        m.reconcile(self.client, self.state, preview['plan_sha256'], self.save)
        self.assertEqual(self.state['counts']['missing'], 0)

    def test_symlink_original_and_partial_inventory_cannot_enable_cleanup(self):
        original = self.workspace / 'memory/atlas.md'
        original.unlink()
        original.symlink_to(self.database)
        self.assertIsNone(self.prepare())
        self.assertEqual(self.state['counts']['failed'], 1)
        with self.assertRaisesRegex(m.ImportFailure, 'complete_refresh_required'):
            m.reconciliation_preview(self.client, self.state)

    def test_persisted_failed_attempt_resumes_pinned_bytes_after_source_changes(self):
        state_dir = self.root / 'durable-state'
        args = ['--database', str(self.database), '--workspace', str(self.workspace), '--state-dir', str(state_dir),
                '--collection-id', 'fixture-memory', '--instance-id', 'fixture-importer', '--format', 'json']
        key = 'openclaw-memory/clawd/main/memory/atlas.md/part-0001'
        self.client.fail_key = key
        with patch.object(m, 'Client', return_value=self.client), \
                patch.dict(os.environ, {'GRAPHRAG_TOKEN': 'synthetic-private-token'}), \
                patch.object(m.time, 'sleep', lambda _: None), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(m.main(args), 1)
            pinned = json.loads(m.read_private(state_dir / 'collection.json'))
            expected = pinned['pending']['tasks'][key]['payload']['content']
            self.write('memory/atlas.md', b'Later indexed source, separate from pending draft')
            self.assertEqual(m.main(['--state-dir', str(state_dir), '--resume', '--resume-jobs', '--format', 'json']), 0)
        self.assertEqual(next(iter(self.client.sources.values()))['content'], expected)
        self.assertEqual(len(self.client.receipts), 1)
        stored = json.loads(m.read_private(state_dir / 'collection.json'))
        freshness = json.loads(m.read_private(state_dir / 'freshness.json'))
        self.assertNotIn('pending', stored)
        self.assertEqual(freshness['status'], 'complete')
        self.assertEqual(freshness['pending_parts'], 0)
        self.assertIsNotNone(freshness['last_success_at'])
        self.assertEqual((state_dir / 'collection.json').stat().st_mode & 0o777, 0o600)
        self.assertNotIn('content', json.dumps(freshness))

    def test_interrupted_snapshot_retains_failure_evidence_without_admission(self):
        state_dir = self.root / 'interrupted-state'
        args = ['--database', str(self.database), '--workspace', str(self.workspace), '--state-dir', str(state_dir), '--format', 'json']
        with patch.object(m, 'snapshot', side_effect=KeyboardInterrupt), \
                patch.object(m.Client, 'initialize', side_effect=AssertionError('network')), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(m.main(args), 130)
        freshness = json.loads(m.read_private(state_dir / 'freshness.json'))
        self.assertEqual(freshness['status'], 'paused')
        self.assertIsNotNone(freshness['last_attempt_at'])
        self.assertEqual(freshness['pending_parts'], 0)

    def test_state_and_lock_symlinks_cannot_read_or_overwrite_external_files(self):
        external = self.root / 'external.json'
        external.write_text('{"private":"unrelated"}')
        external.chmod(0o600)
        state_dir = self.root / 'unsafe-state'
        state_dir.mkdir(mode=0o700)
        (state_dir / 'collection.json').symlink_to(external)
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(m.main(['--state-dir', str(state_dir), '--status', '--format', 'json']), 1)
        self.assertNotIn('unrelated', output.getvalue())
        self.assertEqual(external.read_text(), '{"private":"unrelated"}')
        (state_dir / 'collection.json').unlink()
        (state_dir / 'refresh.lock').symlink_to(external)
        with self.assertRaises(OSError), m.collection_lock(state_dir):
            pass
        self.assertEqual(external.read_text(), '{"private":"unrelated"}')

    def test_wrong_authenticated_credential_fails_before_any_upload(self):
        self.client.authenticated_instance = 'foreign-reader'
        args = ['--database', str(self.database), '--workspace', str(self.workspace),
                '--state-dir', str(self.root / 'wrong-caller'), '--instance-id', 'fixture-importer', '--format', 'json']
        with patch.object(m, 'Client', return_value=self.client), \
                patch.dict(os.environ, {'GRAPHRAG_TOKEN': 'synthetic-private-token'}), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(m.main(args), 1)
        self.assertEqual(json.loads(output.getvalue())['error_code'], 'authenticated_instance_mismatch')
        self.assertEqual(self.client.receipts, {})
        self.assertFalse(any(name == 'upload_source' for name, _ in self.client.calls))

    def test_older_service_without_authenticated_status_fails_closed(self):
        with patch.object(self.client, 'call', side_effect=m.ImportFailure('invalid_input')):
            with self.assertRaisesRegex(m.ImportFailure, 'service_status_required'):
                m.verify_principal(self.client, 'fixture-importer')


if __name__ == '__main__':
    unittest.main()
