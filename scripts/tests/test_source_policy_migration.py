"""Private reviewed migration plans over fictional SQLite and MCP state."""
import copy
import contextlib
import io
import os
from unittest.mock import patch
import json
import unittest
import test_openclaw_memory_refresh as fixtures
m = fixtures.m

class MigrationService(fixtures.FakeService):
    target_policy = 'b' * 64
    def call(self, name, value):
        result = super().call(name, value)
        if name == 'get_source':
            for job in reversed(list(self.jobs.values())):
                if job['admission']['source_id'] == result['id'] and job.get('published'):
                    result['latest_upload_request_id'] = job['payload']['request_id']; break
            result['configured_processing_policy_sha256'] = self.target_policy
            current = not result['extract_entities'] or result['processing_policy_sha256'] == self.target_policy
            result.update(processing_policy_current=current, ingestion_policy_current=True,
                          extraction_policy_current=current if result['extract_entities'] else None)
        if name in ('get_job', 'resume_job') and result['status'] == 'completed':
            job = self.jobs[value['id']]
            intent = job['payload'].get('policy_migration')
            if job['payload']['extract_entities']:
                self.sources[job['admission']['source_id']]['processing_policy_sha256'] = self.target_policy
            if intent and not job.get('migration_published'):
                self.sources[job['admission']['source_id']]['processing_policy_sha256'] = intent['target_policy_sha256']
                job['action'] = result['result']['action'] = 'updated'
                self.extraction_calls += 1
                job['migration_published'] = True
        return result

class MigrationTests(unittest.TestCase):
    setUp = fixtures.RefreshTests.setUp
    tearDown = fixtures.RefreshTests.tearDown
    write = fixtures.RefreshTests.write
    prepare = fixtures.RefreshTests.prepare
    run_refresh = fixtures.RefreshTests.run_refresh
    save = fixtures.RefreshTests.save

    def seed(self, false_neighbor=False):
        self.config['extract_entities'] = True
        self.state = m.new_state(self.config)
        self.prepare(); self.run_refresh()
        if false_neighbor:
            self.write('memory/other.md', b'# Unselected\n\nDefault false source.')
            self.config['extract_entities'] = False
            self.prepare(); self.run_refresh()
        self.client = self.clone_client()
        for key, policy in self.state['source_policies'].items():
            policy['retained_extraction_drift'] = policy['extract_entities']
        self.key = next(key for key, policy in self.state['source_policies'].items() if policy['extract_entities'])
        return m.snapshot(self.database, self.workspace)[0]

    def clone_client(self):
        service = MigrationService()
        service.__dict__.update(self.client.__dict__)
        return service

    def test_explicit_subset_lost_ack_then_unchanged_and_later_edit(self):
        documents = self.seed(True)
        neighbor = next(key for key in self.state['documents'] if key != self.key)
        neighbor_entry = copy.deepcopy(self.state['documents'][neighbor])
        before = copy.deepcopy(self.state)
        plan = m.migration_preview(self.client, self.state, documents, [self.key])
        self.assertEqual(self.state, before)
        self.assertFalse(plan['targets'][0]['inspected']['content_changed'])
        prior_jobs = len(self.client.jobs); prior_embedding = self.client.embedding_calls
        m.migration_apply(self.client, self.state, documents, plan, plan['plan_sha256'], self.save)
        self.assertEqual(self.state['source_policies'][self.key]['processing_policy_sha256'], 'a' * 64)
        self.client.lost_ack = True
        m.run_refresh(self.client, self.state, self.save, self.args)
        self.assertEqual(len(self.client.jobs), prior_jobs + 1)
        self.assertEqual(self.client.embedding_calls, prior_embedding)
        self.assertFalse(self.state['source_policies'][self.key]['retained_extraction_drift'])
        self.assertEqual(self.state['source_policies'][self.key]['processing_policy_sha256'], 'b' * 64)
        self.assertEqual(self.state['documents'][neighbor], neighbor_entry)
        jobs = len(self.client.jobs); extraction = self.client.extraction_calls
        self.prepare(); self.run_refresh()
        self.assertEqual(len(self.client.jobs), jobs)
        self.assertEqual(self.client.extraction_calls, extraction)
        self.write('memory/atlas.md', b'# Atlas\n\nLater indexed edit after migration.')
        self.prepare(); self.run_refresh()
        self.assertEqual(len(self.client.jobs), jobs + 1)
        self.assertNotIn('policy_migration', list(self.client.jobs.values())[-1]['payload'])

    def test_whole_plan_preflight_refuses_input_owner_revision_target_and_registry_drift(self):
        documents = self.seed()
        plan = m.migration_preview(self.client, self.state, documents, [self.key])
        original = copy.deepcopy(self.state)
        for changed in ('input', 'owner', 'revision', 'target', 'registry'):
            with self.subTest(changed=changed):
                state = copy.deepcopy(original)
                sources = copy.deepcopy(self.client.sources)
                current = copy.deepcopy(documents)
                source = next(iter(self.client.sources.values()))
                if changed == 'input': current[0]['raw'] += b'\nchanged indexed input'
                if changed == 'owner': source['instance_id'] = 'foreign-owner'
                if changed == 'revision': source['revision'] = 'c' * 64
                if changed == 'target': self.client.target_policy = 'c' * 64
                if changed == 'registry': state['documents'][self.key]['payload_hash'] = 'd' * 64
                before_jobs = len(self.client.jobs)
                with self.assertRaises(m.ImportFailure):
                    m.migration_apply(self.client, state, current, plan, plan['plan_sha256'], lambda: None)
                self.assertEqual(state, original if changed != 'registry' else state)
                self.assertEqual(len(self.client.jobs), before_jobs)
                self.client.sources = sources
                self.client.target_policy = 'b' * 64

    def test_failure_checkpoint_retains_policy_and_exact_explicit_resume(self):
        documents = self.seed()
        plan = m.migration_preview(self.client, self.state, documents, [self.key])
        m.migration_apply(self.client, self.state, documents, plan, plan['plan_sha256'], self.save)
        self.client.fail_key = self.key
        with self.assertRaisesRegex(m.ImportFailure, 'job_requires_explicit_resume'):
            m.run_refresh(self.client, self.state, self.save, self.args)
        self.assertTrue(self.state['source_policies'][self.key]['retained_extraction_drift'])
        saved = copy.deepcopy(self.state)
        tasks = self.state['pending']['tasks']
        exact = copy.deepcopy(tasks[self.key]['payload'])
        jobs = len(self.client.jobs)
        self.client.target_policy = 'c' * 64
        with self.assertRaisesRegex(m.ImportFailure, 'migration_target_changed'):
            m.migration_resume_preflight(self.client, self.state)
        self.client.target_policy = 'b' * 64
        self.state = copy.deepcopy(saved)
        m.migration_resume_preflight(self.client, self.state)
        self.args.resume_jobs = True
        m.run_refresh(self.client, self.state, self.save, self.args)
        self.assertEqual(len(self.client.jobs), jobs)
        self.assertEqual(list(self.client.jobs.values())[-1]['payload'], exact)
        self.assertEqual(m.evidence(self.state)['retained_extraction_policy_parts'], 0)

    def test_lost_ack_pending_attempt_recovers_exact_receipt_without_ready_preflight(self):
        documents = self.seed()
        plan = m.migration_preview(self.client, self.state, documents, [self.key])
        m.migration_apply(self.client, self.state, documents, plan, plan['plan_sha256'], self.save)
        payload = self.state['pending']['tasks'][self.key]['payload']
        admitted = self.client.call('upload_source', payload)
        # Server prepared/committed while the client received no admission.
        self.client.call('get_job', {'id': admitted['job_id']})
        source = self.client.sources[admitted['source_id']]
        source.update(status='pending', successful_generation=source['generation'] - 1)
        self.assertFalse(self.state['pending']['tasks'][self.key]['entry'])
        jobs = len(self.client.jobs)
        m.migration_resume_preflight(self.client, self.state)
        source.update(status='ready', successful_generation=source['generation'])
        m.run_refresh(self.client, self.state, self.save, self.args)
        self.assertEqual(len(self.client.jobs), jobs)
        self.assertEqual(self.state['documents'][self.key]['entry']['admission']['job_id'], admitted['job_id'])
        self.assertEqual(m.evidence(self.state)['retained_extraction_policy_parts'], 0)

    def test_plan_and_pending_payload_tamper_are_refused(self):
        documents = self.seed()
        plan = m.migration_preview(self.client, self.state, documents, [self.key])
        changed = copy.deepcopy(plan); changed['targets'][0]['payload']['content'] = 'unreviewed'
        with self.assertRaisesRegex(m.ImportFailure, 'migration_plan_mismatch'):
            m.migration_apply(self.client, self.state, documents, changed, plan['plan_sha256'], self.save)
        m.migration_apply(self.client, self.state, documents, plan, plan['plan_sha256'], self.save)
        self.state['pending']['tasks'][self.key]['payload']['content'] = 'unreviewed'
        with self.assertRaisesRegex(m.ImportFailure, 'migration_plan_mismatch'):
            m.migration_resume_preflight(self.client, self.state)
        self.assertEqual(len(self.client.jobs), 1)

    def test_cli_preview_is_read_only_and_apply_requires_exact_private_digest(self):
        self.seed(True)
        directory = self.root / 'state'
        directory.mkdir(mode=0o700)
        m.atomic_json(directory / 'collection.json', self.state)
        m.atomic_json(directory / 'freshness.json', m.evidence(self.state))
        selection = self.root / 'selection.json'
        selection.write_text(json.dumps([self.key])); selection.chmod(0o600)
        command = ['--database', str(self.database), '--workspace', str(self.workspace), '--state-dir', str(directory),
                   '--collection-id', 'fixture-memory', '--source-host', 'clawd', '--source-agent', 'main',
                   '--instance-id', 'fixture-importer', '--server', self.config['server'], '--format', 'json', '--migrate-policy']
        before = [(directory / name).read_bytes() for name in ('collection.json', 'freshness.json')]
        jobs = len(self.client.jobs)
        with patch.object(m, 'Client', side_effect=lambda *args, **kwargs: self.client), patch.dict(os.environ, {'GRAPHRAG_TOKEN': 'fictional-private-token'}), contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(m.main(command + ['--selection-file', str(selection)]), 0)
        preview = json.loads(output.getvalue())
        self.assertEqual(preview['status'], 'policy_migration_preview')
        self.assertEqual(preview['provider_calls'], 0)
        self.assertEqual(before, [(directory / name).read_bytes() for name in ('collection.json', 'freshness.json')])
        self.assertEqual(len(self.client.jobs), jobs)
        self.assertNotIn(self.key, output.getvalue())
        self.assertEqual((directory / 'policy-migration-preview.json').stat().st_mode & 0o777, 0o600)
        with patch.object(m, 'Client', side_effect=lambda *args, **kwargs: self.client), patch.dict(os.environ, {'GRAPHRAG_TOKEN': 'fictional-private-token'}), contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(m.main(command + ['--yes', '--plan-sha256', 'f' * 64]), 1)
        self.assertEqual(len(self.client.jobs), jobs)
        with patch.object(m, 'Client', side_effect=lambda *args, **kwargs: self.client), patch.dict(os.environ, {'GRAPHRAG_TOKEN': 'fictional-private-token'}), contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(m.main(command + ['--yes', '--plan-sha256', preview['plan_sha256']]), 0)
        registered = json.loads((directory / 'collection.json').read_text())
        self.assertEqual(registered['status'], 'complete')
        self.assertEqual(registered['source_policies'][self.key]['processing_policy_sha256'], 'b' * 64)
        self.assertNotIn('fictional-private-token', (directory / 'collection.json').read_text())

    def test_selection_bound_default_false_and_missing_original(self):
        documents = self.seed(True)
        neighbor = next(key for key in self.state['documents'] if key != self.key)
        for selection in ([], [self.key] * 2, ['unknown'], [self.key + str(i) for i in range(9)], [neighbor]):
            with self.subTest(selection=selection), self.assertRaises(m.ImportFailure):
                m.migration_preview(self.client, self.state, documents, selection)
        changed = copy.deepcopy(documents)
        changed[0]['raw'] += b'\nIndexed combined edit.'
        plan = m.migration_preview(self.client, self.state, changed, [self.key])
        self.assertTrue(plan['targets'][0]['inspected']['content_changed'])

if __name__ == '__main__': unittest.main()
