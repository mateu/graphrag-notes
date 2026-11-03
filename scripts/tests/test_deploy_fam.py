"""Offline integration tests for the deploy state machine (no launchd or real DB)."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import plistlib
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location(
    'deploy_fam', Path(__file__).resolve().parents[1] / 'deploy_fam.py')
deploy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(deploy)
OLD = 'a' * 40
NEW = 'b' * 40
COUNTS = {'note': 3, 'embedding': 3}


class FakeProcess:
    def __init__(self, harness, argv):
        self.harness = harness
        self.db = Path(argv[argv.index('--db-path') + 1])
        self.harness.open_database(Path(argv[0]), self.db, 'rehearsal serve')
        self.harness.rehearsal_process = self
        self.alive = True
        self.terminated = False
        self.killed = False

    def poll(self):
        return None if self.alive else 0

    def terminate(self):
        self.terminated = True
        if not self.harness.rehearsal_stubborn:
            self.alive = False

    def wait(self, timeout=None):
        if self.alive:
            raise subprocess.TimeoutExpired('fake rehearsal', timeout)
        return 0

    def kill(self):
        self.killed = True
        self.alive = False


class Harness(deploy.Deploy):
    """Run real orchestration/filesystem operations against modeled CLI boundaries.

    Every old-binary database open is checked, not just final rollback state.
    A schema file stands in for the on-disk compatibility boundary.
    """
    def __init__(self, args):
        super().__init__(args)
        self.calls = []
        self.opens = []
        self.fail_once = None
        self.fail_always = set()
        self.migration_diff = True
        self.head = NEW
        self.object_type = 'commit'
        self.running = True
        self.pid = 987654321
        self.shutdown_error = False
        self.candidate_error = False
        self.rehearsal_process = None
        self.rehearsal_stubborn = False
        self.bad_embeddings = False
        self.count_drift = False
        self.stop_port_stuck = False

    def fault(self, name):
        self.calls.append(name)
        self.events.append({'command': name, 'exit': 0, 'seconds': 0})
        if name in self.fail_always or name == self.fail_once:
            self.fail_once = None
            raise deploy.Failure('injected ' + name)

    def open_database(self, binary, db, operation):
        flavor = binary.read_text()
        schema = db / 'schema'
        version = int(schema.read_text()) if schema.exists() else 1
        self.opens.append((flavor, db, version, operation))
        if flavor == 'old' and version != 1:
            raise AssertionError('OLD BINARY OPENED A MIGRATED DATABASE')
        if flavor == 'candidate':
            schema.write_text('2')
        return 2 if flavor == 'candidate' else 1

    def run(self, name, command, **kwargs):
        self.fault(name)
        argv = [str(x) for x in command]
        if name.startswith('git '):
            op = argv[3]
            if op == 'cat-file':
                return self.object_type
            if op == 'diff':
                return 'crates/db/src/migrations/v002_next.rs\n' if self.migration_diff else ''
            if op == 'worktree' and argv[4] == 'add':
                source = Path(argv[6])
                migrations = source / 'crates/db/src/migrations'
                migrations.mkdir(parents=True)
                (migrations / 'v002_next.rs').touch()
                return ''
            return ''
        if name == 'verify checkout HEAD':
            return self.head
        if name == 'release build':
            candidate = Path(kwargs['cwd']) / 'target/release/graphrag'
            candidate.parent.mkdir(parents=True)
            candidate.write_text('candidate')
            return ''
        if name == 'toolchain':
            return 'rustc fake\n'
        if name == 'launchctl print':
            deploy.require(self.running, 'fake service stopped')
            return f'pid = {self.pid}\n'
        if name == 'launchctl bootout':
            self.running = False
            if self.shutdown_error:
                with self.log.open('a') as log:
                    log.write('ERROR old shutdown\n')
            return ''
        if name == 'launchctl bootstrap':
            self.open_database(self.binary, self.db, name)
            self.running = True
            if self.candidate_error and self.binary.read_text() == 'candidate':
                with self.log.open('a') as log:
                    log.write('ERROR candidate startup\n')
            return ''
        if name in ('rollback restore', 'rehearsal restore'):
            db = Path(argv[argv.index('--db-path') + 1])
            if db.exists():
                raise AssertionError('restore must target a fresh database')
            db.mkdir()
            (db / 'schema').write_text('1')
            (db / 'payload').write_text('restored logical records')
            self.open_database(Path(argv[0]), db, name)
            return ''
        if name in ('backup verify', 'backup create', 'schema-version'):
            binary = Path(argv[0])
            db = Path(argv[argv.index('--db-path') + 1])
            if name == 'backup verify':
                archive = Path(argv[argv.index('verify') + 1])
                summary = json.loads((archive / 'summary.json').read_text())
                if self.bad_embeddings:
                    summary['includes_embeddings'] = False
                return json.dumps(summary)
            version = self.open_database(binary, db, name)
            if name == 'schema-version':
                return f'Schema version: {version} (latest supported: {version})'
            archive = Path(argv[argv.index('create') + 1])
            archive.mkdir()
            counts = dict(COUNTS)
            if self.count_drift and binary.read_text() == 'candidate' and db == self.db:
                counts['note'] += 1
            self.write_archive(archive, version, counts)
            return '{}'
        raise AssertionError('unexpected command: ' + name)

    @staticmethod
    def write_archive(archive, version=1, counts=None):
        (archive / 'summary.json').write_text(json.dumps({
            'schema_version': version, 'includes_embeddings': True,
            'record_counts': counts if counts is not None else COUNTS}))
        (archive / 'records.jsonl').write_text(json.dumps({
            'table': 'processing_job', 'record': {'status': 'completed'}}) + '\n')

    def port_open(self, port):
        if port == self.a.port:
            return self.running or self.stop_port_stuck
        return bool(self.rehearsal_process and self.rehearsal_process.alive)

    def fake_kill(self, pid, sig):
        if pid != self.pid or sig != 0:
            raise AssertionError('unexpected signal')
        if not self.running:
            raise ProcessLookupError(pid)

    def probe(self, binary, port):
        self.fault('rehearsal probe' if port == self.a.rehearsal_port else 'live probe')


class DeployIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='test-deploy-fam-')
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        state = root / 'state'
        for directory in ('bin', 'database', 'config', 'logs', 'seed'):
            (state / directory).mkdir(parents=True)
        (state / 'bin/graphrag').write_text('old')
        (state / 'database/schema').write_text('1')
        (state / 'database/payload').write_text('original database')
        (state / 'config/config.toml').write_text('# fake config')
        (state / 'credentials').write_text('fake credential')
        (state / 'logs/service.log').write_text('ERROR historical log\n')
        Harness.write_archive(state / 'seed')
        self.prior = {'status': 'success', 'source_commit_full': OLD,
                      'binary_sha256': deploy.digest(state / 'bin/graphrag'),
                      'schema_version_deployed': 1,
                      'pre_deploy_backups': {'portable': str(state / 'seed')}}
        (state / 'bin/graphrag.deployment.json').write_text(json.dumps(self.prior))
        plist = root / 'service.plist'
        with plist.open('wb') as target:
            plistlib.dump({'Label': 'fake.graphrag', 'ProgramArguments': [
                str(state / 'bin/graphrag'), '--config', str(state / 'config/config.toml'),
                '--db-path', str(state / 'database'), 'serve', '--listen', '127.0.0.1:32001',
                '--credentials-file', str(state / 'credentials')],
                'StandardErrorPath': str(state / 'logs/service.log')}, target)
        self.args = SimpleNamespace(state=state, repo=root / 'repo', sha=NEW, plist=plist,
                                    label='fake.graphrag', port=32001, rehearsal_port=32002,
                                    backup=None, dry_run=False, cargo='fake-cargo',
                                    rustc='fake-rustc', query='fixture')
        self.args.repo.mkdir()
        self.h = Harness(self.args)
        self.h.token = 'fake-token'
        self.addCleanup(patch.stopall)
        patch.object(deploy.os, 'kill', self.h.fake_kill).start()
        patch.object(deploy.shutil, 'disk_usage', return_value=SimpleNamespace(free=10**12)).start()
        patch.object(deploy.subprocess, 'Popen', side_effect=lambda argv, **kw: FakeProcess(self.h, argv)).start()
        # A missed fake boundary must never execute an actual host command.
        patch.object(deploy.subprocess, 'run', side_effect=AssertionError('real subprocess forbidden')).start()

    def execute(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.h.execute()
        return output.getvalue()

    def assert_clean(self):
        self.assertFalse((self.h.root / 'deploy.lock').exists())
        self.assertEqual(list(self.h.root.glob('deploy-rehearsal-*')), [])
        if self.h.work:
            self.assertFalse(self.h.work.parent.exists())
            self.assertIn('git worktree', self.h.calls)
        if self.h.rehearsal_process:
            self.assertFalse(self.h.rehearsal_process.alive)
            self.assertTrue(self.h.rehearsal_process.terminated)

    def assert_prior_live(self):
        self.assertEqual(self.h.binary.read_text(), 'old')
        self.assertEqual(json.loads(self.h.record_path.read_text()), self.prior)
        self.assertEqual((self.h.db / 'schema').read_text(), '1')
        self.assertTrue(self.h.running)
        self.assert_clean()

    def report(self):
        return json.loads((self.h.backup / 'deployment.json').read_text())

    def test_success_rehearses_migrations_and_records_evidence(self):
        self.assertIn('PASS', self.execute())
        record = json.loads(self.h.record_path.read_text())
        self.assertEqual(record['source_commit_full'], NEW)
        self.assertEqual(record['binary_sha256'], deploy.digest(self.h.binary))
        self.assertEqual(record['record_counts_before'], record['record_counts_after'])
        self.assertEqual(record['rehearsal']['status'], 'passed')
        self.assertTrue(record['rehearsal']['artifacts_removed'])
        self.assertEqual(record, self.report())
        self.assertLess(self.h.calls.index('rehearsal probe'), self.h.calls.index('launchctl bootout'))
        self.assert_clean()

    def test_unchanged_migration_paths_skip_rehearsal(self):
        self.h.migration_diff = False
        self.execute()
        self.assertNotIn('rehearsal restore', self.h.calls)
        self.assertIsNone(self.h.rehearsal_process)
        self.assert_clean()

    def test_old_shutdown_errors_are_excluded_from_candidate_log_gate(self):
        self.h.shutdown_error = True
        self.execute()
        record = json.loads(self.h.record_path.read_text())
        self.assertEqual(self.h.log.read_bytes()[record['candidate_log_offset']:], b'')
        self.assertEqual(record['status'], 'success')

    def test_candidate_error_rolls_back_with_fresh_portable_database(self):
        self.h.candidate_error = True
        with self.assertRaisesRegex(deploy.Failure, 'candidate log ERROR'):
            self.execute()
        self.assert_prior_live()
        self.assertEqual(self.report()['rollback_status'], 'restored portable')
        self.assertEqual((self.h.backup / 'failed-database/schema').read_text(), '2')
        self.assertEqual((self.h.db / 'payload').read_text(), 'restored logical records')
        self.assertEqual(self.h.calls.count('launchctl bootout'), 2)
        self.assertTrue(all(version == 1 for flavor, _, version, _ in self.h.opens if flavor == 'old'))

    def test_failure_before_live_open_uses_binary_only_rollback(self):
        self.h.fail_once = 'backup create'  # rehearsal also creates a backup: skip it here
        self.h.migration_diff = False
        with self.assertRaisesRegex(deploy.Failure, 'injected backup create'):
            self.execute()
        self.assert_prior_live()
        self.assertEqual(self.report()['rollback_status'], 'binary only')
        self.assertNotIn('rollback restore', self.h.calls)
        self.assertEqual((self.h.db / 'payload').read_text(), 'original database')

    def test_live_open_failure_is_marked_for_portable_rollback(self):
        self.h.migration_diff = False
        self.h.fail_once = 'schema-version'
        with self.assertRaisesRegex(deploy.Failure, 'injected schema-version'):
            self.execute()
        self.assert_prior_live()
        self.assertTrue(self.h.opened)
        self.assertIn('rollback restore', self.h.calls)

    def test_count_drift_rolls_back_before_candidate_start(self):
        self.h.count_drift = True
        with self.assertRaisesRegex(deploy.Failure, 'live record counts changed'):
            self.execute()
        self.assert_prior_live()
        self.assertEqual(self.h.calls.count('launchctl bootout'), 1)
        self.assertEqual(self.report()['rollback_status'], 'restored portable')

    def test_live_probe_failure_rolls_back(self):
        self.h.fail_once = 'live probe'
        with self.assertRaisesRegex(deploy.Failure, 'injected live probe'):
            self.execute()
        self.assert_prior_live()
        self.assertEqual(self.report()['rollback_status'], 'restored portable')

    def test_rollback_restore_failure_keeps_candidate_database_stopped(self):
        self.h.candidate_error = True
        self.h.fail_always.add('rollback restore')
        with self.assertRaisesRegex(deploy.Failure, 'manual recovery|rollback failed'):
            self.execute()
        self.assertFalse(self.h.running)
        self.assertEqual((self.h.db / 'schema').read_text(), '2')
        self.assertEqual(self.h.binary.read_text(), 'candidate')
        self.assertIn('manual recovery required', self.report()['rollback_status'])
        self.assert_clean()

    def test_ambiguous_stop_does_not_open_database_or_restart(self):
        self.h.migration_diff = False
        self.h.stop_port_stuck = True
        with self.assertRaisesRegex(deploy.Failure, 'port still owned'):
            self.execute()
        self.assertEqual(self.h.opens, [])
        self.assertNotIn('launchctl bootstrap', self.h.calls)
        self.assertFalse(self.h.stopped)
        self.assert_clean()

    def test_invalid_sha_is_refused_before_any_command(self):
        for sha in ('main', 'b' * 39, 'g' * 40, NEW + '\n'):
            with self.subTest(sha=sha):
                self.args.sha = sha
                with self.assertRaisesRegex(deploy.Failure, '40-hex'):
                    self.execute()
        self.assertEqual(self.h.calls, [])
        self.assert_prior_live()

    def test_non_commit_is_refused(self):
        self.h.object_type = 'tree'
        with self.assertRaisesRegex(deploy.Failure, 'not a commit'):
            self.execute()
        self.assertNotIn('release build', self.h.calls)
        self.assert_prior_live()

    def test_ancestry_failure_is_refused_before_build(self):
        # Fail each independent ancestry check, preserving the real preflight order.
        original = self.h.run
        for rejected in (NEW, OLD):
            with self.subTest(ancestor=rejected):
                def run(name, command, **kw):
                    if name == 'git merge-base' and command[4] == '--is-ancestor' and command[5] == rejected:
                        raise deploy.Failure('injected ancestry failure')
                    return original(name, command, **kw)
                with patch.object(self.h, 'run', side_effect=run):
                    with self.assertRaisesRegex(deploy.Failure, 'ancestry failure'):
                        self.execute()
                self.assertNotIn('release build', self.h.calls)
                self.assert_prior_live()

    def test_detached_head_mismatch_cleans_worktree_without_stopping_service(self):
        self.h.head = OLD
        with self.assertRaisesRegex(deploy.Failure, 'HEAD does not match'):
            self.execute()
        self.assertNotIn('release build', self.h.calls)
        self.assertNotIn('launchctl bootout', self.h.calls)
        self.assert_prior_live()

    def test_fetch_failure_has_no_stale_main_fallback(self):
        self.h.fail_once = 'git fetch'
        with self.assertRaisesRegex(deploy.Failure, 'injected git fetch'):
            self.execute()
        self.assertNotIn('git cat-file', self.h.calls)
        self.assertNotIn('release build', self.h.calls)
        self.assert_prior_live()

    def test_disk_shortage_refuses_before_build(self):
        with patch.object(deploy.shutil, 'disk_usage', return_value=SimpleNamespace(free=0)):
            with self.assertRaisesRegex(deploy.Failure, 'insufficient disk'):
                self.execute()
        self.assertNotIn('release build', self.h.calls)
        self.assert_prior_live()

    def test_backup_verify_command_failure_refuses_before_build(self):
        self.h.fail_once = 'backup verify'
        with self.assertRaisesRegex(deploy.Failure, 'injected backup verify'):
            self.execute()
        self.assertNotIn('release build', self.h.calls)
        self.assert_prior_live()

    def test_backup_without_embeddings_refused(self):
        self.h.bad_embeddings = True
        with self.assertRaisesRegex(deploy.Failure, 'lacks embeddings'):
            self.execute()
        self.assert_prior_live()

    def test_backup_with_runnable_jobs_refused(self):
        (self.h.root / 'seed/records.jsonl').write_text(json.dumps({
            'table': 'processing_job', 'record': {'status': 'queued'}}) + '\n')
        with self.assertRaisesRegex(deploy.Failure, 'unfinished jobs'):
            self.execute()
        self.assert_prior_live()

    def test_dry_run_uses_cached_refs_without_build_or_service_mutation(self):
        self.args.dry_run = True
        before = deploy.tree_manifest(self.h.root)
        self.assertIn('DRY RUN', self.execute())
        self.assertNotIn('git fetch', self.h.calls)
        self.assertNotIn('release build', self.h.calls)
        self.assertEqual(before, deploy.tree_manifest(self.h.root))
        self.assert_prior_live()

    def test_no_op_does_not_build_or_stop(self):
        self.args.sha = OLD
        before = deploy.tree_manifest(self.h.root)
        self.assertIn('NO-OP', self.execute())
        self.assertNotIn('release build', self.h.calls)
        self.assertEqual(before, deploy.tree_manifest(self.h.root))
        self.assert_prior_live()

    def test_unsuccessful_same_sha_is_not_a_no_op(self):
        record = dict(self.prior, status='failed')
        self.h.record_path.write_text(json.dumps(record))
        self.args.sha = OLD
        with self.assertRaisesRegex(deploy.Failure, 'same SHA not successful'):
            self.execute()
        self.assertNotIn('release build', self.h.calls)

    def test_rehearsal_failure_is_mandatory_and_always_cleans_process_and_artifacts(self):
        self.h.fail_once = 'rehearsal probe'
        self.h.rehearsal_stubborn = True
        with self.assertRaisesRegex(deploy.Failure, 'injected rehearsal probe'):
            self.execute()
        self.assertTrue(self.h.rehearsal_process.killed)
        self.assertNotIn('launchctl bootout', self.h.calls)
        self.assert_prior_live()

    def test_rehearsal_restore_failure_always_cleans_artifacts(self):
        self.h.fail_once = 'rehearsal restore'
        with self.assertRaisesRegex(deploy.Failure, 'injected rehearsal restore'):
            self.execute()
        self.assertIsNone(self.h.rehearsal_process)
        self.assertNotIn('launchctl bootout', self.h.calls)
        self.assert_prior_live()


class LogGateTests(unittest.TestCase):
    def test_only_new_error_records_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / 'service.log'
            log.write_bytes(b'ERROR old\n')
            offset = log.stat().st_size
            with log.open('ab') as target:
                target.write(b'INFO candidate ready\nNOTERROR harmless\n')
            deploy.log_gate(log, offset)
            with log.open('ab') as target:
                target.write(b'ERROR candidate\n')
            with self.assertRaisesRegex(deploy.Failure, 'candidate log ERROR'):
                deploy.log_gate(log, offset)

    def test_missing_or_truncated_log_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / 'service.log'
            with self.assertRaisesRegex(deploy.Failure, 'missing or rotated'):
                deploy.log_gate(log, 0)
            log.write_bytes(b'x')
            with self.assertRaisesRegex(deploy.Failure, 'missing or rotated'):
                deploy.log_gate(log, 2)


if __name__ == '__main__':
    unittest.main()
