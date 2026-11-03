#!/usr/bin/env python3
"""Human-initiated, fail-closed fam deployment. No shell commands or secret logs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import plistlib
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager


class Failure(Exception):
    pass


def require(ok, reason):
    if not ok:
        raise Failure(reason)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    temporary = path.with_name(path.name + '.new')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temporary, path)


def tree_manifest(path):
    return {str(p.relative_to(path)): (p.stat().st_size, digest(p))
            for p in sorted(path.rglob('*')) if p.is_file()}


def log_gate(path, offset):
    require(path.exists() and path.stat().st_size >= offset, 'candidate log missing or rotated')
    with path.open('rb') as source:
        source.seek(offset)
        require(not re.search(rb'\bERROR\b', source.read()), 'candidate log ERROR')


@contextmanager
def deploy_lock(root):
    lock = root / 'deploy.lock'
    try:
        lock.mkdir(mode=0o700)
    except FileExistsError:
        raise Failure('deploy lock held; reconcile owner, never delete blindly') from None
    try:
        (lock / 'pid').write_text(str(os.getpid()) + '\n')
        yield
    finally:
        shutil.rmtree(lock)


class Deploy:
    def __init__(self, args):
        self.a = args
        self.root = args.state.resolve()
        self.repo = args.repo.resolve()
        self.binary = self.root / 'bin/graphrag'
        self.db = self.root / 'database'
        self.config = self.root / 'config/config.toml'
        self.record_path = self.root / 'bin/graphrag.deployment.json'
        self.events = []
        self.stopped = False
        self.opened = False
        self.installed = False
        self.candidate_started = False
        self.backup = None
        self.work = None
        self.locked = False
        self.rehearsal_result = 'not required: migration paths unchanged'
        self.service = f'gui/{os.getuid()}/{args.label}'
        self.domain = f'gui/{os.getuid()}'
        self.env = dict(os.environ)
        self.env.update(RUSTC_WRAPPER='', CARGO_BUILD_JOBS='2')
        # Auth is only inherited by read probes, never by builds/local DB commands.
        self.token = self.env.pop('GRAPHRAG_TOKEN', '')

    def run(self, name, command, *, timeout=600, auth=False, cwd=None):
        started = time.monotonic()
        env = dict(self.env)
        if auth:
            env['GRAPHRAG_TOKEN'] = self.token
        try:
            p = subprocess.run([str(x) for x in command], cwd=cwd, env=env,
                               stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, timeout=timeout)
        except (OSError, subprocess.TimeoutExpired):
            raise Failure(name + ' could not complete (child output suppressed)') from None
        self.events.append({'command': name, 'exit': p.returncode,
                            'seconds': round(time.monotonic() - started, 3)})
        require(p.returncode == 0, name + ' failed (child output suppressed)')
        return p.stdout.decode()

    def git(self, *args):
        return self.run('git ' + args[0], ['git', '-C', self.repo, *args]).strip()

    def local(self, binary, db, *args):
        return [binary, '--config', self.config, '--db-path', db, *args]

    def verify(self, binary, archive):
        summary = json.loads(self.run('backup verify', self.local(binary, self.db,
                             'backup', 'verify', archive, '--format', 'json')))
        require(summary['includes_embeddings'] is True, 'backup lacks embeddings')
        require(isinstance(summary['record_counts'], dict), 'backup lacks counts')
        # Restarting durable jobs can mutate counts or external state during validation.
        with (archive / 'records.jsonl').open() as records:
            for line in records:
                item = json.loads(line)
                if item['table'] == 'processing_job':
                    require(item['record'].get('status') == 'completed',
                            'backup contains runnable/unfinished jobs; quiesce first')
        return summary

    def archive(self, binary, db, target):
        self.run('backup create', self.local(binary, db, 'backup', 'create', target,
                                            '--include-embeddings', '--format', 'json'))
        return self.verify(binary, target)

    def schema(self, binary, db, expected):
        raw = self.run('schema-version', self.local(binary, db, 'schema-version'))
        match = re.search(r'Schema version: (\d+) \(latest supported: (\d+)\)', raw)
        require(match and int(match[1]) == int(match[2]) == expected,
                'schema does not equal expected target')

    def service_pid(self):
        raw = self.run('launchctl print', ['launchctl', 'print', self.service])
        match = re.search(r'\bpid = (\d+)', raw)
        require(match is not None, 'service not running / already mid-deploy')
        pid = int(match[1])
        os.kill(pid, 0)
        return pid

    def wait_dead(self, pid):
        for _ in range(1000):
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return
            time.sleep(1)
        raise Failure('old process still owns database')

    def stop(self):
        pid = self.service_pid()
        self.run('launchctl bootout', ['launchctl', 'bootout', self.service], timeout=1000)
        self.wait_dead(pid)
        require(not self.port_open(self.a.port), 'service port still owned')

    @staticmethod
    def port_open(port):
        try:
            with socket.create_connection(('127.0.0.1', port), timeout=.2):
                return True
        except OSError:
            return False

    def wait_ready(self, port, proc=None):
        for _ in range(300):
            if proc is not None:
                require(proc.poll() is None, 'rehearsal process exited')
            if self.port_open(port):
                return
            time.sleep(.2)
        raise Failure('readiness timeout')

    def probe(self, binary, port):
        from importlib.util import module_from_spec, spec_from_file_location
        spec = spec_from_file_location('deploy_probe', Path(__file__).with_name('post-deploy-probe.py'))
        probe = module_from_spec(spec)
        spec.loader.exec_module(probe)
        base = [binary, '--server', f'http://127.0.0.1:{port}/mcp']
        for mode, graph in [('keyword', 'off'), ('hybrid', 'on')]:
            raw = self.run('authenticated ' + mode, base + ['search', '--scope', 'notes',
                           '--limit', '2', '--mode', mode, '--graph', graph, '--format',
                           'json', '--', self.a.query], timeout=45, auth=True)
            data = probe.read_data(raw, 'search_notes')
            probe.top_note(data)
            if graph == 'on':
                require(data.get('graph_status') == 'ok', 'hybrid graph not ok')
        raw = self.run('authenticated build_context', base + ['augment', '--scope', 'notes',
                       '--limit', '2', '--graph', 'on', '--format', 'json', '--', self.a.query],
                       timeout=45, auth=True)
        data = probe.read_data(raw, 'build_context')
        require(data.get('chunks') and data.get('diagnostics', {}).get('graph_status') == 'ok',
                'build_context failed')

    def validate_plist(self):
        with self.a.plist.open('rb') as source:
            self.plist = plistlib.load(source)
        require(self.plist['Label'] == self.a.label, 'launchd label mismatch')
        argv = self.plist['ProgramArguments']
        def option(name):
            return argv[argv.index(name) + 1]
        require(str(self.binary) in argv and option('--db-path') == str(self.db)
                and option('--config') == str(self.config)
                and option('--listen') == f'127.0.0.1:{self.a.port}'
                and 'serve' in argv, 'plist does not match state/binary/config/loopback port')
        self.credentials = Path(option('--credentials-file'))
        require(self.credentials.is_file(), 'service credential file missing')
        self.log = Path(self.plist['StandardErrorPath'])
        require(self.log.is_relative_to(self.root), 'service log outside state directory')

    def preflight(self):
        require(re.fullmatch(r'[0-9a-fA-F]{40}', self.a.sha), 'require full 40-hex SHA')
        self.a.sha = self.a.sha.lower()
        require(self.root.is_dir() and self.db.is_dir(), 'existing state/database required')
        require(self.locked or not (self.root / 'deploy.lock').exists(), 'deploy lock held')
        self.prior = json.loads(self.record_path.read_text())
        self.previous = self.prior['source_commit_full']
        require(re.fullmatch(r'[0-9a-f]{40}', self.previous), 'invalid prior source commit')
        require(digest(self.binary) == self.prior['binary_sha256'], 'deployed binary hash mismatch')
        self.validate_plist()
        self.service_pid()
        require(self.port_open(self.a.port), 'service not ready / mid-deploy')
        if not self.a.dry_run:
            self.git('fetch', 'origin')  # no stale-main fallback
        require(self.git('cat-file', '-t', self.a.sha) == 'commit', 'SHA is not a commit')
        self.git('merge-base', '--is-ancestor', self.a.sha, 'origin/main')
        self.git('merge-base', '--is-ancestor', self.previous, self.a.sha)
        self.migrations = bool(self.git('diff', '--name-only', self.previous, self.a.sha,
                                        '--', 'crates/db/src/migrations'))
        if self.previous == self.a.sha:
            require(self.prior.get('status', 'success') == 'success', 'same SHA not successful')
            return False
        require(self.token.strip() and '\n' not in self.token and '\r' not in self.token,
                'GRAPHRAG_TOKEN required in environment')
        self.seed = self.a.backup.resolve() if self.a.backup else Path(
            self.prior['pre_deploy_backups']['portable'])
        self.seed_summary = self.verify(self.binary, self.seed)
        used = sum(p.stat().st_size for p in self.db.rglob('*') if p.is_file())
        portable = sum(p.stat().st_size for p in self.seed.rglob('*') if p.is_file())
        # Cold copy, logical archives, fresh rollback DB, rehearsal and build headroom.
        require(shutil.disk_usage(self.root).free >= 4 * used + 3 * portable + 2 * 1024**3,
                'insufficient disk for cold copy, portable backup, rehearsal and recovery')
        require(self.a.rehearsal_port != self.a.port and self.a.rehearsal_port != 31057,
                'rehearsal must use a non-live port')
        require(not self.port_open(self.a.rehearsal_port), 'rehearsal port already in use')
        return True

    def build(self):
        self.work = Path(tempfile.mkdtemp(prefix='graphrag-deploy-source-')) / 'source'
        self.git('worktree', 'add', '--detach', str(self.work), self.a.sha)
        head = self.run('verify checkout HEAD', ['git', '-C', self.work, 'rev-parse', 'HEAD']).strip()
        require(head == self.a.sha, 'detached checkout HEAD does not match requested SHA')
        self.run('release build', ['nice', '-n', '10', self.a.cargo, 'build', '--release',
                                  '--locked', '-p', 'graphrag-cli'], cwd=self.work, timeout=7200)
        target = Path(self.env.get('CARGO_TARGET_DIR', str(self.work / 'target'))).resolve()
        self.candidate = target / 'release/graphrag'
        self.toolchain = self.run('toolchain', [self.a.rustc, '--version'], cwd=self.work).strip()
        self.expected = max(int(p.name[1:4]) for p in
                            (self.work / 'crates/db/src/migrations').glob('v[0-9][0-9][0-9]_*.rs'))
        self.candidate_hash = digest(self.candidate)

    def rehearse(self):
        with tempfile.TemporaryDirectory(prefix='deploy-rehearsal-', dir=self.root) as directory:
            isolated = Path(directory)
            db = isolated / 'database'
            self.run('rehearsal restore', self.local(self.candidate, db, 'backup', 'restore',
                                                   self.seed, '--db-path', db))
            self.schema(self.candidate, db, self.expected)
            summary = self.archive(self.candidate, db, isolated / 'portable')
            require(summary['record_counts'] == self.seed_summary['record_counts'],
                    'rehearsal changed record counts')
            # Never persist service output or credentials from the rehearsal.
            proc = subprocess.Popen([str(x) for x in self.local(self.candidate, db, 'serve',
                                     '--listen', f'127.0.0.1:{self.a.rehearsal_port}',
                                     '--credentials-file', self.credentials, '--max-job-workers', '1')],
                                    env=self.env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                self.wait_ready(self.a.rehearsal_port, proc)
                self.probe(self.candidate, self.a.rehearsal_port)
            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            require(not self.port_open(self.a.rehearsal_port), 'rehearsal port not released')
        self.rehearsal_result = {'status': 'passed', 'schema': self.expected,
                                 'record_counts': summary['record_counts'],
                                 'port': self.a.rehearsal_port, 'artifacts_removed': True}

    def install(self, source):
        temp = self.binary.with_name('graphrag.new')
        shutil.copy2(source, temp)
        os.replace(temp, self.binary)

    def bootstrap(self):
        self.run('launchctl bootstrap', ['launchctl', 'bootstrap', self.domain, self.a.plist])
        self.wait_ready(self.a.port)
        self.service_pid()

    def cutover(self):
        stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '-' + str(os.getpid())
        self.backup = self.root / 'backups' / ('deploy-' + stamp)
        self.backup.mkdir(parents=True, mode=0o700)
        self.old = self.backup / 'previous-binary'
        shutil.copy2(self.binary, self.old)
        shutil.copy2(self.record_path, self.backup / 'previous-deployment.json')
        # Any ambiguous stop failure is fail-closed; no DB commands until stop returns.
        self.stop()
        self.stopped = True
        shutil.copytree(self.db, self.backup / 'cold-database')
        require(tree_manifest(self.db) == tree_manifest(self.backup / 'cold-database'),
                'cold copy verification failed')
        self.before = self.archive(self.old, self.db, self.backup / 'portable')
        require(self.before['schema_version'] == self.prior['schema_version_deployed'],
                'backup schema disagrees with prior deployment')
        archived_old = self.root / 'bin' / ('graphrag.' + self.previous[:7])
        if archived_old.exists():
            require(digest(archived_old) == digest(self.old), 'previous binary name collision')
        else:
            shutil.copy2(self.old, archived_old)
        atomic_json(self.root / 'bin' / ('graphrag.deployment.' + self.previous[:7] + '.json'), self.prior)
        versioned = self.root / 'bin' / ('graphrag.' + self.a.sha[:7])
        if versioned.exists():
            require(digest(versioned) == self.candidate_hash, 'candidate binary name collision')
        else:
            shutil.copy2(self.candidate, versioned)
        self.install(versioned)
        self.installed = True
        # Mark BEFORE attempting anything that can open/migrate the live database.
        self.opened = True
        self.schema(self.binary, self.db, self.expected)
        after = self.archive(self.binary, self.db, self.backup / 'post-migration-portable')
        require(after['record_counts'] == self.before['record_counts'], 'live record counts changed')
        # Old PID was joined in stop(); exclude all of its shutdown ERRORs.
        offset = self.log.stat().st_size
        self.candidate_started = True  # bootstrap failure can still have spawned a process
        self.bootstrap()
        self.probe(self.binary, self.a.port)
        log_gate(self.log, offset)
        record = {'schema_version': 1, 'status': 'success', 'source_commit_full': self.a.sha,
                  'source_commit': self.a.sha[:7], 'binary_sha256': self.candidate_hash,
                  'schema_version_deployed': self.expected, 'toolchain': self.toolchain,
                  'pre_deploy_backups': {'cold': str(self.backup / 'cold-database'),
                                         'portable': str(self.backup / 'portable')},
                  'record_counts_before': self.before['record_counts'],
                  'record_counts_after': after['record_counts'], 'rehearsal': self.rehearsal_result,
                  'candidate_log_offset': offset, 'command_timings': self.events,
                  'previous_deployment': str(self.backup / 'previous-deployment.json'),
                  'rollback_status': 'not required',
                  'rollback': 'Stop service. Restore verified portable with previous-binary into a fresh '
                              'DB. Preserve failed DB; swap fresh DB in, then previous binary and JSON. '
                              'Never point previous binary at a DB opened by candidate.'}
        atomic_json(self.record_path, record)
        self.report(record)

    def report(self, record):
        atomic_json(self.backup / 'deployment.json', record)
        (self.backup / 'REPORT.md').write_text('# fam deployment\n\n```json\n' +
                                              json.dumps(record, indent=2) + '\n```\n')

    def rollback(self):
        if self.candidate_started:
            self.stop()  # if owner cannot be proven dead, NEVER touch the DB
        if self.opened:
            fresh = self.backup / 'restored-database'
            self.run('rollback restore', self.local(self.old, fresh, 'backup', 'restore',
                     self.backup / 'portable', '--db-path', fresh))
            restored = self.archive(self.old, fresh, self.backup / 'rollback-portable')
            require(restored['record_counts'] == self.before['record_counts'],
                    'rollback restored counts mismatch')
            self.db.rename(self.backup / 'failed-database')
            fresh.rename(self.db)
        self.install(self.old)
        shutil.copy2(self.backup / 'previous-deployment.json', self.record_path)
        self.bootstrap()
        self.report({'status': 'failed', 'source_commit_full': self.a.sha,
                     'rollback_status': 'restored portable' if self.opened else 'binary only',
                     'command_timings': self.events,
                     'failed_database': str(self.backup / 'failed-database') if self.opened else None})

    def execute(self):
        if self.a.dry_run:
            if not self.preflight():
                print('NO-OP: exact SHA and binary already deployed successfully')
            else:
                print('DRY RUN: cached origin/main ancestry, service, backup and disk verified; '
                      'would fetch, build, rehearse if migrations differ, then cut over')
            return
        with deploy_lock(self.root):
            self.locked = True
            try:
                if not self.preflight():
                    print('NO-OP: exact SHA and binary already deployed successfully')
                    return
                self.build()
                if self.migrations:
                    self.rehearse()
                self.cutover()
            except BaseException:
                if self.stopped:
                    try:
                        self.rollback()
                    except BaseException:
                        self.report({'status': 'failed', 'rollback_status': 'FAILED: manual recovery required',
                                     'source_commit_full': self.a.sha, 'command_timings': self.events})
                        raise Failure('rollback failed; reconcile service owner and backup REPORT.md') from None
                raise
            finally:
                self.locked = False
                if self.work:
                    # Remove only our detached source and owned target artifacts.
                    try:
                        self.git('worktree', 'remove', '--force', str(self.work))
                    finally:
                        shutil.rmtree(self.work.parent, ignore_errors=True)
        print('PASS: deployment verified; evidence in timestamped backups/deploy-*')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sha')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--state', type=Path, default=Path.home() / '.graphrag/shared')
    parser.add_argument('--repo', type=Path, default=Path.home() / 'dev/graphrag-notes')
    parser.add_argument('--label', default='com.hunter.graphrag.shared')
    parser.add_argument('--plist', type=Path)
    parser.add_argument('--port', type=int, default=31057)
    parser.add_argument('--rehearsal-port', type=int, default=31201)
    parser.add_argument('--backup', type=Path, help='verified vector-inclusive rehearsal seed archive')
    parser.add_argument('--query', default='OpenClaw', help='known-existing search keyword')
    parser.add_argument('--cargo', default=str(Path.home() / '.cargo/bin/cargo'))
    parser.add_argument('--rustc', default=str(Path.home() / '.cargo/bin/rustc'))
    args = parser.parse_args(argv)
    args.plist = (args.plist or Path.home() / 'Library/LaunchAgents' / (args.label + '.plist')).resolve()
    os.umask(0o077)
    def interrupted(signum, frame):
        raise Failure('interrupted')
    signal.signal(signal.SIGTERM, interrupted)
    try:
        Deploy(args).execute()
        return 0
    except BaseException as error:
        # Only our fixed error vocabulary is safe to render; never child output.
        reason = str(error) if isinstance(error, Failure) else 'validation/operation failed (details suppressed)'
        print('FAIL: ' + reason + '; reconcile backup REPORT.md and service owner before retry',
              file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
