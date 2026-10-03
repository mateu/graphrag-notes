"""Failure/cleanup and isolation contracts for the opt-in native harness."""
import contextlib
import ctypes
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[1] / "validate-native-mcp.py"
SPEC = importlib.util.spec_from_file_location("native_mcp_validation", SCRIPT)
HARNESS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HARNESS)

def helper_module(name):
    spec = importlib.util.spec_from_file_location("native_" + name, SCRIPT.parent / "native-mcp" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

ENVELOPE = helper_module("envelope")
EXTENDED = helper_module("extended")


class NativeRuntimeHarness(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="native-mcp-contract-")
        self.directory = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def executable(self, name, code):
        path = self.directory / name
        path.write_text("#!" + sys.executable + "\n" + code)
        path.chmod(0o700)
        return path

    def run_child(self, path, label="fixture", timeout=2, tokens=()):
        return HARNESS.run_child([str(path)], {}, self.directory, label,
            HARNESS.private_environment(self.directory, "server"), tokens, timeout)

    def test_private_environments_do_not_inherit_personal_configuration_or_secrets(self):
        with mock.patch.dict(os.environ, {"NATIVE_TEST_SECRET": "secret", "OPENCLAW_CONFIG_PATH": "/personal/config", "HERMES_HOME": "/personal/hermes"}):
            node = HARNESS.private_environment(self.directory, "openclaw-a")
            hermes = HARNESS.private_environment(self.directory, "hermes", Path("/installed/code"))
        self.assertNotIn("NATIVE_TEST_SECRET", node)
        self.assertNotIn("NATIVE_TEST_SECRET", hermes)
        self.assertNotIn("HERMES_HOME", node)
        self.assertTrue(Path(node["OPENCLAW_CONFIG_PATH"]).is_relative_to(self.directory))
        self.assertEqual(json.loads(Path(node["OPENCLAW_CONFIG_PATH"]).read_text()), {})
        self.assertTrue(Path(hermes["HERMES_HOME"]).is_relative_to(self.directory))
        self.assertEqual(hermes["PYTHONPATH"], "/installed/code")
        self.assertEqual(stat.S_IMODE(Path(hermes["HERMES_HOME"]).stat().st_mode), 0o700)
        self.assertEqual(stat.S_IMODE(Path(node["OPENCLAW_CONFIG_PATH"]).stat().st_mode), 0o600)

    def test_private_environment_rejects_linked_parents_without_editing_external_profiles(self):
        cases = (("hermes", ()), ("hermes", ("homes",)),
                 ("hermes", ("homes", "hermes")),
                 ("hermes", ("homes", "hermes", "xdg")),
                 ("hermes", ("homes", "hermes", "cache")),
                 ("hermes", ("homes", "hermes", "data")),
                 ("hermes", ("homes", "hermes", "hermes-profile")),
                 ("openclaw-a", ("homes", "openclaw-a", "state")))
        for client, parts in cases:
            with self.subTest(client=client, parts=parts), tempfile.TemporaryDirectory(
                    prefix="native-parent-probe-", dir=self.directory) as temporary:
                root = Path(temporary)
                runspace = root / "runspace"
                runspace.mkdir(mode=0o700)
                HARNESS.private_environment(runspace, client, Path("/installed/code"))
                external = root / "external-profile"
                external.mkdir(mode=0o700)
                original = {"config.yaml": "synthetic existing configuration\n",
                            ".env": "SYNTHETIC_MARKER=preserve\n"}
                for name, content in original.items():
                    (external / name).write_text(content)
                linked = runspace.joinpath(*parts)
                linked.rename(linked.with_name(linked.name + "-original"))
                linked.symlink_to(external, target_is_directory=True)
                with self.assertRaisesRegex(HARNESS.ValidationError, "Unsafe private environment directory"):
                    HARNESS.private_environment(runspace, client, Path("/installed/code"))
                self.assertTrue(linked.is_symlink())
                self.assertEqual(sorted(path.name for path in external.iterdir()), sorted(original))
                for name, content in original.items():
                    self.assertEqual((external / name).read_text(), content)

    def test_private_environment_rejects_unsafe_existing_directories_without_repairing_them(self):
        for parts in ((), ("homes",), ("homes", "hermes"),
                      ("homes", "hermes", "xdg"), ("homes", "hermes", "cache"),
                      ("homes", "hermes", "data"), ("homes", "hermes", "hermes-profile")):
            with self.subTest(parts=parts), tempfile.TemporaryDirectory(
                    prefix="native-mode-probe-", dir=self.directory) as temporary:
                runspace = Path(temporary)
                HARNESS.private_environment(runspace, "hermes", Path("/installed/code"))
                unsafe = runspace.joinpath(*parts)
                unsafe.chmod(0o755)
                with self.assertRaisesRegex(HARNESS.ValidationError, "Unsafe private environment directory"):
                    HARNESS.private_environment(runspace, "hermes", Path("/installed/code"))
                self.assertEqual(stat.S_IMODE(unsafe.stat().st_mode), 0o755)

    def test_private_environment_rejects_non_directory_parent_without_blocking(self):
        for fifo in (False, True):
            with self.subTest(fifo=fifo), tempfile.TemporaryDirectory(
                    prefix="native-special-parent-probe-", dir=self.directory) as temporary:
                runspace = Path(temporary)
                environment = HARNESS.private_environment(runspace, "hermes", Path("/installed/code"))
                profile = Path(environment["HERMES_HOME"])
                profile.rename(profile.with_name("original-profile"))
                if fifo:
                    os.mkfifo(profile, 0o600)
                else:
                    profile.write_text("synthetic existing file")
                started = time.monotonic()
                with self.assertRaisesRegex(HARNESS.ValidationError, "Unsafe private environment directory"):
                    HARNESS.private_environment(runspace, "hermes", Path("/installed/code"))
                self.assertLess(time.monotonic() - started, 1)
                self.assertTrue(stat.S_ISFIFO(profile.lstat().st_mode) if fifo else profile.is_file())

    def test_runtime_credential_leak_is_rejected_and_only_redacted_logs_survive(self):
        credential = "synthetic-bearer-must-not-be-retained"
        binary = self.executable("leaking-runtime", f"import sys\nprint({credential!r})\nprint({credential!r}, file=sys.stderr)\n")
        with self.assertRaisesRegex(HARNESS.ValidationError, "printed a synthetic credential"):
            self.run_child(binary, tokens=[credential])
        for suffix in ("stdout", "stderr"):
            content = (self.directory / ("fixture." + suffix)).read_text()
            self.assertNotIn(credential, content)
            self.assertIn("REDACTED", content)
            self.assertEqual(stat.S_IMODE((self.directory / ("fixture." + suffix)).stat().st_mode), 0o600)

    def test_failed_runtime_preserves_sanitized_diagnostic_and_not_success_evidence(self):
        binary = self.executable("failed-runtime", "import sys\nprint('partial evidence',flush=True)\nprint('runtime entry changed',file=sys.stderr,flush=True)\nsys.exit(7)\n")
        with self.assertRaisesRegex(HARNESS.ValidationError, "failed; inspect its private sanitized log"):
            self.run_child(binary)
        self.assertIn("partial evidence", (self.directory / "fixture.stdout").read_text())
        self.assertIn("runtime entry changed", (self.directory / "fixture.stderr").read_text())

    def test_timeout_stops_child_process_group_and_preserves_partial_output(self):
        child = self.executable("descendant", "import signal,time\nsignal.signal(signal.SIGTERM,lambda *_: exit(0))\ntime.sleep(30)\n")
        pid_path = self.directory / "descendant.pid"
        binary = self.executable("blocked-runtime", "import pathlib,subprocess,time\n"
            + f"child=subprocess.Popen([{str(child)!r}])\npathlib.Path({str(pid_path)!r}).write_text(str(child.pid))\n"
            + "print('entered before timeout',flush=True)\ntime.sleep(30)\n")
        started = time.monotonic()
        with self.assertRaisesRegex(HARNESS.ValidationError, "bounded deadline"):
            self.run_child(binary, timeout=0.3)
        self.assertLess(time.monotonic() - started, 8)
        self.assertIn("entered before timeout", (self.directory / "fixture.stdout").read_text())
        pid = int(pid_path.read_text())
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.02)
        else:
            self.fail("Timed-out native runtime left a live descendant")

    def test_service_startup_failure_closes_process_and_log_handles(self):
        binary = self.executable("bad-service", "import sys\nprint('startup refused',file=sys.stderr,flush=True)\nsys.exit(4)\n")
        real_popen = subprocess.Popen
        spawned = []
        def capture(*args, **kwargs):
            process = real_popen(*args, **kwargs)
            spawned.append((process, kwargs["stdout"], kwargs["stderr"]))
            return process
        with mock.patch.object(HARNESS.subprocess, "Popen", side_effect=capture):
            with self.assertRaisesRegex(HARNESS.ValidationError, "stopped during startup"):
                HARNESS.PrivateServer(binary, self.directory, 1)
        self.assertEqual(len(spawned), 1)
        self.assertIsNotNone(spawned[0][0].poll())
        self.assertTrue(spawned[0][1].closed)
        self.assertTrue(spawned[0][2].closed)
        self.assertIn("startup refused", next(self.directory.glob("server-*.stderr")).read_text())

    def assert_process_stopped(self, pid):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return
            time.sleep(0.02)
        self.fail("Cleanup left a live private descendant")

    def test_successful_runtime_with_leaked_descendant_fails_after_cleanup(self):
        child = self.executable("leaked-child", "import time\ntime.sleep(30)\n")
        pid_path = self.directory / "leaked.pid"
        binary = self.executable("successful-leaking-runtime",
            "import pathlib,subprocess\n"
            + f"child=subprocess.Popen([{str(child)!r}],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)\n"
            + f"pathlib.Path({str(pid_path)!r}).write_text(str(child.pid))\n"
            + "print('parent exited successfully',flush=True)\n")
        with self.assertRaisesRegex(HARNESS.ValidationError, "forced cleanup was required"):
            self.run_child(binary)
        self.assertIn("parent exited successfully", (self.directory / "fixture.stdout").read_text())
        self.assert_process_stopped(int(pid_path.read_text()))

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux child adoption contract")
    def test_clean_runtime_reaps_orphan_zombies_without_relying_on_pid_one(self):
        # This supervisor delays orphan reaping until the inner harness exits.
        # That reproduces a non-reaping PID 1 without leaving test zombies there.
        self.assertEqual(ctypes.CDLL(None).prctl(36, 1, 0, 0, 0), 0)
        pid_path = self.directory / "completed-descendant.pid"
        binary = self.executable("successful-orphaning-runtime",
            "import pathlib,subprocess,sys,time\n"
            + "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(0.02)'],"
              "stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)\n"
            + f"pathlib.Path({str(pid_path)!r}).write_text(str(child.pid))\n"
            + "time.sleep(0.1)\nprint('completed synthetic runtime')\n")
        controller = self.executable("orphan-zombie-controller",
            "import importlib.util,pathlib\n"
            + f"spec=importlib.util.spec_from_file_location('harness',{str(SCRIPT)!r})\n"
            + "harness=importlib.util.module_from_spec(spec);spec.loader.exec_module(harness)\n"
            + f"directory=pathlib.Path({str(self.directory)!r})\n"
            + f"output=harness.run_child([{str(binary)!r}],{{}},directory,'orphan-probe',"
              "harness.private_environment(directory,'server'),(),2)\n"
            + "assert output == 'completed synthetic runtime\\n'\n")
        result = None
        try:
            result = subprocess.run([str(controller)], capture_output=True, text=True, timeout=8)
        finally:
            if pid_path.exists():
                # A broken inner harness leaves its exited child with this
                # supervisor; reap that known fixture before asserting failure.
                try:
                    os.waitpid(int(pid_path.read_text()), 0)
                except ChildProcessError:
                    pass
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assert_process_stopped(int(pid_path.read_text()))

    def test_successful_service_exit_with_leaked_descendant_is_not_clean(self):
        child = self.executable("server-leaked-child", "import time\ntime.sleep(30)\n")
        pid_path = self.directory / "server-leaked.pid"
        binary = self.executable("leaking-service",
            "import pathlib,signal,socket,subprocess,sys,time\n"
            + "signal.signal(signal.SIGINT,lambda *_: sys.exit(0))\n"
            + f"child=subprocess.Popen([{str(child)!r}],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)\n"
            + f"pathlib.Path({str(pid_path)!r}).write_text(str(child.pid))\n"
            + "host,port=sys.argv[sys.argv.index('--listen')+1].split(':')\n"
            + "listener=socket.socket(); listener.bind((host,int(port))); listener.listen()\n"
            + "while True: time.sleep(0.1)\n")
        server = HARNESS.PrivateServer(binary, self.directory, 2)
        with self.assertRaisesRegex(HARNESS.ValidationError, "did not shut down cleanly"):
            server.stop()
        self.assertEqual(server.process.returncode, 0)
        self.assertTrue(all(stream.closed for stream in server.streams))
        self.assert_process_stopped(int(pid_path.read_text()))

    def test_unresponsive_service_shutdown_uses_remaining_budget_and_reaps(self):
        binary = self.executable("unresponsive-service",
            "import signal,socket,sys,time\n"
            + "signal.signal(signal.SIGINT,signal.SIG_IGN); signal.signal(signal.SIGTERM,signal.SIG_IGN)\n"
            + "host,port=sys.argv[sys.argv.index('--listen')+1].split(':')\n"
            + "listener=socket.socket(); listener.bind((host,int(port))); listener.listen()\n"
            + "while True: time.sleep(0.1)\n")
        server = HARNESS.PrivateServer(binary, self.directory, 2)
        started = time.monotonic()
        with self.assertRaisesRegex(HARNESS.ValidationError, "did not shut down cleanly"):
            server.stop(timeout=0.2)
        self.assertLess(time.monotonic() - started, 1.5)
        self.assertLess(server.process.returncode, 0)
        self.assertTrue(all(stream.closed for stream in server.streams))
        self.assert_process_stopped(server.process.pid)

    def test_timeout_cleanup_reaps_signalled_parent_after_eperm_exit_race(self):
        actual_killpg = os.killpg
        for denied_signal in (signal.SIGTERM, signal.SIGKILL):
            with self.subTest(denied_signal=denied_signal):
                injected = []
                label = "denied-signal-" + str(denied_signal)
                binary = self.executable(label,
                    "import signal,time\n"
                    + "signal.signal(signal.SIGTERM,signal.SIG_IGN)\n"
                    + "print('partial evidence before timeout',flush=True)\ntime.sleep(30)\n")
                def deny_after_exit_begins(pid, sent_signal):
                    if sent_signal == denied_signal:
                        # Use a real subprocess exit, then report the Darwin
                        # group-signal race before the caller has reaped it.
                        actual_killpg(pid, signal.SIGKILL)
                        injected.append(pid)
                        raise PermissionError("synthetic group exit race")
                    return actual_killpg(pid, sent_signal)
                with mock.patch.object(HARNESS.os, "killpg", side_effect=deny_after_exit_begins):
                    with self.assertRaisesRegex(HARNESS.ValidationError, "bounded deadline"):
                        self.run_child(binary, label=label, timeout=0.2)
                self.assertTrue(injected)
                self.assert_process_stopped(injected[0])
                self.assertIn("partial evidence", (self.directory / (label + ".stdout")).read_text())

    def test_uncertain_residual_group_cannot_make_successful_runtime_clean(self):
        child = self.executable("uncertain-child", "import time\ntime.sleep(30)\n")
        pid_path = self.directory / "uncertain.pid"
        binary = self.executable("uncertain-successful-parent",
            "import pathlib,subprocess\n"
            + f"child=subprocess.Popen([{str(child)!r}],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)\n"
            + f"pathlib.Path({str(pid_path)!r}).write_text(str(child.pid))\n")
        actual_killpg = os.killpg
        injected = []
        def deny_after_group_exit_begins(pid, sent_signal):
            result = actual_killpg(pid, sent_signal)
            if sent_signal == signal.SIGKILL:
                injected.append(pid)
                raise PermissionError("synthetic uncertain residual group")
            return result
        with mock.patch.object(HARNESS.os, "killpg", side_effect=deny_after_group_exit_begins):
            with self.assertRaisesRegex(HARNESS.ValidationError, "forced cleanup was required"):
                self.run_child(binary)
        self.assertTrue(injected)
        self.assert_process_stopped(int(pid_path.read_text()))

    def test_invalid_deadlines_are_rejected_before_starting_runtimes(self):
        required = ["--binary", "/absent", "--openclaw-root", "/absent", "--hermes-root", "/absent"]
        for flag in ("--command-timeout", "--deadline-seconds"):
            for value in ("0", "-1", "nan", "inf"):
                with self.subTest(flag=flag, value=value):
                    with mock.patch.object(HARNESS, "PrivateServer") as server:
                        with contextlib.redirect_stderr(io.StringIO()):
                            with self.assertRaises(SystemExit) as error:
                                HARNESS.main([*required,flag,value])
                    self.assertEqual(error.exception.code, 2)
                    server.assert_not_called()

    def installed_arguments(self):
        binary=self.executable("fixture-binary", "pass\n")
        openclaw=self.directory/"openclaw"
        entry=openclaw/"dist/agents/agent-bundle-mcp-runtime.js"
        entry.parent.mkdir(parents=True); entry.write_text("export {};\n")
        (openclaw/"package.json").write_text('{"version":"fixture"}')
        hermes=self.directory/"hermes"
        (hermes/"tools").mkdir(parents=True)
        (hermes/"tools/mcp_tool_discovery.py").write_text("")
        python=hermes/"venv/bin/python"
        python.parent.mkdir(parents=True)
        python.symlink_to(sys.executable)
        return ["--binary",str(binary),"--openclaw-root",str(openclaw),
            "--node",str(binary),"--hermes-root",str(hermes),"--hermes-python",str(python)]

    def test_python_venv_executable_symlink_is_preserved(self):
        options=HARNESS.parser_options(self.installed_arguments())
        self.assertTrue(options.hermes_python.is_symlink())
        self.assertEqual(options.hermes_python,self.directory/"hermes/venv/bin/python")

    def test_openclaw_runtime_override_is_bound_to_the_reported_installation(self):
        arguments = self.installed_arguments()
        root = self.directory / "openclaw"
        internal = root / "dist/custom-runtime.js"
        internal.write_text("export {};\n")
        options = HARNESS.parser_options([*arguments,"--openclaw-runtime",str(internal)])
        self.assertEqual(options.openclaw_runtime, internal.resolve())
        foreign = self.directory / "other-installation/runtime.js"
        foreign.parent.mkdir()
        foreign.write_text("export {};\n")
        linked = root / "dist/foreign-runtime.js"
        linked.symlink_to(foreign)
        for override in (foreign, linked):
            with self.subTest(override=override):
                with mock.patch.object(HARNESS, "PrivateServer") as server:
                    with contextlib.redirect_stderr(io.StringIO()) as error:
                        with self.assertRaises(SystemExit):
                            HARNESS.main([*arguments,"--openclaw-runtime",str(override)])
                server.assert_not_called()
                self.assertIn("must resolve within --openclaw-root", error.getvalue())

    def test_native_service_error_survives_hermes_rendering_and_requires_exact_category(self):
        envelope = {"schema_version": 1, "data": None,
                    "error": {"code": "revision_conflict", "message": "Refresh the snapshot.", "retryable": False}}
        rendered = json.dumps({"error": json.dumps(envelope)})
        self.assertEqual(ENVELOPE.checked_result(rendered, "revision_conflict"),
                         {"error_code": "revision_conflict", "retryable": False})
        with self.assertRaises(ValueError):
            ENVELOPE.checked_result(rendered, "not_found")
        with self.assertRaises(ValueError):
            ENVELOPE.checked_result(rendered)
        successful = {"schema_version": 1, "data": {"id": "note:synthetic"}, "error": None}
        self.assertEqual(ENVELOPE.checked_result(json.dumps({"result": json.dumps(successful)})), successful["data"])

    def test_transport_or_partial_envelope_cannot_count_as_native_policy_denial(self):
        for value in ({"error": "revision_conflict: server disconnected"},
                      {"error": "tool approval required"},
                      {"schema_version": 1, "error": {"code": "not_found"}},
                      {"schema_version": 2, "data": None, "error": {"code": "not_found"}}):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    ENVELOPE.checked_result(json.dumps(value), "not_found")

    def test_extended_provider_pause_is_released_when_native_call_fails(self):
        provider = mock.Mock()
        provider.release = threading.Event()
        with self.assertRaisesRegex(RuntimeError, "native runtime failure"):
            with EXTENDED.provider_pause(provider):
                raise RuntimeError("native runtime failure")
        provider.arm.assert_called_once_with()
        self.assertTrue(provider.release.is_set())

    def test_failed_extended_native_call_retains_attempt_without_claiming_success(self):
        report = {}
        with self.assertRaisesRegex(RuntimeError, "native failed"):
            EXTENDED.run_extended(mock.Mock(side_effect=RuntimeError("native failed")), mock.Mock(),
                                  HARNESS.require, lambda: 10, report)
        self.assertEqual(report["native_calls"], [{"client": "openclaw", "instance_id": "writer",
            "tool": "capture_note", "expected_error": None, "status": "attempted"}])
        self.assertFalse(report["connection_decisions_exercised"])

    def test_log_cleanup_rejects_symbolic_and_hard_links_without_copying_external_content(self):
        with tempfile.TemporaryDirectory(prefix="native-external-fixture-") as external:
            target = Path(external) / "synthetic-external.txt"
            target.write_text("synthetic external content must never become evidence")
            (self.directory / "child.stdout").symlink_to(target)
            os.link(target, self.directory / "child.stderr")
            ordinary = self.directory / "server.stdout"
            ordinary.write_text("ordinary log with generated-secret")
            errors = HARNESS.sanitize_logs(self.directory, ["generated-secret"])
            self.assertTrue(errors)
            self.assertFalse((self.directory / "child.stdout").exists())
            self.assertFalse((self.directory / "child.stderr").exists())
            self.assertEqual(target.read_text(), "synthetic external content must never become evidence")
            self.assertIn("REDACTED", ordinary.read_text())
            self.assertNotIn("synthetic external content", ordinary.read_text())

    def test_log_cleanup_rejects_fifo_without_blocking_for_a_writer(self):
        fifo = self.directory / "child.stderr"
        os.mkfifo(fifo, 0o600)
        started = time.monotonic()
        self.assertTrue(HARNESS.sanitize_logs(self.directory, []))
        self.assertLess(time.monotonic() - started, 1)
        self.assertFalse(fifo.exists())

    def test_executable_snapshot_and_hash_survive_replacement_of_supplied_binary(self):
        original = self.executable("supplied-binary", "print('original-version')\n")
        snapshot, digest = HARNESS.pin_binary(original, self.directory)
        replacement = self.executable("replacement-binary", "print('new-version')\n")
        os.replace(replacement, original)
        self.assertNotEqual(HARNESS.binary_digest(original), digest)
        self.assertEqual(HARNESS.binary_digest(snapshot), digest)
        self.assertEqual(stat.S_IMODE(snapshot.stat().st_mode), 0o500)
        self.assertEqual(stat.S_IMODE(snapshot.parent.stat().st_mode), 0o700)
        self.assertEqual(self.run_child(snapshot), "original-version\n")
        self.assertEqual(self.run_child(snapshot, label="snapshot-restart"), "original-version\n")

    def test_rejected_directory_symlink_is_unlinked_without_following_target_metadata(self):
        with tempfile.TemporaryDirectory(prefix="native-external-directory-") as external:
            target = Path(external)
            marker = target / "untouched.txt"
            marker.write_text("synthetic external marker")
            link = self.directory / "child.stderr"
            link.symlink_to(target, target_is_directory=True)
            actual_is_dir = Path.is_dir
            def checked_is_dir(path):
                # Python 3.12 glob checks its root before listing entries.
                # Only that known private root may be inspected; checking a
                # rejected link or its external target remains a failure.
                if path != self.directory:
                    raise AssertionError("A rejected log must not follow its target")
                return actual_is_dir(path)
            with mock.patch.object(Path, "is_dir", autospec=True, side_effect=checked_is_dir):
                self.assertTrue(HARNESS.sanitize_logs(self.directory, []))
            self.assertFalse(link.is_symlink())
            self.assertEqual(marker.read_text(), "synthetic external marker")

    def test_runspace_root_requires_exact_private_mode_and_write_search_access(self):
        arguments = self.installed_arguments()
        root = self.directory / "runspace"
        root.mkdir(mode=0o700)
        for mode in (0o000, 0o500, 0o600, 0o701, 0o1700):
            with self.subTest(mode=mode):
                root.chmod(mode)
                try:
                    with contextlib.redirect_stderr(io.StringIO()) as error:
                        with self.assertRaises(SystemExit):
                            HARNESS.parser_options([*arguments,"--runspace-root",str(root)])
                    self.assertIn("mode 0700",error.getvalue())
                finally:
                    root.chmod(0o700)
        actual_access = os.access
        def denied_access(path, mode):
            return False if Path(path) == root else actual_access(path, mode)
        with mock.patch.object(HARNESS.os, "access", side_effect=denied_access):
            with contextlib.redirect_stderr(io.StringIO()) as error:
                with self.assertRaises(SystemExit):
                    HARNESS.parser_options([*arguments,"--runspace-root",str(root)])
        self.assertIn("write/search access",error.getvalue())
        self.assertEqual(HARNESS.parser_options([*arguments,"--runspace-root",str(root)]).runspace_root,root.resolve())


if __name__ == "__main__":
    unittest.main()
