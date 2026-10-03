#!/usr/bin/env python3
"""Opt-in installed native MCP runtime smoke with synthetic, private data.

This runs two OpenClaw runtime contexts and the Hermes registry on ONE host.
It neither deploys to another machine nor starts conversational/LLM sessions.
No existing client configuration, credentials, or corpus is used.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import ctypes
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import secrets
import shutil
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import time

SCRIPTS = Path(__file__).resolve().parent
RUNTIMES = SCRIPTS / "native-mcp"
LINUX_SUBREAPER_PID = None


class ValidationError(RuntimeError):
    """A static, credential-free diagnostic suitable for an evidence report."""


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def private_write(path, text, directory_fd=None):
    if directory_fd is None:
        descriptor, temporary = tempfile.mkstemp(prefix=".native-write-", dir=path.parent)
    else:
        temporary = ".native-write-" + secrets.token_hex(16)
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                             0o600, dir_fd=directory_fd)
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(text)
            stream.flush()
        # Atomic replacement also avoids following a surprising symlink left
        # by an installed runtime. Existing external files are never opened.
        if directory_fd is None:
            os.replace(temporary, path)
        else:
            os.replace(temporary, path.name, src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
    finally:
        if directory_fd is None:
            Path(temporary).unlink(missing_ok=True)
        else:
            try:
                os.unlink(temporary, dir_fd=directory_fd)
            except FileNotFoundError:
                pass


@contextmanager
def private_directory(path, parent_fd=None, create=True):
    """Open only an owned mode-0700 directory, without following child links."""
    descriptor = None
    try:
        if create:
            try:
                os.mkdir(path, mode=0o700, dir_fd=parent_fd)
            except FileExistsError:
                pass
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_NONBLOCK,
                             dir_fd=parent_fd)
        metadata = os.fstat(descriptor)
        require(stat.S_ISDIR(metadata.st_mode) and stat.S_IMODE(metadata.st_mode) == 0o700
                and metadata.st_uid == os.geteuid(), "Unsafe private environment directory")
    except OSError:
        if descriptor is not None:
            os.close(descriptor)
        raise ValidationError("Unsafe private environment directory") from None
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
        raise
    try:
        yield descriptor
    finally:
        os.close(descriptor)


def redact(text, tokens):
    for token in tokens:
        text = text.replace(token, "[REDACTED SYNTHETIC CREDENTIAL]")
    return text


def sanitize_logs(directory, tokens):
    """Reject child-created links/special files without reading their targets."""
    rejected = False
    for path in directory.glob("*.std*"):
        descriptor = None
        try:
            metadata = path.lstat()
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise ValidationError("Unsafe retained log path")
            # The second check closes the lstat/open race; nonblocking prevents
            # a swapped-in FIFO from hanging cleanup before fstat rejects it.
            descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise ValidationError("Unsafe retained log path")
            with os.fdopen(descriptor, "r", encoding="utf-8", errors="replace") as stream:
                descriptor = None
                text = stream.read()
            private_write(path, redact(text, tokens))
        except (OSError, ValidationError):
            rejected = True
            # Unlink removes only the local entry, including a link/FIFO; an
            # empty directory can be removed without traversing its contents.
            try:
                path.unlink(missing_ok=True)
            except OSError:
                try:
                    # rmdir never follows a final symlink; avoid is_dir/stat
                    # even after rejection, which could inspect its target.
                    path.rmdir()
                except OSError:
                    pass
        finally:
            if descriptor is not None:
                os.close(descriptor)
    return ["Rejected an unsafe retained log path without reading it"] if rejected else []


def binary_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pin_binary(source, directory):
    """Hash and execute one private snapshot even if the caller rebuilds."""
    folder = directory / "binary"
    folder.mkdir(mode=0o700)
    snapshot = folder / "graphrag"
    descriptor = os.open(source, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        require(stat.S_ISREG(os.fstat(descriptor).st_mode), "Supplied executable is no longer a regular file")
        with os.fdopen(descriptor, "rb") as origin:
            descriptor = None
            digest = hashlib.sha256()
            with snapshot.open("xb") as destination:
                for block in iter(lambda: origin.read(1024 * 1024), b""):
                    destination.write(block)
                    digest.update(block)
                destination.flush()
                os.fsync(destination.fileno())
        snapshot.chmod(0o500)
        return snapshot, digest.hexdigest()
    finally:
        if descriptor is not None:
            os.close(descriptor)


def private_environment(directory, client, hermes_root=None):
    home = directory / "homes" / client
    config = home / "xdg"
    # Intentionally construct an environment instead of copying os.environ:
    # no personal API keys, client defaults, proxy credentials or profile paths.
    environment = {
        "HOME": str(home), "XDG_CONFIG_HOME": str(config),
        "XDG_CACHE_HOME": str(home / "cache"), "XDG_DATA_HOME": str(home / "data"),
        "PATH": os.defpath, "TERM": "dumb", "LANG": "en_US.UTF-8",
        "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1",
    }
    # Runtimes may leave links or alter directory permissions between calls.
    # Validate every private parent and keep writes anchored to its open handle;
    # refusing an unsafe path must never repair or overwrite an external profile.
    with private_directory(directory, create=False) as runspace_fd, \
            private_directory("homes", runspace_fd) as homes_fd, \
            private_directory(client, homes_fd) as home_fd:
        for folder in ("xdg", "cache", "data"):
            with private_directory(folder, home_fd):
                pass
        if client.startswith("openclaw"):
            state = home / "state"
            with private_directory("state", home_fd):
                pass
            config_path = home / "empty-openclaw.json"
            private_write(config_path, "{}\n", directory_fd=home_fd)
            environment.update(OPENCLAW_HOME=str(home), OPENCLAW_STATE_DIR=str(state),
                               OPENCLAW_CONFIG_PATH=str(config_path))
        elif client == "hermes":
            profile = home / "hermes-profile"
            with private_directory("hermes-profile", home_fd) as profile_fd:
                private_write(profile / "config.yaml", "mcp_servers: {}\n", directory_fd=profile_fd)
                private_write(profile / ".env", "", directory_fd=profile_fd)
            environment.update(HERMES_HOME=str(profile), HERMES_REAL_HOME=str(home),
                               PYTHONPATH=str(hermes_root), TERMINAL_HOME_MODE="profile")
    return environment


def enable_linux_subreaper():
    """Own orphaned private descendants instead of relying on Linux PID 1."""
    global LINUX_SUBREAPER_PID
    if platform.system() != "Linux" or LINUX_SUBREAPER_PID == os.getpid():
        return
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        prctl = libc.prctl
        prctl.argtypes = [ctypes.c_int] + [ctypes.c_ulong] * 4
        prctl.restype = ctypes.c_int
        # PR_SET_CHILD_SUBREAPER, available on supported Linux kernels.
        require(prctl(36, 1, 0, 0, 0) == 0, "Linux private descendant adoption is unavailable")
    except (AttributeError, OSError):
        raise ValidationError("Linux private descendant adoption is unavailable") from None
    LINUX_SUBREAPER_PID = os.getpid()


def reap_group_descendants(process):
    # Popen owns the direct child's exit status. Reap adopted children only
    # after Popen has reaped that parent, and only from its private group.
    if platform.system() != "Linux" or process.poll() is None:
        return
    # Bound one probe even if a faulty descendant keeps creating children;
    # the surrounding group loop retains the cleanup deadline.
    for _ in range(256):
        try:
            child, _ = os.waitpid(-process.pid, os.WNOHANG)
        except ChildProcessError:
            return
        if child == 0:
            return


def terminate_group(process, timeout=5):
    """Stop a private subprocess group; report whether intervention was needed."""
    deadline = time.monotonic() + max(0, timeout)
    intervened = process.poll() is None
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except PermissionError as denied:
            # macOS can deny a signal during exit before poll has reaped the
            # parent. Confirm exit with the bounded emergency reap; a parent
            # that still lives is a cleanup failure, never a clean result.
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                raise denied
        try:
            process.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            pass
    # A successful parent can leave descendants behind. Allow a brief bounded
    # interval for exiting children to be reaped before testing the whole group.
    reap_deadline = min(deadline, time.monotonic() + 0.25)
    while True:
        reap_group_descendants(process)
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            process.wait(timeout=max(0, deadline - time.monotonic()))
            return intervened
        except PermissionError:
            # A remaining private group must not count as a clean shutdown,
            # even if the platform will not let us signal it while reaping.
            pass
        if time.monotonic() >= reap_deadline:
            break
        time.sleep(min(0.01, max(0, reap_deadline - time.monotonic())))
    intervened = True
    killed_reap_deadline = max(deadline, time.monotonic() + 1)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except PermissionError as denied:
        # Resolve the same exit race without treating an unconfirmed group as
        # clean: intervened remains true even when parent reaping succeeds.
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            raise denied
    # Deadline expiry must still reap a killed child. This fixed emergency
    # allowance is independent of (and never restarts) the work budget.
    process.wait(timeout=max(0, killed_reap_deadline - time.monotonic()))
    while True:
        reap_group_descendants(process)
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            return intervened
        except PermissionError:
            pass
        require(time.monotonic() < killed_reap_deadline,
                "Private subprocess group remained after bounded forced cleanup")
        time.sleep(min(0.01, max(0, killed_reap_deadline - time.monotonic())))


def run_child(command, payload, directory, label, environment, tokens, timeout):
    enable_linux_subreaper()
    deadline = time.monotonic() + timeout
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, cwd=directory,
                               env=environment, start_new_session=True)
    timed_out = False
    forced_cleanup = False
    cleanup_performed = False
    stdout = stderr = ""
    try:
        stdout, stderr = process.communicate(json.dumps(payload) if payload is not None else None,
                                             timeout=max(0, deadline - time.monotonic()))
    except subprocess.TimeoutExpired:
        timed_out = True
        cleanup_performed = True
        forced_cleanup = terminate_group(process, timeout=0)
        stdout, stderr = process.communicate(timeout=1)
    finally:
        try:
            if not cleanup_performed:
                forced_cleanup = terminate_group(process, timeout=max(0, deadline - time.monotonic())) or forced_cleanup
        finally:
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream is not None:
                    stream.close()
    leaked = any(token in stdout + stderr for token in tokens)
    private_write(directory / f"{label}.stdout", redact(stdout, tokens))
    private_write(directory / f"{label}.stderr", redact(stderr, tokens))
    require(not leaked, "Installed runtime printed a synthetic credential; logs were redacted")
    require(not timed_out, f"{label} exceeded its bounded deadline; private logs retained")
    require(process.returncode == 0, f"{label} failed; inspect its private sanitized log")
    require(not forced_cleanup, f"{label} left a private process group after exiting; forced cleanup was required")
    return stdout


def runtime_result(*args, **kwargs):
    output = run_child(*args, **kwargs)
    lines = [line.removeprefix("HARNESS_RESULT=") for line in output.splitlines()
             if line.startswith("HARNESS_RESULT=")]
    require(len(lines) == 1, "Installed runtime did not return exactly one evidence result")
    try:
        result = json.loads(lines[0])
    except ValueError:
        raise ValidationError("Installed runtime returned malformed evidence") from None
    require(isinstance(result, dict), "Installed runtime returned invalid evidence shape")
    return result


def free_port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


class PrivateServer:
    def __init__(self, binary, directory, timeout):
        enable_linux_subreaper()
        self.directory = directory
        self.port = free_port()
        self.process = None
        self.streams = []
        self.shutdown_timeout = min(15, timeout)
        deadline = time.monotonic() + min(20, timeout)
        try:
            for name in ("stdout", "stderr"):
                path = directory / f"server-{self.port}.{name}"
                fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                self.streams.append(os.fdopen(fd, "w"))
            command = [str(binary), "--config", str(directory / "config.toml"),
                       "--db-path", str(directory / "corpus"), "serve", "--listen",
                       f"127.0.0.1:{self.port}", "--credentials-file", str(directory / "credentials.json")]
            self.process = subprocess.Popen(command, cwd=directory,
                env=private_environment(directory, "server"), stdout=self.streams[0],
                stderr=self.streams[1], start_new_session=True)
            while time.monotonic() < deadline:
                require(self.process.poll() is None, "Synthetic service stopped during startup")
                try:
                    with socket.create_connection(("127.0.0.1", self.port), timeout=0.2):
                        return
                except OSError:
                    time.sleep(0.05)
            raise ValidationError("Synthetic service startup exceeded its bounded deadline")
        except BaseException:
            self.stop(require_clean=False, timeout=max(0, deadline - time.monotonic()))
            raise

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}/mcp"

    def stop(self, require_clean=True, timeout=None):
        budget = self.shutdown_timeout if timeout is None else min(15, max(0, timeout))
        deadline = time.monotonic() + budget
        clean = True
        try:
            if self.process is not None:
                if self.process.poll() is None:
                    self.process.send_signal(signal.SIGINT)
                    try:
                        self.process.wait(timeout=max(0, deadline - time.monotonic()))
                    except subprocess.TimeoutExpired:
                        clean = False
                clean = clean and self.process.poll() == 0
                forced_cleanup = terminate_group(self.process, timeout=max(0, deadline - time.monotonic()))
                clean = clean and not forced_cleanup
        finally:
            for stream in self.streams:
                stream.close()
        if require_clean:
            require(clean, "Synthetic service did not shut down cleanly")


def parser_options(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path, help="Built MCP-capable GraphRAG binary")
    parser.add_argument("--openclaw-root", required=True, type=Path, help="Installed OpenClaw package directory")
    parser.add_argument("--openclaw-runtime", type=Path, help="Override runtime module entry point within the declared OpenClaw installation")
    parser.add_argument("--node", type=Path, default=shutil.which("node"), help="Installed Node executable")
    parser.add_argument("--hermes-root", required=True, type=Path, help="Installed Hermes Agent code directory")
    parser.add_argument("--hermes-python", type=Path, help="Hermes Python/venv executable; defaults to ROOT/venv/bin/python")
    parser.add_argument("--runspace-root", type=Path, help="Existing private directory for fresh retained runs")
    parser.add_argument("--command-timeout", type=float, default=120, help="Per-child work budget, 1–600 seconds; bounded emergency cleanup may follow")
    parser.add_argument("--deadline-seconds", type=float, default=600, help="Whole scenario work budget, 1–1800 seconds; bounded emergency cleanup may follow")
    parser.add_argument("--extended", action="store_true", help="Explicitly grant extra synthetic identities mutation/upload/jobs capabilities and exercise the combined candidate")
    options = parser.parse_args(argv)
    if platform.system() not in ("Darwin", "Linux"):
        parser.error("This harness targets macOS and Linux")
    for key, maximum in (("command_timeout", 600), ("deadline_seconds", 1800)):
        value = getattr(options, key)
        if not math.isfinite(value) or not 1 <= value <= maximum:
            parser.error(f"--{key.replace('_', '-')} must be finite and between 1 and {maximum}")
    options.binary = options.binary.expanduser().resolve()
    options.openclaw_root = options.openclaw_root.expanduser().resolve()
    options.hermes_root = options.hermes_root.expanduser().resolve()
    options.openclaw_runtime = (options.openclaw_runtime or options.openclaw_root / "dist/agents/agent-bundle-mcp-runtime.js").expanduser().resolve()
    if not options.openclaw_runtime.is_relative_to(options.openclaw_root):
        parser.error("--openclaw-runtime must resolve within --openclaw-root so evidence identifies the loaded installation")
    # Python discovers pyvenv.cfg relative to its invoked executable path.
    # Dereferencing a venv/bin/python symlink silently drops its MCP packages.
    options.hermes_python = (options.hermes_python or options.hermes_root / "venv/bin/python").expanduser().absolute()
    if not options.node:
        parser.error("Node is unavailable; supply --node")
    options.node = Path(options.node).expanduser().absolute()
    for executable in (options.binary, options.node, options.hermes_python):
        if not executable.is_file() or not os.access(executable, os.X_OK):
            parser.error("A supplied binary/runtime executable is missing or not executable")
    if not options.openclaw_runtime.is_file() or not (options.openclaw_root / "package.json").is_file() or not (options.hermes_root / "tools/mcp_tool_discovery.py").is_file():
        parser.error("Installed client runtime entry points are missing; verify roots for their installed versions")
    if options.runspace_root:
        options.runspace_root = options.runspace_root.expanduser()
        if (options.runspace_root.is_symlink() or not options.runspace_root.is_dir()
                or stat.S_IMODE(options.runspace_root.stat().st_mode) != 0o700
                or not os.access(options.runspace_root, os.W_OK | os.X_OK)):
            parser.error("--runspace-root must be an existing private regular directory (mode 0700) with write/search access")
        options.runspace_root = options.runspace_root.resolve()
    return options


def main(argv=None):
    options = parser_options(argv)
    os.umask(0o077)
    directory = Path(tempfile.mkdtemp(prefix="graphrag-native-mcp-", dir=options.runspace_root))
    evidence = {"schema_version": 1, "success": False,
                "scope": "installed native tool runtimes on one host", "host_platform": platform.system(),
                "different_computers_tested": False, "real_agent_llm_sessions_tested": False,
                "synthetic_inference": True, "real_user_profiles_used": False,
                "host_backup_restore_tested": False,
                "binary_sha256": None, "binary_snapshot_used": False, "wire_schema_version": 1,
                "writer_capabilities": ["read", "capture"], "mutation_capabilities_granted": options.extended,
                "extended_scenarios_requested": options.extended,
                "cleanup_errors": []}
    tokens = {name: secrets.token_urlsafe(48) for name in ("openclaw-a", "openclaw-b", "hermes", "observer")}
    extended = extended_state = None
    if options.extended:
        spec = importlib.util.spec_from_file_location("native_extended_fixture", RUNTIMES / "extended.py")
        extended = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(extended)
        tokens.update({name: secrets.token_urlsafe(48) for name in extended.EXPLICIT_GRANTS})
    provider = server = None
    deadline = time.monotonic() + options.deadline_seconds
    def timeout():
        remaining = deadline - time.monotonic()
        require(remaining > 0, "Whole native scenario exceeded its bounded deadline")
        return min(options.command_timeout, remaining)
    try:
        options.binary, evidence["binary_sha256"] = pin_binary(options.binary, directory)
        evidence["binary_snapshot_used"] = True
        evidence["binary_version"] = run_child([str(options.binary), "--version"], None,
            directory, "binary-version", private_environment(directory, "server"), tokens.values(), timeout()).strip()
        spec = importlib.util.spec_from_file_location("native_daily_fixture", SCRIPTS / "validate-daily-workflow.py")
        fixture = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixture)
        provider = fixture.OfflineProvider()
        private_write(directory / "config.toml", f'''[database]
path = {json.dumps(str(directory / 'corpus'))}
[inference]
embedding_provider = "ollama"
embedding_url = {json.dumps(provider.endpoint)}
embedding_model = "bge-m3:latest"
extraction_provider = "ollama"
extraction_url = {json.dumps(provider.endpoint)}
extraction_model = "phi4-mini:latest"
ollama_url = {json.dumps(provider.endpoint)}
retry_attempts = 1
cache_enabled = false
processing_concurrency = 1
timeout_secs = {30 if options.extended else 10}
[logging]
level = "warn"
''')
        policy = {"schema_version": 1, "credentials": [{"instance_id": name,
            "token_sha256": hashlib.sha256(token.encode()).hexdigest(),
            "capabilities": extended.EXPLICIT_GRANTS[name] if extended and name in extended.EXPLICIT_GRANTS else
                ["read"] if name == "observer" else ["read", "capture"]}
            for name, token in tokens.items()]}
        private_write(directory / "credentials.json", json.dumps(policy))
        capture = {"request_id": "native-retry-001", "content": "nativeclientatlas synthetic shared client runtime note.",
            "title": "Synthetic native clients", "tags": ["native-client-walkthrough"],
            "provenance": {"uri": "file:///client-only/native-source.md", "label": "synthetic",
                           "metadata": {"session": "synthetic-native-client-session"}}}
        def openclaw(phase, **extra):
            payload = {"phase": phase, "directory": str(directory / ("openclaw-" + phase)),
                "url": server.url, "tokens": tokens, "capture": capture,
                "runtime_entry": str(options.openclaw_runtime), "install_root": str(options.openclaw_root), **extra}
            return runtime_result([str(options.node), str(RUNTIMES / "openclaw-runtime.mjs")], payload,
                directory, "openclaw-" + phase, private_environment(directory, "openclaw-" + phase), tokens.values(), timeout())
        sequence = 0
        def native_calls(client, instance, calls):
            nonlocal sequence
            sequence += 1
            label = f"{client}-extended-{sequence:03d}"
            if client == "openclaw":
                payload = {"phase": "calls", "instance": instance, "calls": calls,
                    "directory": str(directory / label), "url": server.url, "tokens": tokens,
                    "runtime_entry": str(options.openclaw_runtime), "install_root": str(options.openclaw_root)}
                command = [str(options.node), str(RUNTIMES / "openclaw-runtime.mjs")]
                environment = private_environment(directory, label)
            else:
                require(client == "hermes", "Unknown native client adapter")
                payload = {"phase": "calls", "url": server.url, "token": tokens[instance], "calls": calls}
                command = [str(options.hermes_python), str(RUNTIMES / "hermes-runtime.py")]
                environment = private_environment(directory, "hermes", options.hermes_root)
            return runtime_result(command, payload, directory, label, environment, tokens.values(), timeout())
        server = PrivateServer(options.binary, directory, timeout())
        original = openclaw("initial")
        hermes = runtime_result([str(options.hermes_python), str(RUNTIMES / "hermes-runtime.py")],
            {"url": server.url, "token": tokens["hermes"], "capture": capture, "original": original["record"]},
            directory, "hermes", private_environment(directory, "hermes", options.hermes_root), tokens.values(), timeout())
        readonly = openclaw("read-only")
        policy["credentials"] = [entry for entry in policy["credentials"] if entry["instance_id"] != "observer"]
        private_write(directory / "credentials.json", json.dumps(policy))
        revoked = openclaw("revoked")
        if extended:
            evidence["extended"] = {}
            _, extended_state = extended.run_extended(native_calls, provider, require, timeout, evidence["extended"])
        server.stop(timeout=timeout())
        server = None
        server = PrivateServer(options.binary, directory, timeout())
        replay = openclaw("replay", original=original["record"])
        if extended:
            evidence["extended"]["service_restart"] = extended.replay_extended(native_calls, extended_state, require)
        server.stop(timeout=timeout())
        server = None
        exported = directory / "logical-records.jsonl"
        run_child([str(options.binary), "--config", str(directory / "config.toml"), "--db-path",
            str(directory / "corpus"), "export", str(exported), "--output", "json"], None,
            directory, "host-export", private_environment(directory, "server"), tokens.values(), timeout())
        exported_text = exported.read_text()
        require(not any(token in exported_text for token in tokens.values()), "Host export included an authenticated credential")
        rows = [json.loads(line) for line in exported_text.splitlines()]
        receipts = [row["record"] for row in rows if row["table"] == "remote_capture_receipt"]
        foundation_receipts = [row for row in receipts if row["instance_id"] in ("openclaw-a", "openclaw-b", "hermes")]
        require(len(foundation_receipts) == 3 and len({(row["instance_id"], row["request_id"]) for row in foundation_receipts}) == 3,
                "Native retries produced duplicate or missing durable receipts")
        expected_notes = 3 + (len(extended_state["note_ids"]) if extended_state else 0)
        actual_notes = len([row for row in rows if row["table"] == "note"])
        require(actual_notes == expected_notes, "Native retries, edits or uploaded jobs duplicated/resurrected notes")
        require(all(row["payload"]["source"]["uri"] == capture["provenance"]["uri"] for row in foundation_receipts),
                "Host export changed opaque client provenance")
        if extended:
            require(len(receipts) == 4, "Extended native capture replay duplicated durable receipts")
            mutations = [row["record"] for row in rows if row["table"] == "remote_mutation_receipt"]
            require(len(mutations) == 3 and all(row["instance_id"] == "writer" for row in mutations),
                    "Native mutation retries/failed preconditions changed durable receipt counts")
            evidence["extended"]["durable_mutation_receipt_count"] = len(mutations)
        archive = directory / "schema-probe-backup"
        run_child([str(options.binary), "--config", str(directory / "config.toml"), "--db-path",
            str(directory / "corpus"), "backup", "create", str(archive), "--format", "json"], None,
            directory, "schema-backup", private_environment(directory, "server"), tokens.values(), timeout())
        manifest = json.loads((archive / "manifest.json").read_text())
        require(isinstance(manifest.get("schema_version"), int) and manifest["schema_version"] > 0,
                "Host archive did not report its actual database schema")
        evidence.update(success=True, openclaw=original, hermes=hermes,
            readonly_native_runtime=readonly, revoked_native_runtime=revoked, service_restart=replay,
            durable_receipt_count=len(receipts), durable_note_count=actual_notes,
            independent_instances=sorted(row["instance_id"] for row in foundation_receipts),
            provider_request_count=len(provider.requests), source_provenance_preserved_by_export=True,
            database_schema_version=manifest["schema_version"], database_schema_evidence="host portable-backup manifest")
    except (Exception, KeyboardInterrupt) as error:
        evidence["error"] = str(error) if isinstance(error, ValidationError) else "Harness interrupted or an unexpected local/runtime error occurred; inspect private sanitized logs"
    finally:
        if server:
            try:
                server.stop(timeout=max(0, min(options.command_timeout, deadline - time.monotonic())))
            except Exception:
                evidence["cleanup_errors"].append("Synthetic service required forced cleanup")
        if provider:
            try:
                provider.close()
            except Exception:
                evidence["cleanup_errors"].append("Synthetic inference fixture cleanup failed")
        # Server logs also remain private; redact every generated credential
        # defensively before retaining any process evidence.
        evidence["cleanup_errors"].extend(sanitize_logs(directory, tokens.values()))
        evidence["success"] = evidence["success"] and not evidence["cleanup_errors"]
        private_write(directory / "evidence.json", json.dumps(evidence, indent=2) + "\n")
        print(json.dumps({"success": evidence["success"], "evidence": str(directory / "evidence.json"),
                          "error": evidence.get("error"), "cleanup_errors": evidence["cleanup_errors"]}))
    return 0 if evidence["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
