#!/usr/bin/env python3
"""Opt-in synthetic daily-workflow validation using only the public CLI.

Defaults to deterministic localhost inference. --live uses already available
Ollama models; this script never downloads models or starts installed services.
Each binary gets its own private corpus. Evidence remains local, including on
failure. Automated command/ID forwarding is measured; human copying is not.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import http.server
import json
import math
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from urllib.parse import urlsplit

SCHEMA_VERSION = 1
BASELINE_LABEL = "published v0.1.0-rc.1 onboarding baseline before U2-U7"
SAMPLES = Path(__file__).resolve().parents[1] / "samples" / "daily-workflow"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def canonical_id(value, table):
    """Read actual CLI RecordId serialization, including the older envelope."""
    if isinstance(value, str) and value.startswith(table + ":"):
        return value
    if isinstance(value, dict):
        if value.get("tb", value.get("table")) == table:
            key = value.get("id", value.get("key"))
            if isinstance(key, str):
                return table + ":" + key
            if isinstance(key, dict):
                for kind in ("String", "string", "Uuid", "uuid"):
                    if isinstance(key.get(kind), str):
                        return table + ":" + key[kind]
        for key in ("id", "note", "proposal", "edge", "resulting_edge_id"):
            result = canonical_id(value.get(key), table)
            if result:
                return result
    return None


class OfflineProvider:
    def __init__(self):
        self.requests = []
        self.lock = threading.Lock()
        self.pause_next_embedding = False
        self.entered = threading.Event()
        self.release = threading.Event()
        owner = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_GET(self):
                self.respond()

            def do_POST(self):
                self.respond()

            def respond(self):
                length = int(self.headers.get("Content-Length", "0"))
                if length > 16 * 1024 * 1024:
                    self.send_error(413)
                    return
                body = self.rfile.read(length).decode("utf-8")
                embedding = self.path in ("/api/embed", "/api/embeddings")
                with owner.lock:
                    owner.requests.append({"method": self.command, "path": self.path})
                    pause = embedding and owner.pause_next_embedding
                    if pause:
                        owner.pause_next_embedding = False
                if pause:
                    owner.entered.set()
                    if not owner.release.wait(30):
                        self.send_error(503, "bounded fixture pause expired")
                        return
                vector = [1.0] + [0.0] * 1023
                if self.path == "/api/tags":
                    data = {"models": [{"name": "bge-m3:latest"}, {"name": "phi4-mini:latest"}]}
                elif self.path == "/api/embed":
                    inputs = json.loads(body)["input"]
                    data = {"embeddings": [vector] * (len(inputs) if isinstance(inputs, list) else 1)}
                elif self.path == "/api/embeddings":
                    data = {"embedding": vector}
                elif self.path == "/api/chat":
                    data = {"message": {"role": "assistant", "content": '{"entities":[],"relationships":[]}'}, "done": True}
                else:
                    self.send_error(404, "unexpected deterministic provider route")
                    return
                payload = json.dumps(data).encode()
                try:
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.send_header("Connection", "close")
                    self.end_headers()
                    self.wfile.write(payload)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.endpoint = f"http://127.0.0.1:{self.server.server_port}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def arm(self):
        self.entered.clear()
        self.release.clear()
        with self.lock:
            self.pause_next_embedding = True

    def close(self):
        self.release.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(3)


class Workflow:
    def __init__(self, binary, label, options, provider):
        self.binary = Path(binary).expanduser().resolve()
        self.binary_hash = None
        if self.binary.is_file():
            digest = hashlib.sha256()
            with self.binary.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            self.binary_hash = digest.hexdigest()
        self.label = label
        self.options = options
        self.provider = provider
        self.started = time.monotonic()
        self.finished = None
        self.deadline = self.started + options.deadline_seconds
        self.directory = Path(tempfile.mkdtemp(prefix="graphrag-daily-workflow-")).resolve()
        self.config = self.directory / "configuration.toml"
        self.db = self.directory / "corpus.surreal"
        self.folder = self.directory / "notes"
        self.folder.mkdir(mode=0o700)
        for sample in sorted(SAMPLES.glob("*.md")):
            shutil.copyfile(sample, self.folder / sample.name)
        require(len(list(self.folder.glob("*.md"))) >= 3, "sanitized daily-workflow samples are missing")
        self.commands = []
        self.steps = []
        self.transfers = []
        self.active_step = None
        self.error = None
        self.success = False
        self.env = {"HOME": str(self.directory / "home"), "XDG_CONFIG_HOME": str(self.directory / "xdg"),
                    "PATH": "/usr/bin:/bin", "TERM": "dumb"}
        Path(self.env["HOME"]).mkdir(mode=0o700)
        Path(self.env["XDG_CONFIG_HOME"]).mkdir(mode=0o700)
        self.open_log = self.directory / "opened.json"
        self.opener = self.directory / "literal-opener.py"
        self.opener.write_text("import json,pathlib,sys\npathlib.Path(sys.argv[1]).write_text(json.dumps(sys.argv[2:]))\n")
        self.config_text = self.settings(provider.endpoint if provider else options.ollama_url)
        self.config.write_text(self.config_text)
        self.config.chmod(0o600)

    def settings(self, endpoint):
        quote = lambda value: json.dumps(str(value), ensure_ascii=False)
        return f'''[database]
path = {quote(self.db)}
[inference]
embedding_provider = "ollama"
embedding_url = {quote(endpoint)}
embedding_model = "bge-m3:latest"
extraction_provider = "ollama"
extraction_url = {quote(endpoint)}
extraction_model = "phi4-mini:latest"
ollama_url = {quote(endpoint)}
retry_attempts = 1
cache_enabled = false
processing_concurrency = 1
timeout_secs = 30
ollama_timeout_secs = 120
[gardener]
similarity_threshold = 0.0
auto_apply = false
[logging]
level = "info"
[navigation]
opener = {json.dumps([sys.executable, str(self.opener), str(self.open_log), "literal $(touch injected-marker) `id`"], ensure_ascii=False)}
'''

    def remaining(self):
        value = min(self.options.command_timeout, self.deadline - time.monotonic())
        require(value > 0, "workflow deadline expired; evidence retained")
        return value

    def argv(self, args, config=None):
        return [str(self.binary), "--config", str(config or self.config), "--db-path", str(self.db), *map(str, args)]

    def save(self):
        (self.directory / "metrics.json").write_text(json.dumps(self.result(), indent=2) + "\n")

    def result(self):
        return {"label": self.label, "binary": str(self.binary), "evidence_directory": str(self.directory),
                "binary_sha256": self.binary_hash,
                "success": self.success, "error": self.error, "elapsed_seconds": round((self.finished or time.monotonic()) - self.started, 3),
                "actual_cli_command_count": sum(command["executed"] for command in self.commands),
                "cli_subprocess_count": sum(command["executed"] for command in self.commands),
                "workspace_input_count": sum(command.get("workspace_input_count", 0) for command in self.commands),
                "command_attempt_count": len(self.commands), "automated_canonical_id_transfer_count": len(self.transfers),
                "canonical_id_transfers": self.transfers, "observed_human_manual_id_copies": None,
                "observed_human_elapsed_seconds": None, "installation_execution_measured": False,
                "steps": self.steps, "commands": self.commands}

    @contextmanager
    def task(self, name, availability="available", note=None):
        step = {"name": name, "availability": availability, "success": False,
                "note": note, "start_command_index": len(self.commands), "next_action_explained": None}
        self.steps.append(step)
        self.active_step = step
        started = time.monotonic()
        transfer_start = len(self.transfers)
        try:
            yield step
            step["success"] = True
        except Exception as error:
            step["error"] = str(error)
            raise
        finally:
            step["elapsed_seconds"] = round(time.monotonic() - started, 3)
            step["actual_cli_command_count"] = sum(command["executed"] for command in self.commands[step["start_command_index"]:])
            step["cli_subprocess_count"] = step["actual_cli_command_count"]
            step["workspace_input_count"] = sum(command.get("workspace_input_count", 0) for command in self.commands[step["start_command_index"]:])
            step["automated_canonical_id_transfer_count"] = len(self.transfers) - transfer_start
            self.active_step = None
            self.save()

    def unavailable(self, name, reason):
        self.steps.append({"name": name, "availability": "unavailable", "success": None,
                           "note": reason, "actual_cli_command_count": 0, "cli_subprocess_count": 0,
                           "workspace_input_count": 0, "automated_canonical_id_transfer_count": 0,
                           "next_action_explained": None, "elapsed_seconds": 0.0})

    def record(self, label, args, output, elapsed, executed=True, error=None, workspace_inputs=()):
        number = len(self.commands) + 1
        stem = f"{number:03d}-{re.sub('[^a-z0-9-]', '-', label.lower())}"
        stdout = self.directory / (stem + ".stdout")
        stderr = self.directory / (stem + ".stderr")
        stdout.write_text(output.stdout or "")
        stderr.write_text(output.stderr or "")
        record = {"label": label, "task": self.active_step["name"] if self.active_step else None,
                  "argv": list(map(str, args)), "executed": executed, "exit_code": output.returncode,
                  "elapsed_seconds": round(elapsed, 3), "stdout_path": str(stdout), "stderr_path": str(stderr)}
        record["workspace_input_count"] = len(workspace_inputs) if executed else 0
        if workspace_inputs:
            record["workspace_inputs"] = list(workspace_inputs)
        if error:
            record["error"] = error
        self.commands.append(record)
        return output

    def run(self, label, args, *, input=None, code=0, config=None, ids=(), exact=False):
        argv = [str(self.binary), *map(str, args)] if exact else self.argv(args, config)
        workspace_inputs = (input or "").splitlines() if args and args[0] in ("workspace", "interactive") else []
        started = time.monotonic()
        try:
            output = subprocess.run(argv, input=input, text=True, capture_output=True,
                                    timeout=self.remaining(), env=self.env, cwd=self.directory)
        except subprocess.TimeoutExpired as error:
            decode = lambda text: text.decode(errors="replace") if isinstance(text, bytes) else text or ""
            output = subprocess.CompletedProcess(argv, None, decode(error.stdout), decode(error.stderr))
            self.record(label, argv, output, time.monotonic() - started, error="command timeout", workspace_inputs=workspace_inputs)
            raise RuntimeError(f"{label} timed out; inspect retained stdout/stderr") from error
        except OSError as error:
            output = subprocess.CompletedProcess(argv, None, "", str(error))
            self.record(label, argv, output, time.monotonic() - started, executed=False, error=str(error), workspace_inputs=workspace_inputs)
            raise RuntimeError(f"Cannot execute {self.binary}: {error}") from error
        self.record(label, argv, output, time.monotonic() - started, workspace_inputs=workspace_inputs)
        for value, source in ids:
            require(isinstance(value, str) and ":" in value and value in map(str, args), "ID transfer must be present in executed CLI arguments")
            self.transfers.append({"id": value, "from_command": source, "to_command": label,
                                   "transport": "automated argv", "observed_human_copy": False})
        require(output.returncode == code, f"{label} exited {output.returncode}, expected {code}; inspect {self.commands[-1]['stderr_path']}")
        return output

    def data(self, label, args, **kwargs):
        output = self.run(label, args, **kwargs)
        envelope = json.loads(output.stdout)
        if args[0] == "jobs":
            return envelope  # Existing public jobs JSON intentionally has no CLI envelope.
        require(envelope.get("schema_version") == 1, f"{label} did not return schema-version-1 JSON")
        return envelope["data"]

    def find(self, label, *, candidate, uri=None):
        args = ["search", "dailyatlas", "--scope", "notes", "--graph", "off", "--format", "json"]
        if candidate:
            args.extend(["--mode", "keyword"])
        if uri:
            args.extend(["--source-uri", uri])
        data = self.data(label, args)
        require(data["results"], f"{label} found no synthetic notes")
        return data["results"][0]

    def note_id(self):
        data = self.data("legacy-capture-id", ["notes", "list", "--limit", "1", "--format", "json"])
        value = canonical_id(data["notes"][0], "note")
        require(value, "actual legacy notes.list output has no canonical note ID")
        return value

    def interrupt_sync(self):
        """Observe actual CLI import progress after job creation, then SIGINT."""
        args = ["sync", "daily", "--format", "json"]
        argv = self.argv(args)
        started = time.monotonic()
        if self.provider:
            self.provider.arm()
        process = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   env=self.env, cwd=self.directory)
        stdout, stderr = bytearray(), bytearray()
        imported, acknowledged = threading.Event(), threading.Event()

        def read(stream, destination, progress=False):
            while True:
                chunk = stream.read(1)
                if not chunk:
                    return
                destination.extend(chunk)
                if progress:
                    if b"Ingesting markdown from:" in destination:
                        imported.set()
                    if b"Cancellation requested;" in destination:
                        acknowledged.set()

        readers = [threading.Thread(target=read, args=(process.stdout, stdout), daemon=True),
                   threading.Thread(target=read, args=(process.stderr, stderr, True), daemon=True)]
        for reader in readers:
            reader.start()
        error = None
        try:
            require(imported.wait(self.remaining()), "sync never reported first public file import")
            if self.provider:
                require(self.provider.entered.wait(self.remaining()), "sync never reached deterministic embedding pause")
            require(process.poll() is None, "sync completed before interruption; no cancelled measurement claimed")
            process.send_signal(signal.SIGINT)
            require(acknowledged.wait(min(20, self.remaining())), "sync did not acknowledge cancellation")
            if self.provider:
                self.provider.release.set()
            process.wait(timeout=self.remaining())
        except Exception as failure:
            error = str(failure)
            if process.poll() is None:
                process.kill()
            process.wait(10)
        finally:
            if self.provider:
                self.provider.release.set()
            for reader in readers:
                reader.join(5)
            output = subprocess.CompletedProcess(argv, process.returncode, stdout.decode(errors="replace"), stderr.decode(errors="replace"))
            self.record("interrupted-sync", argv, output, time.monotonic() - started, error=error)
        require(error is None, error)
        require(output.returncode == 5, "interrupted sync must report incomplete work with exit 5")
        report = json.loads(output.stdout)["data"]
        require(report["cancelled"] and report["job_id"] and any(row["status"] == "pending" for row in report["files"]), "sync did not preserve a cancelled durable job with pending files")
        require("sync --resume" in output.stderr or any("--resume" in (row.get("retry_command") or "") for row in report["files"]), "interrupted sync omitted a next-action resume hint")
        return report["job_id"]

    def execute(self, candidate):
        if not candidate:
            # Navigation configuration did not exist in the published binary.
            self.config.write_text(self.config_text.split("[navigation]")[0])
        with self.task("binary_and_onboarding_probe", note="Uses supplied binary; installation itself is measured separately"):
            version = self.run("binary-version", ["--version"]).stdout.strip()
            help_text = self.run("binary-help", ["--help"]).stdout
            self.steps[-1]["version"] = version
            if not candidate:
                require(version == "graphrag 0.1.0-rc.1", "--before-binary must be the actual published 0.1.0-rc.1 baseline")
            self.run("onboarding-preview", ["init", "--format", "json"])
            if candidate:
                require(all(re.search(r"^\s+" + command + r"\s", help_text, re.M) for command in ("capture", "folders", "sync", "inspect", "open", "workspace")), "candidate CLI is missing a required U2-U7 command")

        body = "dailyatlas pilot decision: capture the next action and inspect its original source."
        with self.task("capture_and_edit", "available" if candidate else "legacy_equivalent", "Legacy add has no recoverable capture draft" if not candidate else None):
            if candidate:
                note = self.data("capture", ["capture", "--stdin", "--title", "Daily pilot decision", "--format", "json"], input=body)["id"]
                source = "capture"
            else:
                self.run("legacy-add", ["add", body, "--title", "Daily pilot decision"])
                note = self.note_id()
                source = "legacy-capture-id"
            require(canonical_id(note, "note"), "capture produced no canonical note ID")
            edited = body + " The revised decision keeps the pilot small and reviewable."
            self.data("edit", ["notes", "edit", note, "--stdin", "--format", "json"], input=edited, ids=[(note, source)])
            shown = self.data("show-edit", ["notes", "show", note, "--format", "json"], ids=[(note, source)])
            require(shown["content"] == edited, "saved edit does not match submitted content")
            if candidate:
                self.data("capture-peer", ["capture", edited, "--title", "Related pilot decision", "--format", "json"])
            else:
                self.run("legacy-add-peer", ["add", edited, "--title", "Related pilot decision"])

        if candidate:
            with self.task("capture_provider_failure_and_recovery"):
                broken = self.directory / "temporarily-unavailable.toml"
                broken.write_text(self.settings("http://127.0.0.1:1"))
                failed_body = "dailyatlas recoverable synthetic capture after a documented provider failure."
                failure = self.run("failed-capture", ["capture", failed_body, "--format", "json"], code=1, config=broken)
                recovery_line = next((line.split("Recover:", 1)[1].strip() for line in failure.stderr.splitlines() if "Recover:" in line), None)
                require(recovery_line, "provider failure did not explain a copyable Recover command")
                drafts = list(Path(str(self.db) + ".drafts").iterdir())
                require(len(drafts) == 1 and drafts[0].read_text() == failed_body and drafts[0].stat().st_mode & 0o777 == 0o600, "provider failure lost private recovery input")
                retained = self.data("previous-note-after-failure", ["notes", "show", note, "--format", "json"], ids=[(note, source)])
                require(retained["content"] == edited, "provider failure changed previous content")
                broken.write_text(self.config_text)
                replay = shlex.split(recovery_line)
                require(replay[0] == "graphrag", "unrecognized printed recovery executable")
                recovered = self.run("replay-capture-recovery", replay[1:], exact=True)
                recovered_id = json.loads(recovered.stdout)["data"]["id"]
                saved = self.data("inspect-recovered-capture", ["notes", "show", recovered_id, "--format", "json"], ids=[(recovered_id, "replay-capture-recovery")])
                require(saved["content"] == failed_body, "replayed recovery saved different content")
                require(list(Path(str(self.db) + ".drafts").iterdir()) == drafts, "replayed recovery retained an additional preparation draft")
                self.steps[-1]["original_recovery_input_retained"] = True
                self.steps[-1]["next_action_explained"] = True
        else:
            self.unavailable("capture_provider_failure_and_recovery", "Legacy add has no recoverable capture-draft contract")

        uri = "file://" + str((self.folder / "atlas.md").resolve())
        with self.task("folder_initial_unchanged_and_changed_refresh", "available" if candidate else "legacy_equivalent", "Legacy import/reimport replaces folder registration/sync" if not candidate else None):
            if candidate:
                self.data("register-folder", ["folders", "add", "daily", self.folder, "--format", "json"])
                first = self.data("initial-sync", ["sync", "daily", "--format", "json"])
                require(sum(row["status"] == "created" for row in first["files"]) == 3, "initial sync did not create all three sanitized files")
                old_hit = self.find("find-original-source", candidate=True, uri=uri)
                old_ids = self.data("initial-note-ids", ["notes", "list", "--limit", "100", "--format", "json"])
                requests = len(self.provider.requests) if self.provider else None
                unchanged = self.data("unchanged-sync", ["sync", "daily", "--format", "json"])
                require(unchanged["job_id"] is None and all(row["status"] == "unchanged" for row in unchanged["files"]), "unchanged sync created work")
                require(old_ids == self.data("unchanged-note-ids", ["notes", "list", "--limit", "100", "--format", "json"]), "unchanged sync changed note identities")
                if self.provider:
                    require(len(self.provider.requests) == requests, "unchanged sync contacted inference")
            else:
                self.run("legacy-initial-import", ["import", self.folder / "atlas.md"])
                old_hit = self.find("legacy-find-original", candidate=False, uri=uri)
                self.run("legacy-unchanged-import", ["import", self.folder / "atlas.md"])
            path = self.folder / "atlas.md"
            path.write_text(path.read_text() + "\n## Update\n\ndailyatlas refresh records a revised fictional pilot milestone with source provenance.\n")
            if candidate:
                changed = self.data("changed-sync", ["sync", "daily", "--format", "json"])
                require(sum(row["status"] == "updated" for row in changed["files"]) == 1, "changed sync did not refresh exactly one source")
                old_id = canonical_id(old_hit, "note")
                self.run("stale-original-refused", ["inspect", old_id, "--revision", old_hit["navigation"]["revision"], "--format", "json"], code=3, ids=[(old_id, "find-original-source")])
            else:
                self.run("legacy-changed-reimport", ["sources", "reimport", uri])

        with self.task("find_inspect_and_open", "available" if candidate else "legacy_equivalent", "Legacy search + notes show; original-source open is unavailable" if not candidate else None):
            hit = self.find("find-current-source", candidate=candidate, uri=uri)
            hit_id = canonical_id(hit, "note")
            require(hit_id, "search produced no canonical source note ID")
            if candidate:
                revision = hit["navigation"]["revision"]
                record = self.data("inspect-found-source", ["inspect", hit_id, "--revision", revision, "--format", "json"], ids=[(hit_id, "find-current-source")])
                require(record["provenance"]["source_uri"] == uri, "inspection changed source identity")
                opened = self.data("open-found-source", ["open", hit_id, "--revision", revision, "--format", "json"], ids=[(hit_id, "find-current-source")])
                require(opened["launched"] and json.loads(self.open_log.read_text()) == ["literal $(touch injected-marker) `id`", str(self.folder / "atlas.md")], "source opener did not receive literal argv")
                require(not (self.directory / "injected-marker").exists(), "opener unexpectedly executed a shell")
            else:
                self.data("legacy-show-found-note", ["notes", "show", hit_id, "--format", "json"], ids=[(hit_id, "find-current-source")])
        if not candidate:
            self.unavailable("provider_free_keyword_and_source_open", "Published binary has no --mode keyword, inspect, or open commands")
        else:
            with self.task("terminal_workspace_selection", note="Six entered workspace lines are counted separately from its one subprocess; no canonical ID is typed"):
                requests = len(self.provider.requests) if self.provider else None
                output = self.run("workspace-browse", ["workspace"], input="keyword dailyatlas refresh\nselect 1\ninspect\nopen\ncopy\nquit\n")
                require("Selected result 1" in output.stdout and "Copied context:" in output.stderr and "Error:" not in output.stdout, "workspace did not inspect/open/copy its selected result")
                if self.provider:
                    require(len(self.provider.requests) == requests, "offline workspace contacted inference")
        if not candidate:
            self.unavailable("terminal_workspace_selection", "Published binary has the prior thin interactive shell, not the U7 selection workspace")

        with self.task("reuse_augmentation", "available" if candidate else "legacy_equivalent", "Legacy human augmentation is available; --raw is unavailable" if not candidate else None):
            args = ["augment", "dailyatlas pilot decision", "--graph", "off", "--limit", "3", "--max-tokens", "600"]
            if candidate:
                args.append("--raw")
            augmented = self.run("reuse-context", args)
            require("[C1]" in augmented.stdout, "augmentation produced no cited synthetic context")
            if candidate:
                require(not any(header in augmented.stdout for header in ("Augmentation Context", "Context sources:", "Packing diagnostics")), "raw stdout contains CLI diagnostics")
            (self.directory / "reusable-context.txt").write_text(augmented.stdout)

        with self.task("review_accept_and_undo", "available" if candidate else "legacy_equivalent", "Legacy proposal list/show replaces side-by-side inbox cards" if not candidate else None):
            self.run("scan-links", ["garden", "scan"])
            if candidate:
                cards = self.data("review-links", ["garden", "review", "--format", "json"])["proposals"]
                require(cards, "no synthetic link proposals to review")
                proposal = cards[0]["id"]
                for endpoint in ("from", "to"):
                    selected = cards[0][endpoint]
                    self.data("inspect-link-" + endpoint, ["inspect", selected["id"], "--revision", selected["revision"], "--format", "json"], ids=[(selected["id"], "review-links")])
            else:
                listed = self.run("legacy-list-links", ["garden", "proposals", "list"])
                match = re.search(r"proposed_edge:[A-Za-z0-9_-]+", listed.stdout)
                require(match, "legacy proposal list printed no canonical proposal ID")
                proposal = match.group(0)
            self.run("accept-link", ["garden", "proposals", "accept", proposal, "--yes", "--reason", "synthetic daily workflow review"], ids=[(proposal, "review-links" if candidate else "legacy-list-links")])
            if candidate:
                accepted = self.data("accepted-link-audit", ["garden", "review", "--id", proposal, "--format", "json"], ids=[(proposal, "review-links")])["proposals"][0]
                require(accepted["status"] == "accepted" and accepted["acceptance_is_manual"], "acceptance lost manual review audit")
                edge = accepted["resulting_edge_id"]
                edge_source = "accepted-link-audit"
            else:
                audit = self.run("legacy-accepted-audit", ["garden", "proposals", "show", proposal], ids=[(proposal, "legacy-list-links")])
                match = re.search(r"Accepted edge:\s+([a-z_]+:[A-Za-z0-9_-]+)", audit.stdout)
                require(match, "legacy accepted audit printed no canonical resulting edge ID")
                edge, edge_source = match.group(1), "legacy-accepted-audit"
            self.run("undo-link", ["edges", "undo", edge, "--yes"], ids=[(edge, edge_source)])
            if candidate:
                retired = self.data("retired-link-audit", ["garden", "review", "--id", proposal, "--format", "json"], ids=[(proposal, "review-links")])["proposals"][0]
                require(retired["status"] == "superseded" and retired["resulting_edge_id"] is None and retired["reviewed_at"] == accepted["reviewed_at"], "undo lost original review audit or retained active edge")
            else:
                self.run("legacy-retired-audit", ["garden", "proposals", "show", proposal], ids=[(proposal, "legacy-list-links")])

        if candidate:
            with self.task("interrupted_folder_work_and_recovery"):
                for index in range(8):
                    (self.folder / f"recover-{index:02}.md").write_text(f"# Recovery {index}\n\ndailyatlas synthetic recovery item {index} records a separate pilot checkpoint and a documented next action.\n")
                job = self.interrupt_sync()
                cancelled = self.data("inspect-cancelled-job", ["jobs", "show", job, "--format", "json"], ids=[(job, "interrupted-sync")])
                require(cancelled["id"] == job and cancelled["status"] == "cancelled", "job inspection did not preserve cancelled identity/status")
                resumed = self.data("resume-folder-work", ["sync", "--resume", job, "--format", "json"], ids=[(job, "interrupted-sync")])
                require(not resumed["cancelled"] and all(row["status"] in ("created", "updated", "unchanged") for row in resumed["files"]), "resume did not complete every pinned file")
                completed = self.data("inspect-completed-job", ["jobs", "show", job, "--format", "json"], ids=[(job, "interrupted-sync")])
                require(completed["id"] == job and completed["status"] == "completed", "resume did not complete its original durable job")
                final = self.data("recovery-unchanged-sync", ["sync", "daily", "--format", "json"])
                require(final["job_id"] is None and all(row["status"] == "unchanged" for row in final["files"]), "recovered work created duplicate pending imports")
                self.steps[-1]["next_action_explained"] = True
        else:
            self.unavailable("interrupted_folder_work_and_recovery", "Published baseline has no folder sync or sync --resume; its generic durable job APIs are not a folder-work equivalent")
        self.success = True
        self.finished = time.monotonic()
        self.save()


def parser():
    result = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    result.add_argument("--binary", required=True, help="candidate CLI executable (source-built or installed)")
    result.add_argument("--before-binary", help="actual published 0.1.0-rc.1 executable; gets a separate corpus")
    mode = result.add_mutually_exclusive_group()
    mode.add_argument("--offline", action="store_true", help="deterministic loopback provider doubles (default)")
    mode.add_argument("--live", action="store_true", help="use existing Ollama bge-m3:latest/phi4-mini:latest; no downloads")
    result.add_argument("--ollama-url", default="http://127.0.0.1:11434", help="existing Ollama endpoint for --live")
    result.add_argument("--report", type=Path, help="also write the local opt-in metrics JSON to this path")
    result.add_argument("--command-timeout", type=float, default=180, help="maximum seconds per CLI execution")
    result.add_argument("--deadline-seconds", type=float, default=1800, help="maximum seconds per binary workflow")
    return result


def main(argv=None):
    arguments = parser().parse_args(argv)
    if not all(math.isfinite(value) and value > 0 for value in (arguments.command_timeout, arguments.deadline_seconds)):
        parser().error("timeouts must be positive")
    if arguments.report and (arguments.report.expanduser().exists() or arguments.report.expanduser().is_symlink()):
        parser().error("--report must name a new file; existing files and symlinks are never replaced")
    endpoint = urlsplit(arguments.ollama_url)
    if endpoint.scheme not in ("http", "https") or not endpoint.hostname or endpoint.username or endpoint.password:
        parser().error("--ollama-url requires an HTTP(S) endpoint without embedded credentials")
    provider = None if arguments.live else OfflineProvider()
    workflows = []
    try:
        for binary, label, candidate in [(arguments.binary, "candidate daily-workflow acceptance", True)] + ([(arguments.before_binary, BASELINE_LABEL, False)] if arguments.before_binary else []):
            workflow = Workflow(binary, label, arguments, provider)
            workflows.append(workflow)
            try:
                workflow.execute(candidate)
            except Exception as error:
                workflow.error = str(error)
                workflow.success = False
                workflow.finished = time.monotonic()
                workflow.save()
            print(f"{'PASS' if workflow.success else 'FAIL'} {label}: {workflow.directory}", flush=True)
        report = {"schema_version": SCHEMA_VERSION, "mode": "live" if arguments.live else "offline",
                  "platform": platform.platform(), "success": all(item.success for item in workflows),
                  "measurement_scope": "scripted CLI acceptance on synthetic corpora; not a human usability study",
                  "comparison_policy": "Compare like tasks and availability; total counts span different feature coverage and are not a usability score",
                  "command_count_policy": "CLI subprocesses and scripted workspace input lines supplied to stdin are counted separately; supplied lines are not observed human actions or proof of execution after a failure; stdin note content is not a workspace command",
                  "observed_human_manual_id_copies": None, "installation_execution_measured": False,
                  "runs": [workflow.result() for workflow in workflows]}
        destination = arguments.report.expanduser().absolute() if arguments.report else workflows[0].directory / "comparison.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(json.dumps(report, indent=2) + "\n")
        except OSError as error:
            print(f"Cannot create new report {destination}: {error}; per-run evidence remains in the printed directories", file=sys.stderr)
            return 1
        print("Local metrics: " + str(destination), flush=True)
        for workflow in workflows:
            if workflow.error:
                print(workflow.label + ": " + workflow.error, file=sys.stderr)
        return 0 if report["success"] else 1
    finally:
        if provider:
            provider.close()


if __name__ == "__main__":
    raise SystemExit(main())
