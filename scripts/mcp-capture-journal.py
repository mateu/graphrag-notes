#!/usr/bin/env python3
"""Private durable capture boundary for the stateless GraphRAG HTTP MCP service.

Python 3.11+, macOS/Linux, standard library only. No database ownership, model
calls, credential persistence, or automatic recovery on startup.
"""
import argparse
import contextlib
import datetime
import fcntl
import hashlib
import http.client
import http.server
import ipaddress
import json
import os
from pathlib import Path
import re
import stat
import sys
import threading
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import uuid

MAX_REQUEST = 256 * 1024
MAX_RESPONSE = 2 * 1024 * 1024
MAX_ENTRY = 512 * 1024
DIGEST = re.compile(r"[0-9a-f]{64}\Z")
FORWARD_HEADERS = {"authorization", "accept", "content-type", "mcp-protocol-version",
                   "mcp-session-id", "mcp-method", "last-event-id"}
RESPONSE_HEADERS = {"content-type", "mcp-session-id", "mcp-protocol-version", "retry-after"}


class JournalError(Exception):
    """Static codes only; never interpolate requests, credentials, or exceptions."""


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def decode(raw):
    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise JournalError("duplicate_json_key_refused")
            result[key] = value
        return result
    def invalid_constant(_):
        raise JournalError("nonfinite_json_number_refused")
    return json.loads(raw, object_pairs_hook=object_pairs, parse_constant=invalid_constant)


def digest(value):
    return hashlib.sha256(encode(value)).hexdigest()


def timestamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")


def endpoint(value):
    try:
        url = urllib.parse.urlsplit(value)
        host = url.hostname
        port = url.port
        loopback = host == "localhost"
        if host and not loopback:
            try:
                loopback = ipaddress.ip_address(host).is_loopback
            except ValueError:
                pass
        if not host or url.username is not None or url.password is not None or url.query or url.fragment:
            raise ValueError()
        if url.scheme != "https" and not (url.scheme == "http" and loopback):
            raise ValueError()
        host = host.lower()
        if ":" in host:
            host = "[" + host + "]"
        if port is not None and port != (443 if url.scheme == "https" else 80):
            host += ":" + str(port)
        return urllib.parse.urlunsplit((url.scheme.lower(), host, "/mcp" if url.path in {"", "/"} else url.path, "", ""))
    except (ValueError, TypeError):
        raise JournalError("invalid_endpoint_use_https_or_loopback_tunnel") from None


def identity(value):
    if (not isinstance(value, str) or not value or value.strip() != value or len(value) > 128
            or len(value.encode("utf-8")) > 256 or any(unicodedata.category(c) == "Cc" for c in value)):
        raise JournalError("invalid_identity")
    return value


def capture_arguments(arguments):
    if not isinstance(arguments, dict) or set(arguments) - {"request_id", "content", "title", "tags", "provenance"}:
        raise JournalError("invalid_capture_arguments")
    identity(arguments.get("request_id"))
    content = arguments.get("content")
    if not isinstance(content, str) or not content.strip() or len(content.encode("utf-8")) > 65536:
        raise JournalError("invalid_capture_content")
    title = arguments.get("title")
    tags = arguments.get("tags", [])
    if title is not None and (not isinstance(title, str) or len(title) > 512):
        raise JournalError("invalid_capture_title")
    if not isinstance(tags, list) or len(tags) > 32 or any(not isinstance(t, str) or not t.strip() or len(t) > 64 for t in tags):
        raise JournalError("invalid_capture_tags")
    # The upstream service remains authoritative for bounded provenance validation.
    if arguments.get("provenance") is not None and not isinstance(arguments["provenance"], dict):
        raise JournalError("invalid_capture_provenance")
    return arguments


def journal_id(server, principal, request_id):
    return digest([server, principal, request_id])


class Journal:
    def __init__(self, directory):
        directory = Path(directory).expanduser().absolute()
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        info = os.fstat(self.fd)
        if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o700:
            os.close(self.fd)
            raise JournalError("journal_directory_must_be_owned_mode_0700")

    def close(self):
        os.close(self.fd)

    @staticmethod
    def name(entry_id):
        if not isinstance(entry_id, str) or not DIGEST.fullmatch(entry_id):
            raise JournalError("invalid_journal_id")
        return entry_id + ".json"

    def private_open(self, name, create=False):
        flags = (os.O_RDWR if create else os.O_RDONLY) | os.O_NOFOLLOW | os.O_NONBLOCK
        if create:
            flags |= os.O_CREAT
        fd = os.open(name, flags, 0o600, dir_fd=self.fd)
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o600:
            os.close(fd)
            raise JournalError("journal_file_must_be_owned_private_regular")
        return fd

    @contextlib.contextmanager
    def lock(self, entry_id):
        self.name(entry_id)
        fd = self.private_open(entry_id + ".lock", create=True)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            os.close(fd)

    def read(self, entry_id):
        try:
            fd = self.private_open(self.name(entry_id))
        except FileNotFoundError:
            return None
        with os.fdopen(fd, "rb") as source:
            raw = source.read(MAX_ENTRY + 1)
        if len(raw) > MAX_ENTRY:
            raise JournalError("journal_entry_too_large")
        try:
            entry = decode(raw)
            args = capture_arguments(entry["arguments"])
            valid = (type(entry["schema_version"]) is int and entry["schema_version"] == 1 and entry["journal_id"] == entry_id
                     and entry["endpoint"] == endpoint(entry["endpoint"])
                     and entry["instance_id"] == identity(entry["instance_id"])
                     and entry["payload_sha256"] == digest(args)
                     and entry_id == journal_id(entry["endpoint"], entry["instance_id"], args["request_id"])
                     and entry["state"] in {"pending", "uncertain", "receipt_known", "verified"})
            if not valid:
                raise JournalError("journal_identity_or_payload_mismatch")
            return entry
        except (ValueError, TypeError, KeyError, UnicodeError):
            raise JournalError("invalid_journal_entry") from None

    def write(self, entry):
        raw = encode(entry)
        if len(raw) > MAX_ENTRY:
            raise JournalError("journal_entry_too_large")
        temporary = ".pending-" + uuid.uuid4().hex
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=self.fd)
        try:
            with os.fdopen(fd, "wb") as output:
                output.write(raw)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, self.name(entry["journal_id"]), src_dir_fd=self.fd, dst_dir_fd=self.fd)
            os.fsync(self.fd)
        finally:
            try:
                os.unlink(temporary, dir_fd=self.fd)
            except FileNotFoundError:
                pass

    def begin(self, server, principal, arguments):
        capture_arguments(arguments)
        entry_id = journal_id(server, principal, arguments["request_id"])
        prior = self.read(entry_id)
        if prior:
            if prior["arguments"] != arguments or prior["payload_sha256"] != digest(arguments):
                raise JournalError("request_id_payload_mismatch_keep_original_draft")
            return prior
        entry = {"schema_version": 1, "journal_id": entry_id, "endpoint": server,
                 "instance_id": principal, "arguments": arguments, "payload_sha256": digest(arguments),
                 "state": "pending", "created_at": timestamp(), "updated_at": timestamp(),
                 "receipt": None, "verified_at": None, "error_code": None}
        self.write(entry)
        return entry

    def transition(self, entry, state, error=None):
        entry["state"], entry["error_code"], entry["updated_at"] = state, error, timestamp()
        self.write(entry)

    def summaries(self):
        result = []
        for name in sorted(os.listdir(self.fd)):
            if name.endswith(".json") and DIGEST.fullmatch(name[:-5]):
                # Atomic replacement makes each snapshot consistent without
                # waiting for a long in-flight capture's identity lock.
                entry = self.read(name[:-5])
                if entry is not None:
                    result.append(summary(entry))
        return result

    def cleanup(self, entry_id):
        with self.lock(entry_id):
            entry = self.read(entry_id)
            if entry is None or entry["state"] != "verified":
                raise JournalError("cleanup_requires_verified_entry")
            os.unlink(self.name(entry_id), dir_fd=self.fd)
            os.fsync(self.fd)


def summary(entry):
    record = (entry.get("receipt") or {}).get("record", {})
    return {"journal_id": entry["journal_id"], "state": entry["state"],
            "independently_verified": entry["state"] == "verified", "instance_id": entry["instance_id"],
            "request_id": entry["arguments"]["request_id"], "record_id": record.get("id"),
            "revision": record.get("revision"), "created_at": entry["created_at"],
            "updated_at": entry["updated_at"], "verified_at": entry["verified_at"], "error_code": entry["error_code"]}


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class Upstream:
    def __init__(self, server, timeout=300):
        self.server = endpoint(server)
        self.timeout = timeout
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())

    def send(self, body, headers, method="POST"):
        selected = {k: v for k, v in headers.items() if k.lower() in FORWARD_HEADERS}
        request = urllib.request.Request(self.server, data=body, headers=selected, method=method)
        try:
            try:
                response = self.opener.open(request, timeout=self.timeout)
            except urllib.error.HTTPError as error:
                response = error
            with response:
                raw = response.read(MAX_RESPONSE + 1)
                if len(raw) > MAX_RESPONSE:
                    raise JournalError("upstream_response_too_large")
                return response.status, dict(response.headers), raw
        except (OSError, urllib.error.URLError, TimeoutError, http.client.HTTPException):
            raise JournalError("upstream_response_uncertain_recover_exact_draft") from None

    def call(self, tool, arguments, headers, meta=None):
        params = {"name": tool, "arguments": arguments}
        if meta is not None:
            params["_meta"] = meta
        selected = dict(headers)
        selected.update({"Content-Type": "application/json", "Accept": "application/json", "Mcp-Method": "tools/call"})
        return self.send(encode({"jsonrpc": "2.0", "id": "journal-check", "method": "tools/call", "params": params}), selected)


def tool_data(response, expected_id="journal-check"):
    status, _, body = response
    try:
        value = decode(body)
        result = value["result"]
        envelope = result["structuredContent"]
        if status != 200 or value.get("id") != expected_id or value.get("jsonrpc") != "2.0" or result.get("isError", False) or type(envelope["schema_version"]) is not int or envelope["schema_version"] != 1 or envelope["error"] is not None:
            raise JournalError("upstream_tool_not_successful")
        return envelope["data"]
    except (ValueError, TypeError, KeyError, UnicodeError, AttributeError):
        raise JournalError("incompatible_upstream_result") from None


def authenticated_principal(upstream, headers, expected, meta=None):
    value = tool_data(upstream.call("service_status", {}, headers, meta))
    if (not isinstance(value, dict) or type(value.get("schema_version")) is not int or value.get("schema_version") != 1 or value.get("read_only") is not True
            or value.get("inference_probed") is not False or value.get("instance_id") != expected):
        raise JournalError("authenticated_principal_mismatch_check_endpoint_and_credential")
    return expected


def receipt_matches(receipt, entry):
    args = entry["arguments"]
    try:
        record = receipt["record"]
        if (receipt["request_id"] != args["request_id"] or not isinstance(receipt["replayed"], bool)
                or record["content"] != args["content"] or record["tags"] != args.get("tags", [])
                or (args.get("title") is not None and record["title"] != args["title"])
                or not re.fullmatch(r"note:[A-Za-z0-9_\-]+", record["id"])
                or not DIGEST.fullmatch(record["revision"])
                or record["provenance"]["instance_id"] != entry["instance_id"]
                or record["provenance"].get("source") != args.get("provenance")):
            raise JournalError("capture_receipt_mismatch_keep_original_draft")
        # Replays must return the same immutable original receipt, even if the note
        # has since changed. Readback of its pinned revision then fails explicitly.
        prior = entry.get("receipt")
        if prior and (prior["request_id"] != receipt["request_id"] or prior["record"] != record):
            raise JournalError("capture_original_receipt_changed")
    except (TypeError, KeyError):
        raise JournalError("incompatible_capture_receipt_keep_original_draft") from None


class CaptureBoundary:
    def __init__(self, journal, upstream, principal):
        self.journal, self.upstream, self.principal = journal, upstream, identity(principal)

    def submit(self, request, headers):
        args = capture_arguments(request["params"]["arguments"])
        entry_id = journal_id(self.upstream.server, self.principal, args["request_id"])
        with self.journal.lock(entry_id):
            meta = request["params"].get("_meta")
            authenticated_principal(self.upstream, headers, self.principal, meta)
            entry = self.journal.begin(self.upstream.server, self.principal, args)
            # fsync this state before transmission. A crash anywhere after this
            # point leaves the exact original payload explicitly recoverable.
            self.journal.transition(entry, "uncertain")
            try:
                response = self.upstream.send(encode(request), headers)
            except JournalError as error:
                self.journal.transition(entry, "uncertain", str(error))
                raise
            try:
                receipt = tool_data(response, request["id"])
            except JournalError as error:
                self.journal.transition(entry, "uncertain", str(error))
                return response
            try:
                receipt_matches(receipt, entry)
            except JournalError as error:
                self.journal.transition(entry, "uncertain", str(error))
                raise
            entry["receipt"] = receipt
            self.journal.transition(entry, "receipt_known")
            self.verify(entry, headers, meta)
            return response

    def verify(self, entry, headers, meta=None):
        record = entry["receipt"]["record"]
        try:
            value = tool_data(self.upstream.call("get_record", {"id": record["id"], "revision": record["revision"], "neighbors": 0}, headers, meta))
            if any(value.get(field) != record.get(field) for field in ("id", "revision", "title", "content", "provenance")):
                raise JournalError("revision_pinned_readback_mismatch")
        except (JournalError, AttributeError):
            self.journal.transition(entry, "receipt_known", "independent_readback_failed_recover_explicitly")
            return
        entry["verified_at"] = timestamp()
        self.journal.transition(entry, "verified")

    def recover(self, entry_id, headers):
        # Read within the same lock used by submission; submit acquires it again
        # only after checking immutable binding, so no payload editing is allowed.
        with self.journal.lock(entry_id):
            entry = self.journal.read(entry_id)
            if entry is None:
                raise JournalError("journal_entry_not_found")
            if entry["endpoint"] != self.upstream.server or entry["instance_id"] != self.principal:
                raise JournalError("recovery_binding_mismatch_no_submission")
            arguments = entry["arguments"]
        request = {"jsonrpc": "2.0", "id": "journal-recovery", "method": "tools/call", "params": {"name": "capture_note", "arguments": arguments}}
        self.submit(request, headers)
        with self.journal.lock(entry_id):
            return summary(self.journal.read(entry_id))


def failure_response(request_id, code):
    return encode({"jsonrpc": "2.0", "id": request_id, "error": {"code": -32000,
                   "message": code, "data": {"guidance": "Keep the original draft; use the private journal list/recover commands."}}})


def proxy_server(boundary, port):
    class BoundedServer(http.server.ThreadingHTTPServer):
        daemon_threads = True

        def __init__(self, *args):
            self.slots = threading.BoundedSemaphore(16)
            super().__init__(*args)

        def get_request(self):
            connection, address = super().get_request()
            connection.settimeout(10)
            return connection, address

        def process_request(self, request, client_address):
            if not self.slots.acquire(blocking=False):
                self.shutdown_request(request)
                return
            try:
                super().process_request(request, client_address)
            except BaseException:
                self.slots.release()
                raise

        def process_request_thread(self, request, client_address):
            try:
                super().process_request_thread(request, client_address)
            finally:
                self.slots.release()

    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_):
            pass

        def dispatch(self):
            request_id = None
            try:
                if self.path != "/mcp" or self.headers.get("Origin") is not None:
                    raise JournalError("loopback_mcp_route_required_no_browser_origin")
                if self.headers.get("Host") not in {"127.0.0.1:" + str(self.server.server_port), "localhost:" + str(self.server.server_port)}:
                    raise JournalError("loopback_host_required")
                authorization = self.headers.get("Authorization", "")
                if not authorization.startswith("Bearer ") or not authorization[7:] or any(c.isspace() for c in authorization[7:]):
                    raise JournalError("bearer_credential_required")
                if self.headers.get("Transfer-Encoding"):
                    raise JournalError("content_length_required")
                length = int(self.headers.get("Content-Length", "0"))
                if length < 0 or length > MAX_REQUEST:
                    raise JournalError("request_too_large")
                body = self.rfile.read(length) if length else None
                if body is not None and len(body) != length:
                    raise JournalError("incomplete_request_body")
                request = decode(body) if body else None
                if body and not isinstance(request, dict):
                    raise JournalError("single_jsonrpc_object_required")
                if isinstance(request, dict):
                    request_id = request.get("id")
                is_capture = isinstance(request, dict) and request.get("method") == "tools/call" and isinstance(request.get("params"), dict) and request["params"].get("name") == "capture_note"
                if is_capture:
                    if request.get("jsonrpc") != "2.0" or type(request_id) not in {str, int} or self.command != "POST":
                        raise JournalError("capture_requires_jsonrpc_request_id")
                    response = boundary.submit(request, dict(self.headers))
                else:
                    # No added tools or broader catalog: upstream policy and
                    # authorization remain authoritative for every forwarded RPC.
                    response = boundary.upstream.send(body, dict(self.headers), self.command)
                status_code, headers, result = response
            except JournalError as error:
                self.close_connection = True
                status_code, headers, result = 200, {"Content-Type": "application/json"}, failure_response(request_id, str(error))
            except Exception:
                self.close_connection = True
                status_code, headers, result = 200, {"Content-Type": "application/json"}, failure_response(request_id, "journal_or_request_failure_no_new_identity")
            self.send_response(status_code)
            for name, value in headers.items():
                if name.lower() in RESPONSE_HEADERS:
                    self.send_header(name, value)
            self.send_header("Content-Length", str(len(result)))
            self.end_headers()
            try:
                self.wfile.write(result)
            except (BrokenPipeError, ConnectionResetError):
                pass

        do_POST = dispatch
        do_GET = dispatch
        do_DELETE = dispatch

    return BoundedServer(("127.0.0.1", port), Handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--journal-dir", type=Path, default=Path.home() / ".graphrag/capture-journal")
    commands = parser.add_subparsers(dest="command", required=True)
    proxy = commands.add_parser("proxy", help="Forward native MCP calls; journal captures before submission")
    proxy.add_argument("--server", required=True)
    proxy.add_argument("--instance-id", required=True)
    proxy.add_argument("--port", type=int, default=3301)
    commands.add_parser("list", help="List private recovery metadata without note text")
    recover = commands.add_parser("recover", help="Explicitly replay an original journal payload and independently verify")
    recover.add_argument("journal_id")
    recover.add_argument("--server", required=True)
    recover.add_argument("--instance-id", required=True)
    recover.add_argument("--credential-env", default="GRAPHRAG_TOKEN")
    cleanup = commands.add_parser("cleanup", help="Explicitly remove one independently verified local journal entry")
    cleanup.add_argument("journal_id")
    cleanup.add_argument("--yes", action="store_true", required=True)
    options = parser.parse_args(argv)
    journal = None
    try:
        journal = Journal(options.journal_dir)
        if options.command == "proxy":
            boundary = CaptureBoundary(journal, Upstream(options.server), options.instance_id)
            with proxy_server(boundary, options.port) as server:
                print("Capture journal proxy ready at http://127.0.0.1:" + str(server.server_port)
                      + "/mcp; startup never replays drafts.", flush=True)
                server.serve_forever()
        elif options.command == "list":
            print(json.dumps({"schema_version": 1, "entries": journal.summaries()}, indent=2))
        elif options.command == "recover":
            token = os.environ.get(options.credential_env, "")
            if not token or any(c.isspace() for c in token):
                raise JournalError("credential_environment_missing_or_invalid")
            headers = {"Authorization": "Bearer " + token, "MCP-Protocol-Version": "2025-11-25"}
            result = CaptureBoundary(journal, Upstream(options.server), options.instance_id).recover(options.journal_id, headers)
            print(json.dumps(result, indent=2))
            return 0 if result["independently_verified"] else 2
        else:
            journal.cleanup(options.journal_id)
            print(json.dumps({"journal_id": options.journal_id, "cleaned": True}))
        return 0
    except KeyboardInterrupt:
        return 0
    except JournalError as error:
        print(str(error), file=sys.stderr)
        return 2
    except Exception:
        print("journal_storage_or_configuration_failure_keep_original_draft", file=sys.stderr)
        return 2
    finally:
        if journal is not None:
            journal.close()


if __name__ == "__main__":
    sys.exit(main())
