#!/usr/bin/env python3
"""Fail-closed, read-only remote deployment check; never render corpus output."""
import argparse
import ipaddress
import json
import math
import os
import re
import subprocess
import sys
from urllib.parse import urlsplit


class ProbeFailure(Exception):
    pass


class Parser(argparse.ArgumentParser):
    def error(self, message):
        # argparse's default includes raw arguments, which may contain private text.
        self.exit(2, "FAIL setup: invalid arguments (see --help)\n")


def endpoint(value):
    try:
        url = urlsplit(value)
        loopback = url.hostname == "localhost"
        if not loopback:
            try:
                loopback = ipaddress.ip_address(url.hostname).is_loopback
            except ValueError:
                pass
        if (url.scheme not in ("http", "https") or not url.hostname
                or url.username is not None or url.password is not None
                or url.query or url.fragment or url.path != "/mcp"
                or (url.scheme == "http" and not loopback)):
            raise ValueError
        _ = url.port
    except (ValueError, TypeError):
        raise ProbeFailure("invalid endpoint") from None
    return value


def run_cli(base, arguments, timeout, allowed_codes=(0,)):
    try:
        result = subprocess.run(base + arguments, stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        raise ProbeFailure("client timeout") from None
    except OSError:
        raise ProbeFailure("CLI could not start") from None
    if result.returncode not in allowed_codes:
        # Do not forward stderr: it can contain query recovery commands or IDs.
        raise ProbeFailure("CLI returned nonzero")
    return result.stdout


def read_data(raw, command):
    try:
        outer = json.loads(raw)
        if (not isinstance(outer, dict) or type(outer.get("schema_version")) is not int
                or outer["schema_version"] != 1 or outer.get("success") is not True
                or outer.get("command") != command or outer.get("errors") != []):
            raise ValueError
        envelope = outer["data"]
        if (not isinstance(envelope, dict)
                or type(envelope.get("schema_version")) is not int
                or envelope["schema_version"] != 1 or "error" not in envelope
                or envelope["error"] is not None or not isinstance(envelope["data"], dict)):
            raise ValueError
        return envelope["data"]
    except (ValueError, KeyError, TypeError, UnicodeError):
        raise ProbeFailure("invalid or unsuccessful JSON response") from None


def top_note(data):
    records = data.get("records")
    if not isinstance(records, list) or not records:
        raise ProbeFailure("no notes returned")
    top = records[0]
    if (not isinstance(top, dict) or top.get("hit_type") != "note"
            or not isinstance(top.get("id"), str) or not top["id"].startswith("note:")
            or not top["id"][5:].strip()):
        raise ProbeFailure("invalid top note")
    return top["id"]


def main(argv=None):
    parser = Parser(description=__doc__)
    parser.add_argument("--server", required=True, help="HTTPS /mcp endpoint or loopback tunnel")
    parser.add_argument("--query", required=True, help="known-existing keyword for the deployed corpus")
    parser.add_argument("--cli", default="graphrag", help="CLI executable path (not a shell command)")
    parser.add_argument("--credential-env", default="GRAPHRAG_TOKEN", help="inherited token variable name")
    parser.add_argument("--timeout", type=float, default=15,
                        help="client wall-clock timeout in seconds per command (default: 15)")
    args = parser.parse_args(argv)
    step = "setup"
    try:
        endpoint(args.server)
        if not math.isfinite(args.timeout) or args.timeout <= 0 or not args.query.strip():
            raise ProbeFailure("invalid query or timeout")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", args.credential_env):
            raise ProbeFailure("invalid credential environment variable name")
        token = os.environ.get(args.credential_env, "")
        if not token.strip() or "\r" in token or "\n" in token:
            raise ProbeFailure("credential environment variable missing or invalid")
        base = [args.cli, "--server", args.server, "--credential-env", args.credential_env]
        step = "doctor"
        raw = run_cli(base, ["doctor", "--format", "json"], args.timeout, (0, 1))
        try:
            report = json.loads(raw)
            if (report.get("schema_version") != 1 or report.get("exit_code") not in (0, 1)
                    or report.get("read_only") is not True):
                raise ValueError
            print(f"doctor: exit {report['exit_code']}")
            checks = {"transport": report["transport"], "authorization": report["authorization"],
                      **{name: report["application"][name] for name in
                         ("storage", "embeddings", "extraction", "sources", "backup")},
                      "jobs": report["jobs"]["readiness"]}
            for name, check in checks.items():
                state, summary = check["state"], check["summary"]
                if not isinstance(state, str) or not isinstance(summary, str):
                    raise ValueError
                # Doctor summaries are provider-free readiness evidence, not corpus output.
                safe_summary = "".join(c for c in summary if c.isprintable())
                print(f"doctor {name}: {state}: {safe_summary}")
        except (ValueError, KeyError, TypeError, AttributeError, UnicodeError):
            raise ProbeFailure("invalid doctor response") from None
        step = "keyword_graph_off"
        search = ["search", "--scope", "notes", "--limit", "1", "--format", "json"]
        raw = run_cli(base, search + ["--mode", "keyword", "--graph", "off", "--", args.query], args.timeout)
        identifier = top_note(read_data(raw, "search_notes"))
        step = "inspect_top_hit"
        raw = run_cli(base, ["inspect", identifier, "--format", "json"], args.timeout)
        inspected = read_data(raw, "get_record")
        if inspected.get("id") != identifier or inspected.get("hit_type") != "note":
            raise ProbeFailure("inspection did not match top note")
        step = "hybrid_graph_on"
        raw = run_cli(base, search + ["--mode", "hybrid", "--graph", "on", "--", args.query], args.timeout)
        data = read_data(raw, "search_notes")
        top_note(data)
        if data.get("graph_status") != "ok":
            raise ProbeFailure("graph_status is not ok")
    except ProbeFailure as error:
        print(f"FAIL {step}: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print(f"FAIL {step}: interrupted", file=sys.stderr)
        return 1
    print("PASS doctor keyword_graph_off inspect_top_hit hybrid_graph_on")
    return 0


if __name__ == "__main__":
    sys.exit(main())
