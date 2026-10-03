#!/usr/bin/env python3
"""Provision private per-instance MCP credentials without printing bearer tokens."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import tempfile


def instance_id(value):
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", value):
        raise ValueError("instance IDs must contain 1–64 letters, digits, dots, dashes or underscores")
    return value


def private_write(path, text):
    fd, temporary = tempfile.mkstemp(prefix=".credential-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def provision(directory, writers, readers):
    writers, readers = [instance_id(i) for i in writers], [instance_id(i) for i in readers]
    identities = writers + readers
    if not identities or len(set(identities)) != len(identities):
        raise ValueError("provide at least one unique client ID")
    directory.mkdir(mode=0o700, parents=True, exist_ok=False)
    policy = {"schema_version": 1, "credentials": []}
    for identity in identities:
        token = secrets.token_urlsafe(32)
        private_write(directory / f"{identity}.env", f"GRAPHRAG_TOKEN='{token}'\n")
        policy["credentials"].append({
            "instance_id": identity,
            "token_sha256": hashlib.sha256(token.encode()).hexdigest(),
            "capabilities": ["read", "capture"] if identity in writers else ["read"],
        })
    private_write(directory / "credentials.json", json.dumps(policy, indent=2) + "\n")


def rotate(directory, identity):
    identity = instance_id(identity)
    metadata = directory.lstat()
    if not directory.is_dir() or directory.is_symlink() or metadata.st_mode & 0o077:
        raise ValueError("credential directory must be a private regular directory (mode 0700)")
    policy_path = directory / "credentials.json"
    if policy_path.is_symlink() or not policy_path.is_file():
        raise ValueError("credential policy must be a regular file")
    policy = json.loads(policy_path.read_text())
    entries = [entry for entry in policy["credentials"] if entry["instance_id"] == identity]
    if policy.get("schema_version") != 1 or len(entries) != 1:
        raise ValueError("rotation requires exactly one existing instance in schema version 1")
    token = secrets.token_urlsafe(32)
    entries[0]["token_sha256"] = hashlib.sha256(token.encode()).hexdigest()
    private_write(directory / f"{identity}.env", f"GRAPHRAG_TOKEN='{token}'\n")
    private_write(policy_path, json.dumps(policy, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--client", action="append", default=[], help="read/capture instance ID; repeat for each")
    parser.add_argument("--read-only", action="append", default=[], help="read-only instance ID; repeat for each")
    parser.add_argument("--rotate", metavar="ID", help="replace one existing instance's token, retaining its identity")
    args = parser.parse_args()
    try:
        if args.rotate:
            if args.client or args.read_only:
                raise ValueError("rotation cannot provision new clients")
            rotate(args.directory, args.rotate)
        else:
            provision(args.directory, args.client, args.read_only)
    except (OSError, ValueError, KeyError, TypeError):
        parser.exit(1, "Credential setup failed; check private directory, unique client IDs and existing policy. No tokens are printed.\n")
    print(f"Private credential files ready in {args.directory}; transfer only each client's own .env file.")


if __name__ == "__main__":
    main()
