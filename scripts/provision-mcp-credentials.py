#!/usr/bin/env python3
"""Provision private per-instance MCP credentials without printing bearer tokens."""
import argparse
import fcntl
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


def validate_identities(identities):
    # Instance IDs are ASCII. Token/recovery filenames must remain distinct
    # on the case-insensitive filesystems used by the supported macOS build.
    if (not identities or len(identities) > 128
            or len({identity.lower() for identity in identities}) != len(identities)):
        raise ValueError("provide 1–128 client IDs unique ignoring ASCII letter case")


def private_write(path, text):
    fd, temporary = tempfile.mkstemp(prefix=".credential-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        Path(temporary).unlink(missing_ok=True)


def provision(directory, writers, readers, grants=()):
    writers, readers = [instance_id(i) for i in writers], [instance_id(i) for i in readers]
    identities = writers + readers
    validate_identities(identities)
    capabilities = {identity: (["read", "capture"] if identity in writers else ["read"]) for identity in identities}
    allowed = {"read", "capture", "edit", "delete", "accept", "reject", "undo"}
    for grant in grants:
        identity, separator, names = grant.partition("=")
        if not separator or identity not in capabilities:
            raise ValueError("grants require an existing client ID and INSTANCE=CAPABILITY syntax")
        for name in names.split(","):
            if name not in allowed or name in capabilities[identity]:
                raise ValueError("grant capabilities must be known and unique for each instance")
            capabilities[identity].append(name)
    directory.mkdir(mode=0o700, parents=True, exist_ok=False)
    policy = {"schema_version": 1, "credentials": []}
    for identity in identities:
        token = secrets.token_urlsafe(32)
        private_write(directory / f"{identity}.env", f"GRAPHRAG_TOKEN='{token}'\n")
        policy["credentials"].append({
            "instance_id": identity,
            "token_sha256": hashlib.sha256(token.encode()).hexdigest(),
            "capabilities": capabilities[identity],
        })
    private_write(directory / "credentials.json", json.dumps(policy, indent=2) + "\n")


def rotate(directory, identity):
    identity = instance_id(identity)
    metadata = directory.lstat()
    if not directory.is_dir() or directory.is_symlink() or metadata.st_mode & 0o077:
        raise ValueError("credential directory must be a private regular directory (mode 0700)")
    # Serialize rotations so each pending record refers to one policy generation.
    lock_fd = os.open(directory / ".rotation.lock", os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        _rotate_locked(directory, identity)
    finally:
        os.close(lock_fd)


def _rotate_locked(directory, identity):
    policy_path = directory / "credentials.json"
    if policy_path.is_symlink() or not policy_path.is_file():
        raise ValueError("credential policy must be a regular file")
    policy = json.loads(policy_path.read_text())
    # Refuse an older colliding policy before writing either its replacement
    # policy or a staged token; rotating one entry could overwrite the other.
    validate_identities([instance_id(entry["instance_id"]) for entry in policy["credentials"]])
    entries = [entry for entry in policy["credentials"] if entry["instance_id"] == identity]
    if policy.get("schema_version") != 1 or len(entries) != 1:
        raise ValueError("rotation requires exactly one existing instance in schema version 1")
    pending_path = directory / f"{identity}.rotation.json"
    if pending_path.exists() or pending_path.is_symlink():
        if (pending_path.is_symlink() or not pending_path.is_file()
                or pending_path.stat().st_mode & 0o077):
            raise ValueError("rotation recovery must be a private regular file")
        pending = json.loads(pending_path.read_text())
        token = pending["token"]
        if (pending.get("instance_id") != identity
                or entries[0]["token_sha256"] not in (pending["old_hash"], pending["new_hash"])
                or hashlib.sha256(token.encode()).hexdigest() != pending["new_hash"]):
            raise ValueError("rotation recovery does not match the current credential")
    else:
        token = secrets.token_urlsafe(32)
        pending = {"instance_id": identity, "old_hash": entries[0]["token_sha256"],
                   "new_hash": hashlib.sha256(token.encode()).hexdigest(), "token": token}
        # Persist the replacement token before revoking the old one. A crash
        # at any later step can finish with this exact token on the next run.
        private_write(pending_path, json.dumps(pending) + "\n")
    entries[0]["token_sha256"] = pending["new_hash"]
    private_write(policy_path, json.dumps(policy, indent=2) + "\n")
    private_write(directory / f"{identity}.env", f"GRAPHRAG_TOKEN='{token}'\n")
    pending_path.unlink()
    directory_fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--client", action="append", default=[], help="read/capture instance ID; repeat for each")
    parser.add_argument("--read-only", action="append", default=[], help="read-only by default instance ID; use --grant for explicit additional permissions")
    parser.add_argument("--grant", action="append", default=[], metavar="INSTANCE=CAPABILITY[,CAPABILITY]", help="explicit additional grants for a provisioned client; repeat for each instance")
    parser.add_argument("--rotate", metavar="ID", help="replace one existing instance's token, retaining its identity")
    args = parser.parse_args()
    try:
        if args.rotate:
            if args.client or args.read_only or args.grant:
                raise ValueError("rotation cannot provision new clients")
            rotate(args.directory, args.rotate)
        else:
            provision(args.directory, args.client, args.read_only, args.grant)
    except (OSError, ValueError, KeyError, TypeError):
        parser.exit(1, "Credential setup failed; check private directory, unique client IDs and existing policy. Retry the same --rotate command to recover an interrupted rotation. No tokens are printed.\n")
    print(f"Private credential files ready in {args.directory}; transfer only each client's own .env file.")


if __name__ == "__main__":
    main()
