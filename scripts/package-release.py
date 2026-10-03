#!/usr/bin/env python3
"""Validate and package native releases; never create tags or publish releases.

Python 3.11+ and system Git/native inspection tools are required. No third-party
Python packages, inference services, downloads, or Cargo execution are used.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib
from pathlib import Path

TARGETS = {
    "aarch64-apple-darwin": ("Mach-O", "arm64"),
    "x86_64-apple-darwin": ("Mach-O", "x86_64"),
    "x86_64-unknown-linux-gnu": ("ELF", "x86-64"),
}
RELEASE_GATES = (
    "workspace_tests", "clippy", "offline_workflow", "native_live_workflow", "local_asset_install"
)
HELP_CHECKS = {
    (): ("capture", "folders", "sync", "inspect", "open", "augment", "garden", "workspace", "backup", "jobs"),
    ("search",): ("--mode", "keyword", "--scope", "--explain"),
    ("inspect",): ("--revision", "--neighbors"),
    ("open",): ("--opener", "--revision"),
    ("capture",): ("--editor", "--title"),
    ("sync",): ("--resume", "--dry-run"),
    ("augment",): ("--format", "--raw"),
    ("garden", "review"): ("--interactive", "--all-statuses"),
    ("workspace",): ("workspace",),
    ("backup",): ("create", "verify", "restore"),
    ("jobs",): ("resume", "cancel"),
}


class ReleaseError(Exception):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReleaseError(message)


def run(argv: list[str], repo: Path | None = None, env: dict | None = None) -> str:
    result = subprocess.run(argv, cwd=repo, env=env, text=True, capture_output=True)
    require(result.returncode == 0, f"{' '.join(argv)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def sha256(path: Path) -> str:
    with path.open("rb") as contents:
        return hashlib.file_digest(contents, "sha256").hexdigest()


def regular_file(path: Path, label: str) -> Path:
    require(path.is_file() and not path.is_symlink(), f"{label} must be a regular, non-symlink file: {path}")
    return path


def json_file(path: Path) -> dict:
    regular_file(path, "JSON input")
    data = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(data, dict), f"JSON input must be an object: {path}")
    return data


def compile_input(path: str) -> bool:
    parts = Path(path).parts
    return path in {"Cargo.toml", "Cargo.lock", "rust-toolchain.toml"} or (
        parts and parts[0] == ".cargo"
    ) or (
        len(parts) >= 3 and parts[0] == "crates"
        and (parts[2] == "src" or parts[2] in {"Cargo.toml", "build.rs"})
    )


def input_identity(repo: Path, commit: str) -> str:
    entries = run(["git", "ls-tree", "-r", commit], repo).splitlines()
    inputs = [entry for entry in entries if compile_input(entry.split("\t", 1)[1])]
    return hashlib.sha256(("\n".join(inputs) + "\n").encode()).hexdigest()


def validate_source(repo: Path, tag: str, expected_commit: str, clean: bool = True, require_tag: bool = False) -> dict:
    require(re.fullmatch(r"v[0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z.-]+)?", tag) is not None,
            "tag must be a version such as v0.1.0-rc.2")
    require(re.fullmatch(r"[0-9a-f]{40}", expected_commit) is not None,
            "expected commit must be a complete 40-character SHA")
    head = run(["git", "rev-parse", "HEAD"], repo)
    require(head == expected_commit, f"checkout HEAD {head} differs from expected commit {expected_commit}")
    if clean:
        require(not run(["git", "status", "--porcelain", "--untracked-files=all"], repo),
                "release packaging requires a clean checkout (including untracked files)")
    manifest = tomllib.loads(regular_file(repo / "Cargo.toml", "workspace manifest").read_text())
    workspace = manifest["workspace"]
    version = workspace["package"]["version"]
    require(tag == f"v{version}", f"tag {tag} differs from workspace version {version}")
    packages = {}
    for member in workspace["members"]:
        package = tomllib.loads(regular_file(repo / member / "Cargo.toml", "member manifest").read_text())["package"]
        require(package.get("version") == {"workspace": True}, f"{member} must inherit the workspace version")
        require(package["name"] not in packages, "workspace contains duplicate package names")
        packages[package["name"]] = version
    lock = tomllib.loads(regular_file(repo / "Cargo.lock", "lockfile").read_text())
    for name in packages:
        matches = [entry for entry in lock["package"] if entry["name"] == name and "source" not in entry]
        require(len(matches) == 1 and matches[0]["version"] == version,
                f"Cargo.lock must contain exactly one local {name} at version {version}")
    toolchain = tomllib.loads(regular_file(repo / "rust-toolchain.toml", "toolchain manifest").read_text())["toolchain"]["channel"]
    require(toolchain == workspace["package"]["rust-version"], "toolchain and workspace MSRV differ")
    existing = run(["git", "tag", "--list", tag], repo)
    if existing:
        require(run(["git", "rev-parse", f"refs/tags/{tag}^{{commit}}"], repo) == head,
                f"existing tag {tag} points to another commit; never move a release tag")
    require(not require_tag or bool(existing), f"release tag {tag} must already exist")
    return {
        "commit": head, "tree": run(["git", "rev-parse", "HEAD^{tree}"], repo),
        "compile_inputs_sha256": input_identity(repo, head), "version": version,
        "tag": tag, "workspace_packages": packages, "rust_toolchain": toolchain,
        "source_date_epoch": int(run(["git", "show", "-s", "--format=%ct", head], repo)),
    }


def smoke_binary(binary: Path, version: str) -> dict:
    regular_file(binary, "binary")
    require(os.access(binary, os.X_OK), "binary must be executable")
    with tempfile.TemporaryDirectory(prefix="graphrag-release-smoke-") as temp:
        env = {**os.environ, "HOME": temp, "TEI_URL": "http://127.0.0.1:1", "TGI_URL": "http://127.0.0.1:1"}
        require(run([str(binary), "--version"], Path(temp), env) == f"graphrag {version}",
                "binary version does not exactly match the workspace/tag")
        outputs = {}
        for command, tokens in HELP_CHECKS.items():
            text = run([str(binary), *command, "--help"], Path(temp), env)
            require(all(token in text for token in tokens), f"binary lacks expected {' '.join(command) or 'root'} help/features")
            outputs[" ".join(command) or "root"] = hashlib.sha256(text.encode()).hexdigest()
    return {"status": "passed", "help_sha256": outputs}


def validate_native_facts(target: str, description: str, libraries: str, load_commands: str = "", glibc_versions: str = "") -> dict:
    require(target in TARGETS, "unsupported release target")
    family, architecture = TARGETS[target]
    require(family in description and architecture in description and "executable" in description,
            "binary architecture/format differs from the native release target")
    require("universal" not in description.lower(), "release archive requires a single native architecture")
    if family == "Mach-O":
        paths = [line.strip().split(" (", 1)[0] for line in libraries.splitlines()[1:] if line.strip()]
        require(paths and all(path.startswith(("/usr/lib/", "/System/Library/")) for path in paths),
                "macOS release binary requires a non-system shared library")
        minimums = re.findall(r"\bminos\s+([0-9.]+)", load_commands)
        require(minimums == ["15.0"], "macOS release binary must declare deployment target 15.0")
        return {"runtime_libraries": paths, "minimum_macos": "15.0", "system_libraries_only": True}
    require("not found" not in libraries and not re.search(r"lib(?:ssl|crypto)\.so", libraries),
            "Linux release has missing libraries or dynamically linked OpenSSL")
    paths = re.findall(r"(?:=>\s+)?(/[^\s]+)", libraries)
    require(all(path.startswith(("/lib/", "/lib64/", "/usr/lib/")) for path in paths),
            "Linux release requires a non-system shared library")
    versions = [tuple(map(int, value.split("."))) for value in re.findall(r"GLIBC_([0-9.]+)", glibc_versions)]
    require(not versions or max(versions) <= (2, 35), "Linux release requires glibc newer than Ubuntu 22.04 (2.35)")
    return {"runtime_libraries": libraries.splitlines(), "maximum_required_glibc": ".".join(map(str, max(versions))) if versions else None,
            "glibc_baseline": "2.35", "system_libraries_only": True}


def native_facts(binary: Path, target: str, repo: Path, toolchain: str) -> dict:
    rustc = run(["rustc", "--version"], repo)
    verbose = run(["rustc", "-vV"], repo)
    require(rustc.startswith(f"rustc {toolchain} "), "active Rust toolchain differs from the declared exact toolchain")
    require(f"host: {target}" in verbose.splitlines(), "build must be native to the selected target")
    description = run(["file", "-b", str(binary)])
    if target.endswith("apple-darwin"):
        require(os.environ.get("MACOSX_DEPLOYMENT_TARGET") == "15.0", "MACOSX_DEPLOYMENT_TARGET must be 15.0")
        libraries = run(["otool", "-L", str(binary)])
        loads = run(["otool", "-l", str(binary)])
        facts = validate_native_facts(target, description, libraries, loads)
        facts["build_host_macos"] = run(["sw_vers", "-productVersion"])
    else:
        facts = validate_native_facts(target, description, run(["ldd", str(binary)]),
                                      glibc_versions=run(["readelf", "--version-info", str(binary)]))
    require(os.environ.get("OPENSSL_STATIC") == "1", "OPENSSL_STATIC must be 1")
    facts.update({"target": target, "binary_description": description, "rustc": rustc,
                  "build_environment": {name: os.environ.get(name) for name in (
                      "OPENSSL_STATIC", "MACOSX_DEPLOYMENT_TARGET", "CARGO_BUILD_JOBS", "RUSTC_WRAPPER")}})
    return facts


def write_json(path: Path, data: dict) -> None:
    with path.open("x", encoding="utf-8") as output:
        output.write(json.dumps(data, indent=2, sort_keys=True) + "\n")


def record_build(args) -> dict:
    repo = args.repo.resolve()
    source = validate_source(repo, args.tag, args.expected_commit, clean=False)
    changed = run(["git", "diff", "--name-only", "HEAD"], repo).splitlines()
    changed += run(["git", "ls-files", "--others", "--exclude-standard"], repo).splitlines()
    require(not any(compile_input(path) for path in changed), "compiled inputs changed while building the candidate")
    binary = args.binary.absolute()
    smoke = smoke_binary(binary, source["version"])
    facts = native_facts(binary, args.target, repo, source["rust_toolchain"])
    record = {"schema_version": 1, "source": source, "native": facts,
              "binary_sha256": sha256(binary), "binary_smoke": smoke}
    write_json(args.output, record)
    return record


def validation_checks(path: Path | None, require_gates: bool, context: dict | None = None) -> dict:
    checks = {name: {"status": "not_run"} for name in (*RELEASE_GATES, "published_asset_install")}
    if path:
        supplied = json_file(path).get("checks")
        require(isinstance(supplied, dict), "validation evidence needs a checks object")
        for name, check in supplied.items():
            require(re.fullmatch(r"[a-z][a-z0-9_]*", name) is not None and isinstance(check, dict), "invalid validation check")
            status = check.get("status")
            require(status in {"passed", "not_run", "deferred"}, "failed or unknown validation status cannot form a release package")
            public = {"status": status}
            if status == "passed":
                evidence = regular_file(Path(check.get("evidence_file", "")), "passed-check evidence")
                public.update({"evidence_name": evidence.name, "evidence_sha256": sha256(evidence)})
                require(context is not None, "passed evidence must be bound to source/binary provenance")
                if name in {"offline_workflow", "native_live_workflow"}:
                    report = json_file(evidence)
                    mode = "live" if name == "native_live_workflow" else "offline"
                    require(report.get("success") is True and report.get("mode") == mode,
                            f"{name} evidence must be a successful {mode} workflow report")
                    matches = [item for item in report.get("runs", [])
                               if item.get("binary_sha256") == context["binary_sha256"]]
                    require(len(matches) == 1 and matches[0].get("success") is True,
                            f"{name} evidence does not match the sealed binary hash")
                    steps = matches[0].get("steps", [])
                    require(steps and all(step.get("success") is True for step in steps)
                            and any(step.get("version") == f"graphrag {context['version']}" for step in steps),
                            f"{name} evidence lacks passed steps and the exact binary version")
                    public.update({"origin": "validated_workflow_report", "binary_sha256": context["binary_sha256"],
                                   "tested_source_commit": context["build_source_commit"]})
                elif name in {"local_asset_install", "published_asset_install"}:
                    report = json_file(evidence)
                    require(report.get("status") == "passed" and report.get("version") == context["version"]
                            and report.get("binary_sha256") == context["binary_sha256"]
                            and report.get("archive_sha256") == context["archive_sha256"]
                            and report.get("source_commit") == context["source_commit"],
                            f"{name} evidence differs from the final source/version/binary/archive")
                    transport = report.get("transport", "")
                    require(isinstance(transport, str) and (
                        transport == "published HTTPS assets" if name == "published_asset_install"
                        else transport == "local curl fixture; published download not tested"
                    ), f"{name} evidence has the wrong transport claim")
                    public.update({"origin": "validated_install_report", "transport": transport,
                                   "binary_sha256": context["binary_sha256"], "archive_sha256": context["archive_sha256"]})
                else:
                    commit = check.get("source_commit")
                    require(commit in {context["source_commit"], context["build_source_commit"]},
                            f"{name} source-check evidence must identify the final or compiled source commit")
                    public.update({"origin": "operator_recorded_source_check", "tested_source_commit": commit})
            if check.get("detail"):
                require(isinstance(check["detail"], str) and len(check["detail"]) <= 2000, "validation detail must be short text")
                public["detail"] = check["detail"]
            checks[name] = public
    if require_gates:
        require(all(checks[name]["status"] == "passed" for name in RELEASE_GATES),
                "publication preparation requires passed workspace/clippy/offline/live/local-install evidence")
    return checks


def deterministic_archive(path: Path, binary: Path, sample: Path, epoch: int) -> None:
    with path.open("xb") as raw, gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=epoch) as compressed:
        with tarfile.open(fileobj=compressed, mode="w", format=tarfile.USTAR_FORMAT) as archive:
            for source, name, mode in [(binary, "graphrag", 0o755), (sample, "samples/first-notes.md", 0o644)]:
                info = tarfile.TarInfo(name)
                info.size = source.stat().st_size
                info.mode, info.mtime = mode, epoch
                info.uid = info.gid = 0
                with source.open("rb") as contents:
                    archive.addfile(info, contents)


def inspect_archive(path: Path, expected_binary: str, expected_sample: str) -> None:
    with tarfile.open(path, "r:gz") as archive:
        members = archive.getmembers()
        require([member.name for member in members] == ["graphrag", "samples/first-notes.md"],
                "archive contents violate the installer allowlist")
        for member, expected, mode in zip(members, [expected_binary, expected_sample], [0o755, 0o644]):
            require(member.isfile() and member.mode == mode, "archive members must be regular files with fixed permissions")
            contents = archive.extractfile(member)
            require(contents is not None and hashlib.file_digest(contents, "sha256").hexdigest() == expected,
                    "archive contents differ from the validated binary/sample")


def publish_directory(output: Path, staged: Path) -> None:
    require(not output.exists() and not output.is_symlink(), f"output already exists; choose a fresh directory: {output}")
    output.mkdir()  # Exclusive creation protects an output created after the preflight.
    for path in sorted(staged.iterdir()):
        os.link(path, output / path.name)  # No-clobber publication on the same filesystem.


def package(args) -> dict:
    repo, output = args.repo.resolve(), args.output.absolute()
    require(output.parent.is_dir(), "output parent must already exist")
    require(not output.exists() and not output.is_symlink(), "output already exists; never overwrite release artifacts")
    source = validate_source(repo, args.tag, args.expected_commit)
    record = json_file(args.build_record)
    require(record.get("schema_version") == 1, "unsupported build record")
    built = record["source"]
    require(built["version"] == source["version"] and built["tag"] == source["tag"], "build record version/tag differs from release source")
    require(input_identity(repo, built["commit"]) == built["compile_inputs_sha256"] == source["compile_inputs_sha256"],
            "compiled source inputs differ from the final release commit; rebuild the binary")
    binary = regular_file(args.binary.absolute(), "binary")
    require(sha256(binary) == record["binary_sha256"], "binary differs from the sealed native build record")
    require(record["native"]["target"] in TARGETS and record["binary_smoke"]["status"] == "passed", "build record lacks a supported native smoke test")
    smoke_binary(binary, source["version"])
    sample = regular_file(repo / "samples/first-notes.md", "starter sample")
    target = record["native"]["target"]
    asset = f"graphrag-notes-{args.tag}-{target}.tar.gz"
    binary_hash, sample_hash = sha256(binary), sha256(sample)
    with tempfile.TemporaryDirectory(prefix=".graphrag-package-", dir=output.parent) as temp:
        staged = Path(temp)
        deterministic_archive(staged / asset, binary, sample, source["source_date_epoch"])
        inspect_archive(staged / asset, binary_hash, sample_hash)
        archive_hash = sha256(staged / asset)
        checks = validation_checks(args.validation_file, args.require_gates, {
            "source_commit": source["commit"], "build_source_commit": built["commit"],
            "version": source["version"], "binary_sha256": binary_hash, "archive_sha256": archive_hash,
        })
        info = {"schema_version": 1, "version": source["version"], "tag": args.tag,
                "source_commit": source["commit"], "source_tree": source["tree"],
                "compile_inputs_sha256": source["compile_inputs_sha256"],
                "build_source_commit": built["commit"], "build_source_tree": built["tree"],
                "workspace_packages": source["workspace_packages"], "source_date_epoch": source["source_date_epoch"],
                **record["native"], "binary_sha256": binary_hash, "sample_sha256": sample_hash,
                "archive": asset, "archive_sha256": archive_hash,
                "validation": {**checks, "binary_version_and_help": record["binary_smoke"], "archive_integrity": {"status": "passed"}}}
        write_json(staged / "BUILDINFO.json", info)
        (staged / "SHA256SUMS").write_text(
            "".join(f"{sha256(staged / name)}  {name}\n" for name in sorted([asset, "BUILDINFO.json"])), encoding="utf-8")
        publish_directory(output, staged)
    return info


def assemble(args) -> dict:
    output = args.output.absolute()
    require(output.parent.is_dir(), "assembly output parent must exist")
    expected = set(args.targets)
    require(expected and expected <= TARGETS.keys(), "assembly targets must be supported native targets")
    records = [json_file(path) for path in sorted(args.input.glob("*/BUILDINFO.json"))]
    require(len(records) == len(expected) and {record["target"] for record in records} == expected,
            "assembly is missing targets or contains duplicate/unexpected targets")
    require(all(record["tag"] == args.tag for record in records), "assembled artifacts have inconsistent tags")
    require(len({(record["source_commit"], record["source_tree"], record["compile_inputs_sha256"]) for record in records}) == 1,
            "assembled artifacts have inconsistent source provenance")
    with tempfile.TemporaryDirectory(prefix=".graphrag-assemble-", dir=output.parent) as temp:
        staged = Path(temp)
        for path in sorted(args.input.glob("*/BUILDINFO.json")):
            record = json_file(path)
            asset = record["archive"]
            require(asset == f"graphrag-notes-{args.tag}-{record['target']}.tar.gz", "unsafe or mismatched archive filename")
            archive = regular_file(path.parent / asset, "archive")
            manifest = regular_file(path.parent / "SHA256SUMS", "per-target checksum manifest")
            entries = {}
            for line in manifest.read_text().splitlines():
                match = re.fullmatch(r"([0-9a-f]{64})  ([^/]+)", line)
                require(match is not None and match[2] not in entries, "invalid or duplicate per-target checksum entry")
                entries[match[2]] = match[1]
            require(set(entries) == {"BUILDINFO.json", asset}, "per-target checksum manifest must identify archive and metadata")
            require(entries["BUILDINFO.json"] == sha256(path), "assembly metadata checksum differs from manifest")
            require(entries[asset] == record["archive_sha256"], "assembly manifest and metadata disagree on archive hash")
            require(sha256(archive) == record["archive_sha256"], "assembly archive checksum differs from BUILDINFO")
            inspect_archive(archive, record["binary_sha256"], record["sample_sha256"])
            shutil.copyfile(archive, staged / asset)
            shutil.copyfile(path, staged / f"BUILDINFO-{record['target']}.json")
            require(sha256(staged / asset) == entries[asset]
                    and sha256(staged / f"BUILDINFO-{record['target']}.json") == entries["BUILDINFO.json"],
                    "assembly inputs changed during copy")
        names = sorted(path.name for path in staged.iterdir())
        (staged / "SHA256SUMS").write_text("".join(f"{sha256(staged / name)}  {name}\n" for name in names), encoding="utf-8")
        publish_directory(output, staged)
    return {"tag": args.tag, "targets": sorted(expected), "output": str(output)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ["validate-source", "record-build", "package"]:
        sub = commands.add_parser(name)
        sub.add_argument("--repo", type=Path, default=Path.cwd())
        sub.add_argument("--tag", required=True)
        sub.add_argument("--expected-commit", required=True)
        if name == "validate-source":
            sub.add_argument("--require-tag", action="store_true")
            sub.add_argument("--github-output", type=Path)
        else:
            sub.add_argument("--binary", type=Path, required=True)
            sub.add_argument("--output", type=Path, required=True)
        if name == "record-build":
            sub.add_argument("--target", choices=TARGETS, required=True)
        if name == "package":
            sub.add_argument("--build-record", type=Path, required=True)
            sub.add_argument("--validation-file", type=Path)
            sub.add_argument("--require-gates", action="store_true")
    sub = commands.add_parser("assemble")
    sub.add_argument("--input", type=Path, required=True)
    sub.add_argument("--output", type=Path, required=True)
    sub.add_argument("--tag", required=True)
    sub.add_argument("--targets", nargs="+", required=True)
    args = parser.parse_args()
    try:
        if args.command == "validate-source":
            result = validate_source(args.repo.resolve(), args.tag, args.expected_commit, require_tag=args.require_tag)
            if args.github_output:
                with args.github_output.open("a") as output:
                    output.write(f"tag={result['tag']}\nversion={result['version']}\ncommit={result['commit']}\n")
        else:
            result = {"record-build": record_build, "package": package, "assemble": assemble}[args.command](args)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (ReleaseError, OSError, ValueError, KeyError, TypeError) as error:
        print(f"Release preparation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
