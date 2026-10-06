#!/usr/bin/env python3
"""Offline release checks using temporary repositories and native-tool fixtures."""

import argparse
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).with_name("package-release.py")
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location("release", SCRIPT)
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="graphrag-release-tests-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        self.version = "0.1.0-rc.3"
        self.tag = f"v{self.version}"
        (self.repo / "Cargo.toml").write_text(
            '[workspace]\nmembers = ["crates/cli", "crates/core"]\n'
            '[workspace.package]\nversion = "0.1.0-rc.3"\nrust-version = "1.97.1"\n')
        (self.repo / "rust-toolchain.toml").write_text('[toolchain]\nchannel = "1.97.1"\n')
        lock = 'version = 4\n'
        for name in ["cli", "core"]:
            member = self.repo / "crates" / name
            (member / "src").mkdir(parents=True)
            (member / "Cargo.toml").write_text(f'[package]\nname = "graphrag-{name}"\nversion.workspace = true\n')
            (member / "src/lib.rs").write_text("// fixture compiled input\n")
            lock += f'[[package]]\nname = "graphrag-{name}"\nversion = "{self.version}"\n'
        (self.repo / "Cargo.lock").write_text(lock)
        (self.repo / "samples").mkdir()
        (self.repo / "samples/first-notes.md").write_text("# Sanitized sample\nAtlas launch fixture.\n")
        for relative in release.RELEASE_PAYLOADS:
            path = self.repo / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("Public fictional release fixture: " + relative + "\n")
        self.git("init", "-q")
        self.git("config", "user.name", "Offline fixture")
        self.git("config", "user.email", "fixture@example.invalid")
        self.commit()
        self.binary = self.root / "graphrag"
        tokens = " ".join(token for values in release.HELP_CHECKS.values() for token in values)
        self.binary.write_text(f'#!/bin/sh\nif [ "$1" = "--version" ]; then printf "graphrag {self.version}\\n"; else printf "%s\\n" "{tokens}"; fi\n')
        self.binary.chmod(0o755)
        self.build_record = self.root / "build.json"
        platform = (os.uname().sysname, os.uname().machine)
        self.target = {
            ("Darwin", "arm64"): "aarch64-apple-darwin",
            ("Darwin", "x86_64"): "x86_64-apple-darwin",
            ("Linux", "x86_64"): "x86_64-unknown-linux-gnu",
        }.get(platform)
        if self.target is None:
            self.skipTest(f"No native release fixture for {platform}")
        family, architecture = release.TARGETS[self.target]
        self.native_outputs = {
            ("file", "-b"): f"{family} 64-bit executable {architecture}",
            ("otool", "-L"): "fixture:\n /usr/lib/libSystem.B.dylib (compatibility version 1.0.0)",
            ("otool", "-l"): " minos 15.0",
            ("ldd",): "linux-vdso.so.1 (0x00000001)\nlibc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00000002)",
            ("readelf", "--version-info"): "GLIBC_2.35",
        }
        original_run = release.run

        def fixture_tools(argv, repo=None, env=None):
            for prefix, output in self.native_outputs.items():
                if tuple(argv[:len(prefix)]) == prefix:
                    return output
            return original_run(argv, repo, env)

        native_tools = patch.object(release, "run", side_effect=fixture_tools)
        native_tools.start()
        self.addCleanup(native_tools.stop)
        self.facts = {**release.inspect_binary(self.binary, self.target), "rustc": "rustc 1.97.1 fixture",
                      "build_environment": {"OPENSSL_STATIC": "1", "MACOSX_DEPLOYMENT_TARGET": "15.0",
                                            "CARGO_BUILD_JOBS": "2", "RUSTC_WRAPPER": ""}}
        if self.target.endswith("apple-darwin"):
            self.facts["build_host_macos"] = "15.0"
        self.seal_build()

    def git(self, *args):
        return subprocess.check_output(["git", *args], cwd=self.repo, text=True, stderr=subprocess.DEVNULL).strip()

    def commit(self):
        self.git("add", ".")
        self.git("commit", "-qm", "fixture")
        self.commit_id = self.git("rev-parse", "HEAD")

    def seal_build(self):
        args = argparse.Namespace(repo=self.repo, tag=self.tag, expected_commit=self.commit_id,
                                 binary=self.binary, output=self.build_record, target=self.facts["target"])
        with patch.object(release, "native_facts", return_value=self.facts):
            release.record_build(args)

    def package_args(self, output="dist", **kwargs):
        return argparse.Namespace(repo=self.repo, tag=self.tag, expected_commit=self.commit_id,
                                  binary=self.binary, output=self.root / output, build_record=self.build_record,
                                  validation_file=None, require_gates=False, **kwargs)

    def cargo_rows(self, target):
        versions_features = {
            "graphrag-cli": (self.version, []),
            "graphrag-db": (self.version, ["default", "rocksdb"]),
            "surrealdb": ("3.2.4", ["kv-mem", "kv-rocksdb"]),
            "surrealdb-core": ("3.2.4", ["kv-mem", "kv-rocksdb"]),
            "surrealdb-types": ("3.2.4", ["default"]),
        }
        rows = []
        for name, (version, features) in versions_features.items():
            rows.append({"reason": "compiler-artifact", "package_id": f"registry+https://fixture.invalid#index#{name}@{version}",
                         "target": {"kind": ["bin"] if name == "graphrag-cli" else ["lib"],
                                    "name": "graphrag" if name == "graphrag-cli" else name.replace("-", "_")},
                         "features": features, "profile": {"opt_level": "3", "test": False,
                                                              "debug_assertions": False, "overflow_checks": False},
                         "executable": str(self.binary) if name == "graphrag-cli" else None})
        return rows + [{"reason": "build-finished", "success": True}]

    def write_cargo_rows(self, rows, name="cargo.jsonl"):
        path = self.root / name
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        return path

    def rc5_build(self):
        old = self.version
        self.version, self.tag = "0.1.0-rc.5", "v0.1.0-rc.5"
        for path in (self.repo / "Cargo.toml", self.repo / "Cargo.lock", self.binary):
            path.write_text(path.read_text().replace(old, self.version))
        self.commit()
        self.build_record = self.root / "rc5-build.json"
        messages = self.write_cargo_rows(self.cargo_rows(self.target))
        args = argparse.Namespace(repo=self.repo, tag=self.tag, expected_commit=self.commit_id,
                                 binary=self.binary, output=self.build_record, target=self.target, cargo_messages=messages)
        with patch.object(release, "native_facts", return_value=self.facts):
            release.record_build(args)
        return messages

    def test_compiler_feature_proof_follows_actual_target_packages_and_omits_private_paths(self):
        for target in release.TARGETS:
            messages = self.write_cargo_rows(self.cargo_rows(target))
            proof = release.cargo_feature_proof(messages, self.binary, target, self.version)
            self.assertEqual(proof["binary_sha256"], release.sha256(self.binary))
            self.assertEqual(proof["cargo_messages_sha256"], release.sha256(messages))
            self.assertEqual(proof["allocator"], "system")
            self.assertNotIn(str(self.root), json.dumps(proof))
            self.assertNotIn("fixture.invalid", json.dumps(proof))

    def test_compiler_executable_bytes_must_match_but_allow_an_immutable_copy(self):
        messages = self.write_cargo_rows(self.cargo_rows(self.target))
        copied = self.root / "immutable-copy"
        copied.write_bytes(self.binary.read_bytes())
        release.cargo_feature_proof(messages, copied, self.target, self.version)
        copied.write_bytes(b"different executable bytes\n")
        with self.assertRaisesRegex(release.ReleaseError, "executable differs"):
            release.cargo_feature_proof(messages, copied, self.target, self.version)

    def test_compiler_failed_missing_duplicate_debug_and_allocator_drift_are_rejected(self):
        target = "aarch64-apple-darwin"
        original = self.cargo_rows(target)
        mutations = []
        rows = deepcopy(original); rows[-1]["success"] = False; mutations.append(rows)
        mutations.append(deepcopy(original[:-1]))
        mutations.append(deepcopy(original + [original[-1]]))
        mutations.append(deepcopy(original + [original[0]]))
        for field, value in (("opt_level", "0"), ("test", True), ("debug_assertions", True), ("overflow_checks", True)):
            rows = deepcopy(original); rows[0]["profile"][field] = value; mutations.append(rows)
        for index in range(len(original) - 1):
            rows = deepcopy(original); rows.pop(index); mutations.append(rows)
        for index, features in ((0, ["allocator"]), (1, ["allocator", "default", "rocksdb"]),
                                (2, ["allocator", "kv-mem", "kv-rocksdb"]),
                                (3, ["allocator", "kv-mem", "kv-rocksdb"])):
            rows = deepcopy(original); rows[index]["features"] = features; mutations.append(rows)
        rows = deepcopy(original); rows[2]["package_id"] = rows[2]["package_id"].replace("3.2.4", "3.3.0"); mutations.append(rows)
        rows = deepcopy(original); rows.append({"reason": "compiler-message", "message": {"level": "error"}}); mutations.append(rows)
        for index, rows in enumerate(mutations):
            with self.subTest(index=index), self.assertRaises(release.ReleaseError):
                release.cargo_feature_proof(self.write_cargo_rows(rows), self.binary, target, self.version)
        for target in release.TARGETS:
            rows = self.cargo_rows(target); rows[0]["features"] = ["allocator"]
            with self.assertRaises(release.ReleaseError):
                release.cargo_feature_proof(self.write_cargo_rows(rows), self.binary, target, self.version)
            for name, version in (("mimalloc", "0.1.52"), ("libmimalloc-sys", "0.1.49"),
                                  ("tikv-jemallocator", "0.6.1"), ("tikv-jemalloc-sys", "0.6.1")):
                rows = self.cargo_rows(target)
                unwanted = deepcopy(rows[1])
                unwanted.update(package_id=f"registry+https://fixture.invalid#index#{name}@{version}",
                                target={"kind": ["lib"], "name": name.replace("-", "_")}, features=[])
                rows.insert(-1, unwanted)
                with self.assertRaises(release.ReleaseError):
                    release.cargo_feature_proof(self.write_cargo_rows(rows), self.binary, target, self.version)

    def test_rc5_rechecks_sealed_compiler_log_without_depending_on_mutable_cache(self):
        messages = self.rc5_build()
        # Record-time verification saw the actual executable. Packaging can use
        # an immutable copy after a later build replaces that cache path.
        copied = self.root / "sealed-graphrag"
        copied.write_bytes(self.binary.read_bytes()); copied.chmod(0o755)
        rows = [json.loads(line) for line in messages.read_text().splitlines()]
        rows[0]["executable"] = str(self.root / "cargo-cache-artifact")
        cache = Path(rows[0]["executable"]); cache.write_bytes(self.binary.read_bytes())
        messages = self.write_cargo_rows(rows)
        self.build_record = self.root / "copied-build.json"
        args = argparse.Namespace(repo=self.repo, tag=self.tag, expected_commit=self.commit_id,
                                 binary=copied, output=self.build_record, target=self.target, cargo_messages=messages)
        with patch.object(release, "native_facts", return_value=self.facts):
            release.record_build(args)
        self.binary = copied
        cache.write_bytes(b"later unrelated build\n")
        info = release.package(self.package_args(cargo_messages=messages))
        self.assertEqual(info["cargo_features"], json.loads(self.build_record.read_text())["cargo_features"])
        assembled = release.assemble(argparse.Namespace(input=self.root, output=self.root / "assembly",
                                                        tag=self.tag, targets=[self.target]))
        self.assertEqual(assembled["targets"], [self.target])
        messages.write_text(messages.read_text() + "\n")
        with self.assertRaisesRegex(release.ReleaseError, "sealed build record"):
            release.package(self.package_args("changed-log", cargo_messages=messages))
        self.assertFalse((self.root / "changed-log").exists())

    def test_rc5_missing_feature_log_forged_seal_and_assembly_policy_are_rejected(self):
        messages = self.rc5_build()
        with self.assertRaisesRegex(release.ReleaseError, "retained Cargo"):
            release.package(self.package_args("missing-log"))
        args = argparse.Namespace(repo=self.repo, tag=self.tag, expected_commit=self.commit_id,
                                 binary=self.binary, output=self.root / "missing-build.json", target=self.target)
        with patch.object(release, "native_facts", return_value=self.facts), self.assertRaisesRegex(release.ReleaseError, "retained Cargo"):
            release.record_build(args)
        original = json.loads(self.build_record.read_text())
        altered = deepcopy(original); altered["cargo_features"]["cargo_messages_sha256"] = "0" * 64
        self.build_record.write_text(json.dumps(altered))
        with self.assertRaisesRegex(release.ReleaseError, "sealed build record"):
            release.package(self.package_args("forged-seal", cargo_messages=messages))
        self.build_record.write_text(json.dumps(original))
        release.package(self.package_args("native", cargo_messages=messages))
        info_path = self.root / "native/BUILDINFO.json"
        info = json.loads(info_path.read_text()); del info["cargo_features"]
        info_path.write_text(json.dumps(info))
        with self.assertRaisesRegex(release.ReleaseError, "compiler feature proof"):
            release.assemble(argparse.Namespace(input=self.root, output=self.root / "bad-assembly",
                                               tag=self.tag, targets=[self.target]))

    def test_concurrent_assembly_metadata_replacement_cannot_bypass_feature_policy(self):
        messages = self.rc5_build()
        release.package(self.package_args("native", cargo_messages=messages))
        folder = self.root / "native"
        info_path = folder / "BUILDINFO.json"
        original = json.loads(info_path.read_bytes())
        validate = release.validate_feature_proof
        replaced = False

        def replace_metadata_after_validation(proof, target, version, binary_hash):
            nonlocal replaced
            validate(proof, target, version, binary_hash)
            if not replaced:
                replaced = True
                altered = deepcopy(original)
                del altered["cargo_features"]
                # Simulate a consistent replacement of metadata, flat identity,
                # and checksums between the initial read and assembly copying.
                info_path.write_text(json.dumps(altered, sort_keys=True) + "\n")
                (folder / "BUILDINFO.identity").write_bytes(release.release_identity(altered, release.sha256(info_path)))
                lines = (folder / "SHA256SUMS").read_text().splitlines()
                updated = []
                for line in lines:
                    _, name = line.split("  ")
                    updated.append(release.sha256(folder / name) + "  " + name + "\n")
                (folder / "SHA256SUMS").write_text("".join(updated))

        args = argparse.Namespace(input=self.root, output=self.root / "raced-assembly", tag=self.tag, targets=[self.target])
        with patch.object(release, "validate_feature_proof", side_effect=replace_metadata_after_validation):
            with self.assertRaisesRegex(release.ReleaseError, "metadata checksum"):
                release.assemble(args)
        self.assertTrue(replaced)
        self.assertFalse(args.output.exists())

    def test_release_sources_keep_system_allocator_and_original_defaults(self):
        import tomllib
        source = SCRIPT.parents[1]
        db = tomllib.loads((source / "crates/db/Cargo.toml").read_text())["features"]
        cli = tomllib.loads((source / "crates/cli/Cargo.toml").read_text()).get("features", {})
        self.assertEqual(db["default"], ["rocksdb"])
        self.assertNotIn("allocator", db)
        self.assertEqual(cli.get("default", []), [])
        self.assertNotIn("allocator", cli)
        lock = tomllib.loads((source / "Cargo.lock").read_text())
        names = {row["name"] for row in lock["package"]}
        self.assertFalse(names & {"mimalloc", "libmimalloc-sys", "tikv-jemallocator", "tikv-jemalloc-sys"})

    def test_version_lock_tag_commit_and_clean_source_checks(self):
        source = release.validate_source(self.repo, self.tag, self.commit_id)
        self.assertEqual(set(source["workspace_packages"]), {"graphrag-cli", "graphrag-core"})
        for tag, commit in [("v9.9.9", self.commit_id), ("../bad", self.commit_id), (self.tag, "0" * 40)]:
            with self.assertRaises(release.ReleaseError):
                release.validate_source(self.repo, tag, commit)
        with self.assertRaises(release.ReleaseError):
            release.validate_source(self.repo, self.tag, self.commit_id, require_tag=True)
        (self.repo / "Cargo.lock").write_text((self.repo / "Cargo.lock").read_text().replace(self.version, "9.9.9", 1))
        self.commit()
        with self.assertRaisesRegex(release.ReleaseError, "Cargo.lock"):
            release.validate_source(self.repo, self.tag, self.commit_id)

    def test_moved_tag_and_non_inherited_member_versions_are_rejected(self):
        self.git("tag", self.tag)
        member = self.repo / "crates/cli/Cargo.toml"
        member.write_text(member.read_text().replace("version.workspace = true", 'version = "0.1.0-rc.2"'))
        self.commit()
        with self.assertRaisesRegex(release.ReleaseError, "inherit"):
            release.validate_source(self.repo, self.tag, self.commit_id)
        member.write_text(member.read_text().replace('version = "0.1.0-rc.2"', "version.workspace = true"))
        self.commit()
        with self.assertRaisesRegex(release.ReleaseError, "another commit"):
            release.validate_source(self.repo, self.tag, self.commit_id)

    def test_dirty_untracked_and_changed_compiled_inputs_are_rejected(self):
        (self.repo / "untracked").write_text("not committed")
        with self.assertRaisesRegex(release.ReleaseError, "clean"):
            release.package(self.package_args())
        (self.repo / "untracked").unlink()
        (self.repo / "crates/core/src/lib.rs").write_text("// changed production source\n")
        dirty_record = argparse.Namespace(repo=self.repo, tag=self.tag, expected_commit=self.commit_id,
                                          binary=self.binary, output=self.root / "dirty-build.json", target=self.facts["target"])
        with self.assertRaisesRegex(release.ReleaseError, "compiled inputs changed"):
            release.record_build(dirty_record)
        self.commit()
        with self.assertRaisesRegex(release.ReleaseError, "compiled source inputs"):
            release.package(self.package_args())
        self.assertFalse((self.root / "dist").exists())

    def test_metadata_only_commits_preserve_build_identity_and_reproducible_archives(self):
        (self.repo / "release-notes.md").write_text("Documentation only.\n")
        self.commit()
        first = release.package(self.package_args("first"))
        second = release.package(self.package_args("second"))
        self.assertEqual(first, second)
        self.assertEqual((self.root / "first" / first["archive"]).read_bytes(),
                         (self.root / "second" / second["archive"]).read_bytes())
        self.assertEqual(first["source_commit"], self.commit_id)
        self.assertNotEqual(first["source_commit"], first["build_source_commit"])
        self.assertEqual(first["validation"]["published_asset_install"]["status"], "not_run")
        with tarfile.open(self.root / "first" / first["archive"]) as archive:
            self.assertEqual(archive.getnames(), ["graphrag", "samples/first-notes.md", *sorted(first["payload_sha256"])])
            self.assertEqual(set(first["payload_sha256"]), {"release/" + p for p in release.RELEASE_PAYLOADS} | {"release/PAYLOADS.json", "release/PAYLOADS.sha256", "release/VERSION", "release/SOURCE-COMMIT"})
            self.assertTrue(all(member.uid == member.gid == 0 for member in archive.getmembers()))

    def test_concurrent_worktree_rewrites_cannot_change_verified_release_payloads(self):
        relatives = ("samples/first-notes.md", "scripts/refresh-openclaw-memory.py", "README.md")
        committed = {name: subprocess.check_output(["git", "show", f"{self.commit_id}:{name}"], cwd=self.repo)
                     for name in relatives}
        pending = {(self.repo / name).resolve(): b"Uncommitted concurrent editor rewrite\n" for name in relatives}
        original_run = release.run

        def rewrite_after_verification(argv, repo=None, env=None):
            value = original_run(argv, repo, env)
            if argv[:2] == ["git", "hash-object"] and Path(argv[2]).resolve() in pending:
                path = Path(argv[2]).resolve()
                path.write_bytes(pending.pop(path))
            return value

        with patch.object(release, "run", side_effect=rewrite_after_verification):
            info = release.package(self.package_args())
        self.assertFalse(pending, "all concurrent rewrites must occur after their successful Git hash checks")
        self.assertEqual(info["source_commit"], self.commit_id)
        self.assertEqual(info["sample_sha256"], hashlib.sha256(committed[relatives[0]]).hexdigest())
        for asset, native in ((info["archive"], True), (info["client_archive"], False)):
            with tarfile.open(self.root / "dist" / asset) as archive:
                if native:
                    self.assertEqual(archive.extractfile(relatives[0]).read(), committed[relatives[0]])
                for relative in relatives[1:]:
                    self.assertEqual(archive.extractfile("release/" + relative).read(), committed[relative])
                    self.assertEqual(info["payload_sha256"]["release/" + relative],
                                     hashlib.sha256(committed[relative]).hexdigest())
        for relative in relatives:
            self.assertEqual((self.repo / relative).read_bytes(), b"Uncommitted concurrent editor rewrite\n")

    def test_source_tree_remains_bound_to_pinned_commit_when_head_moves(self):
        original_commit = self.commit_id
        original_tree = self.git("rev-parse", f"{original_commit}^{{tree}}")
        (self.repo / "another-commit.md").write_text("Concurrent commit fixture\n")
        self.commit()
        next_commit = self.commit_id
        self.git("reset", "--hard", original_commit)
        self.commit_id = original_commit
        original_run = release.run

        def move_head_after_manifest_validation(argv, repo=None, env=None):
            value = original_run(argv, repo, env)
            if argv == ["git", "tag", "--list", self.tag]:
                self.git("update-ref", "HEAD", next_commit)
            return value

        with patch.object(release, "run", side_effect=move_head_after_manifest_validation):
            source = release.validate_source(self.repo, self.tag, original_commit)
        self.assertEqual(self.git("rev-parse", "HEAD"), next_commit)
        self.assertEqual(source["commit"], original_commit)
        self.assertEqual(source["tree"], original_tree)
        self.assertEqual(source["compile_inputs_sha256"], release.input_identity(self.repo, original_commit))

    def test_missing_symlink_modified_and_wrong_version_binaries_are_rejected(self):
        self.binary.write_text("#!/bin/sh\necho graphrag 9.9.9\n")
        with self.assertRaisesRegex(release.ReleaseError, "sealed native"):
            release.package(self.package_args())
        with self.assertRaisesRegex(release.ReleaseError, "version"):
            release.smoke_binary(self.binary, self.version)
        self.binary.unlink()
        self.binary.symlink_to(self.build_record)
        with self.assertRaisesRegex(release.ReleaseError, "non-symlink"):
            release.package(self.package_args())
        self.binary.unlink()
        with self.assertRaisesRegex(release.ReleaseError, "regular"):
            release.package(self.package_args())

    def test_missing_sample_and_missing_feature_help_are_rejected(self):
        (self.repo / "samples/first-notes.md").unlink()
        self.commit()
        with self.assertRaisesRegex(release.ReleaseError, "starter sample"):
            release.package(self.package_args())
        self.binary.write_text(f'#!/bin/sh\nif [ "$1" = "--version" ]; then echo graphrag {self.version}; else echo obsolete; fi\n')
        with self.assertRaisesRegex(release.ReleaseError, "help/features"):
            release.smoke_binary(self.binary, self.version)

    def test_existing_output_never_changes_and_gate_evidence_is_explicit(self):
        output = self.root / "dist"
        output.mkdir()
        (output / "keep").write_text("existing asset")
        with self.assertRaisesRegex(release.ReleaseError, "overwrite"):
            release.package(self.package_args())
        self.assertEqual((output / "keep").read_text(), "existing asset")
        with self.assertRaisesRegex(release.ReleaseError, "passed workspace"):
            release.validation_checks(None, True)
        evidence = self.root / "private-local-evidence.log"
        evidence.write_text("sanitized fixture checks passed\n")
        validation = self.root / "validation.json"
        context = {"source_commit": self.commit_id, "build_source_commit": self.commit_id,
                   "version": self.version, "binary_sha256": release.sha256(self.binary), "archive_sha256": "a" * 64}
        checks = {name: {"status": "passed", "source_commit": self.commit_id,
                         "evidence_file": str(evidence)} for name in release.RELEASE_GATES}
        for name in ["offline_workflow", "native_live_workflow"]:
            report = self.root / f"{name}.json"
            report.write_text(json.dumps({"success": True, "mode": "offline" if name == "offline_workflow" else "live",
                                          "runs": [{"binary_sha256": context["binary_sha256"], "success": True,
                                                    "steps": [{"success": True, "version": f"graphrag {self.version}"}]}]}))
            checks[name]["evidence_file"] = str(report)
        local_report = self.root / "local-install.json"
        local_report.write_text(json.dumps({"status": "passed", **context,
                                           "transport": "local curl fixture; published download not tested"}))
        checks["local_asset_install"]["evidence_file"] = str(local_report)
        validation.write_text(json.dumps({"checks": checks}))
        public = release.validation_checks(validation, True, context)
        self.assertNotIn(str(self.root), json.dumps(public))
        self.assertEqual(public["workspace_tests"]["evidence_sha256"], release.sha256(evidence))
        checks["workspace_tests"]["status"] = "failed"
        validation.write_text(json.dumps({"checks": checks}))
        with self.assertRaisesRegex(release.ReleaseError, "failed"):
            release.validation_checks(validation, False, context)

    def test_passed_runtime_reports_must_match_binary_version_source_and_transport(self):
        context = {"source_commit": self.commit_id, "build_source_commit": self.commit_id,
                   "version": self.version, "binary_sha256": release.sha256(self.binary), "archive_sha256": "a" * 64}
        report = self.root / "arbitrary-passed.json"
        report.write_text(json.dumps({"status": "passed"}))
        validation = self.root / "validation.json"
        validation.write_text(json.dumps({"checks": {"offline_workflow": {"status": "passed", "evidence_file": str(report)}}}))
        with self.assertRaisesRegex(release.ReleaseError, "workflow report"):
            release.validation_checks(validation, False, context)
        report.write_text(json.dumps({"success": True, "mode": "offline", "runs": [
            {"success": True, "binary_sha256": "0" * 64, "steps": [{"success": True, "version": f"graphrag {self.version}"}]}]}))
        with self.assertRaisesRegex(release.ReleaseError, "sealed binary hash"):
            release.validation_checks(validation, False, context)
        wrong_source = {"status": "passed", **context, "source_commit": "0" * 40,
                        "transport": "local curl fixture; published download not tested"}
        report.write_text(json.dumps(wrong_source))
        validation.write_text(json.dumps({"checks": {"local_asset_install": {"status": "passed", "evidence_file": str(report)}}}))
        with self.assertRaisesRegex(release.ReleaseError, "final source"):
            release.validation_checks(validation, False, context)
        wrong_source["source_commit"] = self.commit_id
        report.write_text(json.dumps(wrong_source))
        validation.write_text(json.dumps({"checks": {"published_asset_install": {"status": "passed", "evidence_file": str(report)}}}))
        with self.assertRaisesRegex(release.ReleaseError, "transport"):
            release.validation_checks(validation, False, context)

    def test_unsafe_runtime_dependencies_wrong_architecture_and_deployment_are_rejected(self):
        description = "Mach-O 64-bit executable arm64"
        libraries = "fixture:\n /usr/lib/libSystem.B.dylib (compatibility version 1.0.0)"
        self.assertEqual(release.validate_native_facts("aarch64-apple-darwin", description, libraries, " minos 15.0")["minimum_macos"], "15.0")
        for desc, libs, loads in [
            ("Mach-O 64-bit executable x86_64", libraries, "minos 15.0"),
            (description, libraries.replace("/usr/lib/", "/opt/homebrew/lib/"), "minos 15.0"),
            (description, libraries, "minos 27.0"),
            (description, "fixture:\n @rpath/libcustom.dylib", "minos 15.0"),
        ]:
            with self.assertRaises(release.ReleaseError):
                release.validate_native_facts("aarch64-apple-darwin", desc, libs, loads)
        for libraries, glibc in [("libssl.so.3 => /lib/libssl.so.3", "GLIBC_2.35"),
                                  ("libmissing.so => not found", ""), ("/lib/libc.so.6", "GLIBC_2.36")]:
            with self.assertRaises(release.ReleaseError):
                release.validate_native_facts("x86_64-unknown-linux-gnu", "ELF 64-bit executable x86-64", libraries, glibc_versions=glibc)

    def test_tampered_build_record_native_source_and_smoke_are_rejected(self):
        original = json.loads(self.build_record.read_text())
        other = next(target for target in release.TARGETS if target != self.target)
        cases = [
            ("native", "target", other, "architecture/format"),
            ("native", "binary_description", "forged description", "native facts differ"),
            ("native", "runtime_libraries", ["/opt/private/libssl.dylib"], "native facts differ"),
            ("native", "system_libraries_only", False, "native facts differ"),
            ("native", "source_commit", "0" * 40, "unexpected native fields"),
            ("native", "rustc", "rustc 1.96.0 forged", "historical toolchain"),
            ("source", "tree", "0" * 40, "source tree differs"),
            ("source", "source_date_epoch", original["source"]["source_date_epoch"] + 1, "source timestamp differs"),
            ("source", "rust_toolchain", "1.96.0", "toolchain/packages differ"),
            ("source", "workspace_packages", {"graphrag-cli": "9.9.9"}, "toolchain/packages differ"),
            ("source", "commit", "HEAD", "complete commit SHA"),
            ("binary_smoke", "status", "not_run", "smoke results differ"),
            ("binary_smoke", "help_sha256", {"root": "0" * 64}, "smoke results differ"),
        ]
        if self.target.endswith("apple-darwin"):
            cases.append(("native", "minimum_macos", "27.0", "native facts differ"))
        else:
            cases.extend([
                ("native", "maximum_required_glibc", "2.36", "native facts differ"),
                ("native", "glibc_baseline", "2.36", "native facts differ"),
            ])
        for index, (section, field, value, diagnostic) in enumerate(cases):
            with self.subTest(section=section, field=field):
                altered = json.loads(json.dumps(original))
                altered[section][field] = value
                self.build_record.write_text(json.dumps(altered))
                output = f"tampered-{index}"
                with self.assertRaisesRegex(release.ReleaseError, diagnostic):
                    release.package(self.package_args(output))
                self.assertFalse((self.root / output).exists())
        self.build_record.write_text(json.dumps(original))
        info = release.package(self.package_args("untampered"))
        self.assertEqual(info["target"], self.target)
        self.assertEqual(info["build_source_tree"], original["source"]["tree"])
        self.assertEqual(info["validation"]["binary_version_and_help"], release.smoke_binary(self.binary, self.version))

    def test_packaging_reinspects_changed_intrinsic_binary_facts(self):
        # The sealed record stays unchanged. New native-tool output must be
        # rejected, even though the executable fixture's byte hash is identical.
        if self.target.endswith("apple-darwin"):
            key, output = ("otool", "-l"), " minos 27.0"
        else:
            key, output = ("readelf", "--version-info"), "GLIBC_2.36"
        self.native_outputs[key] = output
        with self.assertRaisesRegex(release.ReleaseError, "deployment target|glibc newer"):
            release.package(self.package_args())
        self.assertFalse((self.root / "dist").exists())

    def test_binary_replacement_after_smoke_cannot_change_sealed_archive(self):
        actual_smoke = release.smoke_binary

        def replace_after_smoke(binary, version):
            smoke = actual_smoke(binary, version)
            binary.write_text(binary.read_text() + "# replaced after inspection\n")
            return smoke

        with patch.object(release, "smoke_binary", side_effect=replace_after_smoke):
            with self.assertRaisesRegex(release.ReleaseError, "contents differ"):
                release.package(self.package_args())
        self.assertFalse((self.root / "dist").exists())

    def test_linux_runtime_facts_ignore_only_aslr_addresses(self):
        libraries = "linux-vdso.so.1 (0x123456)\nlibc.so.6 => /lib/libc.so.6 (0x456789)"
        first = release.validate_native_facts("x86_64-unknown-linux-gnu", "ELF 64-bit executable x86-64",
                                             libraries, glibc_versions="GLIBC_2.35")
        second = release.validate_native_facts("x86_64-unknown-linux-gnu", "ELF 64-bit executable x86-64",
                                              libraries.replace("0x123456", "0xabcdef").replace("0x456789", "0x987654"),
                                              glibc_versions="GLIBC_2.35")
        self.assertEqual(first, second)
        self.assertIn("libc.so.6 => /lib/libc.so.6", first["runtime_libraries"])

    def test_archive_links_extra_paths_and_altered_payload_are_rejected(self):
        info = release.package(self.package_args())
        bad = self.root / "malicious.tar.gz"
        with tarfile.open(bad, "w:gz") as archive:
            member = tarfile.TarInfo("../escape")
            member.type, member.linkname = tarfile.SYMTYPE, "/etc/passwd"
            archive.addfile(member)
        with self.assertRaisesRegex(release.ReleaseError, "allowlist"):
            release.inspect_archive(bad, info["binary_sha256"], info["sample_sha256"])
        with self.assertRaisesRegex(release.ReleaseError, "differ"):
            release.inspect_archive(self.root / "dist" / info["archive"], "0" * 64, info["sample_sha256"], info["payload_sha256"])

    def test_assembly_requires_all_targets_consistent_provenance_and_valid_checksums(self):
        release.package(self.package_args("native"))
        other = next(target for target in release.TARGETS if target != self.target)
        args = argparse.Namespace(input=self.root, output=self.root / "assembly", tag=self.tag,
                                  targets=[self.target, other])
        with self.assertRaisesRegex(release.ReleaseError, "missing targets"):
            release.assemble(args)
        args.targets = [self.target]
        valid_tag = args.tag
        args.tag = "../unsafe"
        with self.assertRaisesRegex(release.ReleaseError, "versioned release tag"):
            release.assemble(args)
        args.tag = valid_tag
        info_path = self.root / "native/BUILDINFO.json"
        original = info_path.read_text()
        value = json.loads(original); value['version'] = '9.9.9'
        info_path.write_text(json.dumps(value))
        with self.assertRaisesRegex(release.ReleaseError, "versions/tags"):
            release.assemble(args)
        info_path.write_text(original)
        assembled = release.assemble(args)
        self.assertEqual(assembled["targets"], args.targets)
        self.assertTrue((args.output / f"BUILDINFO-{self.target}.json").is_file())
        info_path = self.root / "native/BUILDINFO.json"
        info_path.write_text(info_path.read_text() + "\n")
        args.output = self.root / "tampered-assembly"
        with self.assertRaisesRegex(release.ReleaseError, "metadata checksum"):
            release.assemble(args)

    def install_fixture(self, dist, force=False, clients_only=False):
        tools = self.root / "installer-tools"
        tools.mkdir(exist_ok=True)
        curl = tools / "curl"
        curl.write_text("""#!/bin/bash
set -euo pipefail
output='' url=''
while [ "$#" -gt 0 ]; do
    case "$1" in
      -o) output="$2"; shift 2 ;;
      --proto|--proto-redir) shift 2 ;;
      --tlsv1.2|-fsSL) shift ;;
      https://*) url="$1"; shift ;;
      *) exit 2 ;;
    esac
done
case "$url" in
  https://github.com/mateu/graphrag-notes/releases/download/*) cp "$GRN_TEST_DIST/${url##*/}" "$output" ;;
  *) exit 22 ;;
esac
""")
        curl.chmod(0o755)
        no_python = tools / "python3"
        no_python.write_text("#!/bin/sh\nexit 99\n")
        no_python.chmod(0o755)
        env = dict(os.environ, PATH=str(tools) + os.pathsep + os.environ['PATH'], GRN_TEST_DIST=str(dist))
        return subprocess.run(["bash", str(SCRIPT.with_name("install.sh")), "--version", self.version,
                               "--bin-dir", str(self.root / "installed-bin"), "--data-dir", str(self.root / "installed-data"),
                               *(["--force"] if force else []), *(["--clients-only"] if clients_only else [])],
                              env=env, capture_output=True, text=True)

    def test_platform_neutral_clients_share_native_payloads_without_installing_binary(self):
        info = release.package(self.package_args())
        clients = self.root / "dist" / info['client_archive']
        self.assertEqual(release.sha256(clients), info['client_archive_sha256'])
        with tarfile.open(clients) as client_source, tarfile.open(self.root / 'dist' / info['archive']) as native_source:
            self.assertEqual(client_source.getnames(), sorted(info['payload_sha256']))
            self.assertNotIn('graphrag', client_source.getnames())
            self.assertNotIn('samples/first-notes.md', client_source.getnames())
            for name in client_source.getnames():
                self.assertEqual(client_source.extractfile(name).read(), native_source.extractfile(name).read())
        installed = self.install_fixture(self.root / 'dist', clients_only=True)
        self.assertEqual(installed.returncode, 0, installed.stderr)
        self.assertFalse((self.root / 'installed-bin/graphrag').exists())
        self.assertFalse((self.root / 'installed-data/samples').exists())
        bundle = self.root / 'installed-data/releases' / self.tag
        self.assertEqual((bundle / 'VERSION').read_text(), self.version + '\n')
        self.assertEqual((bundle / 'scripts/native-mcp/envelope.py').read_bytes(),
                         (self.repo / 'scripts/native-mcp/envelope.py').read_bytes())

    def test_installed_readme_links_and_environment_template_exist_without_overwriting_env(self):
        template = SCRIPT.parent.parent / ".env.example"
        readme = SCRIPT.parent.parent / "README.md"
        (self.repo / ".env.example").write_bytes(template.read_bytes())
        (self.repo / "README.md").write_bytes(readme.read_bytes())
        self.commit()
        info = release.package(self.package_args())
        self.assertIn("release/.env.example", info["payload_sha256"])
        self.assertNotIn("release/.env", info["payload_sha256"])
        data = self.root / "installed-data"
        data.mkdir()
        environment = data / ".env"
        environment.write_text("# Existing operator environment fixture; preserve\n")
        for clients_only in (False, True):
            installed = self.install_fixture(self.root / "dist", clients_only=clients_only)
            self.assertEqual(installed.returncode, 0, installed.stderr)
            bundle = data / "releases" / self.tag
            self.assertEqual((bundle / ".env.example").read_bytes(), template.read_bytes())
            for target in re.findall(r"\]\(([^)]+)\)", (bundle / "README.md").read_text()):
                if not target.startswith(("https://", "http://", "#")):
                    self.assertTrue((bundle / target.split("#", 1)[0]).is_file(), target)
            self.assertEqual(environment.read_text(), "# Existing operator environment fixture; preserve\n")
            self.assertFalse((bundle / ".env").exists())

    def test_packaged_openclaw_sources_preserve_documented_dependencies(self):
        source = SCRIPT.parents[1]
        plugin_files = [relative for relative in release.RELEASE_PAYLOADS
                        if relative.startswith("clients/openclaw-fast-notes/")]
        self.assertEqual(len(plugin_files), 13)
        source_payloads = plugin_files + ["docs/validation/openclaw-dispatch-108.md",
                                         "docs/validation/openclaw-browser-108.md",
                                         "docs/validation/openclaw-browser-108.json"]
        for relative in source_payloads:
            (self.repo / relative).write_bytes((source / relative).read_bytes())
        self.commit()
        self.build_record = self.root / "openclaw-client-build.json"
        self.seal_build()
        info = release.package(self.package_args())
        installed = self.install_fixture(self.root / "dist", clients_only=True)
        self.assertEqual(installed.returncode, 0, installed.stderr)
        bundle = self.root / "installed-data/releases" / self.tag
        plugin = bundle / "clients/openclaw-fast-notes"
        package = json.loads((plugin / "package.json").read_bytes())
        lock = json.loads((plugin / "package-lock.json").read_bytes())
        manifest = json.loads((plugin / "openclaw.plugin.json").read_bytes())
        self.assertEqual(package["version"], manifest["version"])
        self.assertEqual(package["version"], lock["packages"][""]["version"])
        for filename in package["files"] + ["package-lock.json"]:
            self.assertTrue((plugin / filename).is_file(), filename)
        for relative in source_payloads:
            contents = (bundle / relative).read_bytes()
            self.assertEqual(contents, (source / relative).read_bytes())
            self.assertEqual(info["payload_sha256"]["release/" + relative],
                             hashlib.sha256(contents).hexdigest())
        self.assertEqual(len(list(plugin.glob("*.test.mjs"))), 6)
        for document in (plugin / "README.md", bundle / "docs/validation/openclaw-dispatch-108.md",
                         bundle / "docs/validation/openclaw-browser-108.md"):
            for target in re.findall(r"\]\(([^)]+)\)", document.read_text()):
                if not target.startswith(("https://", "http://", "#")):
                    self.assertTrue((document.parent / target.split("#", 1)[0]).is_file(), target)
        self.assertTrue((plugin / "mcp-read-pool.mjs").is_file())
        self.assertFalse((bundle / ".env").exists())

    def test_packaged_clients_run_without_changing_versioned_bundle(self):
        # Exercise real importing entrypoints after packaging, with no providers.
        for relative in ("scripts/validate-native-mcp.py", "scripts/validate-daily-workflow.py",
                         "scripts/native-mcp/extended.py", "scripts/native-mcp/envelope.py",
                         "scripts/refresh-openclaw-memory.py", "scripts/openclaw_memory_refresh.py",
                         "scripts/benchmark-search.py", "scripts/evaluate-retrieval.py"):
            if not (SCRIPT.parents[1] / relative).exists():
                continue  # Draft dependencies are required before final source packaging.
            (self.repo / relative).write_bytes((SCRIPT.parents[1] / relative).read_bytes())
        self.commit()
        self.build_record = self.root / "client-use-build.json"
        self.seal_build()
        release.package(self.package_args())
        installed = self.install_fixture(self.root / "dist", clients_only=True)
        self.assertEqual(installed.returncode, 0, installed.stderr)
        bundle = self.root / "installed-data/releases" / self.tag
        driver = bundle / "scripts/validate-native-mcp.py"
        environment = {k: v for k, v in os.environ.items() if k not in
                       ("PYTHONDONTWRITEBYTECODE", "PYTHONPYCACHEPREFIX")}
        invoked = subprocess.run([sys.executable, str(driver), "--help"],
                                 capture_output=True, text=True, env=environment)
        self.assertEqual(invoked.returncode, 0, invoked.stderr)
        for name in ("refresh-openclaw-memory.py", "benchmark-search.py"):
            if (SCRIPT.parents[1] / "scripts" / name).exists():
                invoked = subprocess.run([sys.executable, str(bundle / "scripts" / name), "--help"],
                                         capture_output=True, text=True, env=environment)
                self.assertEqual(invoked.returncode, 0, invoked.stderr)
        loader = """import importlib.util,runpy,sys
runpy.run_path(sys.argv[1])
for path in sys.argv[2:]:
 spec=importlib.util.spec_from_file_location('installed_fixture',path)
 module=importlib.util.module_from_spec(spec)
 spec.loader.exec_module(module)
"""
        loaded = subprocess.run([sys.executable, "-c", loader, str(driver),
                                 str(bundle / "scripts/native-mcp/extended.py"),
                                 str(bundle / "scripts/validate-daily-workflow.py")],
                                capture_output=True, text=True, env=environment)
        self.assertEqual(loaded.returncode, 0, loaded.stderr)
        self.assertEqual(list(bundle.rglob("__pycache__")), [])
        reinstalled = self.install_fixture(self.root / "dist", clients_only=True)
        self.assertEqual(reinstalled.returncode, 0, reinstalled.stderr)

    def test_client_bundle_installs_without_python_and_refuses_edited_version(self):
        info = release.package(self.package_args())
        installed = self.install_fixture(self.root / "dist")
        self.assertEqual(installed.returncode, 0, installed.stderr)
        bundle = self.root / "installed-data/releases" / self.tag
        self.assertEqual((bundle / "VERSION").read_text(), self.version + "\n")
        self.assertEqual((bundle / "SOURCE-COMMIT").read_text(), self.commit_id + "\n")
        for path in release.RELEASE_PAYLOADS:
            self.assertEqual((bundle / path).read_bytes(), (self.repo / path).read_bytes())
        client = bundle / "scripts/refresh-openclaw-memory.py"
        client.write_text("operator edited versioned client\n")
        refused = self.install_fixture(self.root / "dist", force=True)
        self.assertNotEqual(refused.returncode, 0)
        self.assertIn("installed release bundle differs", refused.stderr)
        self.assertEqual(release.sha256(self.root / "installed-bin/graphrag"), info['binary_sha256'])
        self.assertEqual(client.read_text(), "operator edited versioned client\n")
        self.assertFalse((bundle.parent / (".install-" + self.tag + ".lock")).exists())

    def test_installer_refuses_client_or_metadata_tampering_before_binary_publication(self):
        info = release.package(self.package_args())
        dist = self.root / "dist"
        archive = dist / info['archive']
        with tarfile.open(archive) as source:
            payloads = {name: source.extractfile(name).read() for name in info['payload_sha256']}
        payloads['release/scripts/refresh-openclaw-memory.py'] += b"tampered\n"
        archive.unlink()
        release.deterministic_archive(archive, self.binary, self.repo / "samples/first-notes.md", 1, payloads)
        manifest = dist / "SHA256SUMS"
        original_manifest = manifest.read_text()
        manifest.write_text(original_manifest.replace(info['archive_sha256'], release.sha256(archive)))
        refused = self.install_fixture(dist)
        self.assertNotEqual(refused.returncode, 0)
        self.assertIn("release identity archive checksum differs", refused.stderr)
        self.assertFalse((self.root / "installed-bin/graphrag").exists())
        # Metadata is independently pinned by SHA256SUMS as well.
        (dist / "BUILDINFO.json").write_text((dist / "BUILDINFO.json").read_text() + "\n")
        refused = self.install_fixture(dist)
        self.assertIn("BUILDINFO checksum does not match", refused.stderr)
        self.assertFalse((self.root / "installed-bin/graphrag").exists())

    def rewrite_checksum(self, dist, name):
        manifest = dist / "SHA256SUMS"
        lines = manifest.read_text().splitlines()
        manifest.write_text("".join((release.sha256(dist / name) + "  " + name if line.split()[1] == name else line) + "\n" for line in lines))

    def test_installer_cross_binds_native_and_client_metadata_bytes(self):
        release.package(self.package_args())
        dist = self.root / "dist"
        for name, clients in [("BUILDINFO.json", False), ("CLIENTINFO.json", True)]:
            metadata = dist / name
            original = metadata.read_bytes()
            value = json.loads(original)
            value.update(version="9.9.9", source_commit="0" * 40)
            metadata.write_text(json.dumps(value))
            self.rewrite_checksum(dist, name)
            refused = self.install_fixture(dist, clients_only=clients)
            self.assertNotEqual(refused.returncode, 0)
            self.assertIn(name[:-5] + " bytes differ from release identity", refused.stderr)
            self.assertFalse((self.root / "installed-bin/graphrag").exists())
            self.assertFalse((self.root / "installed-data/releases" / self.tag).exists())
            metadata.write_bytes(original)
            self.rewrite_checksum(dist, name)

    def test_installer_refuses_mixed_flat_source_target_and_archive_identity(self):
        release.package(self.package_args())
        dist = self.root / "dist"
        for name, clients in [("BUILDINFO.identity", False), ("CLIENTINFO.identity", True)]:
            identity = dist / name
            original = identity.read_text()
            for key, replacement in [("source_commit", "0" * 40), ("target", "unexpected"),
                                     ("version", "9.9.9"), ("archive_sha256", "0" * 64)]:
                identity.write_text("".join((key + "=" + replacement if line.startswith(key + "=") else line) + "\n" for line in original.splitlines()))
                self.rewrite_checksum(dist, name)
                refused = self.install_fixture(dist, clients_only=clients)
                self.assertNotEqual(refused.returncode, 0, key)
                self.assertFalse((self.root / "installed-data/releases" / self.tag).exists())
            identity.write_text(original + "source_commit=" + "0" * 40 + "\n")
            self.rewrite_checksum(dist, name)
            self.assertIn("release identity fields are invalid", self.install_fixture(dist, clients_only=clients).stderr)
            identity.write_text(original)
            self.rewrite_checksum(dist, name)

    def test_real_packaged_archive_works_with_existing_installer_local_transport(self):
        # Only native-inspection output is doubled; the archive, checksum,
        # installer publication, refusal and sample preservation are real.
        info = release.package(self.package_args())
        evidence = self.root / "installed.json"
        subprocess.run(["bash", str(SCRIPT.with_name("verify-local-release.sh")), str(self.root / "dist"), self.version, str(evidence)], check=True, capture_output=True)
        installed = json.loads(evidence.read_text())
        self.assertEqual(installed["archive_sha256"], info["archive_sha256"])
        self.assertEqual(installed["binary_sha256"], info["binary_sha256"])
        self.assertGreaterEqual(installed["elapsed_seconds"], 0)
        self.assertEqual(installed["installer_invocations"], 3)
        self.assertIn("published download not tested", installed["transport"])
        validation = self.root / "final-validation.json"
        checks = {"local_asset_install": {"status": "passed", "evidence_file": str(evidence)}}
        for name, mode in [("offline_workflow", "offline"), ("native_live_workflow", "live")]:
            report = self.root / f"{name}.json"
            report.write_text(json.dumps({"success": True, "mode": mode, "runs": [
                {"success": True, "binary_sha256": info["binary_sha256"],
                 "steps": [{"success": True, "version": f"graphrag {self.version}"}]}]}))
            checks[name] = {"status": "passed", "evidence_file": str(report)}
        validation.write_text(json.dumps({"checks": checks}))
        args = self.package_args("sealed-final")
        args.validation_file = validation
        final = release.package(args)
        self.assertEqual(final["archive_sha256"], info["archive_sha256"])
        final_evidence = self.root / "final-installed.json"
        subprocess.run(["bash", str(SCRIPT.with_name("verify-local-release.sh")), str(args.output), self.version,
                        str(final_evidence)], check=True, capture_output=True)
        self.assertEqual(json.loads(final_evidence.read_text())["source_commit"], final["source_commit"])
        manifest = args.output / "SHA256SUMS"
        manifest.write_text(manifest.read_text().replace(final["archive_sha256"], "0" * 64))
        failure = subprocess.run(["bash", str(SCRIPT.with_name("verify-local-release.sh")), str(args.output), self.version,
                                  str(self.root / "failed-install.json")], capture_output=True, text=True)
        self.assertNotEqual(failure.returncode, 0)
        self.assertIn("archive checksum does not match", failure.stderr)


class WorkflowInvocationTests(unittest.TestCase):
    """Run the actual workflow shell with fake Cargo/packaging interfaces."""

    @staticmethod
    def workflow_run(name):
        path = SCRIPT.parent.parent / ".github/workflows/release.yml"
        lines = path.read_text().splitlines()
        start = lines.index(f"      - name: {name}")
        while lines[start] != "        run: |":
            start += 1
        block = []
        for line in lines[start + 1:]:
            if line and not line.startswith("          "):
                break
            block.append(line[10:] if line else "")
        return "\n".join(block) + "\n"

    def select_policy(self, root, version):
        (root / "Cargo.toml").write_text(f'[workspace.package]\nversion="{version}"\n')
        output = root / "outputs"
        env = {**os.environ, "GITHUB_OUTPUT": str(output)}
        result = subprocess.run(["bash", "-e", "-c", self.workflow_run("Select the checked-out release interface")],
                                cwd=root, env=env, capture_output=True, text=True)
        return result, output.read_text().strip().split("=", 1)[1] if output.exists() else None

    def invocation_fixture(self, root, version, target):
        result, policy = self.select_policy(root, version)
        self.assertEqual(result.returncode, 0, result.stderr)
        tools = root / "fake-tools"
        tools.mkdir()
        log = root / "calls.jsonl"
        runner = root / "runner"
        runner.mkdir()
        # The checked-out historical Cargo/package interfaces reject new flags.
        # These probes execute no actual Cargo, binary, installation or native tool.
        probe = r'''import json, os, pathlib, sys
kind = pathlib.Path(sys.argv[0]).name
args = sys.argv[1:]
modern = os.environ["EXPECTED_MODERN"] == "true"
with pathlib.Path(os.environ["CALL_LOG"]).open("a") as stream:
    stream.write(json.dumps({"command": kind, "args": args}) + "\n")
assert all(args), "empty argument"
if kind == "cargo":
    assert "--features" not in args, "experimental allocator is forbidden"
    print(json.dumps({"reason": "build-finished", "success": True}))
else:
    assert args[:2] in (["scripts/package-release.py", "record-build"], ["scripts/package-release.py", "package"])
    assert ("--cargo-messages" in args) == modern, "cargo-messages unavailable or missing"
    if modern:
        log = pathlib.Path(args[args.index("--cargo-messages") + 1])
        assert log == pathlib.Path(os.environ["RUNNER_TEMP"]) / ("cargo-" + os.environ["BUILD_TARGET"] + ".jsonl")
        assert log.is_file()
'''
        for command in ("cargo", "python3"):
            script = tools / command
            script.write_text(f"#!{sys.executable}\n" + probe)
            script.chmod(0o755)
        return {**os.environ, "PATH": str(tools) + os.pathsep + os.environ["PATH"],
                "BUILD_TARGET": target, "BUILD_FEATURES": "allocator",
                "COMPILER_FEATURE_PROOF": policy, "EXPECTED_MODERN": "true" if version != "0.1.0-rc.4" else "false",
                "CALL_LOG": str(log), "RUNNER_TEMP": str(runner), "RELEASE_TAG": "v" + version,
                "RELEASE_COMMIT": "a" * 40}, log

    def test_checked_out_policy_matches_release_contract(self):
        cases = [(f"0.1.0-rc.{n}", n >= 5) for n in range(1, 7)]
        cases += [("0.1.0-rc.12", True), ("0.1.0", True), ("0.2.0-rc.1", True), ("0.0.9", False)]
        for version, expected in cases:
            with self.subTest(version=version), tempfile.TemporaryDirectory() as temp:
                result, policy = self.select_policy(Path(temp), version)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(policy, str(expected).lower())
                self.assertEqual(release.requires_feature_proof(version), expected)
        with tempfile.TemporaryDirectory() as temp:
            result, policy = self.select_policy(Path(temp), "invalid")
            self.assertNotEqual(result.returncode, 0)
            self.assertIsNone(policy)

    def test_historical_and_current_real_workflow_invocations(self):
        steps = ["Build locked CLI", "Seal exact native binary and check version, help, and runtime dependencies",
                 "Package binary with starter notes"]
        for version in ("0.1.0-rc.4", "0.1.0-rc.5"):
            for target in release.TARGETS:
                with self.subTest(version=version, target=target), tempfile.TemporaryDirectory() as temp:
                    root = Path(temp)
                    env, log = self.invocation_fixture(root, version, target)
                    for name in steps:
                        result = subprocess.run(["bash", "-e", "-c", self.workflow_run(name)], cwd=root,
                                                env=env, capture_output=True, text=True)
                        self.assertEqual(result.returncode, 0, result.stderr)
                    calls = [json.loads(line) for line in log.read_text().splitlines()]
                    self.assertEqual(len(calls), 3)
                    self.assertEqual([item["command"] for item in calls], ["cargo", "python3", "python3"])

    def test_historical_arm_ungated_feature_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            env, _ = self.invocation_fixture(root, "0.1.0-rc.4", "aarch64-apple-darwin")
            former = self.workflow_run("Build locked CLI").replace(
                "cargo build --locked", "cargo build --features allocator --locked", 1)
            result = subprocess.run(["bash", "-e", "-c", former], cwd=root,
                                    env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("experimental allocator is forbidden", result.stderr)

    def test_historical_packaging_ungated_messages_are_rejected(self):
        for name in ("Seal exact native binary and check version, help, and runtime dependencies",
                     "Package binary with starter notes"):
            with self.subTest(step=name), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                env, _ = self.invocation_fixture(root, "0.1.0-rc.4", "x86_64-unknown-linux-gnu")
                env["COMPILER_FEATURE_PROOF"] = "true"
                result = subprocess.run(["bash", "-e", "-c", self.workflow_run(name)], cwd=root,
                                        env=env, capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("cargo-messages unavailable or missing", result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
