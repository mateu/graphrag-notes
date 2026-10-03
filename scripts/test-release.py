#!/usr/bin/env python3
"""Offline release checks using temporary repositories and native-tool fixtures."""

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import unittest
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
        self.version = "0.1.0-rc.2"
        self.tag = f"v{self.version}"
        (self.repo / "Cargo.toml").write_text(
            '[workspace]\nmembers = ["crates/cli", "crates/core"]\n'
            '[workspace.package]\nversion = "0.1.0-rc.2"\nrust-version = "1.97.1"\n')
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
        self.git("init", "-q")
        self.git("config", "user.name", "Offline fixture")
        self.git("config", "user.email", "fixture@example.invalid")
        self.commit()
        self.binary = self.root / "graphrag"
        tokens = " ".join(token for values in release.HELP_CHECKS.values() for token in values)
        self.binary.write_text(f'#!/bin/sh\nif [ "$1" = "--version" ]; then printf "graphrag {self.version}\\n"; else printf "%s\\n" "{tokens}"; fi\n')
        self.binary.chmod(0o755)
        self.build_record = self.root / "build.json"
        self.facts = {"target": "aarch64-apple-darwin", "rustc": "rustc 1.97.1 fixture",
                      "minimum_macos": "15.0", "runtime_libraries": ["/usr/lib/libSystem.B.dylib"],
                      "system_libraries_only": True}
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
            self.assertEqual(archive.getnames(), ["graphrag", "samples/first-notes.md"])
            self.assertTrue(all(member.uid == member.gid == 0 for member in archive.getmembers()))

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
            release.inspect_archive(self.root / "dist" / info["archive"], "0" * 64, info["sample_sha256"])

    def test_assembly_requires_all_targets_consistent_provenance_and_valid_checksums(self):
        release.package(self.package_args("arm"))
        args = argparse.Namespace(input=self.root, output=self.root / "assembly", tag=self.tag,
                                  targets=["aarch64-apple-darwin", "x86_64-apple-darwin"])
        with self.assertRaisesRegex(release.ReleaseError, "missing targets"):
            release.assemble(args)
        args.targets = ["aarch64-apple-darwin"]
        assembled = release.assemble(args)
        self.assertEqual(assembled["targets"], args.targets)
        self.assertTrue((args.output / "BUILDINFO-aarch64-apple-darwin.json").is_file())
        info_path = self.root / "arm/BUILDINFO.json"
        info_path.write_text(info_path.read_text() + "\n")
        args.output = self.root / "tampered-assembly"
        with self.assertRaisesRegex(release.ReleaseError, "metadata checksum"):
            release.assemble(args)

    def test_real_packaged_archive_works_with_existing_installer_local_transport(self):
        # Only native-inspection output is doubled; the archive, checksum,
        # installer publication, refusal and sample preservation are real.
        if os.uname().sysname == "Linux":
            data = json.loads(self.build_record.read_text())
            data["native"]["target"] = "x86_64-unknown-linux-gnu"
            self.build_record.write_text(json.dumps(data))
        elif os.uname().machine == "x86_64":
            data = json.loads(self.build_record.read_text())
            data["native"]["target"] = "x86_64-apple-darwin"
            self.build_record.write_text(json.dumps(data))
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
