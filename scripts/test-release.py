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
        assembled = release.assemble(args)
        self.assertEqual(assembled["targets"], args.targets)
        self.assertTrue((args.output / f"BUILDINFO-{self.target}.json").is_file())
        info_path = self.root / "native/BUILDINFO.json"
        info_path.write_text(info_path.read_text() + "\n")
        args.output = self.root / "tampered-assembly"
        with self.assertRaisesRegex(release.ReleaseError, "metadata checksum"):
            release.assemble(args)

    def install_fixture(self, dist, force=False):
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
                               *(["--force"] if force else [])], env=env, capture_output=True, text=True)

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
        self.assertIn("release payload checksum does not match", refused.stderr)
        self.assertFalse((self.root / "installed-bin/graphrag").exists())
        # Metadata is independently pinned by SHA256SUMS as well.
        (dist / "BUILDINFO.json").write_text((dist / "BUILDINFO.json").read_text() + "\n")
        refused = self.install_fixture(dist)
        self.assertIn("BUILDINFO checksum does not match", refused.stderr)
        self.assertFalse((self.root / "installed-bin/graphrag").exists())

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
