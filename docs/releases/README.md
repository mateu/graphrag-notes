# Release preparation and provenance

Workspace package, local lockfile package, binary and tag versions must agree exactly. The performance and retained-policy follow-up candidate uses `0.1.0-rc.4` / `v0.1.0-rc.4`. The published rc.3 remains the immutable shared daily-use reliability checkpoint; neither candidate renames the old roadmap's v0.2 target. Publish the candidate only after review and merge, as a prerelease with `latest=false`. Never move an existing tag, overwrite an output directory, replace an existing release or upload with a clobber flag.

Python 3.11+ provides the packaging and offline checks without additional Python packages. Native inspection also uses Git, Rust, `file`, and `otool`/`sw_vers` on macOS or `ldd`/`readelf` on Linux. Packaging runs only version/help commands, with a temporary HOME and unavailable inference endpoints. It does not open the user's database, start providers, install models, run Cargo or publish anything.

## Build and seal a native binary

Use an isolated target directory and the declared Rust 1.97.1 toolchain. On Apple Silicon, the build environment is:

```bash
export CARGO_TARGET_DIR=/absolute/private/target-native
export CARGO_BUILD_JOBS=2 RUSTC_WRAPPER=''
export OPENSSL_STATIC=1 OPENSSL_DIR=/opt/homebrew/opt/openssl@3
export LIBCLANG_PATH=/opt/homebrew/opt/llvm/lib
export MACOSX_DEPLOYMENT_TARGET=15.0
cargo build --locked --release -p graphrag-cli --bin graphrag
python3 scripts/package-release.py record-build \
  --tag v0.1.0-rc.4 --expected-commit BUILD_COMMIT_SHA \
  --target aarch64-apple-darwin \
  --binary "$CARGO_TARGET_DIR/release/graphrag" \
  --output /absolute/private/native-build.json
```

Use a complete commit SHA and a fresh build-record filename. `record-build` rejects dirty or untracked compiled inputs, a mismatched version/toolchain/architecture, missing features, a wrong deployment target, or non-system runtime libraries. Docs and validation scripts can still be prepared while building. Seal only the binary produced by the recorded locked build, after that build exits successfully.

The compiled-input identity includes workspace manifests/lock/toolchain, `.cargo` configuration and workspace crate sources/manifests/build scripts. Final documentation or integration-test commits may advance the release commit without rebuilding only when this identity remains identical. Binary bytes must still match the sealed build hash. Packaging rechecks the recorded build commit/tree/timestamp, package versions and toolchain against Git and the validated compiled inputs. It rereads architecture, runtime libraries and deployment requirements from the binary, compares those facts and the current version/help results with the sealed record, and rejects inconsistent records. Git provenance documents this relationship; it does not promise bit-identical compiler outputs across different machines or SDKs.

## Package and verify local assets

Commit the reviewed sources before packaging. The final checkout must be clean, including untracked files; the expected SHA must be HEAD. Output parents must exist, and all output/evidence destinations must be fresh.

```bash
python3 scripts/package-release.py package \
  --tag v0.1.0-rc.4 --expected-commit FINAL_COMMIT_SHA \
  --binary "$CARGO_TARGET_DIR/release/graphrag" \
  --build-record /absolute/private/native-build.json \
  --output /absolute/private/preflight-assets
bash scripts/verify-local-release.sh \
  /absolute/private/preflight-assets 0.1.0-rc.4 \
  /absolute/private/local-install.json
```

The installer verification uses the real archive and checksums, the unchanged installer, a fresh HOME and system tools in PATH. Only the download transport is doubled to read those local assets. It checks installed payload equality, exact version, overwrite refusal, explicit force reinstallation and edited-sample/configuration preservation. The versioned client/document bundle must match byte for byte. The installer uses curl, tar and SHA tools without invoking Python; Python 3.11+ is needed only to run the bundled clients or release tooling. A version bundle is published once under `DATA_DIR/releases/vVERSION`, with an exclusive per-version installer lock; an edited existing bundle is refused rather than replaced. After an interrupted installer, inspect its stopped process before removing a stale lock. Historical two-file archives remain installable. This demonstrates local asset compatibility; published HTTPS installation is verified separately after release creation.

Archives contain `graphrag`, `samples/first-notes.md` and a fixed allowlist of public clients/documentation under `release/`, preserving helper imports and relative links. `release/VERSION`, `SOURCE-COMMIT`, `PAYLOADS.json` and `PAYLOADS.sha256` bind that bundle to the selected version and source. Checksum-pinned `BUILDINFO.identity`/`CLIENTINFO.identity` are bounded flat installer contracts, derived and validated against the semantic JSON during packaging/assembly. The installer cross-checks selected version/tag/target/archive, source commit, JSON byte digest and native binary/sample hashes with these fields; this detects mixed metadata/assets using shell and SHA tools. It does not parse or execute metadata as code. All members are regular files, with modes 755/644, zero UID/GID, fixed commit timestamps and a gzip header without a filename or wall-clock timestamp. Repackaging identical inputs at the same final commit produces identical archive bytes. BUILDINFO records every payload hash; corpus data, credentials, private evaluation cases/reports and models never enter the archive. Packaging needs the native inspection tools for the selected target, but does not require the build environment variables to be exported again. `rustc`, build-host version and build-environment entries remain historical operator-recorded build facts; packaging does not claim a new build at the final commit. Linux runtime inspection normalizes ASLR addresses while preserving library names/paths and required glibc versions.

## Attach validation evidence

Prepare a local JSON file with a `checks` object. Each check uses `passed`, `not_run` or `deferred`; failed/unknown statuses are rejected. Passed checks need an existing evidence file. Runtime evidence must correspond to the exact binary version/hash; source quality checks must identify the tested source commit. Absolute evidence paths remain local; public BUILDINFO includes only evidence basenames, hashes and short descriptions.

```json
{
  "checks": {
    "workspace_tests": {
      "status": "passed",
      "source_commit": "COMPLETE_TESTED_COMMIT_SHA",
      "evidence_file": "/absolute/private/workspace-tests.log"
    },
    "published_asset_install": {"status": "not_run"}
  }
}
```

Run final packaging into another fresh destination, with `--validation-file /absolute/private/validation.json --require-gates`. The required gates are `workspace_tests`, `clippy`, `offline_workflow`, `native_live_workflow` and `local_asset_install`. The final and preflight archive hashes must agree when they use the same final commit/binary/sample. Published installation remains explicitly `not_run` until it actually occurs; report that later result in the release notes without silently replacing sealed assets.

Offline regression command:

```bash
python3 scripts/test-release.py
```

The tests use temporary Git repositories, executable fixtures and native-tool fixtures. They verify source/version/lock consistency, compiled-input identity, reproducibility, no overwrite, malicious/missing archive inputs, runtime/deployment constraints, explicit evidence states, matrix assembly and the existing installer. They are independent of native Cargo builds and live inference.

## GitHub workflow

Stable `v*` tags without a prerelease suffix build the complete macOS ARM, macOS Intel and Linux x86_64 matrix. Candidate tags require explicit dispatch; dispatch can prepare full-matrix artifacts without publishing. Native jobs validate package/lock/tag consistency, build locked binaries, record native runtime facts and create the same deterministic packages. Assembly keeps each target's metadata separate, verifies hashes/provenance and produces one checksum manifest. The platform-neutral `clients.tar.gz`, `CLIENTINFO.json` and `CLIENTINFO.identity` must match across native jobs and are included once in the assembled output. Native BUILDINFO binds the same client archive/hash and public payloads. A Linux client can use the tagged installer with `--version 0.1.0-rc.4 --clients-only`; it installs matching scripts/docs without implying a native Linux executable is published.

The release workflow does not replace the normal source quality gates or a documented live walkthrough. BUILDINFO keeps any validation absent from that job explicit. The rc.4 candidate can be built and packaged locally for Apple Silicon while Intel/Linux native release validation remains tracked in [#66](https://github.com/mateu/graphrag-notes/issues/66). Publication uses an existing verified tag; `gh release create` fails if that release already exists.
