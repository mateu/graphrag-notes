# Getting started

The published [0.1.0-rc.2 prerelease](https://github.com/mateu/graphrag-notes/releases/tag/v0.1.0-rc.2)
includes capture, folder sync, keyword search, source navigation, review, and
the terminal workspace. Its binary targets **macOS Apple Silicon, macOS 15 or
newer**. The published `v0.1.0-rc.1` remains a historical first-run baseline
with import and hybrid search.
Native Linux and Intel macOS release acceptance is tracked in
[follow-up #66](https://github.com/mateu/graphrag-notes/issues/66); Windows is
outside this candidate's scope. You need memory and disk space for the two
models and internet access for downloads. The sample contains no personal notes.

## 1. Install the CLI

| Use case | Installation path |
| --- | --- |
| Apple Silicon daily-use binary | [Install rc.2](#published-daily-use-prerelease) |
| Development, Intel Macs, or Linux | [Build from source](#build-from-source) |
| Reproduce the published first-run baseline | [Install rc.1](#published-onboarding-baseline) |

### Published daily-use prerelease

Download the installer from the exact `v0.1.0-rc.2` tag, inspect it, and install
this candidate explicitly:

```bash
curl -fsSL https://raw.githubusercontent.com/mateu/graphrag-notes/v0.1.0-rc.2/scripts/install.sh \
  -o /tmp/graphrag-install.sh
bash /tmp/graphrag-install.sh --help
bash /tmp/graphrag-install.sh --version 0.1.0-rc.2
export PATH="$HOME/.local/bin:$PATH"
graphrag --version
```

Expect `graphrag 0.1.0-rc.2`. The installer verifies `SHA256SUMS` and installs
the binary to `~/.local/bin/graphrag` and starter notes to
`~/.local/share/graphrag-notes/samples/first-notes.md`. Use `--bin-dir` and
`--data-dir` for other destinations. It refuses to replace an existing binary;
follow [the upgrade steps](#upgrade-an-existing-installation) before using
`--force`. Existing starter notes are preserved.

The latest-release lookup selects stable releases and excludes prereleases,
so keep the explicit `--version 0.1.0-rc.2`. See the
[release guide](releases/0.1.0-rc.2.md) for archive contents and verification.
For a new configuration, continue with step 2. For an existing one, inspect it
with `graphrag init` without `--backend` or `--write`.

### Published onboarding baseline

The published `v0.1.0-rc.1` targets macOS 15 or newer on Apple Silicon. It
predates the daily-use commands covered below and in the other user guides.
To reproduce its onboarding path, download the installer from the exact tag,
inspect it, and select the prerelease explicitly:

```bash
curl -fsSL https://raw.githubusercontent.com/mateu/graphrag-notes/v0.1.0-rc.1/scripts/install.sh \
  -o /tmp/graphrag-install.sh
bash /tmp/graphrag-install.sh --help
bash /tmp/graphrag-install.sh --version 0.1.0-rc.1
export PATH="$HOME/.local/bin:$PATH"
graphrag --version
```

The installer verifies the archive against the candidate's `SHA256SUMS` before
installing the CLI and sample notes. It installs the binary to
`~/.local/bin/graphrag` and the sample to
`~/.local/share/graphrag-notes/samples/first-notes.md` by default; `--bin-dir` and
`--data-dir` let you choose other destinations. Check that `graphrag --version` reports `graphrag 0.1.0-rc.1`.

The installer's default latest-release lookup selects stable releases and
excludes prereleases, so this candidate requires `--version 0.1.0-rc.1`. There
is no default stable-binary installation path until a stable release is
published. Find published versions in
[GitHub Releases](https://github.com/mateu/graphrag-notes/releases).

The first candidate is scoped to an Apple Silicon archive. Intel Macs and
Linux use the source-build fallback below. The broader stable release workflow
is configured to build macOS Apple Silicon/Intel (macOS 15+) and Linux x86_64
(glibc 2.35+, Ubuntu 22.04 build baseline); Intel/Linux native builds,
published-asset installation, and live walkthroughs remain tracked in #66.

For a manual candidate install, download
`graphrag-notes-v0.1.0-rc.1-aarch64-apple-darwin.tar.gz` and `SHA256SUMS`. The
archive contains the executable and `samples/first-notes.md`; configuration is
created explicitly with `graphrag init`. Compute the archive hash with
`shasum -a 256 ARCHIVE`, replacing `ARCHIVE` with its downloaded filename, and
compare it with the matching line in `SHA256SUMS`. The hashes must match exactly
before extracting and installing it.

### Build from source

Build the published candidate's source for development, Intel Macs, or Linux,
whose native release acceptance remains in #66. If you already have the
candidate checkout, run setup from that checkout instead of cloning again.

Install [Rust](https://rustup.rs/) 1.97.1 or newer. The repository toolchain pins
1.97.1. Install these native build prerequisites before running setup:

| Platform | Native prerequisites |
| --- | --- |
| macOS | Xcode Command Line Tools (`xcode-select --install`), CMake, pkg-config, and Clang/libclang. Homebrew can provide them with `brew install cmake pkg-config llvm openssl@3`. |
| Ubuntu/Debian | `sudo apt-get install build-essential cmake pkg-config libssl-dev clang libclang-dev` |
| Other Linux | Install the equivalent compiler/build tools, CMake, pkg-config, OpenSSL development headers, and Clang/libclang packages for your distribution. |

```bash
git clone --branch v0.1.0-rc.2 https://github.com/mateu/graphrag-notes.git
cd graphrag-notes
./setup.sh
export PATH="$PWD/target/release:$PATH"
graphrag --version
graphrag --help
```

`setup.sh` reports missing prerequisites and builds the CLI incrementally. It
does not run a package manager, install Rust, download models, or start services.
`sccache` is optional. The PATH command above applies to the current terminal;
add the absolute `target/release` directory to your shell configuration if you
want to use this source build from other directories. Build dependencies may
need to be downloaded on the first build. Omit `--branch v0.1.0-rc.2` when
cloning if you want current development source instead of the tagged candidate.

## 2. Preview and create configuration

For a new installation, choose the Ollama preset:

```bash
graphrag init --backend ollama
graphrag init --backend ollama --write
```

The preview displays the configuration path, database path, provider endpoints,
model names, prerequisites, and next commands. It neither opens the database nor
contacts providers. `--write` creates only a new configuration file; it refuses
to overwrite an existing one and does not start services or install models.

Default paths are:

| Item | macOS | Linux |
| --- | --- | --- |
| Configuration | `~/Library/Application Support/graphrag/config.toml` | `$XDG_CONFIG_HOME/graphrag/config.toml`, or `~/.config/graphrag/config.toml` when XDG_CONFIG_HOME is unset |
| Database | `~/.graphrag/data-v3` | `~/.graphrag/data-v3` |

`graphrag init` prints the actual selected paths. A different location can be
selected explicitly before creating a new configuration:

```bash
graphrag --config ./local-config.toml --db-path ./local-notes-data \
  init --backend ollama
graphrag --config ./local-config.toml --db-path ./local-notes-data \
  init --backend ollama --write
```

Use `--config ./local-config.toml` for subsequent commands with that example.

**Existing installation:** run `graphrag init` without `--backend` or `--write`
to inspect your effective settings. A preset is intended for a new config and
is rejected when the selected configuration already exists. `init` preserves
runtime precedence: defaults, selected TOML, environment overrides, then CLI
overrides. The explicit backend preset changes the new-config defaults;
environment and CLI overrides still apply. A newly written file saves the
defaults/preset and any explicit `--db-path`; it does not copy transient
environment overrides. The preview shows effective settings and the settings
that will be saved. In JSON, `file_settings` describes that new-file template.
Check the preview if you previously exported `TEI_*`, `TGI_*`, `OLLAMA_URL`, or
`GRAPHRAG_*` variables.

## 3. Start Ollama and download models

Install [Ollama](https://ollama.com/download), then start it if it is not already
running. Keep this command running in a separate terminal:

```bash
ollama serve
```

In your GraphRAG terminal, download the selected models:

```bash
ollama pull bge-m3:latest
ollama pull phi4-mini:latest
graphrag init --check
```

The Ollama preset uses `http://localhost:11434`, `bge-m3:latest` for embeddings,
and `phi4-mini:latest` for entity extraction. The embeddings must have 1024
dimensions to match the current schema. If you customized model names, use the
model-specific commands printed by `init`.

`init --check` reuses provider diagnostics without opening the database. A
missing service or model reports a next command. It performs a real embedding
probe and checks the extraction provider, so it can take longer than the offline
preview. Machine-readable previews and checks use `graphrag init --format json`
and `graphrag init --check --format json`.

## 4. Import the sample and find a result

Run a full read-only diagnostic first:

```bash
graphrag doctor
```

A fresh installation can report a warning that the database does not exist.
This is expected before the first import. Once provider checks pass, import the
bundled notes from the binary installation:

```bash
graphrag import "$HOME/.local/share/graphrag-notes/samples/first-notes.md"
graphrag search "What is the Atlas project launch plan?" --limit 3
```

For a source build, run from the repository and use its sample path instead:

```bash
graphrag import samples/first-notes.md
graphrag search "What is the Atlas project launch plan?" --limit 3
```

Expect a hit containing the three launch stages: internal testing, a small pilot,
and opening access after pilot feedback. Result IDs, ranking, and extracted
entities may vary with your model. The first import creates the database and
applies application schema migrations. Repeating an unchanged file import is a
no-op. Finish with `graphrag doctor` to check the initialized database and active
embedding identity.

In rc.2 and current source, each result prints an
`Inspect` command for its full content, source heading/lines, and chat context,
plus an `Open source` command when it names a local file. Copy these commands;
they retain the selected configuration/database and check that the result has
not changed. Use full `note:`, `message:`, or `conversation:` IDs rather than
display numbers. Inspection and source opening work with providers stopped.
See [the navigation contract](cli-contract.md) for editor configuration and
bounded neighboring messages. The published `v0.1.0-rc.1` binary predates these
commands.

You can now use `graphrag add "your note"`, `graphrag import your-notes.md`, and
`graphrag search "your question"`. Database-only commands such as
`graphrag notes list` and `graphrag stats` work without running inference services.
You can also
search already indexed content offline using
`graphrag search "Atlas launch" --mode keyword --scope all`. This explicit mode
searches notes, original messages, and conversation summaries without requiring
vectors; hybrid remains the default. This mode requires current source or the
published rc.2 binary. Continue with the [daily workflow](daily-workflow.md),
or see [keyword search](keyword-search.md) for its exact contract.

## Using TEI and TGI instead

The runtime default remains TEI/TGI. The preset below gives a new installation
provider endpoints matching the repository's Docker Compose stack:

```bash
graphrag init --backend tei-tgi
graphrag init --backend tei-tgi --write
docker compose up -d
graphrag init --check
graphrag doctor
graphrag import samples/first-notes.md
graphrag search "What is the Atlas project launch plan?" --limit 3
```

Run Docker Compose from the repository. This checked-in stack requires a Linux
host with NVIDIA GPU support and the NVIDIA container runtime; use Ollama for
the macOS walkthrough. The stack serves TEI on `http://localhost:8081` and TGI
on `http://localhost:8082`, using `intfloat/e5-large-v2` for embeddings and
`Sreenington/Phi-3-mini-4k-instruct-AWQ` for extraction. Configure any model-access
token required by the
Docker stack in your local `.env`; do not commit it. Setup previews show the
selected model identities, but TEI/TGI model downloads and serving are managed
by the Docker stack.

## Upgrade an existing installation

For the published rc.2 installation, follow the commands below. The
[rc.3 release guide](releases/0.1.0-rc.3.md) records shared-service candidate
availability, matching installed clients, and publication/upgrade validation.
Once rc.3 is published, select its version explicitly in the same installer flow.

Before changing binaries, stop the service/workspace that owns the corpus,
quiesce remote clients, retain pending job/request recovery state, and use the working installation to create and verify
a portable backup at a new path:

```bash
graphrag backup create ./before-upgrade --include-embeddings
graphrag backup verify ./before-upgrade
```

Keep your usual `--config` and `--db-path` flags when needed. Including vectors
requires recorded embedding identity; otherwise omit `--include-embeddings`
and plan to reindex before hybrid retrieval after restore. The new binary
applies additive schema migrations when it opens the existing database.
`rc.1` supports schema 14, rc.2 supports schema 15, and the rc.3 source
candidate includes schema 19. An older binary cannot open the upgraded corpus. Returning
to an older binary requires a compatible pre-upgrade archive restored into a
fresh database. Do not lower schema metadata or remove fields with SQL.

After verifying the backup, download and inspect the rc.2 installer as shown
above. Deliberately replace the installed binary with:

```bash
bash /tmp/graphrag-install.sh --version 0.1.0-rc.2 --force
export PATH="$HOME/.local/bin:$PATH"
graphrag --version
graphrag init
```

Plain `init` inspects the existing configuration. Do not add `--backend` or
`--write` to that command. To keep a separate executable instead of replacing
one, select another directory and invoke that binary explicitly:

```bash
bash /tmp/graphrag-install.sh --version 0.1.0-rc.2 \
  --bin-dir "$HOME/.local/graphrag-rc2/bin"
"$HOME/.local/graphrag-rc2/bin/graphrag" --version
```

Separate executable directories do not make an upgraded database compatible
with the older binary; use a fresh database when comparing versions.

Portable backups preserve logical identities and source content while
redacting host-local paths. Restore may therefore need deliberate source
reconfiguration; keep original Markdown files separately. Follow the
[operating runbooks](operations.md) for restore and embedding reindexing.
The validation harness compares separate fresh databases for the old and new
binaries; it never opens an upgraded corpus with the old binary.

rc.3 clients/docs install below `DATA_DIR/releases/v0.1.0-rc.3`, preserving
configuration and edited starter notes. Installation requires curl, tar and SHA
tools; optional Python clients require Python 3.11+ to run. Keep capture journals,
refresh collection state, tokens and private relevance cases outside that public
version bundle. An existing modified version bundle is refused; inspect an
interrupted installer before removing its per-version installation lock.

## If setup stops

| Symptom | Next action |
| --- | --- |
| `cargo` or native build prerequisites missing | Install the prerequisite reported by `setup.sh`, then run it again. |
| `graphrag` not found | Add the source `target/release` directory or installer `~/.local/bin` directory to PATH, or use the absolute executable path. |
| Installed binary already exists | Verify a backup, then follow the explicit `--force` upgrade or choose another `--bin-dir`. |
| Config already exists | Run `graphrag init` to inspect it. Edit that file deliberately, or select a new path with `--config`; `--write` never overwrites it. |
| Config validation fails | Correct the reported setting in the selected TOML or environment, then run `graphrag config validate`. |
| Ollama is unreachable | Start `ollama serve` on the configured host, then rerun `graphrag init --check`. |
| Ollama model is missing | Run the `ollama pull MODEL` command printed for the selected model, then rerun the check. |
| Fresh database is absent | Once providers pass, import the sample to create it. |
| Database is locked | Stop the other GraphRAG process using that database path, then retry one command at a time. |
| Existing database uses a different embedding model | Keep its previous provider/model or follow the explicit reindex runbook; do not use a preset to change the database implicitly. |

The [operating runbooks](operations.md) cover backups, existing database engine
migration, schema diagnostics, and explicit embedding-model changes.
