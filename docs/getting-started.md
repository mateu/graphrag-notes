# Getting started

This path takes a new local installation from an installed CLI to a searchable
fictional notebook. It uses Ollama on macOS or Linux. You need enough free memory
and disk space for the two models, and internet access to download build
dependencies and models. The sample contains no personal notes.

## 1. Install the CLI

### Build from source today

Versioned release artifacts are produced by the release workflow when a release
tag is published. Until those assets exist, use the source build below.

Install [Rust](https://rustup.rs/) 1.97.1 or newer. The repository toolchain pins
1.97.1. Install these native build prerequisites before running setup:

| Platform | Native prerequisites |
| --- | --- |
| macOS | Xcode Command Line Tools (`xcode-select --install`), CMake, pkg-config, and Clang/libclang. Homebrew can provide them with `brew install cmake pkg-config llvm openssl@3`. |
| Ubuntu/Debian | `sudo apt-get install build-essential cmake pkg-config libssl-dev clang libclang-dev` |
| Other Linux | Install the equivalent compiler/build tools, CMake, pkg-config, OpenSSL development headers, and Clang/libclang packages for your distribution. |

```bash
git clone https://github.com/mateu/graphrag-notes.git
cd graphrag-notes
./setup.sh
export PATH="$PWD/target/release:$PATH"
graphrag --help
```

`setup.sh` reports missing prerequisites and builds the CLI incrementally. It
does not run a package manager, install Rust, download models, or start services.
`sccache` is optional. The PATH command above applies to the current terminal;
add the absolute `target/release` directory to your shell configuration if you
want to use this source build from other directories.

### Install a published binary

Once a binary release is published, the installer selects the matching macOS
(Apple Silicon or Intel) or Linux x86_64 archive, verifies it against the release's
`SHA256SUMS`, and installs the CLI plus sample notes. Download and inspect the
installer, then run it:

```bash
curl -fsSL https://raw.githubusercontent.com/mateu/graphrag-notes/main/scripts/install.sh \
  -o /tmp/graphrag-install.sh
bash /tmp/graphrag-install.sh --help
bash /tmp/graphrag-install.sh
export PATH="$HOME/.local/bin:$PATH"
graphrag --help
```

Release binaries target macOS 15 or newer for Apple Silicon and Intel, and
Linux x86_64 with glibc 2.35 or newer (built on Ubuntu 22.04). For an older
operating system or an unsupported architecture, use the source-build fallback.

The installer defaults to the latest published release. To select a version,
pass `--version` with an existing release number or `v` tag from
[GitHub Releases](https://github.com/mateu/graphrag-notes/releases). It installs
the binary to `~/.local/bin/graphrag` and the sample to
`~/.local/share/graphrag-notes/samples/first-notes.md` by default; `--bin-dir` and
`--data-dir` let you choose other destinations. Run the version you installed
with `graphrag --version`. If no matching release is available for your platform,
use the source build.

For a manual install, each release archive is named
`graphrag-notes-v<VERSION>-<RUST_TARGET>.tar.gz` and contains the executable and
`samples/first-notes.md`. Create configuration explicitly with `graphrag init`.
Download its `SHA256SUMS` alongside the archive and verify the archive before extracting it. On macOS, compute the
archive hash with `shasum -a 256 ARCHIVE`; on Linux, use `sha256sum ARCHIVE`.
Replace `ARCHIVE` with your downloaded filename and compare the output with its
matching line in `SHA256SUMS`; the hashes must match exactly.

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
bundled notes from the repository:

```bash
graphrag import samples/first-notes.md
graphrag search "What is the Atlas project launch plan?" --limit 3
```

For the binary installer, use the installed sample path instead:

```bash
graphrag import "$HOME/.local/share/graphrag-notes/samples/first-notes.md"
graphrag search "What is the Atlas project launch plan?" --limit 3
```

Expect a hit containing the three launch stages: internal testing, a small pilot,
and opening access after pilot feedback. Result IDs, ranking, and extracted
entities may vary with your model. The first import creates the database and
applies application schema migrations. Repeating an unchanged file import is a
no-op. Finish with `graphrag doctor` to check the initialized database and active
embedding identity.

You can now use `graphrag add "your note"`, `graphrag import your-notes.md`, and
`graphrag search "your question"`. Database-only commands such as
`graphrag notes list` and `graphrag stats` work without running inference services.

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

## If setup stops

| Symptom | Next action |
| --- | --- |
| `cargo` or native build prerequisites missing | Install the prerequisite reported by `setup.sh`, then run it again. |
| `graphrag` not found | Add the source `target/release` directory or installer `~/.local/bin` directory to PATH, or use the absolute executable path. |
| Config already exists | Run `graphrag init` to inspect it. Edit that file deliberately, or select a new path with `--config`; `--write` never overwrites it. |
| Config validation fails | Correct the reported setting in the selected TOML or environment, then run `graphrag config validate`. |
| Ollama is unreachable | Start `ollama serve` on the configured host, then rerun `graphrag init --check`. |
| Ollama model is missing | Run the `ollama pull MODEL` command printed for the selected model, then rerun the check. |
| Fresh database is absent | Once providers pass, import the sample to create it. |
| Database is locked | Stop the other GraphRAG process using that database path, then retry one command at a time. |
| Existing database uses a different embedding model | Keep its previous provider/model or follow the explicit reindex runbook; do not use a preset to change the database implicitly. |

The [operating runbooks](operations.md) cover backups, existing database engine
migration, schema diagnostics, and explicit embedding-model changes.
