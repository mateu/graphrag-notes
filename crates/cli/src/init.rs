//! First-run planning and explicit, no-clobber configuration creation.
//!
//! This path returns before application bootstrap: neither preview nor provider
//! checks open the database. A preset is a file-layer template, so the usual
//! environment and CLI precedence remains visible rather than being replaced.

use crate::app::inference_provider_config;
use crate::cli::{Cli, Commands, DoctorFormat, SetupBackend};
use crate::{doctor, output};
use anyhow::{bail, Context, Result};
use graphrag_config::{default_config_path, CliOverrides, ConfigError, RuntimeConfig};
use serde::Serialize;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

#[derive(Debug)]
pub(crate) struct InitCheckError(pub i32);

impl std::fmt::Display for InitCheckError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "Provider checks need attention; follow the setup commands above"
        )
    }
}

impl std::error::Error for InitCheckError {}

#[derive(Serialize, PartialEq, Eq)]
struct ProviderSettings {
    provider: String,
    model: String,
    endpoint: String,
}

impl ProviderSettings {
    fn embedding(config: &RuntimeConfig) -> Self {
        Self {
            provider: config.inference.embedding_provider.clone(),
            model: config.inference.embedding_model.clone(),
            endpoint: doctor::redact_endpoint(&config.inference.embedding_url),
        }
    }

    fn extraction(config: &RuntimeConfig) -> Self {
        Self {
            provider: config.inference.extraction_provider.clone(),
            model: config.inference.extraction_model.clone(),
            endpoint: doctor::redact_endpoint(&config.inference.extraction_url),
        }
    }
}

#[derive(Serialize)]
struct FileSettings {
    database_path: PathBuf,
    embedding: ProviderSettings,
    extraction: ProviderSettings,
}

#[derive(Serialize)]
struct InitReport {
    config_path: PathBuf,
    config_exists: bool,
    config_written: bool,
    database_path: PathBuf,
    embedding: ProviderSettings,
    extraction: ProviderSettings,
    /// Only the new template is persisted, not environment overrides.
    file_settings: Option<FileSettings>,
    prerequisites: Vec<String>,
    next_commands: Vec<String>,
    sample_path: Option<PathBuf>,
    diagnostics: Option<doctor::DoctorReport>,
}

struct InitPlan {
    config: RuntimeConfig,
    template: Option<RuntimeConfig>,
    report: InitReport,
}

pub(crate) async fn run(cli: &Cli) -> Result<()> {
    let Commands::Init {
        backend,
        write,
        check,
        format,
    } = &cli.command
    else {
        unreachable!("init handler requires the init command")
    };
    if cli.memory {
        bail!("init requires persistent settings; --memory cannot be used");
    }
    let env = |key: &str| std::env::var(key).ok();
    let mut plan = plan(cli, *backend, *write, &env, default_config_path())?;
    if *write {
        write_new_config(
            &plan.report.config_path,
            plan.template.as_ref().expect("new configuration template"),
        )?;
        plan.report.config_written = true;
        plan.report.config_exists = true;
    }
    if *check {
        plan.report.diagnostics =
            Some(doctor::run_setup(&inference_provider_config(&plan.config)).await);
    }
    let diagnostic_exit = plan
        .report
        .diagnostics
        .as_ref()
        .map(|report| report.exit_code);
    let output_format = match format {
        DoctorFormat::Human => output::OutputFormat::Human,
        DoctorFormat::Json => output::OutputFormat::Json,
    };
    let mut envelope = output::OutputEnvelope::success("init", &plan.report);
    if let Some(report) = &plan.report.diagnostics {
        envelope.success = report.exit_code == doctor::EXIT_HEALTHY;
        for check in &report.checks {
            let message = format!("{}: {}", check.name, check.summary);
            match check.status {
                doctor::Status::Warning => envelope.warnings.push(message),
                doctor::Status::Failed => envelope.errors.push(message),
                doctor::Status::Healthy => {}
            }
        }
    }
    output::print_envelope(output_format, &envelope, |writer| {
        render_human(writer, &plan.report)
    })?;
    if let Some(exit_code) = diagnostic_exit.filter(|code| *code != 0) {
        return Err(InitCheckError(exit_code).into());
    }
    Ok(())
}

fn plan(
    cli: &Cli,
    backend: Option<SetupBackend>,
    write: bool,
    env: &impl Fn(&str) -> Option<String>,
    default_path: Option<PathBuf>,
) -> Result<InitPlan> {
    let config_path = cli
        .config
        .clone()
        .or_else(|| {
            env("GRAPHRAG_CONFIG")
                .filter(|value| !value.trim().is_empty())
                .map(|value| expand_home(&value, env))
        })
        .or(default_path)
        .context("init requires --config PATH because no platform config directory is available")?;
    if config_path.as_os_str().is_empty() {
        bail!("init requires a non-empty --config PATH");
    }
    // symlink_metadata also treats dangling symlinks as existing targets;
    // preview/read errors must never turn them into writable new configs.
    let exists = match std::fs::symlink_metadata(&config_path) {
        Ok(_) => true,
        Err(error) if error.kind() == io::ErrorKind::NotFound => false,
        Err(error) => return Err(error).context("Cannot inspect config target"),
    };
    if exists && write {
        bail!("Refusing to overwrite existing config {}. Run `graphrag init` without --write to inspect it.", config_path.display());
    }
    if exists && backend.is_some() {
        bail!("--backend cannot replace an existing config; preview without --backend or select a new --config PATH");
    }

    let overrides = CliOverrides {
        database_path: cli.db_path.clone(),
    };
    let (config, template) = if exists {
        (
            RuntimeConfig::load_with_env_and_default_path(
                Some(&config_path),
                &overrides,
                env,
                None,
            )?,
            None,
        )
    } else {
        let mut template = RuntimeConfig::default();
        if matches!(backend, Some(SetupBackend::Ollama)) {
            template.inference.embedding_provider = "ollama".into();
            template.inference.extraction_provider = "ollama".into();
            template.inference.embedding_url = template.inference.ollama_url.clone();
            template.inference.extraction_url = template.inference.ollama_url.clone();
        }
        if matches!(backend, Some(SetupBackend::TeiTgi)) {
            // Match the explicit Docker Compose preset, without changing the
            // historic defaults for callers that choose no preset.
            template.inference.embedding_model = "intfloat/e5-large-v2".into();
            template.inference.extraction_model = "Sreenington/Phi-3-mini-4k-instruct-AWQ".into();
        }
        // Persist only explicit setup choices; an environment override is
        // transient and must not be silently copied into a new file.
        if let Some(path) = &cli.db_path {
            template.database.path = path.clone();
        }
        template.validate()?;
        let effective = template.resolve_file_template(&overrides, env)?;
        (effective, Some(template))
    };

    let sample_path = find_sample(env);
    let db_flag = cli.db_path.as_ref().map_or(String::new(), |path| {
        format!(" --db-path {}", shell_quote(&path.to_string_lossy()))
    });
    let command = format!(
        "graphrag --config {}{db_flag}",
        shell_quote(&config_path.to_string_lossy()),
    );
    let mut prerequisites = vec![
        "The release binary needs no Rust toolchain; source builds need Rust 1.97.1+, a C/C++ toolchain, CMake, libclang, pkg-config and OpenSSL development files.".into(),
        "Inference models are installed explicitly; setup never installs software or models.".into(),
    ];
    if config.inference.embedding_provider == "ollama"
        || config.inference.extraction_provider == "ollama"
    {
        prerequisites.push("Install Ollama if needed, then start it: ollama serve".into());
        if config.inference.embedding_provider == "ollama" {
            prerequisites.push(format!(
                "On the Ollama host, install the embedding model: ollama pull {}",
                shell_quote(&config.inference.embedding_model)
            ));
        }
        if config.inference.extraction_provider == "ollama" {
            prerequisites.push(format!(
                "On the Ollama host, install the extraction model: ollama pull {}",
                shell_quote(&config.inference.extraction_model)
            ));
        }
    }
    if config.inference.embedding_provider == "tei" || config.inference.extraction_provider == "tgi"
    {
        prerequisites.push("Start the configured TEI/TGI services. For the repository Docker setup: docker compose up -d (see docs/getting-started.md for hardware requirements).".into());
    }
    prerequisites.push("Embeddings must have 1024 dimensions. Run doctor to verify model compatibility before ingestion.".into());
    let mut next_commands = Vec::new();
    if !exists && !write {
        let preset = match backend {
            Some(SetupBackend::Ollama) => " --backend ollama",
            Some(SetupBackend::TeiTgi) => " --backend tei-tgi",
            None => "",
        };
        next_commands.push(format!("{command} init{preset} --write"));
    }
    next_commands.push(format!("{command} config validate"));
    next_commands.push(format!("{command} doctor"));
    if let Some(path) = &sample_path {
        next_commands.push(format!(
            "{command} import {}",
            shell_quote(&path.to_string_lossy())
        ));
        next_commands.push(format!("{command} search 'project Atlas'"));
    } else {
        prerequisites.push("The release archive includes samples/first-notes.md; the installer places it under ~/.local/share/graphrag-notes/samples. Import that file or your own Markdown file after doctor.".into());
    }
    let report = InitReport {
        config_path,
        config_exists: exists,
        config_written: false,
        database_path: config.database.path.clone(),
        embedding: ProviderSettings::embedding(&config),
        extraction: ProviderSettings::extraction(&config),
        file_settings: template.as_ref().map(|config| FileSettings {
            database_path: config.database.path.clone(),
            embedding: ProviderSettings::embedding(config),
            extraction: ProviderSettings::extraction(config),
        }),
        prerequisites,
        next_commands,
        sample_path,
        diagnostics: None,
    };
    Ok(InitPlan {
        config,
        template,
        report,
    })
}

fn expand_home(value: &str, env: &impl Fn(&str) -> Option<String>) -> PathBuf {
    if let Some(tail) = value.strip_prefix("~/") {
        if let Some(home) = env("HOME") {
            return PathBuf::from(home).join(tail);
        }
    }
    PathBuf::from(value)
}

fn find_sample(env: &impl Fn(&str) -> Option<String>) -> Option<PathBuf> {
    let mut candidates = vec![std::env::current_dir().ok()?.join("samples/first-notes.md")];
    if let Ok(executable) = std::env::current_exe() {
        if let Some(parent) = executable.parent() {
            candidates.push(parent.join("samples/first-notes.md"));
        }
    }
    if let Some(home) = env("HOME") {
        candidates
            .push(PathBuf::from(home).join(".local/share/graphrag-notes/samples/first-notes.md"));
    }
    candidates.into_iter().find(|path| path.is_file())
}

fn shell_quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\\''"))
}

fn write_new_config(path: &Path, config: &RuntimeConfig) -> Result<()> {
    let content = config.redacted_toml()?;
    let parent = path
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    std::fs::create_dir_all(parent).context("Cannot create config parent directory")?;
    let mut staged = tempfile::NamedTempFile::new_in(parent).context("Cannot stage new config")?;
    staged
        .write_all(content.as_bytes())
        .context("Cannot write staged config")?;
    staged
        .as_file()
        .sync_all()
        .context("Cannot flush staged config")?;
    staged.persist_noclobber(path).map_err(|error| {
        if error.error.kind() == io::ErrorKind::AlreadyExists {
            anyhow::Error::new(ConfigError::Validation(format!(
                "Refusing to overwrite existing config {}",
                path.display()
            )))
        } else {
            anyhow::Error::new(error.error).context("Cannot publish new config")
        }
    })?;
    Ok(())
}

fn render_human(writer: &mut dyn Write, report: &InitReport) -> io::Result<()> {
    writeln!(writer, "GraphRAG first-run setup")?;
    writeln!(
        writer,
        "Config: {} ({})",
        report.config_path.display(),
        if report.config_written {
            "created"
        } else if report.config_exists {
            "existing; unchanged"
        } else {
            "preview; not created"
        }
    )?;
    writeln!(
        writer,
        "Database: {} (not opened)",
        report.database_path.display()
    )?;
    print_provider(writer, "Embeddings", &report.embedding)?;
    print_provider(writer, "Extraction", &report.extraction)?;
    if let Some(settings) = &report.file_settings {
        writeln!(
            writer,
            "\nNew file settings (environment overrides are not saved):"
        )?;
        writeln!(writer, "  Database: {}", settings.database_path.display())?;
        print_provider(writer, "  Embeddings", &settings.embedding)?;
        print_provider(writer, "  Extraction", &settings.extraction)?;
    }
    if let Some(diagnostics) = &report.diagnostics {
        writeln!(writer, "\nProvider-only checks:")?;
        write!(writer, "{}", diagnostics.render_human())?;
    } else {
        writeln!(
            writer,
            "\nProviders were not contacted. Add --check for provider-only diagnostics."
        )?;
    }
    writeln!(writer, "\nPrerequisites and recovery:")?;
    for prerequisite in &report.prerequisites {
        writeln!(writer, "  {prerequisite}")?;
    }
    writeln!(writer, "\nNext commands (start providers first):")?;
    for command in &report.next_commands {
        writeln!(writer, "  {command}")?;
    }
    Ok(())
}

fn print_provider(
    writer: &mut dyn Write,
    label: &str,
    provider: &ProviderSettings,
) -> io::Result<()> {
    writeln!(
        writer,
        "{label}: {} / {} at {}",
        provider.provider, provider.model, provider.endpoint
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn file_preset_and_environment_keep_normal_precedence() {
        let temp = tempfile::tempdir().unwrap();
        let cli = Cli::parse_from([
            "graphrag",
            "--db-path",
            "/explicit-db",
            "init",
            "--backend",
            "ollama",
        ]);
        let env = |key: &str| match key {
            "TEI_URL" => Some("http://remote:11434".into()),
            "GRAPHRAG_DB_PATH" => Some("/environment-db".into()),
            _ => None,
        };
        let plan = plan(
            &cli,
            Some(SetupBackend::Ollama),
            false,
            &env,
            Some(temp.path().join("config.toml")),
        )
        .unwrap();
        assert_eq!(plan.config.database.path, PathBuf::from("/explicit-db"));
        assert_eq!(plan.config.inference.embedding_url, "http://remote:11434");
        let template = plan.template.unwrap();
        assert_eq!(template.inference.embedding_url, "http://localhost:11434");
        assert_eq!(template.database.path, PathBuf::from("/explicit-db"));
        assert!(!temp.path().join("config.toml").exists());
    }

    #[test]
    fn atomic_publication_refuses_a_competing_file_or_symlink() {
        let temp = tempfile::tempdir().unwrap();
        let target = temp.path().join("config.toml");
        std::fs::write(&target, "existing user settings").unwrap();
        assert!(write_new_config(&target, &RuntimeConfig::default()).is_err());
        assert_eq!(
            std::fs::read_to_string(&target).unwrap(),
            "existing user settings"
        );
        assert_eq!(std::fs::read_dir(temp.path()).unwrap().count(), 1);
        #[cfg(unix)]
        {
            let symlink = temp.path().join("dangling.toml");
            std::os::unix::fs::symlink(temp.path().join("absent.toml"), &symlink).unwrap();
            assert!(write_new_config(&symlink, &RuntimeConfig::default()).is_err());
            assert!(!temp.path().join("absent.toml").exists());
        }
    }

    #[test]
    fn endpoints_are_redacted_and_shell_commands_quote_arguments() {
        let temp = tempfile::tempdir().unwrap();
        let cli = Cli::parse_from(["graphrag", "init"]);
        let env = |key: &str| {
            (key == "TEI_URL").then(|| "http://user:secret@localhost:8081/?token=secret".into())
        };
        let path = temp.path().join("a'b space.toml");
        let plan = plan(&cli, None, false, &env, Some(path)).unwrap();
        let json = serde_json::to_string(&plan.report).unwrap();
        assert!(!json.contains("secret"));
        assert!(plan.report.next_commands[0].contains("a'\\''b space.toml"));
    }

    #[test]
    fn preview_matches_resolution_after_the_complete_template_is_saved() {
        let temp = tempfile::tempdir().unwrap();
        let cli = Cli::parse_from(["graphrag", "init"]);
        // An explicit endpoint in TOML takes priority over OLLAMA_URL alone.
        // Legacy chunk fallbacks also differ between absent and explicit fields.
        let env = |key: &str| match key {
            "OLLAMA_URL" => Some("http://remote:11434".into()),
            "GRAPHRAG_LIBRARIAN_MAX_CHUNK_SIZE" => Some("1".into()),
            _ => None,
        };
        let path = temp.path().join("config.toml");
        let error = plan(
            &cli,
            Some(SetupBackend::Ollama),
            false,
            &env,
            Some(path.clone()),
        );
        assert!(
            error.is_err(),
            "explicit serialized chunk fields cannot be silently clamped"
        );
        let env = |key: &str| (key == "OLLAMA_URL").then(|| "http://remote:11434".into());
        let plan = plan(
            &cli,
            Some(SetupBackend::Ollama),
            false,
            &env,
            Some(path.clone()),
        )
        .unwrap();
        write_new_config(&path, plan.template.as_ref().unwrap()).unwrap();
        let reopened = RuntimeConfig::load_with_env_and_default_path(
            Some(&path),
            &CliOverrides::default(),
            &env,
            None,
        )
        .unwrap();
        assert_eq!(plan.config, reopened);
    }
}
