//! Blocking argv editors and durable, private recovery drafts.
use super::navigation::{shell_quote, validate_argv};
use anyhow::{Context, Result};
use clap::Args;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

#[derive(Debug, Default, Args)]
pub struct EditorOptions {
    /// Edit a durable draft with a blocking editor before saving.
    #[arg(long, conflicts_with_all = ["stdin", "content_file"])]
    pub editor: bool,
    /// Editor executable path; otherwise VISUAL, EDITOR, then vi. No shell parsing.
    #[arg(long, value_name = "EXECUTABLE", requires = "editor")]
    pub editor_command: Option<PathBuf>,
    /// Literal editor argument; repeat as needed (use --editor-arg=--wait).
    #[arg(
        long,
        value_name = "ARG",
        requires = "editor",
        allow_hyphen_values = true
    )]
    pub editor_arg: Vec<String>,
    /// Persistent recovery directory; defaults to DATABASE.drafts beside the database.
    #[arg(long, value_name = "PATH")]
    pub draft_dir: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EditorOutcome {
    Changed,
    Unchanged,
    Cancelled,
}

#[derive(Debug)]
pub struct Draft {
    pub path: PathBuf,
    pub recovery_command: String,
    retained: bool,
}

impl Draft {
    pub fn save(
        bytes: &[u8],
        directory: &Path,
        recovery: impl FnOnce(&Path) -> String,
    ) -> Result<Self> {
        let mut builder = std::fs::DirBuilder::new();
        builder.recursive(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::DirBuilderExt;
            builder.mode(0o700);
        }
        builder
            .create(directory)
            .with_context(|| format!("cannot create draft directory {}", directory.display()))?;
        let mut file = tempfile::Builder::new()
            .prefix("graphrag-")
            .suffix(".md")
            .tempfile_in(directory)?;
        file.write_all(bytes)?;
        file.as_file().sync_all()?;
        let (_, path) = file.keep().map_err(|error| error.error)?;
        let recovery_command = recovery(&path);
        Ok(Self {
            path,
            recovery_command,
            retained: true,
        })
    }

    pub fn read(&self) -> Result<String> {
        self.secure()?;
        let bytes = std::fs::read(&self.path)?;
        String::from_utf8(bytes).map_err(|_| {
            super::capture::CaptureValidationError("draft content must be valid UTF-8".into())
                .into()
        })
    }

    fn secure(&self) -> Result<()> {
        let metadata = std::fs::symlink_metadata(&self.path)?;
        if metadata.file_type().is_symlink() || !metadata.is_file() {
            return Err(super::capture::CaptureValidationError(
                "draft must remain a regular file, not a symlink".into(),
            )
            .into());
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&self.path, std::fs::Permissions::from_mode(0o600))?;
        }
        Ok(())
    }

    pub fn edit(&self, options: &EditorOptions, original: &str) -> Result<(EditorOutcome, String)> {
        let executable = options
            .editor_command
            .clone()
            .or_else(|| {
                std::env::var_os("VISUAL")
                    .or_else(|| std::env::var_os("EDITOR"))
                    .map(PathBuf::from)
            })
            .unwrap_or_else(|| PathBuf::from("vi"));
        let mut argv = vec![executable
            .to_str()
            .context("editor executable path must be valid UTF-8")?
            .to_string()];
        argv.extend(options.editor_arg.iter().cloned());
        validate_argv(&argv)?;
        eprintln!("Editing draft: {}", self.path.display());
        let status = Command::new(&argv[0])
            .args(&argv[1..])
            .arg(&self.path)
            .stdin(Stdio::inherit())
            .stdout(Stdio::from(std::io::stderr()))
            .stderr(Stdio::inherit())
            .status()
            .with_context(|| format!("failed to launch editor {:?}", argv[0]))?;
        if !status.success() {
            // A cancelled editor can leave non-UTF-8 recovery bytes. Preserve
            // the file without interpreting or validating discarded input.
            self.secure()?;
            return Ok((EditorOutcome::Cancelled, String::new()));
        }
        // Editors often save through a rename; read the pathname after exit.
        let content = self.read()?;
        let outcome = if content == original {
            EditorOutcome::Unchanged
        } else {
            EditorOutcome::Changed
        };
        Ok((outcome, content))
    }

    pub fn discard(&mut self, note_id: Option<&str>) -> bool {
        // Completion already succeeded. Never emit a retry command or turn
        // cleanup failure into a failed capture that might be retried twice.
        self.retained = false;
        match std::fs::remove_file(&self.path) {
            Ok(()) => true,
            Err(error) => {
                eprintln!("Operation completed{}; draft retained at {} because cleanup failed: {error}. Remove it when no longer needed.", note_id.map(|id| format!(" for note {id}")).unwrap_or_default(), self.path.display());
                false
            }
        }
    }
}

impl Drop for Draft {
    fn drop(&mut self) {
        if self.retained {
            eprintln!(
                "Draft retained: {}\nRecover: {}",
                self.path.display(),
                self.recovery_command
            );
        }
    }
}

pub fn directory(options: &EditorOptions, database: &Path) -> Result<PathBuf> {
    let mut default = database.as_os_str().to_os_string();
    default.push(".drafts");
    let directory = absolute(options.draft_dir.as_deref().unwrap_or(Path::new(&default)))?;
    recovery_path(&directory)?;
    Ok(directory)
}

pub fn format_flag(format: crate::output::OutputFormat) -> &'static str {
    match format {
        crate::output::OutputFormat::Human => "human",
        crate::output::OutputFormat::Json => "json",
        crate::output::OutputFormat::Jsonl => "jsonl",
    }
}

fn absolute(path: &Path) -> Result<PathBuf> {
    Ok(if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()?.join(path)
    })
}

pub fn base_command(config: Option<&Path>, database: &Path) -> Result<String> {
    let mut base = "graphrag".to_string();
    if let Some(path) = config {
        base.push_str(&format!(
            " --config {}",
            shell_quote(recovery_path(&absolute(path)?)?)
        ));
    }
    base.push_str(&format!(
        " --db-path {}",
        shell_quote(recovery_path(&absolute(database)?)?)
    ));
    Ok(base)
}

fn recovery_path(path: &Path) -> Result<&str> {
    path.to_str().ok_or_else(|| {
        super::capture::CaptureValidationError(format!(
            "configuration, database, and draft paths must be valid UTF-8 for copyable recovery commands: {}",
            path.display()
        )).into()
    })
}
