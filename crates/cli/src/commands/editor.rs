//! Blocking argv editors and durable, private recovery drafts.
use super::navigation::{shell_quote, validate_argv};
use anyhow::{Context, Result};
use clap::Args;
use std::io::{Read, Write};
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
    recovery: Option<RecoveredInput>,
}

#[derive(Debug)]
struct RecoveredInput {
    file: std::fs::File,
    bytes: Vec<u8>,
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
            recovery: None,
        })
    }

    /// Only an explicit recovery request transfers ownership of an input file.
    /// Ordinary content files always go through `save` and remain caller-owned.
    pub fn save_or_recover(
        bytes: &[u8],
        directory: &Path,
        recovery_path: Option<&Path>,
        recovery: impl FnOnce(&Path) -> String,
    ) -> Result<Self> {
        let Some(path) = recovery_path else {
            return Self::save(bytes, directory, recovery);
        };
        let path = absolute(path)?;
        let mut file = Self::open_recovery_file(&path)?;
        if !Self::matches_recovery_bytes(&mut file, bytes)? {
            anyhow::bail!("recovery draft changed while preparing the retry; retain it and review before sending");
        }
        Ok(Self {
            recovery_command: recovery(&path),
            path,
            retained: true,
            recovery: Some(RecoveredInput {
                file,
                bytes: bytes.to_vec(),
            }),
        })
    }

    /// Check before opening recovery input so a FIFO cannot block the retry.
    pub fn validate_recovery_path(path: &Path) -> Result<()> {
        Self::open_recovery_file(path).map(|_| ())
    }

    pub fn read_recovery_input(path: &Path, max_read_bytes: usize) -> Result<Vec<u8>> {
        let mut bytes = Vec::new();
        Self::open_recovery_file(path)?
            .take(max_read_bytes as u64)
            .read_to_end(&mut bytes)?;
        Ok(bytes)
    }

    /// Pin the final remote payload after any editor changes. Acknowledgment
    /// cleanup must preserve later saves for fresh drafts as well as retries.
    /// Embedded editor flows keep their existing ownership and cleanup rules.
    pub fn pin_remote_submission(&mut self, bytes: &[u8]) -> Result<()> {
        if let Some(recovered) = &self.recovery {
            if recovered.bytes != bytes {
                anyhow::bail!("remote submission differs from the adopted recovery draft; retain it and review before sending");
            }
            return Ok(());
        }
        let mut file = Self::open_recovery_file(&self.path)?;
        if !Self::matches_recovery_bytes(&mut file, bytes)? {
            anyhow::bail!("remote draft changed while preparing submission; retain it and review before sending");
        }
        self.recovery = Some(RecoveredInput {
            file,
            bytes: bytes.to_vec(),
        });
        Ok(())
    }

    fn open_recovery_file(path: &Path) -> Result<std::fs::File> {
        #[cfg(unix)]
        {
            use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
            let file = std::fs::OpenOptions::new()
                .read(true)
                .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
                .open(path)
                .context("recovery draft must be a private regular file without symlinks")?;
            let metadata = file.metadata()?;
            if !metadata.is_file() {
                anyhow::bail!("recovery draft must be a regular file, not a directory or device");
            }
            if metadata.permissions().mode() & 0o077 != 0 {
                anyhow::bail!("recovery draft must be private (mode 0600)");
            }
            Ok(file)
        }
        #[cfg(not(unix))]
        {
            let _ = path;
            anyhow::bail!("recovery draft adoption requires Unix private-file identity checks; preserve the original file");
        }
    }

    fn matches_recovery_bytes(file: &mut std::fs::File, expected: &[u8]) -> Result<bool> {
        let mut bytes = Vec::new();
        file.take(expected.len().saturating_add(1) as u64)
            .read_to_end(&mut bytes)?;
        Ok(bytes == expected)
    }

    pub fn read(&self) -> Result<String> {
        let bytes = if let Some(recovered) = &self.recovery {
            // Recovery content was already read with no-follow, bounded I/O
            // and pinned at adoption. Never reopen its mutable pathname.
            recovered.bytes.clone()
        } else {
            self.secure()?;
            std::fs::read(&self.path)?
        };
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
        self.discard_after_claim(note_id, |_| {})
    }

    fn matches_recovered_path(path: &Path, recovered: &RecoveredInput) -> Result<bool> {
        let mut current = Self::open_recovery_file(path)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            let original = recovered.file.metadata()?;
            let metadata = current.metadata()?;
            if original.dev() != metadata.dev() || original.ino() != metadata.ino() {
                return Ok(false);
            }
        }
        Self::matches_recovery_bytes(&mut current, &recovered.bytes)
    }

    fn restore_claimed_entry(entry: &Path, path: &Path) -> std::io::Result<()> {
        #[cfg(any(target_os = "macos", target_os = "linux"))]
        {
            use std::ffi::CString;
            use std::os::unix::ffi::OsStrExt;
            let from = CString::new(entry.as_os_str().as_bytes()).map_err(|_| {
                std::io::Error::new(std::io::ErrorKind::InvalidInput, "draft path contains NUL")
            })?;
            let to = CString::new(path.as_os_str().as_bytes()).map_err(|_| {
                std::io::Error::new(std::io::ErrorKind::InvalidInput, "draft path contains NUL")
            })?;
            // Rename the entry itself, including directories and symlinks,
            // only if the destination remains absent at the atomic write.
            #[cfg(target_os = "macos")]
            let result = unsafe { libc::renamex_np(from.as_ptr(), to.as_ptr(), libc::RENAME_EXCL) };
            #[cfg(target_os = "linux")]
            let result = unsafe {
                libc::renameat2(
                    libc::AT_FDCWD,
                    from.as_ptr(),
                    libc::AT_FDCWD,
                    to.as_ptr(),
                    libc::RENAME_NOREPLACE,
                )
            };
            if result == 0 {
                Ok(())
            } else {
                // Unsupported kernels/filesystems also retain the quarantine;
                // a check followed by plain rename could overwrite a new save.
                Err(std::io::Error::last_os_error())
            }
        }
        #[cfg(not(any(target_os = "macos", target_os = "linux")))]
        {
            let _ = (entry, path);
            Err(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "atomic draft restoration is unsupported on this platform",
            ))
        }
    }

    fn discard_after_claim(
        &mut self,
        note_id: Option<&str>,
        after_claim: impl FnOnce(&Path),
    ) -> bool {
        // Completion already succeeded. Never emit a retry command or turn
        // cleanup failure into a failed capture that might be retried twice.
        self.retained = false;
        if let Some(recovered) = &self.recovery {
            let claimed = (|| -> Result<(PathBuf, PathBuf)> {
                let directory = tempfile::Builder::new()
                    .prefix("graphrag-ack-")
                    .tempdir_in(self.path.parent().context("draft parent missing")?)?;
                let entry = directory.path().join("draft.md");
                // Claim this directory entry atomically, before validating it.
                // A later save to the original path must never be unlinked.
                std::fs::rename(&self.path, &entry)?;
                // Rejecting this entry must not let TempDir drop delete it.
                let directory = directory.keep();
                Ok((directory, entry))
            })();
            let Ok((directory, entry)) = claimed else {
                eprintln!("Operation completed{}; recovery draft changed or is no longer a private regular file and was retained at {}. Review it before submitting a new request.", note_id.map(|id| format!(" for note {id}")).unwrap_or_default(), self.path.display());
                return false;
            };
            after_claim(&entry);
            let unchanged = Self::matches_recovered_path(&entry, recovered).unwrap_or(false);
            if unchanged && std::fs::remove_file(&entry).is_ok() {
                let _ = std::fs::remove_dir(directory);
                return true;
            }
            // Restore any replacement entry without overwriting a later save.
            // Keep the private quarantine if another save won the original path.
            if Self::restore_claimed_entry(&entry, &self.path).is_ok() {
                let _ = std::fs::remove_dir(directory);
            } else {
                self.path = entry;
            }
            eprintln!("Operation completed{}; recovery draft changed or cleanup failed and was retained at {}. Review it before submitting a new request.", note_id.map(|id| format!(" for note {id}")).unwrap_or_default(), self.path.display());
            return false;
        }
        match std::fs::remove_file(&self.path) {
            Ok(()) => true,
            Err(error) => {
                eprintln!("Operation completed{}; draft retained at {} because cleanup failed: {error}. Remove it when no longer needed.", note_id.map(|id| format!(" for note {id}")).unwrap_or_default(), self.path.display());
                false
            }
        }
    }

    pub fn recoverable_command(&self) -> Option<&str> {
        // Never advertise a retry that would follow a rejected editor symlink
        // or attempt to read a directory/device as note content.
        if let Some(recovered) = &self.recovery {
            if !Self::matches_recovered_path(&self.path, recovered).unwrap_or(false) {
                return None;
            }
        }
        std::fs::symlink_metadata(&self.path)
            .is_ok_and(|metadata| metadata.file_type().is_file())
            .then_some(self.recovery_command.as_str())
    }
}

impl Drop for Draft {
    fn drop(&mut self) {
        if self.retained {
            eprintln!("Draft retained: {}", self.path.display());
            if let Some(command) = self.recoverable_command() {
                eprintln!("Recover: {command}");
            } else {
                eprintln!("Recovery command withheld: draft path is missing, changed, or is not the original regular file. Inspect the path and retained bytes, and restore the intended draft before retrying; submit unsent edits with a new request.");
            }
        }
    }
}

pub fn directory(options: &EditorOptions, database: &Path) -> Result<PathBuf> {
    let mut default = database.as_os_str().to_os_string();
    default.push(".drafts");
    recovery_directory(options.draft_dir.as_deref().unwrap_or(Path::new(&default)))
}

pub fn recovery_directory(path: &Path) -> Result<PathBuf> {
    let directory = absolute(path)?;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(unix)]
    fn replace_draft_entry(path: &Path, kind: &str) {
        use std::os::unix::fs::{symlink, PermissionsExt};
        match kind {
            "directory" => {
                std::fs::create_dir(path).unwrap();
                std::fs::write(path.join("unsent.md"), b"unsent directory contents").unwrap();
            }
            "symlink" => symlink("never-follow-this-dangling-link", path).unwrap(),
            "fifo" => {
                let name = std::ffi::CString::new(path.as_os_str().as_encoded_bytes()).unwrap();
                assert_eq!(unsafe { libc::mkfifo(name.as_ptr(), 0o600) }, 0);
            }
            "regular" => {
                std::fs::write(path, b"another unsent save").unwrap();
                std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600)).unwrap();
            }
            _ => panic!("unknown fixture kind"),
        }
    }

    #[cfg(unix)]
    fn assert_same_entry(path: &Path, before: &std::fs::Metadata, kind: &str) {
        use std::os::unix::fs::{FileTypeExt, MetadataExt};
        let after = std::fs::symlink_metadata(path).unwrap();
        assert_eq!((after.dev(), after.ino()), (before.dev(), before.ino()));
        assert_eq!(after.file_type(), before.file_type());
        match kind {
            "directory" => assert_eq!(
                std::fs::read(path.join("unsent.md")).unwrap(),
                b"unsent directory contents"
            ),
            "symlink" => assert_eq!(
                std::fs::read_link(path).unwrap(),
                Path::new("never-follow-this-dangling-link")
            ),
            "fifo" => assert!(after.file_type().is_fifo()),
            "regular" => assert_eq!(std::fs::read(path).unwrap(), b"another unsent save"),
            _ => panic!("unknown fixture kind"),
        }
    }

    #[cfg(any(target_os = "macos", target_os = "linux"))]
    #[test]
    fn remote_cleanup_restores_nonregular_replacements_to_unoccupied_original_path() {
        for kind in ["directory", "symlink", "fifo"] {
            let temp = tempfile::tempdir().unwrap();
            let mut draft =
                Draft::save(b"submitted body", temp.path(), |_| "retry".into()).unwrap();
            draft.pin_remote_submission(b"submitted body").unwrap();
            let path = draft.path.clone();
            std::fs::remove_file(&path).unwrap();
            replace_draft_entry(&path, kind);
            let before = std::fs::symlink_metadata(&path).unwrap();
            assert!(!draft.discard(Some("note:committed")), "{kind}");
            assert_eq!(draft.path, path, "{kind}");
            assert_same_entry(&path, &before, kind);
            assert_eq!(std::fs::read_dir(temp.path()).unwrap().count(), 1);
            assert!(draft.recoverable_command().is_none());
        }
    }

    #[cfg(any(target_os = "macos", target_os = "linux"))]
    #[test]
    fn remote_cleanup_restores_symlink_without_touching_its_target() {
        use std::os::unix::fs::{symlink, MetadataExt};
        let temp = tempfile::tempdir().unwrap();
        let target = temp.path().join("sensitive-canary.md");
        std::fs::write(&target, b"unrelated sensitive fixture content").unwrap();
        let target_before = std::fs::metadata(&target).unwrap();
        let mut draft = Draft::save(b"submitted body", temp.path(), |_| "retry".into()).unwrap();
        draft.pin_remote_submission(b"submitted body").unwrap();
        let path = draft.path.clone();
        std::fs::remove_file(&path).unwrap();
        symlink(&target, &path).unwrap();
        let link_before = std::fs::symlink_metadata(&path).unwrap();
        assert!(!draft.discard(Some("note:committed")));
        assert_eq!(draft.path, path);
        assert_eq!(
            std::fs::symlink_metadata(&path).unwrap().ino(),
            link_before.ino()
        );
        assert_eq!(std::fs::read_link(&path).unwrap(), target);
        assert_eq!(
            std::fs::metadata(&target).unwrap().ino(),
            target_before.ino()
        );
        assert_eq!(
            std::fs::read(target).unwrap(),
            b"unrelated sensitive fixture content"
        );
        assert!(draft.recoverable_command().is_none());
    }

    #[cfg(any(target_os = "macos", target_os = "linux"))]
    #[test]
    fn remote_cleanup_never_overwrites_a_new_entry_while_restoring_replacements() {
        for replacement in ["directory", "symlink", "fifo"] {
            for occupied in ["regular", "directory", "symlink", "fifo"] {
                let temp = tempfile::tempdir().unwrap();
                let mut draft =
                    Draft::save(b"submitted body", temp.path(), |_| "retry".into()).unwrap();
                draft.pin_remote_submission(b"submitted body").unwrap();
                let path = draft.path.clone();
                std::fs::remove_file(&path).unwrap();
                replace_draft_entry(&path, replacement);
                let replacement_before = std::fs::symlink_metadata(&path).unwrap();
                let mut occupied_before = None;
                assert!(!draft.discard_after_claim(Some("note:committed"), |_| {
                    replace_draft_entry(&path, occupied);
                    occupied_before = Some(std::fs::symlink_metadata(&path).unwrap());
                }));
                assert_same_entry(&path, &occupied_before.unwrap(), occupied);
                assert_ne!(draft.path, path);
                assert_same_entry(&draft.path, &replacement_before, replacement);
                let retained = draft.path.clone();
                assert!(draft.recoverable_command().is_none());
                drop(draft);
                assert!(std::fs::symlink_metadata(retained).is_ok());
            }
        }
    }

    #[cfg(unix)]
    #[test]
    fn adopted_recovery_refuses_changed_bytes_and_preserves_later_unsent_edits() {
        let temp = tempfile::tempdir().unwrap();
        let mut original = Draft::save(b"submitted body", temp.path(), |_| "retry".into()).unwrap();
        let path = original.path.clone();
        assert!(
            Draft::save_or_recover(b"different body", temp.path(), Some(&path), |_| "retry"
                .into())
            .is_err()
        );
        let mut recovery =
            Draft::save_or_recover(b"submitted body", temp.path(), Some(&path), |_| {
                "retry".into()
            })
            .unwrap();
        assert_eq!(recovery.path, path);
        assert_eq!(std::fs::read_dir(temp.path()).unwrap().count(), 1);
        std::fs::write(&path, b"new unsent edits").unwrap();
        assert!(recovery.recoverable_command().is_none());
        assert!(!recovery.discard(Some("note:committed")));
        assert_eq!(std::fs::read(&path).unwrap(), b"new unsent edits");
        original.retained = false;
    }

    #[cfg(unix)]
    #[test]
    fn recovery_ownership_refuses_exposed_files_symlinks_and_directories() {
        use std::os::unix::fs::{symlink, PermissionsExt};
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("ordinary-input.md");
        std::fs::write(&path, b"caller input").unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap();
        assert!(
            Draft::save_or_recover(b"caller input", temp.path(), Some(&path), |_| "retry"
                .into())
            .is_err()
        );
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
        let link = temp.path().join("link.md");
        symlink(&path, &link).unwrap();
        assert!(Draft::validate_recovery_path(&link).is_err());
        assert!(Draft::validate_recovery_path(temp.path()).is_err());
        assert_eq!(std::fs::read(&path).unwrap(), b"caller input");
    }

    #[cfg(unix)]
    #[test]
    fn adopted_recovery_reads_immutable_snapshot_without_following_replaced_path() {
        use std::os::unix::fs::{symlink, PermissionsExt};
        for kind in ["regular", "symlink", "fifo"] {
            let temp = tempfile::tempdir().unwrap();
            let mut original =
                Draft::save(b"submitted body", temp.path(), |_| "retry".into()).unwrap();
            let path = original.path.clone();
            let recovered =
                Draft::save_or_recover(b"submitted body", temp.path(), Some(&path), |_| {
                    "retry".into()
                })
                .unwrap();
            std::fs::remove_file(&path).unwrap();
            match kind {
                "regular" => {
                    std::fs::write(&path, b"unsent replacement body").unwrap();
                    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600))
                        .unwrap();
                }
                "symlink" => {
                    let other = temp.path().join("must-not-read.md");
                    std::fs::write(&other, b"unrelated sensitive fixture content").unwrap();
                    symlink(other, &path).unwrap();
                }
                _ => {
                    let name = std::ffi::CString::new(path.as_os_str().as_encoded_bytes()).unwrap();
                    assert_eq!(unsafe { libc::mkfifo(name.as_ptr(), 0o600) }, 0);
                }
            }
            assert_eq!(recovered.read().unwrap(), "submitted body", "{kind}");
            assert!(recovered.recoverable_command().is_none());
            assert!(std::fs::symlink_metadata(&path).is_ok());
            original.retained = false;
        }
    }

    #[cfg(unix)]
    #[test]
    fn recovery_preserves_same_bytes_replacement_and_refuses_fifo_without_reading() {
        let temp = tempfile::tempdir().unwrap();
        let mut original = Draft::save(b"submitted body", temp.path(), |_| "retry".into()).unwrap();
        let path = original.path.clone();
        let mut recovered =
            Draft::save_or_recover(b"submitted body", temp.path(), Some(&path), |_| {
                "retry".into()
            })
            .unwrap();
        let mut replacement =
            Draft::save(b"submitted body", temp.path(), |_| "retry".into()).unwrap();
        std::fs::rename(&replacement.path, &path).unwrap();
        assert!(recovered.recoverable_command().is_none());
        assert!(!recovered.discard(Some("note:committed")));
        assert_eq!(std::fs::read(&path).unwrap(), b"submitted body");
        original.retained = false;
        replacement.retained = false;
        let fifo = temp.path().join("fifo");
        let fifo_name = std::ffi::CString::new(fifo.as_os_str().as_encoded_bytes()).unwrap();
        // A read-only FIFO with no writer would hang without O_NONBLOCK.
        assert_eq!(unsafe { libc::mkfifo(fifo_name.as_ptr(), 0o600) }, 0);
        assert!(Draft::read_recovery_input(&fifo, 65_537).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn recovery_cleanup_claims_entry_before_validation_and_preserves_later_original_path_saves() {
        use std::os::unix::fs::PermissionsExt;
        let temp = tempfile::tempdir().unwrap();
        let mut original = Draft::save(b"submitted body", temp.path(), |_| "retry".into()).unwrap();
        let path = original.path.clone();
        let mut recovered =
            Draft::save_or_recover(b"submitted body", temp.path(), Some(&path), |_| {
                "retry".into()
            })
            .unwrap();
        assert!(recovered.discard_after_claim(Some("note:committed"), |_| {
            // Same bytes cannot make a newly saved inode belong to this retry.
            std::fs::write(&path, b"submitted body").unwrap();
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
        }));
        assert_eq!(std::fs::read(&path).unwrap(), b"submitted body");
        assert_eq!(std::fs::read_dir(temp.path()).unwrap().count(), 1);
        original.retained = false;
    }

    #[cfg(unix)]
    #[test]
    fn changed_claimed_recovery_survives_when_original_path_is_occupied() {
        use std::os::unix::fs::PermissionsExt;
        let temp = tempfile::tempdir().unwrap();
        let mut original = Draft::save(b"submitted body", temp.path(), |_| "retry".into()).unwrap();
        let path = original.path.clone();
        let mut recovered =
            Draft::save_or_recover(b"submitted body", temp.path(), Some(&path), |_| {
                "retry".into()
            })
            .unwrap();
        assert!(
            !recovered.discard_after_claim(Some("note:committed"), |claimed| {
                std::fs::write(claimed, b"unsent edits to the claimed inode").unwrap();
                std::fs::write(&path, b"another unsent save").unwrap();
                std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
            })
        );
        assert_eq!(std::fs::read(&path).unwrap(), b"another unsent save");
        assert_ne!(recovered.path, path);
        assert_eq!(
            std::fs::read(&recovered.path).unwrap(),
            b"unsent edits to the claimed inode"
        );
        let retained_path = recovered.path.clone();
        assert!(recovered.recoverable_command().is_none());
        drop(recovered);
        assert!(
            retained_path.is_file(),
            "rejection must persist the quarantine directory"
        );
        original.retained = false;
    }
}
