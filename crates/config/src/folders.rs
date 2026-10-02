//! Explicit folder declarations and lossless config registration.

use crate::{ConfigError, RuntimeConfig};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default, deny_unknown_fields)]
pub struct FolderConfig {
    pub path: PathBuf,
    pub recursive: bool,
    pub include: Vec<String>,
    pub exclude: Vec<String>,
}

impl Default for FolderConfig {
    fn default() -> Self {
        Self {
            path: PathBuf::new(),
            recursive: true,
            include: vec!["**/*.md".into(), "**/*.markdown".into()],
            exclude: vec![
                "**/.git/**".into(),
                "**/.obsidian/**".into(),
                "**/.graphrag/**".into(),
            ],
        }
    }
}

pub fn selected_config_path(explicit: Option<&Path>) -> Option<PathBuf> {
    explicit
        .map(Path::to_path_buf)
        .or_else(|| {
            std::env::var("GRAPHRAG_CONFIG")
                .ok()
                .filter(|value| !value.trim().is_empty())
                .map(|value| crate::expand_home_directory(Path::new(&value)))
        })
        .or_else(crate::default_config_path)
}

pub(crate) fn validate_folders(
    folders: &BTreeMap<String, FolderConfig>,
) -> Result<(), ConfigError> {
    let mut roots: Vec<(&str, PathBuf)> = Vec::new();
    for (name, folder) in folders {
        if name.is_empty()
            || name.len() > 64
            || !name
                .bytes()
                .next()
                .is_some_and(|byte| byte.is_ascii_alphanumeric())
            || !name
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
        {
            return Err(ConfigError::Validation(
                "folder names require 1-64 ASCII letters, digits, '-' or '_', starting with a letter or digit".into(),
            ));
        }
        if !folder.path.is_absolute() || folder.path.to_str().is_none() {
            return Err(ConfigError::Validation(format!(
                "folder {name} requires an absolute UTF-8 path"
            )));
        }
        if folder.include.is_empty() {
            return Err(ConfigError::Validation(format!(
                "folder {name} requires at least one include glob"
            )));
        }
        for pattern in folder.include.iter().chain(&folder.exclude) {
            glob::Pattern::new(pattern).map_err(|error| {
                ConfigError::Validation(format!(
                    "invalid glob for folder {name}: {pattern:?}: {error}"
                ))
            })?;
        }
        // Resolve existing aliases; missing roots remain configured so a sync
        // can report their unavailable state without deleting their sources.
        let root =
            std::fs::canonicalize(&folder.path).unwrap_or_else(|_| lexical_path(&folder.path));
        if let Some((other, _)) = roots
            .iter()
            .find(|(_, other)| root.starts_with(other) || other.starts_with(&root))
        {
            return Err(ConfigError::Validation(format!(
                "folders {other} and {name} have duplicate or overlapping roots"
            )));
        }
        roots.push((name, root));
    }
    Ok(())
}

fn lexical_path(path: &Path) -> PathBuf {
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                normalized.pop();
            }
            other => normalized.push(other.as_os_str()),
        }
    }
    normalized
}

/// Register a new definition while preserving every existing byte outside the
/// appended folder table. Never serialize redacted runtime settings to disk.
pub fn register_folder(
    config_path: &Path,
    name: &str,
    mut folder: FolderConfig,
) -> Result<FolderConfig, ConfigError> {
    folder.path = std::fs::canonicalize(&folder.path).map_err(|source| ConfigError::ReadFile {
        path: folder.path.clone(),
        source,
    })?;
    if !folder.path.is_dir() {
        return Err(ConfigError::Validation(
            "folder path requires an existing directory".into(),
        ));
    }
    let metadata =
        std::fs::symlink_metadata(config_path).map_err(|source| ConfigError::ReadFile {
            path: config_path.into(),
            source,
        })?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(ConfigError::Validation("folder registration requires a regular config file, not a symlink; use the actual config path".into()));
    }
    let original = std::fs::read(config_path).map_err(|source| ConfigError::ReadFile {
        path: config_path.into(),
        source,
    })?;
    let mut config = RuntimeConfig::from_file(config_path)?;
    if config.folders.contains_key(name) {
        return Err(ConfigError::Validation(format!(
            "folder {name} already exists; edit its TOML declaration explicitly"
        )));
    }
    config.folders.insert(name.into(), folder.clone());
    validate_folders(&config.folders)?;
    let content = std::str::from_utf8(&original)
        .map_err(|_| ConfigError::Validation("config requires valid UTF-8".into()))?;
    let mut document = content
        .parse::<toml_edit::DocumentMut>()
        .map_err(|error| ConfigError::Validation(format!("invalid config: {error}")))?;
    if document.get("folders").is_none() {
        document["folders"] = toml_edit::Item::Table(toml_edit::Table::new());
    }
    let mut table = toml_edit::Table::new();
    table["path"] = toml_edit::value(folder.path.to_str().expect("validated UTF-8 path"));
    table["recursive"] = toml_edit::value(folder.recursive);
    for (key, values) in [("include", &folder.include), ("exclude", &folder.exclude)] {
        let mut array = toml_edit::Array::new();
        for value in values {
            array.push(value.as_str());
        }
        table[key] = toml_edit::value(array);
    }
    let declarations = document
        .get_mut("folders")
        .expect("created folder declarations");
    if let Some(declarations) = declarations.as_table_mut() {
        declarations.insert(name, toml_edit::Item::Table(table));
    } else if let Some(declarations) = declarations.as_inline_table_mut() {
        declarations.insert(
            name,
            toml_edit::Value::InlineTable(table.into_inline_table()),
        );
    } else {
        return Err(ConfigError::Validation(
            "folders requires a TOML table or inline table".into(),
        ));
    }
    let parent = config_path
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let io_error = |source| ConfigError::ReadFile {
        path: config_path.into(),
        source,
    };
    let mut temporary = tempfile::NamedTempFile::new_in(parent).map_err(io_error)?;
    temporary
        .as_file()
        .set_permissions(metadata.permissions())
        .map_err(io_error)?;
    temporary
        .write_all(document.to_string().as_bytes())
        .map_err(io_error)?;
    temporary.as_file().sync_all().map_err(io_error)?;
    persist_config_if_unchanged(config_path, &original, temporary)?;
    Ok(folder)
}

fn persist_config_if_unchanged(
    config_path: &Path,
    original: &[u8],
    temporary: tempfile::NamedTempFile,
) -> Result<(), ConfigError> {
    let io_error = |source| ConfigError::ReadFile {
        path: config_path.into(),
        source,
    };
    // An exclusive sidecar lock coordinates concurrent registrations; the
    // byte comparison also detects editors that do not participate in it.
    let lock_path = config_path.with_extension("toml.folders.lock");
    let lock = std::fs::OpenOptions::new().write(true).create_new(true).open(&lock_path)
        .map_err(|error| ConfigError::Validation(format!("cannot lock config for folder registration: {error}; if no registration is running, remove {} and retry", lock_path.display())))?;
    let result = (|| {
        let current_metadata = std::fs::symlink_metadata(config_path).map_err(io_error)?;
        if current_metadata.file_type().is_symlink()
            || std::fs::read(config_path).map_err(io_error)? != original
        {
            return Err(ConfigError::Validation(
                "config changed during registration; retry with the latest file".into(),
            ));
        }
        temporary
            .persist(config_path)
            .map_err(|error| io_error(error.error))?;
        Ok(())
    })();
    drop(lock);
    let _ = std::fs::remove_file(lock_path);
    result
}

#[cfg(test)]
#[path = "folder_tests.rs"]
mod tests;
