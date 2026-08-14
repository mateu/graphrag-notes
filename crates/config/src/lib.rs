//! Typed, validated runtime configuration with explicit precedence.
//!
//! Resolution order is defaults, an optional TOML file, compatible environment
//! variables, then explicit CLI overrides supplied by the caller.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("failed to read config file {path}: {source}")]
    ReadFile {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("failed to parse config file {path}: {source}")]
    ParseFile {
        path: PathBuf,
        source: toml::de::Error,
    },
    #[error("invalid configuration: {0}")]
    Validation(String),
}

#[derive(Debug, Clone, Default)]
pub struct CliOverrides {
    pub database_path: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct RuntimeConfig {
    pub database: DatabaseConfig,
    pub inference: InferenceConfig,
    pub search: SearchConfig,
    pub augment: AugmentConfig,
    pub gardener: GardenerConfig,
    pub librarian: LibrarianConfig,
    pub logging: LoggingConfig,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            database: DatabaseConfig::default(),
            inference: InferenceConfig::default(),
            search: SearchConfig::default(),
            augment: AugmentConfig::default(),
            gardener: GardenerConfig::default(),
            librarian: LibrarianConfig::default(),
            logging: LoggingConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct DatabaseConfig {
    pub path: PathBuf,
}

impl Default for DatabaseConfig {
    fn default() -> Self {
        Self {
            path: PathBuf::from(".graphrag/data-v3"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct InferenceConfig {
    pub embedding_provider: String,
    pub embedding_url: String,
    pub embedding_model: String,
    pub extraction_provider: String,
    pub extraction_url: String,
    pub extraction_model: String,
    pub ollama_url: String,
    pub timeout_secs: u64,
    pub tei_max_batch: usize,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            embedding_provider: "tei".into(),
            embedding_url: "http://localhost:8081".into(),
            embedding_model: "bge-m3:latest".into(),
            extraction_provider: "tgi".into(),
            extraction_url: "http://localhost:8082".into(),
            extraction_model: "phi4-mini:latest".into(),
            ollama_url: "http://localhost:11434".into(),
            timeout_secs: 30,
            tei_max_batch: 32,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct SearchConfig {
    pub default_limit: usize,
    pub vector_weight: f32,
    pub fulltext_weight: f32,
}
impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            default_limit: 10,
            vector_weight: 0.7,
            fulltext_weight: 0.3,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct AugmentConfig {
    pub default_limit: usize,
    pub max_tokens: usize,
    pub max_chunk_tokens: usize,
}
impl Default for AugmentConfig {
    fn default() -> Self {
        Self {
            default_limit: 8,
            max_tokens: 1200,
            max_chunk_tokens: 180,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct GardenerConfig {
    pub similarity_threshold: f32,
    pub auto_apply_threshold: f32,
    pub max_suggestions: usize,
}
impl Default for GardenerConfig {
    fn default() -> Self {
        Self {
            similarity_threshold: 0.7,
            auto_apply_threshold: 0.85,
            max_suggestions: 50,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct LibrarianConfig {
    pub min_chunk_size: usize,
    pub max_chunk_size: usize,
}
impl Default for LibrarianConfig {
    fn default() -> Self {
        Self {
            min_chunk_size: 50,
            max_chunk_size: 1000,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct LoggingConfig {
    pub level: String,
}
impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: "info".into(),
        }
    }
}

impl RuntimeConfig {
    pub fn load(
        explicit_path: Option<&Path>,
        overrides: &CliOverrides,
    ) -> Result<Self, ConfigError> {
        let env = |key: &str| std::env::var(key).ok();
        Self::load_with_env(explicit_path, overrides, &env)
    }

    pub fn load_with_env(
        explicit_path: Option<&Path>,
        overrides: &CliOverrides,
        env: &impl Fn(&str) -> Option<String>,
    ) -> Result<Self, ConfigError> {
        let mut config = match explicit_path
            .map(Path::to_path_buf)
            .or_else(|| env("GRAPHRAG_CONFIG").map(PathBuf::from))
        {
            Some(path) => Self::from_file(&path)?,
            None => default_config_path()
                .filter(|path| path.exists())
                .map(|path| Self::from_file(&path))
                .transpose()?
                .unwrap_or_default(),
        };
        config.apply_env(env)?;
        if let Some(path) = &overrides.database_path {
            config.database.path = path.clone();
        }
        config.validate()?;
        Ok(config)
    }

    pub fn from_file(path: &Path) -> Result<Self, ConfigError> {
        let content = std::fs::read_to_string(path).map_err(|source| ConfigError::ReadFile {
            path: path.into(),
            source,
        })?;
        toml::from_str(&content).map_err(|source| ConfigError::ParseFile {
            path: path.into(),
            source,
        })
    }

    pub fn apply_env(&mut self, env: &impl Fn(&str) -> Option<String>) -> Result<(), ConfigError> {
        set_path(env, "GRAPHRAG_DB_PATH", &mut self.database.path);
        set_string(env, "TEI_PROVIDER", &mut self.inference.embedding_provider);
        set_string(env, "TEI_URL", &mut self.inference.embedding_url);
        set_string(env, "TEI_MODEL", &mut self.inference.embedding_model);
        set_string(env, "TGI_PROVIDER", &mut self.inference.extraction_provider);
        set_string(env, "TGI_URL", &mut self.inference.extraction_url);
        set_string(env, "TGI_MODEL", &mut self.inference.extraction_model);
        set_string(env, "OLLAMA_URL", &mut self.inference.ollama_url);
        set_usize(env, "TEI_MAX_BATCH", &mut self.inference.tei_max_batch)?;
        Ok(())
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.database.path.as_os_str().is_empty() {
            return Err(ConfigError::Validation(
                "database.path must not be empty".into(),
            ));
        }
        for (name, value) in [
            (
                "inference.embedding_provider",
                &self.inference.embedding_provider,
            ),
            ("inference.embedding_model", &self.inference.embedding_model),
            (
                "inference.extraction_provider",
                &self.inference.extraction_provider,
            ),
            (
                "inference.extraction_model",
                &self.inference.extraction_model,
            ),
        ] {
            if value.trim().is_empty() {
                return Err(ConfigError::Validation(format!("{name} must not be empty")));
            }
        }
        if self.inference.timeout_secs == 0
            || self.inference.tei_max_batch == 0
            || self.search.default_limit == 0
            || self.augment.default_limit == 0
            || self.augment.max_tokens == 0
            || self.augment.max_chunk_tokens == 0
            || self.librarian.min_chunk_size == 0
            || self.librarian.max_chunk_size < self.librarian.min_chunk_size
        {
            return Err(ConfigError::Validation("limits must be positive and librarian.max_chunk_size must be at least min_chunk_size".into()));
        }
        if !(0.0..=1.0).contains(&self.search.vector_weight)
            || !(0.0..=1.0).contains(&self.search.fulltext_weight)
            || (self.search.vector_weight + self.search.fulltext_weight - 1.0).abs() > f32::EPSILON
        {
            return Err(ConfigError::Validation(
                "search weights must be in [0, 1] and sum to 1".into(),
            ));
        }
        if !(0.0..=1.0).contains(&self.gardener.similarity_threshold)
            || !(0.0..=1.0).contains(&self.gardener.auto_apply_threshold)
            || self.gardener.auto_apply_threshold < self.gardener.similarity_threshold
        {
            return Err(ConfigError::Validation("Gardener thresholds must be in [0, 1] and auto_apply_threshold must be at least similarity_threshold".into()));
        }
        Ok(())
    }

    pub fn redacted_toml(&self) -> Result<String, ConfigError> {
        toml::to_string_pretty(self).map_err(|error| ConfigError::Validation(error.to_string()))
    }
}

pub fn default_config_path() -> Option<PathBuf> {
    dirs::config_dir().map(|path| path.join("graphrag/config.toml"))
}

fn set_string(env: &impl Fn(&str) -> Option<String>, key: &str, target: &mut String) {
    if let Some(value) = env(key).filter(|value| !value.trim().is_empty()) {
        *target = value;
    }
}
fn set_path(env: &impl Fn(&str) -> Option<String>, key: &str, target: &mut PathBuf) {
    if let Some(value) = env(key).filter(|value| !value.trim().is_empty()) {
        *target = value.into();
    }
}
fn set_usize(
    env: &impl Fn(&str) -> Option<String>,
    key: &str,
    target: &mut usize,
) -> Result<(), ConfigError> {
    if let Some(value) = env(key) {
        *target = value
            .parse()
            .map_err(|_| ConfigError::Validation(format!("{key} must be a positive integer")))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn env(values: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
        let values = values
            .iter()
            .map(|(key, value)| (key.to_string(), value.to_string()))
            .collect::<BTreeMap<_, _>>();
        move |key| values.get(key).cloned()
    }

    #[test]
    fn environment_overrides_defaults_and_cli_wins() {
        let config = RuntimeConfig::load_with_env(
            None,
            &CliOverrides {
                database_path: Some("cli.db".into()),
            },
            &env(&[
                ("TEI_PROVIDER", "ollama"),
                ("TEI_MAX_BATCH", "8"),
                ("GRAPHRAG_DB_PATH", "env.db"),
            ]),
        )
        .unwrap();
        assert_eq!(config.database.path, PathBuf::from("cli.db"));
        assert_eq!(config.inference.embedding_provider, "ollama");
        assert_eq!(config.inference.tei_max_batch, 8);
    }

    #[test]
    fn validation_rejects_invalid_weights() {
        let mut config = RuntimeConfig::default();
        config.search.vector_weight = 0.8;
        assert!(config.validate().is_err());
    }
}
