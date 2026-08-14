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
            // Keep the historical CLI location when no configuration is
            // supplied. `dirs` can fail only on unusual platforms; retaining
            // a relative fallback is better than making configuration
            // resolution unavailable there.
            path: dirs::home_dir()
                .map(|path| path.join(".graphrag/data-v3"))
                .unwrap_or_else(|| PathBuf::from(".graphrag/data-v3")),
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
    /// Maximum in-flight requests per local inference operation/provider.
    pub processing_concurrency: usize,
    /// Includes the initial request. One disables retries.
    pub retry_attempts: usize,
    pub retry_initial_backoff_ms: u64,
    pub retry_max_backoff_ms: u64,
    pub cache_enabled: bool,
    pub tei_max_batch: usize,
    pub tei_prompt_name_query: Option<String>,
    pub tei_prompt_name_passage: Option<String>,
    pub strict_entity_json: bool,
    pub max_entities: usize,
    pub max_relationships: usize,
    pub ollama_timeout_secs: u64,
    pub ollama_options: Option<serde_json::Value>,
    #[serde(skip)]
    embedding_url_from_file: bool,
    #[serde(skip)]
    extraction_url_from_file: bool,
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
            processing_concurrency: 4,
            retry_attempts: 3,
            retry_initial_backoff_ms: 250,
            retry_max_backoff_ms: 5_000,
            cache_enabled: true,
            tei_max_batch: 32,
            tei_prompt_name_query: None,
            tei_prompt_name_passage: None,
            strict_entity_json: true,
            max_entities: 30,
            max_relationships: 15,
            ollama_timeout_secs: 120,
            ollama_options: None,
            embedding_url_from_file: false,
            extraction_url_from_file: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct SearchConfig {
    pub default_limit: usize,
    /// `rrf` is scale-independent for vector and BM25 results. `weighted`
    /// remains available only for controlled legacy comparisons.
    pub fusion_strategy: String,
    pub rrf_k: usize,
    pub vector_weight: f32,
    pub fulltext_weight: f32,
    pub candidate_pool_multiplier: usize,
    pub candidate_pool_min: usize,
    pub candidate_pool_max: usize,
    pub note_weight: f32,
    pub message_weight: f32,
    pub conversation_summary_weight: f32,
}
impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            default_limit: 10,
            fusion_strategy: "rrf".into(),
            rrf_k: 60,
            vector_weight: 0.7,
            fulltext_weight: 0.3,
            candidate_pool_multiplier: 4,
            candidate_pool_min: 50,
            candidate_pool_max: 200,
            note_weight: 1.0,
            message_weight: 1.0,
            conversation_summary_weight: 1.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct AugmentConfig {
    pub default_limit: usize,
    pub max_tokens: usize,
    pub max_chunk_tokens: usize,
    /// Greedy MMR-style selection weight; 0 preserves pure retrieval ranking.
    pub novelty_weight: f32,
    /// Retrieval scores below this value are never selected merely for novelty.
    pub min_relevance: f32,
    /// Local token-set Jaccard threshold for near-duplicate suppression.
    pub near_duplicate_threshold: f32,
}
impl Default for AugmentConfig {
    fn default() -> Self {
        Self {
            default_limit: 8,
            max_tokens: 1200,
            max_chunk_tokens: 180,
            novelty_weight: 0.25,
            min_relevance: 0.0,
            near_duplicate_threshold: 0.85,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct GardenerConfig {
    pub similarity_threshold: f32,
    pub auto_apply_threshold: f32,
    /// Explicit opt-in; a threshold alone never enables mutation.
    pub auto_apply: bool,
    pub max_suggestions: usize,
}
impl Default for GardenerConfig {
    fn default() -> Self {
        Self {
            similarity_threshold: 0.7,
            auto_apply_threshold: 0.85,
            auto_apply: false,
            max_suggestions: 50,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct LibrarianConfig {
    pub min_chunk_size: usize,
    /// Target Markdown chunk size in Unicode scalar values (characters).
    pub target_chunk_size: usize,
    pub max_chunk_size: usize,
    /// Tail characters copied from the preceding chunk when it fits under the
    /// hard maximum.
    pub chunk_overlap: usize,
    pub skip_entity_extraction: bool,
    pub extract_log_each: bool,
    /// Maximum characters sent to entity extraction. Zero preserves the
    /// legacy meaning of no truncation.
    pub extract_max_chars: usize,
    pub extract_progress_every: usize,
    pub extract_progress_every_secs: u64,
    pub import_progress_every: usize,
    pub import_progress_every_secs: u64,
}
impl Default for LibrarianConfig {
    fn default() -> Self {
        Self {
            min_chunk_size: 50,
            target_chunk_size: 700,
            max_chunk_size: 1000,
            chunk_overlap: 100,
            skip_entity_extraction: false,
            extract_log_each: false,
            extract_max_chars: 8000,
            extract_progress_every: 10,
            extract_progress_every_secs: 5,
            import_progress_every: 10,
            import_progress_every_secs: 5,
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
            // The CLI historically defaulted to WARN unless --verbose or
            // RUST_LOG was set.
            level: "warn".into(),
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
        Self::load_with_env_and_default_path(explicit_path, overrides, env, default_config_path())
    }

    /// Like [`RuntimeConfig::load_with_env`], with an injected default path.
    ///
    /// This is public so callers embedding the CLI can use an application-
    /// specific XDG location, and so precedence can be tested without reading
    /// the invoking user's real configuration directory.
    pub fn load_with_env_and_default_path(
        explicit_path: Option<&Path>,
        overrides: &CliOverrides,
        env: &impl Fn(&str) -> Option<String>,
        default_path: Option<PathBuf>,
    ) -> Result<Self, ConfigError> {
        let mut config = match explicit_path.map(Path::to_path_buf).or_else(|| {
            env("GRAPHRAG_CONFIG")
                .filter(|value| !value.trim().is_empty())
                .map(|value| expand_home_directory(Path::new(&value)))
        }) {
            Some(path) => Self::from_file(&path)?,
            None => default_path
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
        let raw: toml::Value =
            toml::from_str(&content).map_err(|source| ConfigError::ParseFile {
                path: path.into(),
                source,
            })?;
        let mut config: Self = raw
            .clone()
            .try_into()
            .map_err(|source| ConfigError::ParseFile {
                path: path.into(),
                source,
            })?;
        let inference = raw.get("inference").and_then(toml::Value::as_table);
        config.inference.embedding_url_from_file =
            inference.is_some_and(|table| table.contains_key("embedding_url"));
        config.inference.extraction_url_from_file =
            inference.is_some_and(|table| table.contains_key("extraction_url"));
        if let Some(librarian) = raw.get("librarian").and_then(toml::Value::as_table) {
            // `target_chunk_size` and `chunk_overlap` were added after the
            // original min/max-only configuration. Derive omitted values from
            // the explicit legacy bounds before validation, rather than
            // rejecting an otherwise valid existing config because today's
            // global defaults do not fit its smaller maximum.
            if !librarian.contains_key("target_chunk_size") {
                config.librarian.target_chunk_size = config.librarian.target_chunk_size.clamp(
                    config.librarian.min_chunk_size,
                    config.librarian.max_chunk_size,
                );
            }
            if !librarian.contains_key("chunk_overlap") {
                config.librarian.chunk_overlap = config
                    .librarian
                    .chunk_overlap
                    .min(config.librarian.max_chunk_size.saturating_sub(1));
            }
        }
        config.normalize_provider_names();
        config.database.path = expand_home_directory(&config.database.path);
        Ok(config)
    }

    pub fn apply_env(&mut self, env: &impl Fn(&str) -> Option<String>) -> Result<(), ConfigError> {
        set_path(env, "GRAPHRAG_DB_PATH", &mut self.database.path);
        set_string(env, "GRAPHRAG_LOG_LEVEL", &mut self.logging.level);

        set_string(env, "TEI_PROVIDER", &mut self.inference.embedding_provider);
        set_string(env, "TEI_URL", &mut self.inference.embedding_url);
        set_string(env, "TEI_MODEL", &mut self.inference.embedding_model);
        set_string(env, "TGI_PROVIDER", &mut self.inference.extraction_provider);
        set_string(env, "TGI_URL", &mut self.inference.extraction_url);
        set_string(env, "TGI_MODEL", &mut self.inference.extraction_model);
        set_string(env, "OLLAMA_URL", &mut self.inference.ollama_url);
        set_usize(env, "TEI_MAX_BATCH", &mut self.inference.tei_max_batch)?;
        set_optional_string(
            env,
            "TEI_PROMPT_NAME_QUERY",
            &mut self.inference.tei_prompt_name_query,
        );
        set_optional_string(
            env,
            "TEI_PROMPT_NAME_PASSAGE",
            &mut self.inference.tei_prompt_name_passage,
        );
        set_bool(
            env,
            "STRICT_ENTITY_JSON",
            &mut self.inference.strict_entity_json,
        )?;
        set_usize(
            env,
            "EXTRACT_MAX_ENTITIES",
            &mut self.inference.max_entities,
        )?;
        set_usize(
            env,
            "EXTRACT_MAX_RELATIONSHIPS",
            &mut self.inference.max_relationships,
        )?;
        set_u64(
            env,
            "TGI_OLLAMA_TIMEOUT_SECS",
            &mut self.inference.ollama_timeout_secs,
        )?;
        set_json_object(
            env,
            "TGI_OLLAMA_OPTIONS",
            &mut self.inference.ollama_options,
        )?;

        set_bool(
            env,
            "SKIP_ENTITY_EXTRACTION",
            &mut self.librarian.skip_entity_extraction,
        )?;
        set_bool(
            env,
            "EXTRACT_LOG_EACH",
            &mut self.librarian.extract_log_each,
        )?;
        set_usize(
            env,
            "EXTRACT_MAX_CHARS",
            &mut self.librarian.extract_max_chars,
        )?;
        set_usize(
            env,
            "EXTRACT_PROGRESS_EVERY",
            &mut self.librarian.extract_progress_every,
        )?;
        set_u64(
            env,
            "EXTRACT_PROGRESS_EVERY_SECS",
            &mut self.librarian.extract_progress_every_secs,
        )?;
        set_usize(
            env,
            "IMPORT_PROGRESS_EVERY",
            &mut self.librarian.import_progress_every,
        )?;
        set_u64(
            env,
            "IMPORT_PROGRESS_EVERY_SECS",
            &mut self.librarian.import_progress_every_secs,
        )?;

        set_u64(
            env,
            "GRAPHRAG_INFERENCE_TIMEOUT_SECS",
            &mut self.inference.timeout_secs,
        )?;
        set_usize(
            env,
            "GRAPHRAG_INFERENCE_CONCURRENCY",
            &mut self.inference.processing_concurrency,
        )?;
        set_usize(
            env,
            "GRAPHRAG_INFERENCE_RETRY_ATTEMPTS",
            &mut self.inference.retry_attempts,
        )?;
        set_u64(
            env,
            "GRAPHRAG_INFERENCE_RETRY_INITIAL_BACKOFF_MS",
            &mut self.inference.retry_initial_backoff_ms,
        )?;
        set_u64(
            env,
            "GRAPHRAG_INFERENCE_RETRY_MAX_BACKOFF_MS",
            &mut self.inference.retry_max_backoff_ms,
        )?;
        set_bool(
            env,
            "GRAPHRAG_INFERENCE_CACHE_ENABLED",
            &mut self.inference.cache_enabled,
        )?;
        set_usize(
            env,
            "GRAPHRAG_SEARCH_DEFAULT_LIMIT",
            &mut self.search.default_limit,
        )?;
        set_string(
            env,
            "GRAPHRAG_SEARCH_FUSION_STRATEGY",
            &mut self.search.fusion_strategy,
        );
        set_usize(env, "GRAPHRAG_SEARCH_RRF_K", &mut self.search.rrf_k)?;
        set_f32(
            env,
            "GRAPHRAG_SEARCH_VECTOR_WEIGHT",
            &mut self.search.vector_weight,
        )?;
        set_f32(
            env,
            "GRAPHRAG_SEARCH_FULLTEXT_WEIGHT",
            &mut self.search.fulltext_weight,
        )?;
        set_usize(
            env,
            "GRAPHRAG_SEARCH_CANDIDATE_POOL_MULTIPLIER",
            &mut self.search.candidate_pool_multiplier,
        )?;
        set_usize(
            env,
            "GRAPHRAG_SEARCH_CANDIDATE_POOL_MIN",
            &mut self.search.candidate_pool_min,
        )?;
        set_usize(
            env,
            "GRAPHRAG_SEARCH_CANDIDATE_POOL_MAX",
            &mut self.search.candidate_pool_max,
        )?;
        set_f32(
            env,
            "GRAPHRAG_SEARCH_NOTE_WEIGHT",
            &mut self.search.note_weight,
        )?;
        set_f32(
            env,
            "GRAPHRAG_SEARCH_MESSAGE_WEIGHT",
            &mut self.search.message_weight,
        )?;
        set_f32(
            env,
            "GRAPHRAG_SEARCH_CONVERSATION_SUMMARY_WEIGHT",
            &mut self.search.conversation_summary_weight,
        )?;
        set_usize(
            env,
            "GRAPHRAG_AUGMENT_DEFAULT_LIMIT",
            &mut self.augment.default_limit,
        )?;
        set_usize(
            env,
            "GRAPHRAG_AUGMENT_MAX_TOKENS",
            &mut self.augment.max_tokens,
        )?;
        set_usize(
            env,
            "GRAPHRAG_AUGMENT_MAX_CHUNK_TOKENS",
            &mut self.augment.max_chunk_tokens,
        )?;
        set_f32(
            env,
            "GRAPHRAG_AUGMENT_NOVELTY_WEIGHT",
            &mut self.augment.novelty_weight,
        )?;
        set_f32(
            env,
            "GRAPHRAG_AUGMENT_MIN_RELEVANCE",
            &mut self.augment.min_relevance,
        )?;
        set_f32(
            env,
            "GRAPHRAG_AUGMENT_NEAR_DUPLICATE_THRESHOLD",
            &mut self.augment.near_duplicate_threshold,
        )?;
        set_f32(
            env,
            "GRAPHRAG_GARDENER_SIMILARITY_THRESHOLD",
            &mut self.gardener.similarity_threshold,
        )?;
        set_f32(
            env,
            "GRAPHRAG_GARDENER_AUTO_APPLY_THRESHOLD",
            &mut self.gardener.auto_apply_threshold,
        )?;
        set_bool(
            env,
            "GRAPHRAG_GARDENER_AUTO_APPLY",
            &mut self.gardener.auto_apply,
        )?;
        set_usize(
            env,
            "GRAPHRAG_GARDENER_MAX_SUGGESTIONS",
            &mut self.gardener.max_suggestions,
        )?;
        set_usize(
            env,
            "GRAPHRAG_LIBRARIAN_MIN_CHUNK_SIZE",
            &mut self.librarian.min_chunk_size,
        )?;
        set_usize(
            env,
            "GRAPHRAG_LIBRARIAN_TARGET_CHUNK_SIZE",
            &mut self.librarian.target_chunk_size,
        )?;
        set_usize(
            env,
            "GRAPHRAG_LIBRARIAN_MAX_CHUNK_SIZE",
            &mut self.librarian.max_chunk_size,
        )?;
        set_usize(
            env,
            "GRAPHRAG_LIBRARIAN_CHUNK_OVERLAP",
            &mut self.librarian.chunk_overlap,
        )?;

        self.normalize_provider_names();

        // The legacy environment setup allowed callers to set only the
        // provider. Apply that fallback only when the provider itself came
        // from the environment so an explicit TOML endpoint remains intact.
        if self
            .inference
            .embedding_provider
            .eq_ignore_ascii_case("ollama")
            && env("TEI_URL").is_none_or(|url| url.trim().is_empty())
            && (env("TEI_PROVIDER")
                .is_some_and(|provider| provider.trim().eq_ignore_ascii_case("ollama"))
                || !self.inference.embedding_url_from_file)
        {
            self.inference.embedding_url = self.inference.ollama_url.clone();
        }
        if self
            .inference
            .extraction_provider
            .eq_ignore_ascii_case("ollama")
            && env("TGI_URL").is_none_or(|url| url.trim().is_empty())
            && (env("TGI_PROVIDER")
                .is_some_and(|provider| provider.trim().eq_ignore_ascii_case("ollama"))
                || !self.inference.extraction_url_from_file)
        {
            self.inference.extraction_url = self.inference.ollama_url.clone();
        }
        if env("TEI_PROVIDER").is_some_and(|provider| {
            !provider.trim().is_empty() && !provider.trim().eq_ignore_ascii_case("ollama")
        }) && env("TEI_URL").is_none_or(|url| url.trim().is_empty())
        {
            self.inference.embedding_url = "http://localhost:8081".into();
        }
        if env("TGI_PROVIDER").is_some_and(|provider| {
            !provider.trim().is_empty() && !provider.trim().eq_ignore_ascii_case("ollama")
        }) && env("TGI_URL").is_none_or(|url| url.trim().is_empty())
        {
            self.inference.extraction_url = "http://localhost:8082".into();
        }
        Ok(())
    }

    fn normalize_provider_names(&mut self) {
        self.inference.embedding_provider = self
            .inference
            .embedding_provider
            .trim()
            .to_ascii_lowercase();
        self.inference.extraction_provider = self
            .inference
            .extraction_provider
            .trim()
            .to_ascii_lowercase();
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
            ("inference.embedding_url", &self.inference.embedding_url),
            ("inference.extraction_url", &self.inference.extraction_url),
            ("inference.ollama_url", &self.inference.ollama_url),
        ] {
            if value.trim().is_empty() {
                return Err(ConfigError::Validation(format!("{name} must not be empty")));
            }
        }
        if !matches!(
            self.inference
                .embedding_provider
                .trim()
                .to_ascii_lowercase()
                .as_str(),
            "tei" | "ollama"
        ) {
            return Err(ConfigError::Validation(
                "inference.embedding_provider must be tei or ollama".into(),
            ));
        }
        if !matches!(
            self.inference
                .extraction_provider
                .trim()
                .to_ascii_lowercase()
                .as_str(),
            "tgi" | "ollama"
        ) {
            return Err(ConfigError::Validation(
                "inference.extraction_provider must be tgi or ollama".into(),
            ));
        }
        if self.inference.timeout_secs == 0
            || self.inference.processing_concurrency == 0
            || self.inference.retry_attempts == 0
            || self.inference.retry_initial_backoff_ms == 0
            || self.inference.retry_max_backoff_ms < self.inference.retry_initial_backoff_ms
            || self.inference.tei_max_batch == 0
            || self.inference.max_entities == 0
            || self.inference.max_relationships == 0
            || self.inference.ollama_timeout_secs == 0
            || self.search.default_limit == 0
            || self.search.rrf_k == 0
            || self.search.candidate_pool_multiplier == 0
            || self.search.candidate_pool_min == 0
            || self.search.candidate_pool_max < self.search.candidate_pool_min
            || self.augment.default_limit == 0
            || self.augment.max_tokens == 0
            || self.augment.max_chunk_tokens == 0
            || self.librarian.min_chunk_size == 0
            || self.librarian.target_chunk_size == 0
            || self.librarian.max_chunk_size < self.librarian.min_chunk_size
            || self.librarian.target_chunk_size < self.librarian.min_chunk_size
            || self.librarian.target_chunk_size > self.librarian.max_chunk_size
            || self.librarian.chunk_overlap >= self.librarian.max_chunk_size
            || (self.librarian.max_chunk_size != usize::MAX
                && self.librarian.max_chunk_size < self.librarian.min_chunk_size.saturating_mul(2))
            || self.librarian.extract_progress_every == 0
            || self.librarian.extract_progress_every_secs == 0
            || self.librarian.import_progress_every == 0
            || self.librarian.import_progress_every_secs == 0
            || self.gardener.max_suggestions == 0
        {
            return Err(ConfigError::Validation("limits must be positive; librarian.min_chunk_size <= target_chunk_size <= max_chunk_size, overlap must be below max_chunk_size, and a bounded max_chunk_size must be at least twice min_chunk_size".into()));
        }
        if !(0.0..=1.0).contains(&self.search.vector_weight)
            || !(0.0..=1.0).contains(&self.search.fulltext_weight)
            || (self.search.vector_weight + self.search.fulltext_weight - 1.0).abs() > f32::EPSILON
        {
            return Err(ConfigError::Validation(
                "search weights must be in [0, 1] and sum to 1".into(),
            ));
        }
        if !matches!(
            self.search
                .fusion_strategy
                .trim()
                .to_ascii_lowercase()
                .as_str(),
            "rrf" | "weighted"
        ) {
            return Err(ConfigError::Validation(
                "search.fusion_strategy must be rrf or weighted".into(),
            ));
        }
        for (name, weight) in [
            ("search.note_weight", self.search.note_weight),
            ("search.message_weight", self.search.message_weight),
            (
                "search.conversation_summary_weight",
                self.search.conversation_summary_weight,
            ),
        ] {
            if !weight.is_finite() || weight < 0.0 {
                return Err(ConfigError::Validation(format!(
                    "{name} must be finite and non-negative"
                )));
            }
        }
        if !(0.0..=1.0).contains(&self.augment.novelty_weight)
            || !(0.0..=1.0).contains(&self.augment.min_relevance)
            || !(0.0..=1.0).contains(&self.augment.near_duplicate_threshold)
        {
            return Err(ConfigError::Validation(
                "augment.novelty_weight, augment.min_relevance, and augment.near_duplicate_threshold must be between 0 and 1".into(),
            ));
        }
        if !(0.0..=1.0).contains(&self.gardener.similarity_threshold)
            || !(0.0..=1.0).contains(&self.gardener.auto_apply_threshold)
            || self.gardener.auto_apply_threshold < self.gardener.similarity_threshold
        {
            return Err(ConfigError::Validation("Gardener thresholds must be in [0, 1] and auto_apply_threshold must be at least similarity_threshold".into()));
        }
        if !matches!(
            self.logging.level.trim().to_ascii_lowercase().as_str(),
            "trace" | "debug" | "info" | "warn" | "error" | "off"
        ) {
            return Err(ConfigError::Validation(
                "logging.level must be trace, debug, info, warn, error, or off".into(),
            ));
        }
        if self
            .inference
            .ollama_options
            .as_ref()
            .is_some_and(|options| !options.is_object())
        {
            return Err(ConfigError::Validation(
                "inference.ollama_options must be a TOML/JSON object".into(),
            ));
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

fn set_optional_string(
    env: &impl Fn(&str) -> Option<String>,
    key: &str,
    target: &mut Option<String>,
) {
    if let Some(value) = env(key) {
        *target = Some(value);
    }
}

fn set_bool(
    env: &impl Fn(&str) -> Option<String>,
    key: &str,
    target: &mut bool,
) -> Result<(), ConfigError> {
    if let Some(value) = env(key) {
        let value = value.trim().to_ascii_lowercase();
        *target = match value.as_str() {
            "1" | "true" | "yes" | "on" => true,
            "0" | "false" | "no" | "off" => false,
            _ => {
                return Err(ConfigError::Validation(format!(
                    "{key} must be a boolean (true/false, yes/no, on/off, or 1/0)"
                )))
            }
        };
    }
    Ok(())
}

fn set_json_object(
    env: &impl Fn(&str) -> Option<String>,
    key: &str,
    target: &mut Option<serde_json::Value>,
) -> Result<(), ConfigError> {
    let Some(raw) = env(key) else {
        return Ok(());
    };
    if raw.trim().is_empty() {
        *target = None;
        return Ok(());
    }
    let value: serde_json::Value = serde_json::from_str(&raw)
        .map_err(|error| ConfigError::Validation(format!("{key} must be valid JSON: {error}")))?;
    if !value.is_object() {
        return Err(ConfigError::Validation(format!(
            "{key} must be a JSON object"
        )));
    }
    *target = Some(value);
    Ok(())
}

fn set_path(env: &impl Fn(&str) -> Option<String>, key: &str, target: &mut PathBuf) {
    if let Some(value) = env(key).filter(|value| !value.trim().is_empty()) {
        *target = expand_home_directory(Path::new(&value));
    }
}

fn set_u64(
    env: &impl Fn(&str) -> Option<String>,
    key: &str,
    target: &mut u64,
) -> Result<(), ConfigError> {
    if let Some(value) = env(key) {
        *target = value
            .parse()
            .map_err(|_| ConfigError::Validation(format!("{key} must be a positive integer")))?;
    }
    Ok(())
}

fn set_f32(
    env: &impl Fn(&str) -> Option<String>,
    key: &str,
    target: &mut f32,
) -> Result<(), ConfigError> {
    if let Some(value) = env(key) {
        *target = value
            .parse()
            .map_err(|_| ConfigError::Validation(format!("{key} must be a number")))?;
    }
    Ok(())
}

fn expand_home_directory(path: &Path) -> PathBuf {
    let Some(path) = path.to_str() else {
        return path.to_path_buf();
    };
    if path == "~" {
        return dirs::home_dir().unwrap_or_else(|| PathBuf::from(path));
    }
    if let Some(suffix) = path.strip_prefix("~/") {
        return dirs::home_dir()
            .map(|home| home.join(suffix))
            .unwrap_or_else(|| PathBuf::from(path));
    }
    PathBuf::from(path)
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
    use std::fs;

    fn env(values: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
        let values = values
            .iter()
            .map(|(key, value)| (key.to_string(), value.to_string()))
            .collect::<BTreeMap<_, _>>();
        move |key| values.get(key).cloned()
    }

    #[test]
    fn environment_overrides_defaults_and_cli_wins() {
        let config = RuntimeConfig::load_with_env_and_default_path(
            None,
            &CliOverrides {
                database_path: Some("cli.db".into()),
            },
            &env(&[
                ("TEI_PROVIDER", "ollama"),
                ("TEI_MAX_BATCH", "8"),
                ("GRAPHRAG_DB_PATH", "env.db"),
            ]),
            None,
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

        config.search.vector_weight = 0.7;
        config.search.fulltext_weight = 0.3;
        config.augment.near_duplicate_threshold = 1.1;
        assert!(config
            .validate()
            .unwrap_err()
            .to_string()
            .contains("near_duplicate_threshold"));
        config.augment.near_duplicate_threshold = 0.85;

        config.librarian.min_chunk_size = 600;
        config.librarian.max_chunk_size = 1000;
        assert!(config
            .validate()
            .unwrap_err()
            .to_string()
            .contains("at least twice min_chunk_size"));
    }

    #[test]
    fn defaults_are_used_when_no_optional_config_file_exists() {
        let missing = tempfile::tempdir().unwrap().path().join("missing.toml");
        let config = RuntimeConfig::load_with_env_and_default_path(
            None,
            &CliOverrides::default(),
            &env(&[]),
            Some(missing),
        )
        .unwrap();

        assert_eq!(config.search.vector_weight, 0.7);
        assert_eq!(config.augment.max_tokens, 1200);
        assert_eq!(config.augment.novelty_weight, 0.25);
        assert_eq!(config.logging.level, "warn");
    }

    #[test]
    fn legacy_librarian_bounds_derive_feasible_chunking_controls() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("legacy.toml");
        fs::write(
            &path,
            "[librarian]\nmin_chunk_size = 40\nmax_chunk_size = 80\n",
        )
        .unwrap();

        let config = RuntimeConfig::from_file(&path).unwrap();
        assert_eq!(config.librarian.min_chunk_size, 40);
        assert_eq!(config.librarian.max_chunk_size, 80);
        assert_eq!(config.librarian.target_chunk_size, 80);
        assert_eq!(config.librarian.chunk_overlap, 79);
        config.validate().unwrap();
    }

    #[test]
    fn environment_overrides_all_augment_tuning_fields() {
        let config = RuntimeConfig::load_with_env_and_default_path(
            None,
            &CliOverrides::default(),
            &env(&[
                ("GRAPHRAG_AUGMENT_NOVELTY_WEIGHT", "0.4"),
                ("GRAPHRAG_AUGMENT_MIN_RELEVANCE", "0.2"),
                ("GRAPHRAG_AUGMENT_NEAR_DUPLICATE_THRESHOLD", "0.7"),
            ]),
            None,
        )
        .unwrap();
        assert_eq!(config.augment.novelty_weight, 0.4);
        assert_eq!(config.augment.min_relevance, 0.2);
        assert_eq!(config.augment.near_duplicate_threshold, 0.7);
    }

    #[test]
    fn explicit_file_then_environment_then_cli_define_precedence() {
        let directory = tempfile::tempdir().unwrap();
        let config_path = directory.path().join("graphrag.toml");
        fs::write(
            &config_path,
            r#"
                [database]
                path = "file.db"
                [inference]
                embedding_provider = "file-provider"
                tei_max_batch = 7
                [search]
                default_limit = 3
            "#,
        )
        .unwrap();

        let config = RuntimeConfig::load_with_env_and_default_path(
            Some(&config_path),
            &CliOverrides {
                database_path: Some("cli.db".into()),
            },
            &env(&[
                ("GRAPHRAG_DB_PATH", "environment.db"),
                ("TEI_PROVIDER", "ollama"),
                ("TEI_MAX_BATCH", "9"),
                ("GRAPHRAG_SEARCH_DEFAULT_LIMIT", "11"),
            ]),
            None,
        )
        .unwrap();

        assert_eq!(config.database.path, PathBuf::from("cli.db"));
        assert_eq!(config.inference.embedding_provider, "ollama");
        assert_eq!(config.inference.tei_max_batch, 9);
        assert_eq!(config.search.default_limit, 11);
    }

    #[test]
    fn missing_explicit_file_is_an_error() {
        let missing = tempfile::tempdir().unwrap().path().join("missing.toml");
        let error = RuntimeConfig::load_with_env_and_default_path(
            Some(&missing),
            &CliOverrides::default(),
            &env(&[]),
            None,
        )
        .unwrap_err();
        assert!(matches!(error, ConfigError::ReadFile { .. }));
    }

    #[test]
    fn invalid_environment_value_identifies_its_field() {
        let error = RuntimeConfig::load_with_env_and_default_path(
            None,
            &CliOverrides::default(),
            &env(&[("GRAPHRAG_AUGMENT_MAX_TOKENS", "zero")]),
            None,
        )
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("GRAPHRAG_AUGMENT_MAX_TOKENS must be a positive integer"));
    }

    #[test]
    fn ollama_provider_keeps_existing_one_variable_setup() {
        let config = RuntimeConfig::load_with_env_and_default_path(
            None,
            &CliOverrides::default(),
            &env(&[
                ("TEI_PROVIDER", "ollama"),
                ("TGI_PROVIDER", "ollama"),
                ("OLLAMA_URL", "http://ollama.example:11434"),
            ]),
            None,
        )
        .unwrap();

        assert_eq!(
            config.inference.embedding_url,
            "http://ollama.example:11434"
        );
        assert_eq!(
            config.inference.extraction_url,
            "http://ollama.example:11434"
        );
    }

    #[test]
    fn provider_names_are_normalized_before_factory_injection() {
        let config = RuntimeConfig::load_with_env_and_default_path(
            None,
            &CliOverrides::default(),
            &env(&[("TEI_PROVIDER", " Ollama "), ("TGI_PROVIDER", " OLLAMA ")]),
            None,
        )
        .unwrap();

        assert_eq!(config.inference.embedding_provider, "ollama");
        assert_eq!(config.inference.extraction_provider, "ollama");
    }

    #[test]
    fn whitespace_padded_ollama_overrides_use_the_legacy_fallback_url() {
        let directory = tempfile::tempdir().unwrap();
        let config_path = directory.path().join("graphrag.toml");
        fs::write(
            &config_path,
            r#"
                [inference]
                embedding_url = "http://tei.example:8081"
                extraction_url = "http://tgi.example:8082"
            "#,
        )
        .unwrap();

        let config = RuntimeConfig::load_with_env_and_default_path(
            Some(&config_path),
            &CliOverrides::default(),
            &env(&[
                ("TEI_PROVIDER", " Ollama "),
                ("TGI_PROVIDER", " OLLAMA "),
                ("TEI_URL", " "),
                ("TGI_URL", "\t"),
                ("OLLAMA_URL", "http://ollama.example:11434"),
            ]),
            None,
        )
        .unwrap();

        assert_eq!(
            config.inference.embedding_url,
            "http://ollama.example:11434"
        );
        assert_eq!(
            config.inference.extraction_url,
            "http://ollama.example:11434"
        );
    }

    #[test]
    fn legacy_inference_and_librarian_environment_settings_are_typed() {
        let config = RuntimeConfig::load_with_env_and_default_path(
            None,
            &CliOverrides::default(),
            &env(&[
                ("TEI_PROMPT_NAME_QUERY", "query"),
                ("TEI_PROMPT_NAME_PASSAGE", "passage"),
                ("STRICT_ENTITY_JSON", "false"),
                ("EXTRACT_MAX_ENTITIES", "7"),
                ("EXTRACT_MAX_RELATIONSHIPS", "4"),
                ("TGI_OLLAMA_TIMEOUT_SECS", "45"),
                ("TGI_OLLAMA_OPTIONS", r#"{"temperature":0,"num_ctx":1024}"#),
                ("SKIP_ENTITY_EXTRACTION", "yes"),
                ("EXTRACT_LOG_EACH", "on"),
                ("EXTRACT_MAX_CHARS", "0"),
                ("EXTRACT_PROGRESS_EVERY", "11"),
                ("EXTRACT_PROGRESS_EVERY_SECS", "6"),
                ("IMPORT_PROGRESS_EVERY", "12"),
                ("IMPORT_PROGRESS_EVERY_SECS", "7"),
                ("GRAPHRAG_GARDENER_AUTO_APPLY", "true"),
            ]),
            None,
        )
        .unwrap();

        assert_eq!(
            config.inference.tei_prompt_name_query.as_deref(),
            Some("query")
        );
        assert_eq!(
            config.inference.tei_prompt_name_passage.as_deref(),
            Some("passage")
        );
        assert!(!config.inference.strict_entity_json);
        assert_eq!(config.inference.max_entities, 7);
        assert_eq!(config.inference.max_relationships, 4);
        assert_eq!(config.inference.ollama_timeout_secs, 45);
        assert_eq!(
            config.inference.ollama_options,
            Some(serde_json::json!({"temperature": 0, "num_ctx": 1024}))
        );
        assert!(config.librarian.skip_entity_extraction);
        assert!(config.librarian.extract_log_each);
        assert_eq!(config.librarian.extract_max_chars, 0);
        assert_eq!(config.librarian.extract_progress_every, 11);
        assert_eq!(config.librarian.extract_progress_every_secs, 6);
        assert_eq!(config.librarian.import_progress_every, 12);
        assert_eq!(config.librarian.import_progress_every_secs, 7);
        assert!(config.gardener.auto_apply);
    }

    #[test]
    fn toml_supports_typed_ollama_options_and_librarian_controls() {
        let directory = tempfile::tempdir().unwrap();
        let config_path = directory.path().join("graphrag.toml");
        fs::write(
            &config_path,
            r#"
                [inference]
                embedding_provider = "ollama"
                embedding_url = "http://remote-embed.example:11434"
                extraction_provider = "ollama"
                extraction_url = "http://remote-extract.example:11434"
                ollama_options = { temperature = 0, num_ctx = 1024 }
                [librarian]
                skip_entity_extraction = true
                extract_max_chars = 0
            "#,
        )
        .unwrap();

        let config = RuntimeConfig::load_with_env_and_default_path(
            Some(&config_path),
            &CliOverrides::default(),
            &env(&[("TEI_PROVIDER", " "), ("TGI_PROVIDER", "\t")]),
            None,
        )
        .unwrap();

        assert_eq!(
            config.inference.ollama_options,
            Some(serde_json::json!({"temperature": 0, "num_ctx": 1024}))
        );
        assert!(config.librarian.skip_entity_extraction);
        assert_eq!(config.librarian.extract_max_chars, 0);
        assert_eq!(
            config.inference.embedding_url,
            "http://remote-embed.example:11434"
        );
        assert_eq!(
            config.inference.extraction_url,
            "http://remote-extract.example:11434"
        );
    }

    #[test]
    fn provider_only_toml_uses_the_shared_ollama_endpoint() {
        let directory = tempfile::tempdir().unwrap();
        let config_path = directory.path().join("graphrag.toml");
        fs::write(
            &config_path,
            r#"
                [inference]
                embedding_provider = "ollama"
                extraction_provider = "ollama"
                ollama_url = "http://remote-ollama.example:11434"
            "#,
        )
        .unwrap();

        let config = RuntimeConfig::load_with_env_and_default_path(
            Some(&config_path),
            &CliOverrides::default(),
            &env(&[]),
            None,
        )
        .unwrap();

        assert_eq!(
            config.inference.embedding_url,
            "http://remote-ollama.example:11434"
        );
        assert_eq!(
            config.inference.extraction_url,
            "http://remote-ollama.example:11434"
        );
    }

    #[test]
    fn invalid_legacy_controls_fail_with_field_specific_errors() {
        let error = RuntimeConfig::load_with_env_and_default_path(
            None,
            &CliOverrides::default(),
            &env(&[("TGI_OLLAMA_OPTIONS", "[]")]),
            None,
        )
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("TGI_OLLAMA_OPTIONS must be a JSON object"));

        let error = RuntimeConfig::load_with_env_and_default_path(
            None,
            &CliOverrides::default(),
            &env(&[("EXTRACT_PROGRESS_EVERY", "0")]),
            None,
        )
        .unwrap_err();
        assert!(error.to_string().contains("limits must be positive"));

        let error = RuntimeConfig::load_with_env_and_default_path(
            None,
            &CliOverrides::default(),
            &env(&[("STRICT_ENTITY_JSON", "tru")]),
            None,
        )
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("STRICT_ENTITY_JSON must be a boolean"));

        let error = RuntimeConfig::load_with_env_and_default_path(
            None,
            &CliOverrides::default(),
            &env(&[("TEI_PROVIDER", "olama")]),
            None,
        )
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("inference.embedding_provider must be tei or ollama"));
    }

    #[test]
    fn home_paths_are_expanded_from_files_and_environment() {
        let directory = tempfile::tempdir().unwrap();
        let config_path = directory.path().join("graphrag.toml");
        fs::write(&config_path, "[database]\npath = \"~/.graphrag/test\"\n").unwrap();
        let config = RuntimeConfig::load_with_env_and_default_path(
            Some(&config_path),
            &CliOverrides::default(),
            &env(&[]),
            None,
        )
        .unwrap();
        let expected = dirs::home_dir().unwrap().join(".graphrag/test");
        assert_eq!(config.database.path, expected);
    }

    #[test]
    fn graphrag_config_environment_path_expands_home_directory() {
        let home = dirs::home_dir().unwrap();
        let file_name = format!(".graphrag-config-test-{}.toml", std::process::id());
        let config_path = home.join(&file_name);
        fs::write(&config_path, "[logging]\nlevel = \"info\"\n").unwrap();

        let config = RuntimeConfig::load_with_env_and_default_path(
            None,
            &CliOverrides::default(),
            &env(&[("GRAPHRAG_CONFIG", &format!("~/{file_name}"))]),
            None,
        )
        .unwrap();
        fs::remove_file(&config_path).unwrap();

        assert_eq!(config.logging.level, "info");
    }

    #[test]
    fn blank_graphrag_config_is_treated_as_absent() {
        let config = RuntimeConfig::load_with_env_and_default_path(
            None,
            &CliOverrides::default(),
            &env(&[("GRAPHRAG_CONFIG", " \t ")]),
            None,
        )
        .unwrap();
        assert_eq!(config.logging.level, "warn");
    }

    #[test]
    fn legacy_provider_overrides_restore_dedicated_default_endpoints() {
        let directory = tempfile::tempdir().unwrap();
        let config_path = directory.path().join("graphrag.toml");
        fs::write(
            &config_path,
            r#"
                [inference]
                embedding_provider = "ollama"
                embedding_url = "http://ollama.example:11434"
                extraction_provider = "ollama"
                extraction_url = "http://ollama.example:11434"
            "#,
        )
        .unwrap();

        let config = RuntimeConfig::load_with_env_and_default_path(
            Some(&config_path),
            &CliOverrides::default(),
            &env(&[("TEI_PROVIDER", "tei"), ("TGI_PROVIDER", "tgi")]),
            None,
        )
        .unwrap();
        assert_eq!(config.inference.embedding_url, "http://localhost:8081");
        assert_eq!(config.inference.extraction_url, "http://localhost:8082");
    }

    #[test]
    fn search_fusion_controls_are_typed_and_validated() {
        let config = RuntimeConfig::load_with_env_and_default_path(
            None,
            &CliOverrides::default(),
            &env(&[
                ("GRAPHRAG_SEARCH_FUSION_STRATEGY", "weighted"),
                ("GRAPHRAG_SEARCH_RRF_K", "42"),
                ("GRAPHRAG_SEARCH_CANDIDATE_POOL_MULTIPLIER", "3"),
                ("GRAPHRAG_SEARCH_CANDIDATE_POOL_MIN", "12"),
                ("GRAPHRAG_SEARCH_CANDIDATE_POOL_MAX", "60"),
                ("GRAPHRAG_SEARCH_NOTE_WEIGHT", "1.2"),
                ("GRAPHRAG_SEARCH_MESSAGE_WEIGHT", "0.8"),
                ("GRAPHRAG_SEARCH_CONVERSATION_SUMMARY_WEIGHT", "0.5"),
            ]),
            None,
        )
        .unwrap();
        assert_eq!(config.search.fusion_strategy, "weighted");
        assert_eq!(config.search.rrf_k, 42);
        assert_eq!(config.search.candidate_pool_max, 60);

        let error = RuntimeConfig::load_with_env_and_default_path(
            None,
            &CliOverrides::default(),
            &env(&[("GRAPHRAG_SEARCH_FUSION_STRATEGY", "learned")]),
            None,
        )
        .unwrap_err();
        assert!(error.to_string().contains("fusion_strategy"));
    }
}
