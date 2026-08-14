//! Local inference clients for embeddings (TEI) and entity extraction (TGI).

use crate::{AgentError, Result};
use async_trait::async_trait;
use graphrag_db::schema::EMBEDDING_DIMENSION;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tracing::{debug, info};

const DEFAULT_TEI_URL: &str = "http://localhost:8081";
const DEFAULT_TEI_PROVIDER: &str = "tei";
const DEFAULT_TGI_URL: &str = "http://localhost:8082";
const DEFAULT_TGI_PROVIDER: &str = "tgi";
const DEFAULT_TEI_MAX_BATCH: usize = 32;
const DEFAULT_INFERENCE_TIMEOUT_SECS: u64 = 30;
const DEFAULT_OLLAMA_TIMEOUT_SECS: u64 = 120;
const DEFAULT_STRICT_ENTITY_JSON: bool = true;
const DEFAULT_MAX_ENTITIES: usize = 30;
const DEFAULT_MAX_RELATIONSHIPS: usize = 15;

/// A shareable embeddings provider used by agents.
///
/// The trait uses `async_trait` so it remains object-safe: callers can pass an
/// `Arc<dyn Embedder>` without propagating a generic provider type through the
/// CLI or agent graph.
#[async_trait]
pub trait Embedder: Send + Sync {
    async fn embed(&self, text: &str, is_query: bool) -> Result<Vec<f32>>;
    async fn embed_batch(&self, texts: &[String], is_query: bool) -> Result<Vec<Vec<f32>>>;
    async fn health(&self) -> Result<bool>;
    fn capabilities(&self) -> InferenceCapabilities;
}

/// A shareable entity-extraction provider used by agents.
#[async_trait]
pub trait EntityExtractor: Send + Sync {
    async fn extract(&self, text: &str) -> Result<EntityExtraction>;
    async fn health(&self) -> Result<bool>;
    fn capabilities(&self) -> InferenceCapabilities;
}

pub type SharedEmbedder = Arc<dyn Embedder>;
pub type SharedEntityExtractor = Arc<dyn EntityExtractor>;

/// Provider metadata exposed without coupling callers to a HTTP client.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InferenceCapabilities {
    pub provider: String,
    pub model: String,
    pub endpoint: String,
    pub known_dimension: Option<usize>,
    /// Canonical, non-secret settings that affect an inference result. This is
    /// folded into durable cache keys so a model setting change cannot reuse a
    /// semantically stale response.
    pub cache_identity: String,
}

/// Production inference adapters selected from the environment in one place.
#[derive(Clone)]
pub struct InferenceProviders {
    pub embedder: SharedEmbedder,
    pub extractor: SharedEntityExtractor,
}

/// Small, typed factory input that configuration code can construct without
/// coupling agents to process environment reads.
#[derive(Debug, Clone, PartialEq)]
pub struct InferenceProviderConfig {
    pub embedding_provider: String,
    pub embedding_url: String,
    pub embedding_model: String,
    pub extraction_provider: String,
    pub extraction_url: String,
    pub extraction_model: String,
    pub timeout_secs: u64,
    pub tei_max_batch: usize,
    pub tei_prompt_name_query: Option<String>,
    pub tei_prompt_name_passage: Option<String>,
    pub strict_entity_json: bool,
    pub max_entities: usize,
    pub max_relationships: usize,
    pub ollama_timeout_secs: u64,
    pub ollama_options: Option<Value>,
}

impl Default for InferenceProviderConfig {
    fn default() -> Self {
        Self {
            embedding_provider: DEFAULT_TEI_PROVIDER.to_string(),
            embedding_url: DEFAULT_TEI_URL.to_string(),
            embedding_model: "unknown".to_string(),
            extraction_provider: DEFAULT_TGI_PROVIDER.to_string(),
            extraction_url: DEFAULT_TGI_URL.to_string(),
            extraction_model: "unknown".to_string(),
            timeout_secs: DEFAULT_INFERENCE_TIMEOUT_SECS,
            tei_max_batch: DEFAULT_TEI_MAX_BATCH,
            tei_prompt_name_query: None,
            tei_prompt_name_passage: None,
            strict_entity_json: DEFAULT_STRICT_ENTITY_JSON,
            max_entities: DEFAULT_MAX_ENTITIES,
            max_relationships: DEFAULT_MAX_RELATIONSHIPS,
            ollama_timeout_secs: DEFAULT_OLLAMA_TIMEOUT_SECS,
            ollama_options: None,
        }
    }
}

impl InferenceProviders {
    /// Construct adapters from resolved settings without consulting the process
    /// environment. This keeps the factory deterministic under test and is the
    /// integration point for typed runtime configuration.
    pub fn from_config(config: &InferenceProviderConfig) -> Self {
        let embedder: SharedEmbedder = if config.embedding_provider.eq_ignore_ascii_case("ollama") {
            Arc::new(
                TeiClient::ollama(&config.embedding_url, &config.embedding_model)
                    .with_runtime_config(config),
            )
        } else {
            Arc::new(
                TeiClient::configured(&config.embedding_url, &config.embedding_model)
                    .with_runtime_config(config),
            )
        };
        let extractor: SharedEntityExtractor =
            if config.extraction_provider.eq_ignore_ascii_case("ollama") {
                Arc::new(
                    TgiClient::ollama(&config.extraction_url, &config.extraction_model)
                        .with_runtime_config(config),
                )
            } else {
                Arc::new(
                    TgiClient::configured(&config.extraction_url, &config.extraction_model)
                        .with_runtime_config(config),
                )
            };

        Self {
            embedder,
            extractor,
        }
    }
}

#[derive(Clone)]
pub struct TeiClient {
    client: Client,
    base_url: String,
    provider: TeiProvider,
    model: String,
    max_batch: usize,
    prompt_name_query: Option<String>,
    prompt_name_passage: Option<String>,
}

impl TeiClient {
    pub fn new(base_url: impl Into<String>) -> Self {
        Self::configured(base_url, "unknown")
    }

    pub fn configured(base_url: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            client: Client::new(),
            base_url: base_url.into(),
            provider: TeiProvider::Tei,
            model: model.into(),
            max_batch: DEFAULT_TEI_MAX_BATCH,
            prompt_name_query: None,
            prompt_name_passage: None,
        }
    }

    pub fn ollama(base_url: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            client: Client::new(),
            base_url: base_url.into(),
            provider: TeiProvider::Ollama,
            model: model.into(),
            max_batch: DEFAULT_TEI_MAX_BATCH,
            prompt_name_query: None,
            prompt_name_passage: None,
        }
    }

    fn with_runtime_config(mut self, config: &InferenceProviderConfig) -> Self {
        self.client = Client::builder()
            .timeout(Duration::from_secs(config.timeout_secs))
            .build()
            .expect("valid inference HTTP client configuration");
        self.max_batch = config.tei_max_batch;
        self.prompt_name_query = config.tei_prompt_name_query.clone();
        self.prompt_name_passage = config.tei_prompt_name_passage.clone();
        self
    }

    pub async fn embed(&self, text: &str, is_query: bool) -> Result<Vec<f32>> {
        if matches!(self.provider, TeiProvider::Ollama) {
            let embedding = self.ollama_embed(text).await?;
            validate_embedding_dim(embedding.len())?;
            return Ok(embedding);
        }

        let prompt_name = if is_query {
            self.prompt_name_query.as_deref()
        } else {
            self.prompt_name_passage.as_deref()
        };

        let url = format!("{}/embed", self.base_url);
        let request = TeiEmbedRequest {
            inputs: text,
            truncate: true,
            prompt_name,
        };

        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await?
            .error_for_status()?
            .json::<Value>()
            .await?;

        let embedding = parse_embedding_response(response)?;
        validate_embedding_dim(embedding.len())?;
        Ok(embedding)
    }

    pub async fn embed_batch(&self, texts: &[String], is_query: bool) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        if matches!(self.provider, TeiProvider::Ollama) {
            let mut results = Vec::with_capacity(texts.len());
            for text in texts {
                let embedding = self.ollama_embed(text).await?;
                validate_embedding_dim(embedding.len())?;
                results.push(embedding);
            }
            return Ok(results);
        }

        let prompt_name = if is_query {
            self.prompt_name_query.as_deref()
        } else {
            self.prompt_name_passage.as_deref()
        };

        let url = format!("{}/embed", self.base_url);
        let mut results = Vec::with_capacity(texts.len());

        for chunk in texts.chunks(self.max_batch) {
            let request = TeiEmbedBatchRequest {
                inputs: chunk,
                truncate: true,
                prompt_name,
            };

            let response = self
                .client
                .post(&url)
                .json(&request)
                .send()
                .await?
                .error_for_status()?
                .json::<Value>()
                .await?;

            let embeddings = parse_embeddings_response(response)?;
            if embeddings.len() != chunk.len() {
                return Err(AgentError::Processing(format!(
                    "Embedding provider returned {} embeddings for a batch of {} inputs",
                    embeddings.len(),
                    chunk.len()
                )));
            }
            for embedding in &embeddings {
                validate_embedding_dim(embedding.len())?;
            }
            results.extend(embeddings);
        }

        Ok(results)
    }

    pub async fn health(&self) -> Result<bool> {
        if matches!(self.provider, TeiProvider::Ollama) {
            let url = format!("{}/api/tags", self.base_url);
            let response = self.client.get(&url).send().await?;
            return Ok(response.status().is_success());
        }

        let url = format!("{}/health", self.base_url);
        let response = self.client.get(&url).send().await?;
        Ok(response.status().is_success())
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Name of the concrete embedding provider selected for this client.
    pub fn provider_name(&self) -> &'static str {
        match self.provider {
            TeiProvider::Tei => "tei",
            TeiProvider::Ollama => "ollama",
        }
    }

    /// Model identifier used by the selected provider.
    pub fn model(&self) -> &str {
        match self.provider {
            // TEI's health/embed API does not expose its serving image or model
            // identifier. Returning the Ollama fallback here would make an eval
            // baseline claim a model that was never used.
            TeiProvider::Tei => "unknown",
            TeiProvider::Ollama => &self.model,
        }
    }

    async fn ollama_embed(&self, text: &str) -> Result<Vec<f32>> {
        let url = format!("{}/api/embeddings", self.base_url);
        let request = OllamaEmbedRequest {
            model: self.model.clone(),
            prompt: text.to_string(),
        };

        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await?
            .error_for_status()?
            .json::<OllamaEmbedResponse>()
            .await?;

        Ok(response.embedding)
    }
}

#[async_trait]
impl Embedder for TeiClient {
    async fn embed(&self, text: &str, is_query: bool) -> Result<Vec<f32>> {
        TeiClient::embed(self, text, is_query).await
    }

    async fn embed_batch(&self, texts: &[String], is_query: bool) -> Result<Vec<Vec<f32>>> {
        TeiClient::embed_batch(self, texts, is_query).await
    }

    async fn health(&self) -> Result<bool> {
        TeiClient::health(self).await
    }

    fn capabilities(&self) -> InferenceCapabilities {
        InferenceCapabilities {
            provider: match self.provider {
                TeiProvider::Tei => DEFAULT_TEI_PROVIDER,
                TeiProvider::Ollama => "ollama",
            }
            .to_string(),
            model: self.model.clone(),
            endpoint: self.base_url.clone(),
            known_dimension: Some(EMBEDDING_DIMENSION),
            cache_identity: format!(
                "provider={};prompt_query={};prompt_passage={}",
                self.provider_name(),
                self.prompt_name_query.as_deref().unwrap_or(""),
                self.prompt_name_passage.as_deref().unwrap_or(""),
            ),
        }
    }
}

#[derive(Clone)]
pub struct TgiClient {
    client: Client,
    base_url: String,
    json_schema: Option<Value>,
    provider: TgiProvider,
    model: String,
    strict_entity_json: bool,
    max_entities: usize,
    max_relationships: usize,
    ollama_timeout_secs: u64,
    ollama_options: Option<Value>,
}

impl TgiClient {
    pub fn new(base_url: impl Into<String>) -> Self {
        Self::configured(base_url, "unknown")
    }

    pub fn configured(base_url: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            client: Client::new(),
            base_url: base_url.into(),
            json_schema: None,
            provider: TgiProvider::Tgi,
            model: model.into(),
            strict_entity_json: DEFAULT_STRICT_ENTITY_JSON,
            max_entities: DEFAULT_MAX_ENTITIES,
            max_relationships: DEFAULT_MAX_RELATIONSHIPS,
            ollama_timeout_secs: DEFAULT_OLLAMA_TIMEOUT_SECS,
            ollama_options: None,
        }
    }

    pub fn ollama(base_url: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            client: Client::new(),
            base_url: base_url.into(),
            json_schema: None,
            provider: TgiProvider::Ollama,
            model: model.into(),
            strict_entity_json: DEFAULT_STRICT_ENTITY_JSON,
            max_entities: DEFAULT_MAX_ENTITIES,
            max_relationships: DEFAULT_MAX_RELATIONSHIPS,
            ollama_timeout_secs: DEFAULT_OLLAMA_TIMEOUT_SECS,
            ollama_options: None,
        }
    }

    fn with_runtime_config(mut self, config: &InferenceProviderConfig) -> Self {
        self.client = Client::builder()
            .timeout(Duration::from_secs(config.timeout_secs))
            .build()
            .expect("valid inference HTTP client configuration");
        self.strict_entity_json = config.strict_entity_json;
        self.max_entities = config.max_entities;
        self.max_relationships = config.max_relationships;
        self.ollama_timeout_secs = config.ollama_timeout_secs;
        self.ollama_options = config.ollama_options.clone();
        self
    }

    pub fn with_json_schema(mut self, schema: Value) -> Self {
        self.json_schema = Some(schema);
        self
    }

    pub async fn extract(&self, text: &str) -> Result<EntityExtraction> {
        let entity_cap = self.max_entities;
        let relationship_cap = self.max_relationships;
        let prompt = format!(
            "Return ONLY valid JSON. No markdown, no extra keys.\n\nSchema:\n{{\"entities\":[{{\"name\":string,\"type\":string}}],\"relationships\":[{{\"source\":string,\"target\":string,\"relationship_type\":string}}]}}\n\nRules:\n- Strings only, double-quoted\n- Keep strings short (1-6 words)\n- If unsure, return empty arrays\n- Max {entity_cap} entities, max {relationship_cap} relationships\n\nText:\n{}",
            text,
            entity_cap = entity_cap,
            relationship_cap = relationship_cap
        );
        match self.provider {
            TgiProvider::Tgi => {
                let generated = self.tgi_generate(prompt).await?;
                let cleaned = normalize_json_payload(&generated);
                let extraction = parse_entity_extraction(&cleaned, self.strict_entity_json)
                    .map_err(|e| {
                        AgentError::Processing(format!(
                            "TGI returned invalid JSON: {} ({})",
                            generated, e
                        ))
                    })?;
                Ok(extraction)
            }
            TgiProvider::Ollama => {
                let options = self.ollama_options.clone();
                let budgets = ollama_predict_budgets(&options);
                let strict = self.strict_entity_json;

                for (idx, budget) in budgets.iter().enumerate() {
                    let attempt_options = merge_options(
                        options.clone(),
                        Some(json!({
                            "num_predict": budget
                        })),
                    );
                    let (generated, done_reason) = self
                        .ollama_chat_generate_with_meta(&prompt, attempt_options)
                        .await?;
                    let cleaned = normalize_json_payload(&generated);
                    if let Ok(extraction) = parse_entity_extraction(&cleaned, strict) {
                        return Ok(extraction);
                    }

                    if strict
                        && idx + 1 < budgets.len()
                        && should_retry_ollama_parse_failure(done_reason.as_deref())
                    {
                        info!(
                            "Ollama returned invalid JSON (done_reason={}); retrying with num_predict={}",
                            done_reason.as_deref().unwrap_or("none"),
                            budgets[idx + 1]
                        );
                        continue;
                    }

                    if strict {
                        return Err(AgentError::Processing(format!(
                            "TGI returned invalid JSON: {}",
                            generated
                        )));
                    }
                }

                debug!("Ollama extraction failed, retrying with entities-only schema");
                let retry_prompt = format!(
                    "Return ONLY valid JSON with the schema {{\"entities\":[{{\"name\":string,\"type\":string}}...],\"relationships\":[]}}.\nAll fields must be strings and double-quoted. Do not include any other keys.\nLimits: up to {entity_cap} entities.\nText:\n{}",
                    text,
                    entity_cap = entity_cap
                );
                let retry_options = merge_options(
                    options,
                    Some(json!({
                        "num_ctx": 512,
                        "num_predict": 128,
                        "temperature": 0,
                        "stop": ["```"]
                    })),
                );
                let (generated_retry, _) = self
                    .ollama_chat_generate_with_meta(&retry_prompt, retry_options)
                    .await?;
                let cleaned_retry = normalize_json_payload(&generated_retry);
                let extraction = parse_entity_extraction(&cleaned_retry, strict).map_err(|e| {
                    AgentError::Processing(format!(
                        "TGI returned invalid JSON: {} ({})",
                        generated_retry, e
                    ))
                })?;
                debug!("Ollama extraction succeeded via entities-only fallback");
                Ok(extraction)
            }
        }
    }

    pub async fn health(&self) -> Result<bool> {
        let url = match self.provider {
            TgiProvider::Tgi => format!("{}/health", self.base_url),
            TgiProvider::Ollama => format!("{}/api/tags", self.base_url),
        };
        let response = self.client.get(&url).send().await?;
        Ok(response.status().is_success())
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    async fn tgi_generate(&self, prompt: String) -> Result<String> {
        let url = format!("{}/generate", self.base_url);
        let request = TgiGenerateRequest {
            inputs: prompt,
            parameters: TgiParameters {
                max_new_tokens: Some(512),
                return_full_text: Some(false),
                stop: Some(vec!["\n\n".to_string()]),
                grammar: self.json_schema.clone(),
            },
        };

        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await?
            .error_for_status()?
            .json::<Value>()
            .await?;

        extract_generated_text(response)
    }

    async fn ollama_chat_generate_with_meta(
        &self,
        prompt: &str,
        options: Option<Value>,
    ) -> Result<(String, Option<String>)> {
        let url = format!("{}/api/chat", self.base_url);
        let system_prompt = "You are a strict JSON generator. Output MUST be a single JSON object matching the provided schema. No prose, no markdown.";
        let request = OllamaChatRequest {
            model: self.model.clone(),
            messages: vec![
                OllamaChatMessage {
                    role: "system".to_string(),
                    content: system_prompt.to_string(),
                },
                OllamaChatMessage {
                    role: "user".to_string(),
                    content: prompt.to_string(),
                },
            ],
            stream: false,
            format: Some(self.entity_extraction_schema()),
            options,
        };

        let response = self
            .client
            .post(&url)
            .json(&request)
            .timeout(Duration::from_secs(self.ollama_timeout_secs))
            .send()
            .await?
            .error_for_status()?
            .json::<OllamaChatResponse>()
            .await?;

        if let Some(done) = response.done {
            info!("Ollama chat done={}", done);
        }

        if let Some(done_reason) = response.done_reason.as_deref() {
            info!("Ollama chat done_reason={}", done_reason);
        }

        if let Some(total_ms) = response.total_duration.map(|ns| ns as f64 / 1_000_000.0) {
            info!("Ollama chat total_duration_ms={:.2}", total_ms);
        }

        let content = response.message.content;
        let trimmed = content.trim_end();
        if !trimmed.ends_with('}') {
            info!(
                "Ollama chat content does not end with '}}' (len={})",
                trimmed.len()
            );
        }

        Ok((content, response.done_reason))
    }

    fn entity_extraction_schema(&self) -> Value {
        entity_extraction_schema(self.max_entities, self.max_relationships)
    }
}

#[async_trait]
impl EntityExtractor for TgiClient {
    async fn extract(&self, text: &str) -> Result<EntityExtraction> {
        TgiClient::extract(self, text).await
    }

    async fn health(&self) -> Result<bool> {
        TgiClient::health(self).await
    }

    fn capabilities(&self) -> InferenceCapabilities {
        InferenceCapabilities {
            provider: match self.provider {
                TgiProvider::Tgi => DEFAULT_TGI_PROVIDER,
                TgiProvider::Ollama => "ollama",
            }
            .to_string(),
            model: self.model.clone(),
            endpoint: self.base_url.clone(),
            known_dimension: None,
            cache_identity: format!(
                "provider={};strict_json={};max_entities={};max_relationships={};ollama_timeout_secs={};ollama_options={};json_schema={}",
                match self.provider { TgiProvider::Tgi => "tgi", TgiProvider::Ollama => "ollama" },
                self.strict_entity_json,
                self.max_entities,
                self.max_relationships,
                self.ollama_timeout_secs,
                canonical_json_identity(self.ollama_options.as_ref()),
                canonical_json_identity(self.json_schema.as_ref()),
            ),
        }
    }
}

/// Deterministic offline embeddings for agent tests.
///
/// Inputs not explicitly configured receive a stable 1024-dimensional vector.
/// Configured inputs are useful for exercising ranking and dimension-error
/// paths without an HTTP inference service.
#[derive(Clone)]
pub struct DeterministicEmbedder {
    embeddings: Arc<Mutex<HashMap<(String, bool), Vec<f32>>>>,
    default_embedding: Vec<f32>,
    generate_per_input: bool,
    health: bool,
    failures_remaining: Arc<Mutex<usize>>,
    failure_message: String,
    batch_length_mismatch: bool,
    provider: String,
    model: String,
}

impl Default for DeterministicEmbedder {
    fn default() -> Self {
        Self {
            embeddings: Arc::new(Mutex::new(HashMap::new())),
            default_embedding: stable_embedding("default"),
            generate_per_input: true,
            health: true,
            failures_remaining: Arc::new(Mutex::new(0)),
            failure_message: "injected embedding failure".to_string(),
            batch_length_mismatch: false,
            provider: "deterministic-test".to_string(),
            model: "fixture".to_string(),
        }
    }
}

impl DeterministicEmbedder {
    pub fn with_embedding(
        self,
        text: impl Into<String>,
        is_query: bool,
        embedding: Vec<f32>,
    ) -> Self {
        self.embeddings
            .lock()
            .expect("deterministic embedder lock poisoned")
            .insert((text.into(), is_query), embedding);
        self
    }

    pub fn with_default_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.default_embedding = embedding;
        self.generate_per_input = false;
        self
    }

    pub fn unhealthy(mut self) -> Self {
        self.health = false;
        self
    }

    pub fn fail_next_requests(self, count: usize, message: impl Into<String>) -> Self {
        *self
            .failures_remaining
            .lock()
            .expect("deterministic embedder lock poisoned") = count;
        let mut this = self;
        this.failure_message = message.into();
        this
    }

    pub fn with_batch_length_mismatch(mut self) -> Self {
        self.batch_length_mismatch = true;
        self
    }

    /// Override the identity exposed to compatibility tests without involving
    /// a live inference provider.
    pub fn with_identity(mut self, provider: impl Into<String>, model: impl Into<String>) -> Self {
        self.provider = provider.into();
        self.model = model.into();
        self
    }

    fn maybe_fail(&self) -> Result<()> {
        let mut remaining = self
            .failures_remaining
            .lock()
            .expect("deterministic embedder lock poisoned");
        if *remaining > 0 {
            *remaining -= 1;
            return Err(AgentError::InferenceService(self.failure_message.clone()));
        }
        Ok(())
    }

    fn embedding_for(&self, text: &str, is_query: bool) -> Vec<f32> {
        self.embeddings
            .lock()
            .expect("deterministic embedder lock poisoned")
            .get(&(text.to_string(), is_query))
            .cloned()
            .unwrap_or_else(|| {
                if self.generate_per_input {
                    stable_embedding(&format!("{is_query}:{text}"))
                } else {
                    self.default_embedding.clone()
                }
            })
    }
}

#[async_trait]
impl Embedder for DeterministicEmbedder {
    async fn embed(&self, text: &str, is_query: bool) -> Result<Vec<f32>> {
        self.maybe_fail()?;
        Ok(self.embedding_for(text, is_query))
    }

    async fn embed_batch(&self, texts: &[String], is_query: bool) -> Result<Vec<Vec<f32>>> {
        self.maybe_fail()?;
        let mut embeddings = texts
            .iter()
            .map(|text| self.embedding_for(text, is_query))
            .collect::<Vec<_>>();
        if self.batch_length_mismatch && !embeddings.is_empty() {
            embeddings.pop();
        }
        Ok(embeddings)
    }

    async fn health(&self) -> Result<bool> {
        Ok(self.health)
    }

    fn capabilities(&self) -> InferenceCapabilities {
        InferenceCapabilities {
            provider: self.provider.clone(),
            model: self.model.clone(),
            endpoint: "offline://deterministic-embedder".to_string(),
            known_dimension: Some(self.default_embedding.len()),
            cache_identity: "deterministic-embedding-v1".to_string(),
        }
    }
}

/// Fixture-driven offline entity extraction for agent tests.
#[derive(Clone)]
pub struct FixtureEntityExtractor {
    fixtures: Arc<Mutex<HashMap<String, std::result::Result<EntityExtraction, String>>>>,
    default: EntityExtraction,
    health: bool,
    failures_remaining: Arc<Mutex<usize>>,
    failure_message: String,
    cache_identity: String,
}

impl Default for FixtureEntityExtractor {
    fn default() -> Self {
        Self {
            fixtures: Arc::new(Mutex::new(HashMap::new())),
            default: EntityExtraction {
                entities: Vec::new(),
                relationships: Vec::new(),
            },
            health: true,
            failures_remaining: Arc::new(Mutex::new(0)),
            failure_message: "injected extraction failure".to_string(),
            cache_identity: "fixture-extraction-v1".to_string(),
        }
    }
}

impl FixtureEntityExtractor {
    pub fn with_fixture(self, text: impl Into<String>, extraction: EntityExtraction) -> Self {
        self.fixtures
            .lock()
            .expect("fixture extractor lock poisoned")
            .insert(text.into(), Ok(extraction));
        self
    }

    /// Configure the given input to model a provider response that failed JSON
    /// validation. The returned error follows the production processing path.
    pub fn with_malformed_fixture(
        self,
        text: impl Into<String>,
        response: impl Into<String>,
    ) -> Self {
        self.fixtures
            .lock()
            .expect("fixture extractor lock poisoned")
            .insert(text.into(), Err(response.into()));
        self
    }

    pub fn with_default(mut self, extraction: EntityExtraction) -> Self {
        self.default = extraction;
        self
    }

    pub fn unhealthy(mut self) -> Self {
        self.health = false;
        self
    }

    pub fn fail_next_requests(self, count: usize, message: impl Into<String>) -> Self {
        *self
            .failures_remaining
            .lock()
            .expect("fixture extractor lock poisoned") = count;
        let mut this = self;
        this.failure_message = message.into();
        this
    }

    /// Adjust only the semantic cache identity in tests, mirroring production
    /// prompt/schema setting changes without an HTTP provider.
    pub fn with_cache_identity(mut self, identity: impl Into<String>) -> Self {
        self.cache_identity = identity.into();
        self
    }
}

#[async_trait]
impl EntityExtractor for FixtureEntityExtractor {
    async fn extract(&self, text: &str) -> Result<EntityExtraction> {
        let mut remaining = self
            .failures_remaining
            .lock()
            .expect("fixture extractor lock poisoned");
        if *remaining > 0 {
            *remaining -= 1;
            return Err(AgentError::InferenceService(self.failure_message.clone()));
        }
        drop(remaining);

        match self
            .fixtures
            .lock()
            .expect("fixture extractor lock poisoned")
            .get(text)
            .cloned()
        {
            Some(Ok(extraction)) => Ok(extraction),
            Some(Err(response)) => Err(AgentError::Processing(format!(
                "fixture extractor returned malformed JSON: {response}"
            ))),
            None => Ok(self.default.clone()),
        }
    }

    async fn health(&self) -> Result<bool> {
        Ok(self.health)
    }

    fn capabilities(&self) -> InferenceCapabilities {
        InferenceCapabilities {
            provider: "fixture-test".to_string(),
            model: "fixture".to_string(),
            endpoint: "offline://fixture-extractor".to_string(),
            known_dimension: None,
            cache_identity: self.cache_identity.clone(),
        }
    }
}

fn canonical_json_identity(value: Option<&Value>) -> String {
    match value {
        Some(value) => serde_json::to_string(value).unwrap_or_else(|_| "<unserializable>".into()),
        None => String::new(),
    }
}

fn stable_embedding(input: &str) -> Vec<f32> {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in input.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    (0..EMBEDDING_DIMENSION)
        .map(|idx| {
            let value = hash.wrapping_add((idx as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15));
            (value as u32) as f32 / u32::MAX as f32
        })
        .collect()
}

#[derive(Clone, Copy)]
enum TgiProvider {
    Tgi,
    Ollama,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityExtraction {
    pub entities: Vec<ExtractedEntity>,
    pub relationships: Vec<ExtractedRelationship>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedEntity {
    pub name: String,
    #[serde(alias = "type", alias = "entity_type")]
    pub entity_type: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedRelationship {
    pub source: String,
    pub target: String,
    #[serde(alias = "type", alias = "relationship_type")]
    pub relationship_type: String,
}

#[derive(Clone, Copy)]
enum TeiProvider {
    Tei,
    Ollama,
}

pub(crate) fn validate_embedding_dim(len: usize) -> Result<()> {
    if len != EMBEDDING_DIMENSION {
        return Err(AgentError::Processing(format!(
            "Embedding dimension {} does not match expected {}. Choose a 1024-dim model or update the schema.",
            len, EMBEDDING_DIMENSION
        )));
    }
    Ok(())
}

#[derive(Serialize)]
struct TeiEmbedRequest<'a> {
    inputs: &'a str,
    truncate: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_name: Option<&'a str>,
}

#[derive(Serialize)]
struct TeiEmbedBatchRequest<'a> {
    inputs: &'a [String],
    truncate: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_name: Option<&'a str>,
}

#[derive(Serialize)]
struct TgiGenerateRequest {
    inputs: String,
    parameters: TgiParameters,
}

#[derive(Serialize)]
struct TgiParameters {
    #[serde(skip_serializing_if = "Option::is_none")]
    max_new_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    return_full_text: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Vec<String>>,
    // Best-effort: TGI may accept a grammar/JSON schema constraint.
    #[serde(skip_serializing_if = "Option::is_none")]
    grammar: Option<Value>,
}

#[derive(Serialize)]
struct OllamaChatRequest {
    model: String,
    messages: Vec<OllamaChatMessage>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    format: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    options: Option<Value>,
}

#[derive(Serialize)]
struct OllamaChatMessage {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct OllamaEmbedRequest {
    model: String,
    prompt: String,
}

#[derive(Deserialize)]
struct OllamaChatResponse {
    message: OllamaChatMessageResponse,
    #[serde(default)]
    done: Option<bool>,
    #[serde(default)]
    done_reason: Option<String>,
    #[serde(default)]
    total_duration: Option<u64>,
}

#[derive(Deserialize)]
struct OllamaChatMessageResponse {
    content: String,
}

#[derive(Deserialize)]
struct OllamaEmbedResponse {
    embedding: Vec<f32>,
}

fn parse_embedding_response(value: Value) -> Result<Vec<f32>> {
    match value {
        Value::Array(items) => {
            if items.is_empty() {
                return Ok(Vec::new());
            }
            if items.first().map(|v| v.is_number()).unwrap_or(false) {
                serde_json::from_value(Value::Array(items)).map_err(|e| {
                    AgentError::Processing(format!("Invalid TEI embedding array: {}", e))
                })
            } else {
                let first = items
                    .into_iter()
                    .next()
                    .ok_or_else(|| AgentError::Processing("Missing embeddings".to_string()))?;
                serde_json::from_value(first).map_err(|e| {
                    AgentError::Processing(format!("Invalid TEI embedding array: {}", e))
                })
            }
        }
        other => Err(AgentError::Processing(format!(
            "Unexpected TEI response format: {}",
            other
        ))),
    }
}

fn parse_embeddings_response(value: Value) -> Result<Vec<Vec<f32>>> {
    match value {
        Value::Array(items) => {
            if items.is_empty() {
                return Ok(Vec::new());
            }
            if items.first().map(|v| v.is_array()).unwrap_or(false) {
                serde_json::from_value(Value::Array(items)).map_err(|e| {
                    AgentError::Processing(format!("Invalid TEI embeddings response: {}", e))
                })
            } else {
                let single: Vec<f32> =
                    serde_json::from_value(Value::Array(items)).map_err(|e| {
                        AgentError::Processing(format!("Invalid TEI embedding array: {}", e))
                    })?;
                Ok(vec![single])
            }
        }
        other => Err(AgentError::Processing(format!(
            "Unexpected TEI response format: {}",
            other
        ))),
    }
}

fn normalize_json_payload(payload: &str) -> String {
    let trimmed = payload.trim();
    if trimmed.is_empty() {
        return trimmed.to_string();
    }

    let without_fence = if trimmed.starts_with("```") {
        let mut lines = trimmed.lines();
        let _ = lines.next(); // drop ``` or ```json
        let mut content = lines.collect::<Vec<_>>().join("\n");
        if content.ends_with("```") {
            content.truncate(content.len().saturating_sub(3));
        }
        content.trim().to_string()
    } else {
        trimmed.to_string()
    };

    if let (Some(start), Some(end)) = (without_fence.find('{'), without_fence.rfind('}')) {
        if start < end {
            return without_fence[start..=end].to_string();
        }
    }

    without_fence
}

fn merge_options(base: Option<Value>, override_value: Option<Value>) -> Option<Value> {
    match (base, override_value) {
        (None, None) => None,
        (Some(value), None) | (None, Some(value)) => Some(value),
        (Some(Value::Object(mut base)), Some(Value::Object(override_obj))) => {
            for (k, v) in override_obj {
                base.insert(k, v);
            }
            Some(Value::Object(base))
        }
        (Some(value), Some(_)) => Some(value),
    }
}

fn ollama_predict_budgets(options: &Option<Value>) -> Vec<u32> {
    let default_budgets = vec![512, 768, 1024];
    let configured = options.as_ref().and_then(|value| {
        value.as_object().and_then(|map| {
            map.get("num_predict")
                .and_then(|value| value.as_u64())
                .and_then(|value| u32::try_from(value).ok())
        })
    });

    let Some(value) = configured else {
        return default_budgets;
    };

    // Expand upward from the configured value rather than collapsing to a single budget.
    let mut budgets = vec![value];
    let step: u32 = 256;
    let max_default = *default_budgets.last().unwrap_or(&1024);
    let max_budget = std::cmp::max(max_default, value.saturating_add(step.saturating_mul(2)));

    let mut next = if value % step == 0 {
        value.saturating_add(step)
    } else {
        ((value / step) + 1).saturating_mul(step)
    };

    while next <= max_budget {
        budgets.push(next);
        next = next.saturating_add(step);
    }

    budgets.sort_unstable();
    budgets.dedup();
    budgets
}

#[cfg(test)]
mod tests {
    use super::*;

    fn extraction() -> EntityExtraction {
        EntityExtraction {
            entities: vec![ExtractedEntity {
                name: "Rust".to_string(),
                entity_type: Some("concept".to_string()),
            }],
            relationships: Vec::new(),
        }
    }

    #[tokio::test]
    async fn deterministic_embeddings_are_stable_and_preserve_batch_order() {
        let embedder = DeterministicEmbedder::default()
            .with_embedding("first", false, vec![1.0; EMBEDDING_DIMENSION])
            .with_embedding("second", false, vec![2.0; EMBEDDING_DIMENSION]);

        assert_eq!(
            embedder.embed("first", false).await.unwrap(),
            embedder.embed("first", false).await.unwrap()
        );
        let batch = embedder
            .embed_batch(&["second".to_string(), "first".to_string()], false)
            .await
            .unwrap();
        assert_eq!(
            batch,
            vec![
                vec![2.0; EMBEDDING_DIMENSION],
                vec![1.0; EMBEDDING_DIMENSION]
            ]
        );
    }

    #[tokio::test]
    async fn deterministic_embedder_supports_injected_failures_health_and_mismatch() {
        let embedder = DeterministicEmbedder::default()
            .fail_next_requests(1, "temporary outage")
            .with_batch_length_mismatch()
            .unhealthy();

        assert!(matches!(
            embedder.embed("text", false).await,
            Err(AgentError::InferenceService(message)) if message == "temporary outage"
        ));
        assert!(!embedder.health().await.unwrap());
        assert_eq!(
            embedder
                .embed_batch(&["one".to_string(), "two".to_string()], false)
                .await
                .unwrap()
                .len(),
            1
        );

        let dimension_mismatch =
            DeterministicEmbedder::default().with_default_embedding(vec![0.0; 3]);
        assert_eq!(
            dimension_mismatch.embed("text", false).await.unwrap().len(),
            3
        );
        assert_eq!(dimension_mismatch.capabilities().known_dimension, Some(3));
    }

    #[tokio::test]
    async fn fixture_extractor_supports_results_malformed_data_and_failures() {
        let extractor = FixtureEntityExtractor::default()
            .with_fixture("known", extraction())
            .with_malformed_fixture("bad", "{not json")
            .fail_next_requests(1, "temporary outage");

        assert!(matches!(
            extractor.extract("known").await,
            Err(AgentError::InferenceService(message)) if message == "temporary outage"
        ));
        assert_eq!(
            extractor.extract("known").await.unwrap().entities[0].name,
            "Rust"
        );
        assert!(matches!(
            extractor.extract("bad").await,
            Err(AgentError::Processing(message)) if message.contains("malformed JSON")
        ));
        assert!(extractor
            .extract("unknown")
            .await
            .unwrap()
            .entities
            .is_empty());
    }

    #[test]
    fn provider_factory_uses_explicit_defaults() {
        let providers = InferenceProviders::from_config(&InferenceProviderConfig::default());
        assert_eq!(providers.embedder.capabilities().provider, "tei");
        assert_eq!(providers.extractor.capabilities().provider, "tgi");
        assert_eq!(
            providers.embedder.capabilities().known_dimension,
            Some(EMBEDDING_DIMENSION)
        );

        let configured = InferenceProviders::from_config(&InferenceProviderConfig {
            embedding_model: "intfloat/e5-large-v2".into(),
            extraction_model: "mistral-small".into(),
            ..InferenceProviderConfig::default()
        });
        assert_eq!(
            configured.embedder.capabilities().model,
            "intfloat/e5-large-v2"
        );
        assert_eq!(configured.extractor.capabilities().model, "mistral-small");
    }

    #[test]
    fn provider_runtime_settings_are_injected_without_environment_reads() {
        let config = InferenceProviderConfig {
            timeout_secs: 17,
            tei_max_batch: 3,
            tei_prompt_name_query: Some("query".into()),
            tei_prompt_name_passage: Some("passage".into()),
            strict_entity_json: false,
            max_entities: 4,
            max_relationships: 2,
            ollama_timeout_secs: 91,
            ollama_options: Some(json!({ "temperature": 0 })),
            ..InferenceProviderConfig::default()
        };

        let tei = TeiClient::configured("http://tei", "model").with_runtime_config(&config);
        assert_eq!(tei.max_batch, 3);
        assert_eq!(tei.prompt_name_query.as_deref(), Some("query"));
        assert_eq!(tei.prompt_name_passage.as_deref(), Some("passage"));

        let tgi = TgiClient::ollama("http://ollama", "model").with_runtime_config(&config);
        assert!(!tgi.strict_entity_json);
        assert_eq!(tgi.ollama_timeout_secs, 91);
        assert_eq!(tgi.ollama_options, Some(json!({ "temperature": 0 })));
        assert_eq!(
            tgi.entity_extraction_schema()["properties"]["entities"]["maxItems"],
            json!(4)
        );
        assert!(parse_entity_extraction("not JSON", true).is_err());
        assert!(parse_entity_extraction("entities: [{\"name\":\"Rust\"}]", false).is_ok());
    }
}

fn should_retry_ollama_parse_failure(done_reason: Option<&str>) -> bool {
    !matches!(done_reason, Some("stop"))
}

fn entity_extraction_schema(entity_cap: usize, relationship_cap: usize) -> Value {
    let max_name_len: usize = 80;
    let max_type_len: usize = 40;
    let max_rel_len: usize = 40;
    json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["entities", "relationships"],
        "properties": {
            "entities": {
                "type": "array",
                "maxItems": entity_cap,
                "items": {
                    "type": "object",
                    "additionalProperties": false,
                    "required": ["name"],
                    "properties": {
                        "name": { "type": "string", "maxLength": max_name_len },
                        "type": { "type": "string", "maxLength": max_type_len }
                    }
                }
            },
            "relationships": {
                "type": "array",
                "maxItems": relationship_cap,
                "items": {
                    "type": "object",
                    "additionalProperties": false,
                    "required": ["source", "target", "relationship_type"],
                    "properties": {
                        "source": { "type": "string", "maxLength": max_name_len },
                        "target": { "type": "string", "maxLength": max_name_len },
                        "relationship_type": { "type": "string", "maxLength": max_rel_len }
                    }
                }
            }
        }
    })
}

fn parse_entity_extraction(payload: &str, strict_entity_json: bool) -> Result<EntityExtraction> {
    let value: Value = match serde_json::from_str(payload) {
        Ok(value) => value,
        Err(_) => {
            if strict_entity_json {
                return Err(AgentError::Processing(format!(
                    "Invalid JSON payload: {}",
                    payload
                )));
            }
            if let Some(entities_json) = extract_json_array(payload, "\"entities\"") {
                let entities_value: Value = match serde_json::from_str(&entities_json) {
                    Ok(value) => value,
                    Err(_) => {
                        let cleaned = clean_json_array(&entities_json);
                        serde_json::from_str(&cleaned).map_err(|e| {
                            AgentError::Processing(format!(
                                "Invalid entities JSON: {} ({})",
                                entities_json, e
                            ))
                        })?
                    }
                };
                let entities = parse_entities_value(&entities_value);
                debug!("Recovered entities from malformed JSON payload");
                return Ok(EntityExtraction {
                    entities,
                    relationships: Vec::new(),
                });
            }
            if let Some(entities_json) = extract_json_array(payload, "entities") {
                let cleaned = clean_json_array(&entities_json);
                if let Ok(entities_value) = serde_json::from_str::<Value>(&cleaned) {
                    let entities = parse_entities_value(&entities_value);
                    debug!("Recovered entities from unquoted entities key");
                    return Ok(EntityExtraction {
                        entities,
                        relationships: Vec::new(),
                    });
                }
            }
            return Err(AgentError::Processing(format!(
                "Invalid JSON payload: {}",
                payload
            )));
        }
    };

    let entities = value
        .get("entities")
        .map(parse_entities_value)
        .unwrap_or_default();

    let relationships = value
        .get("relationships")
        .and_then(|v| v.as_array())
        .map(|items| {
            items
                .iter()
                .filter_map(|item| {
                    let source = item
                        .get("source")
                        .or_else(|| item.get("entity1"))
                        .or_else(|| item.get("from"))
                        .and_then(|v| match v {
                            Value::String(s) => Some(s.to_string()),
                            Value::Array(arr) => arr.first().and_then(value_to_string),
                            Value::Object(obj) => obj.get("name").and_then(value_to_string),
                            _ => None,
                        });
                    let target = item
                        .get("target")
                        .or_else(|| item.get("entity2"))
                        .or_else(|| item.get("to"))
                        .and_then(|v| match v {
                            Value::String(s) => Some(s.to_string()),
                            Value::Array(arr) => arr.first().and_then(value_to_string),
                            Value::Object(obj) => obj.get("name").and_then(value_to_string),
                            _ => None,
                        });
                    let relationship_type = item
                        .get("relationship_type")
                        .or_else(|| item.get("relation_type"))
                        .or_else(|| item.get("type"))
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string());

                    match (source, target, relationship_type) {
                        (Some(source), Some(target), Some(relationship_type)) => {
                            Some(ExtractedRelationship {
                                source,
                                target,
                                relationship_type,
                            })
                        }
                        _ => None,
                    }
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    Ok(EntityExtraction {
        entities,
        relationships,
    })
}

fn extract_json_array(payload: &str, key: &str) -> Option<String> {
    let key_pos = payload.find(key)?;
    let slice = &payload[key_pos..];
    let array_start_rel = slice.find('[')?;
    let array_start = key_pos + array_start_rel;

    let mut depth = 0usize;
    let mut in_string = false;
    let mut escape = false;
    let mut start_idx = None;

    for (offset, ch) in payload[array_start..].char_indices() {
        if in_string {
            if escape {
                escape = false;
            } else if ch == '\\' {
                escape = true;
            } else if ch == '"' {
                in_string = false;
            }
            continue;
        }

        match ch {
            '"' => in_string = true,
            '[' => {
                if depth == 0 {
                    start_idx = Some(array_start + offset);
                }
                depth += 1;
            }
            ']' => {
                if depth == 0 {
                    continue;
                }
                depth -= 1;
                if depth == 0 {
                    let start = start_idx?;
                    let end = array_start + offset;
                    return Some(payload[start..=end].to_string());
                }
            }
            _ => {}
        }
    }

    None
}

fn clean_json_array(payload: &str) -> String {
    payload
        .replace(",]", "]")
        .replace("\n", "")
        .replace("\r", "")
}

fn parse_entities_value(value: &Value) -> Vec<ExtractedEntity> {
    let items = match value {
        Value::Array(items) => items,
        other => {
            if let Some(items) = other.get("entities").and_then(|v| v.as_array()) {
                items
            } else {
                return Vec::new();
            }
        }
    };

    items
        .iter()
        .filter_map(|item| match item {
            Value::String(name) => Some(ExtractedEntity {
                name: name.to_string(),
                entity_type: None,
            }),
            Value::Object(obj) => {
                let name = obj
                    .get("name")
                    .or_else(|| obj.get("entity"))
                    .or_else(|| obj.get("value"))
                    .and_then(value_to_string)?;
                let entity_type = obj
                    .get("type")
                    .or_else(|| obj.get("entity_type"))
                    .or_else(|| obj.get("label"))
                    .or_else(|| obj.get("category"))
                    .and_then(value_to_string);
                Some(ExtractedEntity { name, entity_type })
            }
            _ => None,
        })
        .collect()
}

fn value_to_string(value: &Value) -> Option<String> {
    match value {
        Value::String(s) => Some(s.to_string()),
        Value::Number(n) => Some(n.to_string()),
        Value::Bool(b) => Some(b.to_string()),
        Value::Array(arr) => arr.first().and_then(value_to_string),
        Value::Object(obj) => obj
            .get("name")
            .or_else(|| obj.get("entity"))
            .or_else(|| obj.get("value"))
            .and_then(value_to_string),
        _ => None,
    }
}

fn extract_generated_text(value: Value) -> Result<String> {
    match value {
        Value::Array(mut items) => {
            let first = items
                .pop()
                .ok_or_else(|| AgentError::Processing("Empty TGI response array".to_string()))?;
            extract_generated_text(first)
        }
        Value::Object(mut obj) => {
            if let Some(Value::String(text)) = obj.remove("generated_text") {
                Ok(text)
            } else if let Some(Value::String(text)) = obj.remove("response") {
                Ok(text)
            } else {
                Err(AgentError::Processing(
                    "TGI response missing generated text field".to_string(),
                ))
            }
        }
        other => Err(AgentError::Processing(format!(
            "Unexpected TGI response format: {}",
            other
        ))),
    }
}
