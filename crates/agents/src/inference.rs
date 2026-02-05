//! Local inference clients for embeddings (TEI) and entity extraction (TGI).

use crate::{AgentError, Result};
use graphrag_db::schema::EMBEDDING_DIMENSION;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tracing::debug;
use std::time::Duration;

const DEFAULT_TEI_URL: &str = "http://localhost:8081";
const DEFAULT_TEI_PROVIDER: &str = "tei";
const DEFAULT_OLLAMA_EMBED_MODEL: &str = "nomic-embed-text:latest";
const DEFAULT_TGI_URL: &str = "http://localhost:8082";
const DEFAULT_TGI_PROVIDER: &str = "tgi";
const DEFAULT_OLLAMA_MODEL: &str = "phi4-mini:latest";
const DEFAULT_OLLAMA_FORMAT: &str = "json";
const DEFAULT_TEI_MAX_BATCH: usize = 32;
const DEFAULT_OLLAMA_TIMEOUT_SECS: u64 = 120;
const DEFAULT_OLLAMA_USE_CHAT_SCHEMA: bool = true;
const DEFAULT_STRICT_ENTITY_JSON: bool = true;
const DEFAULT_MAX_ENTITIES: usize = 30;
const DEFAULT_MAX_RELATIONSHIPS: usize = 15;

fn ollama_use_chat_schema() -> bool {
    std::env::var("TGI_OLLAMA_USE_CHAT_SCHEMA")
        .map(|value| {
            let value = value.trim().to_ascii_lowercase();
            matches!(value.as_str(), "1" | "true" | "yes" | "on")
        })
        .unwrap_or(DEFAULT_OLLAMA_USE_CHAT_SCHEMA)
}

fn strict_entity_json() -> bool {
    std::env::var("STRICT_ENTITY_JSON")
        .map(|value| {
            let value = value.trim().to_ascii_lowercase();
            matches!(value.as_str(), "1" | "true" | "yes" | "on")
        })
        .unwrap_or(DEFAULT_STRICT_ENTITY_JSON)
}

fn max_entities() -> usize {
    std::env::var("EXTRACT_MAX_ENTITIES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_MAX_ENTITIES)
}

fn max_relationships() -> usize {
    std::env::var("EXTRACT_MAX_RELATIONSHIPS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_MAX_RELATIONSHIPS)
}

fn env_or_default(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

#[derive(Clone)]
pub struct TeiClient {
    client: Client,
    base_url: String,
    provider: TeiProvider,
    model: String,
}

impl TeiClient {
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            client: Client::new(),
            base_url: base_url.into(),
            provider: TeiProvider::Tei,
            model: DEFAULT_OLLAMA_EMBED_MODEL.to_string(),
        }
    }

    pub fn default_local() -> Self {
        let provider = env_or_default("TEI_PROVIDER", DEFAULT_TEI_PROVIDER);
        if provider.eq_ignore_ascii_case("ollama") {
            let url = env_or_default("TEI_URL", "http://localhost:11434");
            let model = env_or_default("TEI_MODEL", DEFAULT_OLLAMA_EMBED_MODEL);
            Self {
                client: Client::new(),
                base_url: url,
                provider: TeiProvider::Ollama,
                model,
            }
        } else {
            let url = env_or_default("TEI_URL", DEFAULT_TEI_URL);
            Self::new(url)
        }
    }

    pub async fn embed(&self, text: &str, is_query: bool) -> Result<Vec<f32>> {
        if matches!(self.provider, TeiProvider::Ollama) {
            let embedding = self.ollama_embed(text).await?;
            validate_embedding_dim(embedding.len())?;
            return Ok(embedding);
        }

        let prompt_name = if is_query {
            std::env::var("TEI_PROMPT_NAME_QUERY").ok()
        } else {
            std::env::var("TEI_PROMPT_NAME_PASSAGE").ok()
        };

        let url = format!("{}/embed", self.base_url);
        let request = TeiEmbedRequest {
            inputs: text,
            truncate: true,
            prompt_name: prompt_name.as_deref(),
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
            std::env::var("TEI_PROMPT_NAME_QUERY").ok()
        } else {
            std::env::var("TEI_PROMPT_NAME_PASSAGE").ok()
        };

        let max_batch = std::env::var("TEI_MAX_BATCH")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(DEFAULT_TEI_MAX_BATCH);

        let url = format!("{}/embed", self.base_url);
        let mut results = Vec::with_capacity(texts.len());

        for chunk in texts.chunks(max_batch) {
            let request = TeiEmbedBatchRequest {
                inputs: chunk,
                truncate: true,
                prompt_name: prompt_name.as_deref(),
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
            if let Some(first) = embeddings.first() {
                validate_embedding_dim(first.len())?;
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

#[derive(Clone)]
pub struct TgiClient {
    client: Client,
    base_url: String,
    json_schema: Option<Value>,
    provider: TgiProvider,
    model: String,
}

impl TgiClient {
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            client: Client::new(),
            base_url: base_url.into(),
            json_schema: None,
            provider: TgiProvider::Tgi,
            model: DEFAULT_OLLAMA_MODEL.to_string(),
        }
    }

    pub fn default_local() -> Self {
        let provider = env_or_default("TGI_PROVIDER", DEFAULT_TGI_PROVIDER);
        if provider.eq_ignore_ascii_case("ollama") {
            let url = env_or_default("TGI_URL", "http://localhost:11434");
            let model = env_or_default("TGI_MODEL", DEFAULT_OLLAMA_MODEL);
            Self {
                client: Client::new(),
                base_url: url,
                json_schema: None,
                provider: TgiProvider::Ollama,
                model,
            }
        } else {
            let url = env_or_default("TGI_URL", DEFAULT_TGI_URL);
            Self::new(url)
        }
    }

    pub fn with_json_schema(mut self, schema: Value) -> Self {
        self.json_schema = Some(schema);
        self
    }

    pub async fn extract(&self, text: &str) -> Result<EntityExtraction> {
        let entity_cap = max_entities();
        let relationship_cap = max_relationships();
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
                let extraction = parse_entity_extraction(&cleaned).map_err(|e| {
                    AgentError::Processing(format!("TGI returned invalid JSON: {} ({})", generated, e))
                })?;
                Ok(extraction)
            }
            TgiProvider::Ollama => {
                let options = parse_ollama_options()?;
                let generated = self.ollama_generate(&prompt, options.clone()).await?;
                let cleaned = normalize_json_payload(&generated);
                if let Ok(extraction) = parse_entity_extraction(&cleaned) {
                    return Ok(extraction);
                }

                if strict_entity_json() {
                    return Err(AgentError::Processing(format!(
                        "TGI returned invalid JSON: {}",
                        generated
                    )));
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
                let generated_retry = self.ollama_generate(&retry_prompt, retry_options).await?;
                let cleaned_retry = normalize_json_payload(&generated_retry);
                let extraction = parse_entity_extraction(&cleaned_retry).map_err(|e| {
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

    async fn ollama_generate(&self, prompt: &str, options: Option<Value>) -> Result<String> {
        if ollama_use_chat_schema() {
            return self.ollama_chat_generate(prompt, options).await;
        }

        let url = format!("{}/api/generate", self.base_url);
        let format = match std::env::var("TGI_OLLAMA_FORMAT") {
            Ok(value) => {
                let trimmed = value.trim().to_string();
                if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("none") {
                    None
                } else {
                    Some(trimmed)
                }
            }
            Err(_) => Some(DEFAULT_OLLAMA_FORMAT.to_string()),
        };
        let timeout_secs = std::env::var("TGI_OLLAMA_TIMEOUT_SECS")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(DEFAULT_OLLAMA_TIMEOUT_SECS);
        let request = OllamaGenerateRequest {
            model: self.model.clone(),
            prompt: prompt.to_string(),
            stream: false,
            format,
            options,
        };

        let response = self
            .client
            .post(&url)
            .json(&request)
            .timeout(Duration::from_secs(timeout_secs))
            .send()
            .await?
            .error_for_status()?
            .json::<OllamaGenerateResponse>()
            .await?;

        Ok(response.response)
    }

    async fn ollama_chat_generate(&self, prompt: &str, options: Option<Value>) -> Result<String> {
        let url = format!("{}/api/chat", self.base_url);
        let timeout_secs = std::env::var("TGI_OLLAMA_TIMEOUT_SECS")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(DEFAULT_OLLAMA_TIMEOUT_SECS);
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
            format: Some(entity_extraction_schema()),
            options,
        };

        let response = self
            .client
            .post(&url)
            .json(&request)
            .timeout(Duration::from_secs(timeout_secs))
            .send()
            .await?
            .error_for_status()?
            .json::<OllamaChatResponse>()
            .await?;

        if let Some(done) = response.done {
            debug!("Ollama chat done={}", done);
        }

        if let Some(done_reason) = response.done_reason.as_deref() {
            debug!("Ollama chat done_reason={}", done_reason);
        }

        if let Some(total_ms) = response
            .total_duration
            .map(|ns| ns as f64 / 1_000_000.0)
        {
            debug!("Ollama chat total_duration_ms={:.2}", total_ms);
        }

        let content = response.message.content;
        let trimmed = content.trim_end();
        if !trimmed.ends_with('}') {
            debug!(
                "Ollama chat content does not end with '}}' (len={})",
                trimmed.len()
            );
        }

        Ok(content)
    }
}

#[derive(Clone, Copy)]
enum TgiProvider {
    Tgi,
    Ollama,
}

#[derive(Debug, Deserialize)]
pub struct EntityExtraction {
    pub entities: Vec<ExtractedEntity>,
    pub relationships: Vec<ExtractedRelationship>,
}

#[derive(Debug, Deserialize)]
pub struct ExtractedEntity {
    pub name: String,
    #[serde(alias = "type", alias = "entity_type")]
    pub entity_type: Option<String>,
}

#[derive(Debug, Deserialize)]
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

fn validate_embedding_dim(len: usize) -> Result<()> {
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
struct OllamaGenerateRequest {
    model: String,
    prompt: String,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    format: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    options: Option<Value>,
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
struct OllamaGenerateResponse {
    response: String,
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
                let first = items.into_iter().next().ok_or_else(|| {
                    AgentError::Processing("Missing embeddings".to_string())
                })?;
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

fn parse_ollama_options() -> Result<Option<Value>> {
    let raw = match std::env::var("TGI_OLLAMA_OPTIONS") {
        Ok(value) => value,
        Err(_) => return Ok(None),
    };

    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }

    let value: Value = serde_json::from_str(trimmed).map_err(|e| {
        AgentError::Processing(format!("Invalid TGI_OLLAMA_OPTIONS JSON: {}", e))
    })?;

    if !value.is_object() {
        return Err(AgentError::Processing(
            "TGI_OLLAMA_OPTIONS must be a JSON object".to_string(),
        ));
    }

    Ok(Some(value))
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

fn entity_extraction_schema() -> Value {
    let entity_cap = max_entities();
    let relationship_cap = max_relationships();
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

fn parse_entity_extraction(payload: &str) -> Result<EntityExtraction> {
    let value: Value = match serde_json::from_str(payload) {
        Ok(value) => value,
        Err(_) => {
            if strict_entity_json() {
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
                        (Some(source), Some(target), Some(relationship_type)) => Some(
                            ExtractedRelationship {
                                source,
                                target,
                                relationship_type,
                            },
                        ),
                        _ => None,
                    }
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    Ok(EntityExtraction { entities, relationships })
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
