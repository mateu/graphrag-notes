//! Local inference clients for embeddings (TEI) and entity extraction (TGI).

use crate::{AgentError, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;

const DEFAULT_TEI_URL: &str = "http://localhost:8081";
const DEFAULT_TGI_URL: &str = "http://localhost:8082";
const DEFAULT_TGI_PROVIDER: &str = "tgi";
const DEFAULT_OLLAMA_MODEL: &str = "phi4-mini:latest";
const DEFAULT_TEI_MAX_BATCH: usize = 32;

fn env_or_default(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

#[derive(Clone)]
pub struct TeiClient {
    client: Client,
    base_url: String,
}

impl TeiClient {
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            client: Client::new(),
            base_url: base_url.into(),
        }
    }

    pub fn default_local() -> Self {
        let url = env_or_default("TEI_URL", DEFAULT_TEI_URL);
        Self::new(url)
    }

    pub async fn embed(&self, text: &str, is_query: bool) -> Result<Vec<f32>> {
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

        parse_embedding_response(response)
    }

    pub async fn embed_batch(&self, texts: &[String], is_query: bool) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
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
            results.extend(embeddings);
        }

        Ok(results)
    }

    pub async fn health(&self) -> Result<bool> {
        let url = format!("{}/health", self.base_url);
        let response = self.client.get(&url).send().await?;
        Ok(response.status().is_success())
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
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
        let prompt = format!(
            "Extract all unique entities (people, organizations, concepts) and their relationships (supports, contradicts, mentions) from the text below. Return ONLY valid JSON matching this structure: {{ \"entities\": [...], \"relationships\": [...] }}.\n\nText:\n{}",
            text
        );
        let generated = match self.provider {
            TgiProvider::Tgi => {
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

                extract_generated_text(response)?
            }
            TgiProvider::Ollama => {
                let url = format!("{}/api/generate", self.base_url);
                let request = OllamaGenerateRequest {
                    model: self.model.clone(),
                    prompt,
                    stream: false,
                };

                let response = self
                    .client
                    .post(&url)
                    .json(&request)
                    .send()
                    .await?
                    .error_for_status()?
                    .json::<OllamaGenerateResponse>()
                    .await?;

                response.response
            }
        };
        let extraction: EntityExtraction = serde_json::from_str(&generated).map_err(|e| {
            AgentError::Processing(format!("TGI returned invalid JSON: {} ({})", generated, e))
        })?;

        Ok(extraction)
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
}

#[derive(Deserialize)]
struct OllamaGenerateResponse {
    response: String,
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
