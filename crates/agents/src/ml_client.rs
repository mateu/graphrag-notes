//! HTTP client for communicating with Python ML worker

use crate::Result;
use serde::{Deserialize, Serialize};
use tracing::{debug, instrument};

/// Client for the Python ML worker service
#[derive(Clone)]
pub struct MlClient {
    client: reqwest::Client,
    base_url: String,
    ollama_url: Option<String>,
    ollama_model: String,
}

impl MlClient {
    /// Create a new ML client
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: base_url.into(),
            ollama_url: None,
            ollama_model: "phi4-mini:latest".to_string(),
        }
    }
    
    /// Configure Ollama
    pub fn with_ollama(mut self, url: impl Into<String>, model: impl Into<String>) -> Self {
        self.ollama_url = Some(url.into());
        self.ollama_model = model.into();
        self
    }
    
    /// Default client pointing to localhost
    pub fn default_local() -> Self {
        Self::new("http://localhost:8100")
    }
    
    /// Generate embeddings for texts
    #[instrument(skip(self, texts))]
    pub async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        let url = format!("{}/embed", self.base_url);
        
        let request = EmbedRequest { texts };
        
        debug!("Requesting embeddings for {} texts", request.texts.len());
        
        let response: EmbedResponse = self.client
            .post(&url)
            .json(&request)
            .send()
            .await?
            .json()
            .await?;
        
        debug!("Received {} embeddings", response.embeddings.len());
        
        Ok(response.embeddings)
    }
    
    /// Generate embedding for a single text
    pub async fn embed_one(&self, text: impl Into<String>) -> Result<Vec<f32>> {
        let embeddings = self.embed(vec![text.into()]).await?;
        embeddings.into_iter().next().ok_or_else(|| {
            crate::AgentError::MlWorker("No embedding returned".into())
        })
    }
    
    /// Extract entities from text
    #[instrument(skip(self, text))]
    pub async fn extract_entities(&self, text: impl Into<String>) -> Result<Vec<ExtractedEntity>> {
        let url = format!("{}/extract-entities", self.base_url);
        
        let request = ExtractEntitiesRequest { text: text.into() };
        
        let response: ExtractEntitiesResponse = self.client
            .post(&url)
            .json(&request)
            .send()
            .await?
            .json()
            .await?;
        
        Ok(response.entities)
    }
    
    /// Health check
    pub async fn health(&self) -> Result<bool> {
        let url = format!("{}/health", self.base_url);
        
        let response = self.client.get(&url).send().await?;
        
        Ok(response.status().is_success())
    }

    /// Generate a title for the content using Ollama
    #[instrument(skip(self, content))]
    pub async fn generate_title(&self, content: &str) -> Result<Option<String>> {
        let ollama_url = match &self.ollama_url {
            Some(url) => url,
            None => return Ok(None),
        };

        let url = format!("{}/api/generate", ollama_url);
        
        let prompt = format!(
            "Generate a concise, descriptive title (maximum 10 words) for the following note. Do not use quotes or prefixes like 'Title:'. Just the title.\n\nNote content:\n{}", 
            content.chars().take(2000).collect::<String>()
        );

        let request = OllamaGenerateRequest {
            model: self.ollama_model.clone(),
            prompt,
            stream: false,
        };

        debug!("Requesting title generation from Ollama ({})", self.ollama_model);

        match self.client.post(&url).json(&request).send().await {
            Ok(res) => {
                if res.status().is_success() {
                    let body: OllamaGenerateResponse = res.json().await?;
                    let title = body.response.trim().to_string();
                    debug!("Generated title: {}", title);
                    Ok(Some(title))
                } else {
                    debug!("Ollama returned error status: {}", res.status());
                    Ok(None)
                }
            }
            Err(e) => {
                debug!("Failed to contact Ollama: {}", e);
                Ok(None)
            }
        }
    }
}

// ==========================================
// REQUEST/RESPONSE TYPES
// ==========================================

#[derive(Debug, Serialize)]
struct OllamaGenerateRequest {
    model: String,
    prompt: String,
    stream: bool,
}

#[derive(Debug, Deserialize)]
struct OllamaGenerateResponse {
    response: String,
}

#[derive(Debug, Serialize)]
struct EmbedRequest {
    texts: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct EmbedResponse {
    embeddings: Vec<Vec<f32>>,
}

#[derive(Debug, Serialize)]
struct ExtractEntitiesRequest {
    text: String,
}

#[derive(Debug, Deserialize)]
struct ExtractEntitiesResponse {
    entities: Vec<ExtractedEntity>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedEntity {
    pub name: String,
    pub entity_type: String,
    pub start: usize,
    pub end: usize,
    pub confidence: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_client_creation() {
        let client = MlClient::new("http://localhost:8100");
        assert_eq!(client.base_url, "http://localhost:8100");
        assert!(client.ollama_url.is_none());
    }
}
