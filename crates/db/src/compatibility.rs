//! Persisted embedding compatibility metadata.
//!
//! Vector indexes have a fixed dimension.  Before any vector query or write,
//! callers compare their active embedder identity with the corpus identity
//! recorded here.  A model change is deliberately not treated as compatible
//! merely because its vectors have the same length.

use crate::{migrations, DbConnection, DbError, Result};
use serde::{Deserialize, Serialize};
use surrealdb_types::SurrealValue;

const ACTIVE_EMBEDDING_KEY: &str = "active_embedding";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbeddingIdentity {
    pub provider: String,
    pub model: String,
    pub dimension: usize,
}

impl EmbeddingIdentity {
    pub fn new(provider: impl Into<String>, model: impl Into<String>, dimension: usize) -> Self {
        Self {
            provider: provider.into(),
            model: model.into(),
            dimension,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExtractionIdentity {
    pub provider: String,
    pub model: String,
}

impl ExtractionIdentity {
    pub fn new(provider: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            provider: provider.into(),
            model: model.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbeddingMetadata {
    pub application_schema_version: u32,
    pub embedding: EmbeddingIdentity,
    pub extraction: Option<ExtractionIdentity>,
    pub generation: u32,
    pub last_reindex_status: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompatibilityState {
    Empty,
    Matching(EmbeddingMetadata),
}

#[derive(Debug, Deserialize, SurrealValue)]
struct MetadataRecord {
    application_schema_version: i64,
    embedding_provider: String,
    embedding_model: String,
    embedding_dimension: i64,
    extraction_provider: Option<String>,
    extraction_model: Option<String>,
    generation: i64,
    last_reindex_status: Option<String>,
}

/// Return the active corpus metadata, if a successful embedding operation has
/// initialized it.
pub async fn embedding_metadata(db: &DbConnection) -> Result<Option<EmbeddingMetadata>> {
    let records: Vec<MetadataRecord> = db
        .query(
            "SELECT application_schema_version, embedding_provider, embedding_model, \
             embedding_dimension, extraction_provider, extraction_model, generation, \
             last_reindex_status FROM graphrag_metadata WHERE key = $key LIMIT 1",
        )
        .bind(("key", ACTIVE_EMBEDDING_KEY))
        .await?
        .take(0)?;

    records
        .into_iter()
        .next()
        .map(metadata_from_record)
        .transpose()
}

/// Reject incompatible vector work without mutating the database.  The empty
/// state is safe: [`record_embedding_metadata`] establishes it only after an
/// embedding has been successfully produced by the active provider.
pub async fn check_embedding_compatibility(
    db: &DbConnection,
    active: &EmbeddingIdentity,
) -> Result<CompatibilityState> {
    let Some(metadata) = embedding_metadata(db).await? else {
        return Ok(CompatibilityState::Empty);
    };

    let stored = &metadata.embedding;
    if stored.dimension != active.dimension
        || !stored.provider.eq_ignore_ascii_case(&active.provider)
        || stored.model != active.model
    {
        return Err(DbError::EmbeddingCompatibility {
            stored_provider: stored.provider.clone(),
            stored_model: stored.model.clone(),
            stored_dimension: stored.dimension,
            active_provider: active.provider.clone(),
            active_model: active.model.clone(),
            active_dimension: active.dimension,
        });
    }

    Ok(CompatibilityState::Matching(metadata))
}

/// Persist the identity for an empty corpus after a successful embedding
/// probe.  Existing metadata is first checked so concurrent callers cannot
/// overwrite a different model's corpus identity.
pub async fn record_embedding_metadata(
    db: &DbConnection,
    embedding: &EmbeddingIdentity,
    extraction: Option<&ExtractionIdentity>,
) -> Result<CompatibilityState> {
    match check_embedding_compatibility(db, embedding).await? {
        CompatibilityState::Matching(metadata) => {
            if let Some(extraction) = extraction {
                if metadata.extraction.as_ref() != Some(extraction) {
                    db.query(
                        "UPDATE graphrag_metadata SET extraction_provider = $provider, \
                         extraction_model = $model, updated_at = time::now() WHERE key = $key",
                    )
                    .bind(("key", ACTIVE_EMBEDDING_KEY))
                    .bind(("provider", extraction.provider.clone()))
                    .bind(("model", extraction.model.clone()))
                    .await?
                    .check()?;
                    let metadata = embedding_metadata(db).await?.ok_or_else(|| {
                        DbError::QueryFailed(
                            "embedding metadata disappeared while updating provenance".into(),
                        )
                    })?;
                    return Ok(CompatibilityState::Matching(metadata));
                }
            }
            return Ok(CompatibilityState::Matching(metadata));
        }
        CompatibilityState::Empty => {}
    }

    let legacy_vector_records = vector_bearing_record_count(db).await?;
    if legacy_vector_records != 0 {
        return Err(DbError::LegacyEmbeddingMetadata {
            vector_records: legacy_vector_records,
        });
    }

    let insertion = db
        .query(
            "INSERT INTO graphrag_metadata (
                key, application_schema_version, embedding_provider, embedding_model,
                embedding_dimension, extraction_provider, extraction_model, generation,
                last_reindex_at, last_reindex_status, updated_at
            ) VALUES (
                $key, $schema_version, $embedding_provider, $embedding_model,
                $embedding_dimension, $extraction_provider, $extraction_model, 1,
                time::now(), 'initialized', time::now()
            )",
        )
        .bind(("key", ACTIVE_EMBEDDING_KEY))
        .bind(("schema_version", i64::from(migrations::latest_version())))
        .bind(("embedding_provider", embedding.provider.clone()))
        .bind(("embedding_model", embedding.model.clone()))
        .bind((
            "embedding_dimension",
            i64::try_from(embedding.dimension).map_err(|_| {
                DbError::QueryFailed("embedding dimension exceeds database integer range".into())
            })?,
        ))
        .bind((
            "extraction_provider",
            extraction.map(|identity| identity.provider.clone()),
        ))
        .bind((
            "extraction_model",
            extraction.map(|identity| identity.model.clone()),
        ))
        .await?;
    if let Err(error) = insertion.check() {
        // The unique key can only race with another initializer. Re-read and
        // validate in that case; never turn an incompatible second identity
        // into an overwrite of the first writer.
        return match check_embedding_compatibility(db, embedding).await {
            Ok(state) => Ok(state),
            Err(_) => Err(DbError::Surreal(error)),
        };
    }

    check_embedding_compatibility(db, embedding).await
}

fn metadata_from_record(record: MetadataRecord) -> Result<EmbeddingMetadata> {
    Ok(EmbeddingMetadata {
        application_schema_version: u32::try_from(record.application_schema_version).map_err(
            |_| DbError::QueryFailed("stored application schema version is invalid".into()),
        )?,
        embedding: EmbeddingIdentity::new(
            record.embedding_provider,
            record.embedding_model,
            usize::try_from(record.embedding_dimension).map_err(|_| {
                DbError::QueryFailed("stored embedding dimension is invalid".into())
            })?,
        ),
        extraction: match (record.extraction_provider, record.extraction_model) {
            (Some(provider), Some(model)) => Some(ExtractionIdentity::new(provider, model)),
            _ => None,
        },
        generation: u32::try_from(record.generation)
            .map_err(|_| DbError::QueryFailed("stored generation is invalid".into()))?,
        last_reindex_status: record.last_reindex_status,
    })
}

/// Count every persisted vector-bearing record, including records from
/// databases created before compatibility metadata existed.
pub async fn vector_bearing_record_count(db: &DbConnection) -> Result<usize> {
    let mut total = 0_usize;
    for query in [
        "SELECT count() AS count FROM note WHERE embedding IS NOT NONE AND array::len(embedding) > 0 GROUP ALL",
        "SELECT count() AS count FROM entity WHERE embedding IS NOT NONE AND array::len(embedding) > 0 GROUP ALL",
        "SELECT count() AS count FROM message WHERE embedding IS NOT NONE AND array::len(embedding) > 0 GROUP ALL",
        "SELECT count() AS count FROM conversation WHERE summary_embedding IS NOT NONE AND array::len(summary_embedding) > 0 GROUP ALL",
    ] {
        let rows: Vec<serde_json::Value> = db.query(query).await?.take(0)?;
        let count = rows
            .first()
            .and_then(|row| row.get("count"))
            .and_then(serde_json::Value::as_u64)
            .and_then(|count| usize::try_from(count).ok())
            .unwrap_or(0);
        total = total.saturating_add(count);
    }
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::init_memory;

    fn identity(model: &str, dimension: usize) -> EmbeddingIdentity {
        EmbeddingIdentity::new("deterministic-test", model, dimension)
    }

    #[tokio::test]
    async fn empty_database_is_initialized_after_a_successful_probe() {
        let db = init_memory().await.unwrap();
        assert_eq!(
            check_embedding_compatibility(&db, &identity("fixture", 1024))
                .await
                .unwrap(),
            CompatibilityState::Empty
        );

        record_embedding_metadata(
            &db,
            &identity("fixture", 1024),
            Some(&ExtractionIdentity::new("fixture-test", "fixture")),
        )
        .await
        .unwrap();

        let metadata = embedding_metadata(&db).await.unwrap().unwrap();
        assert_eq!(
            metadata.application_schema_version,
            migrations::latest_version()
        );
        assert_eq!(metadata.embedding, identity("fixture", 1024));
    }

    #[tokio::test]
    async fn rejects_dimension_and_same_dimension_model_changes() {
        let db = init_memory().await.unwrap();
        record_embedding_metadata(&db, &identity("fixture", 1024), None)
            .await
            .unwrap();

        let dimension_error = check_embedding_compatibility(&db, &identity("fixture", 768))
            .await
            .unwrap_err();
        assert!(dimension_error.to_string().contains("1024 dimensions"));

        let model_error = check_embedding_compatibility(&db, &identity("new-fixture", 1024))
            .await
            .unwrap_err();
        assert!(model_error.to_string().contains("graphrag reindex --all"));
    }

    #[tokio::test]
    async fn refuses_to_adopt_legacy_vectors_without_metadata() {
        let db = init_memory().await.unwrap();
        db.query(
            "CREATE entity CONTENT {
                entity_type: 'other',
                name: 'legacy',
                canonical_name: 'legacy',
                embedding: [1.0]
            }",
        )
        .await
        .unwrap()
        .check()
        .unwrap();

        let error = record_embedding_metadata(&db, &identity("fixture", 1024), None)
            .await
            .unwrap_err();
        assert!(matches!(
            error,
            DbError::LegacyEmbeddingMetadata { vector_records: 1 }
        ));
        assert!(embedding_metadata(&db).await.unwrap().is_none());
    }
}
