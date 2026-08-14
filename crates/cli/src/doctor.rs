//! Read-only local-stack diagnostics for `graphrag doctor`.

use graphrag_agents::{InferenceProviderConfig, InferenceProviders};
use graphrag_config::RuntimeConfig;
use graphrag_db::{
    compatibility::{
        check_embedding_compatibility, embedding_metadata, CompatibilityState, EmbeddingIdentity,
    },
    connect_persistent, init_memory, migrations, DbConnection,
};
use serde::Serialize;
use std::path::Path;

pub const DOCTOR_SCHEMA_VERSION: u32 = 1;
pub const EXIT_HEALTHY: i32 = 0;
pub const EXIT_WARNING: i32 = 1;
pub const EXIT_FAILED: i32 = 2;

const EXPECTED_TABLES: &[&str] = &[
    "note",
    "entity",
    "source",
    "conversation",
    "message",
    "mentions",
    "graphrag_metadata",
];
#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "lowercase")]
pub enum Status {
    Healthy,
    Warning,
    Failed,
}

#[derive(Debug, Serialize)]
pub struct Check {
    pub name: String,
    pub status: Status,
    pub summary: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct DoctorReport {
    pub schema_version: u32,
    pub status: Status,
    pub exit_code: i32,
    pub read_only: bool,
    pub checks: Vec<Check>,
}

impl DoctorReport {
    pub fn configuration_error(error: impl std::fmt::Display) -> Self {
        Self::from_checks(vec![Check::failed(
            "config",
            "Configuration is invalid",
            error.to_string(),
        )])
    }

    fn from_checks(checks: Vec<Check>) -> Self {
        let status = checks
            .iter()
            .map(|check| check.status)
            .max()
            .unwrap_or(Status::Healthy);
        let exit_code = match status {
            Status::Healthy => EXIT_HEALTHY,
            Status::Warning => EXIT_WARNING,
            Status::Failed => EXIT_FAILED,
        };
        Self {
            schema_version: DOCTOR_SCHEMA_VERSION,
            status,
            exit_code,
            read_only: true,
            checks,
        }
    }

    pub fn render_human(&self) -> String {
        let mut output = format!(
            "GraphRAG doctor: {:?} (exit {})\n",
            self.status, self.exit_code
        );
        for check in &self.checks {
            output.push_str(&format!(
                "[{:?}] {}: {}\n",
                check.status, check.name, check.summary
            ));
            if let Some(detail) = &check.detail {
                output.push_str(&format!("  {detail}\n"));
            }
        }
        output.push_str(
            "Read-only: no migrations, repairs, deletes, or reindexing were performed.\n",
        );
        output
    }
}

impl Check {
    fn healthy(name: impl Into<String>, summary: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            status: Status::Healthy,
            summary: summary.into(),
            detail: None,
        }
    }

    fn warning(
        name: impl Into<String>,
        summary: impl Into<String>,
        detail: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            status: Status::Warning,
            summary: summary.into(),
            detail: Some(detail.into()),
        }
    }

    fn failed(
        name: impl Into<String>,
        summary: impl Into<String>,
        detail: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            status: Status::Failed,
            summary: summary.into(),
            detail: Some(detail.into()),
        }
    }
}

pub async fn run(
    config: &RuntimeConfig,
    provider_config: &InferenceProviderConfig,
    memory: bool,
) -> DoctorReport {
    let mut checks = vec![Check::healthy(
        "config",
        format!(
            "Configuration valid; database path: {}",
            config.database.path.display()
        ),
    )];

    let mut diagnostic_db = None;
    if memory {
        checks.push(Check::warning(
            "database_path",
            "Using an ephemeral in-memory database",
            "No persistent RocksDB directory or lock can be inspected.",
        ));
        match init_memory().await {
            Ok(db) => {
                inspect_database(&mut checks, &db).await;
                diagnostic_db = Some(db);
            }
            Err(error) => checks.push(Check::failed(
                "database_open",
                "Failed to open in-memory database",
                error.to_string(),
            )),
        }
    } else {
        inspect_database_path(&mut checks, &config.database.path);
        if config.database.path.exists() {
            match connect_persistent(&config.database.path).await {
                Ok(db) => {
                    checks.push(Check::healthy(
                        "database_open",
                        "RocksDB opened without a lock conflict",
                    ));
                    inspect_database(&mut checks, &db).await;
                    diagnostic_db = Some(db);
                }
                Err(error) => checks.push(lock_check(&config.database.path, &error.to_string())),
            }
        }
    }

    inspect_providers(&mut checks, provider_config, diagnostic_db.as_ref()).await;
    DoctorReport::from_checks(checks)
}

fn inspect_database_path(checks: &mut Vec<Check>, path: &Path) {
    match std::fs::metadata(path) {
        Ok(metadata) if !metadata.is_dir() => checks.push(Check::failed(
            "database_path",
            "Configured database path is not a directory",
            path.display().to_string(),
        )),
        Ok(metadata) if metadata.permissions().readonly() => checks.push(Check::failed(
            "database_path",
            "Configured database directory is read-only",
            format!(
                "{}; make the directory writable before running GraphRAG",
                path.display()
            ),
        )),
        Ok(_) => checks.push(Check::healthy(
            "database_path",
            format!(
                "Database directory exists and is writable: {}",
                path.display()
            ),
        )),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => checks.push(Check::warning(
            "database_path",
            "Database directory does not exist yet",
            format!("{}; run an ingestion command to create it", path.display()),
        )),
        Err(error) => checks.push(Check::failed(
            "database_path",
            "Cannot inspect database directory",
            format!("{}: {error}", path.display()),
        )),
    }
}

fn lock_check(path: &Path, error: &str) -> Check {
    let is_lock = ["lock", "already in use", "resource temporarily unavailable"]
        .iter()
        .any(|needle| error.to_ascii_lowercase().contains(needle));
    if is_lock {
        Check::failed(
            "database_open",
            "RocksDB could not acquire the database lock",
            format!(
                "{} is likely open in another GraphRAG process; stop that process and retry",
                path.display()
            ),
        )
    } else {
        Check::failed(
            "database_open",
            "RocksDB could not be opened",
            format!("{}: {error}", path.display()),
        )
    }
}

async fn inspect_database(checks: &mut Vec<Check>, db: &DbConnection) {
    match migrations::current_version(db).await {
        Ok(version) if version == migrations::latest_version() => checks.push(Check::healthy(
            "application_schema",
            format!("Application schema version {version} is current"),
        )),
        Ok(version) => checks.push(Check::warning(
            "application_schema",
            format!("Application schema version {version} is behind this binary"),
            format!("Expected migration version {}; run a normal GraphRAG command to apply pending migrations", migrations::latest_version()),
        )),
        Err(error) => checks.push(Check::failed(
            "application_schema",
            "Application schema is incompatible or unreadable",
            error.to_string(),
        )),
    }

    let mut missing_tables = Vec::new();
    let mut missing_indexes = Vec::new();
    for table in EXPECTED_TABLES {
        if table_info(db, table).await.is_err() {
            missing_tables.push(*table);
        }
    }
    for (table, index) in [
        ("note", "idx_note_embedding"),
        ("note", "idx_note_content"),
        ("note", "idx_note_title"),
        ("message", "idx_message_embedding"),
        ("message", "idx_message_content"),
        ("conversation", "idx_conversation_summary_embedding"),
        ("conversation", "idx_conversation_summary"),
        ("conversation", "idx_conversation_title"),
    ] {
        match table_info(db, table).await {
            Ok(info)
                if info
                    .get("indexes")
                    .and_then(serde_json::Value::as_object)
                    .is_some_and(|indexes| indexes.contains_key(index)) => {}
            _ => missing_indexes.push(index),
        }
    }
    if missing_tables.is_empty() && missing_indexes.is_empty() {
        checks.push(Check::healthy(
            "schema_objects",
            "Required tables and vector/full-text indexes are present",
        ));
    } else {
        let mut missing = missing_tables;
        missing.extend(missing_indexes);
        checks.push(Check::failed(
            "schema_objects",
            "Required schema objects are missing",
            format!(
                "Missing: {}. Expected migration version {}",
                missing.join(", "),
                migrations::latest_version()
            ),
        ));
    }

    match embedding_metadata(db).await {
        Ok(None) => checks.push(Check::warning(
            "embedding_metadata",
            "No embedded-corpus metadata has been initialized",
            "Run an ingestion or vector search with a healthy embeddings provider to initialize it.",
        )),
        Ok(Some(metadata)) => checks.push(Check::healthy(
            "embedding_metadata",
            format!("{} / {} ({} dimensions, generation {})", metadata.embedding.provider, metadata.embedding.model, metadata.embedding.dimension, metadata.generation),
        )),
        Err(error) => checks.push(Check::failed("embedding_metadata", "Cannot read embedding metadata", error.to_string())),
    }

    match embedding_integrity_counts(db).await {
        Ok((missing_or_invalid, interrupted_migrations))
            if missing_or_invalid == 0 && interrupted_migrations == 0 =>
        {
            checks.push(Check::healthy(
                "embedding_integrity",
                "No missing or invalid embeddings and no interrupted migration jobs",
            ))
        }
        Ok((missing_or_invalid, interrupted_migrations)) => checks.push(Check::warning(
            "embedding_integrity",
            "Embedding or migration recovery work is needed",
            format!(
                "{missing_or_invalid} missing/invalid embeddings; {interrupted_migrations} interrupted migration attempts. Inspect before any repair or reindex."
            ),
        )),
        Err(error) => checks.push(Check::warning(
            "embedding_integrity",
            "Could not count missing embeddings or interrupted jobs",
            error.to_string(),
        )),
    }
}

async fn inspect_providers(
    checks: &mut Vec<Check>,
    config: &InferenceProviderConfig,
    db: Option<&DbConnection>,
) {
    let providers = InferenceProviders::from_config(config);
    let embedder = providers.embedder;
    let extractor = providers.extractor;
    let embedding_capabilities = embedder.capabilities();
    let extraction_capabilities = extractor.capabilities();

    match embedder.health().await {
        Ok(true) => match embedder
            .embed("graphrag doctor dimension probe", true)
            .await
        {
            Ok(vector) => {
                checks.push(Check::healthy(
                    "embedding_provider",
                    format!(
                        "{} / {} healthy ({} dimensions)",
                        embedding_capabilities.provider,
                        embedding_capabilities.model,
                        vector.len()
                    ),
                ));
                if let Some(db) = db {
                    let active = EmbeddingIdentity::new(
                        embedding_capabilities.provider,
                        embedding_capabilities.model,
                        vector.len(),
                    );
                    match check_embedding_compatibility(db, &active).await {
                        Ok(CompatibilityState::Matching(_)) => checks.push(Check::healthy(
                            "embedding_compatibility",
                            "Active embedding provider matches database metadata",
                        )),
                        Ok(CompatibilityState::Empty) => checks.push(Check::warning(
                            "embedding_compatibility",
                            "Database has no embedded-corpus metadata yet",
                            "Run an ingestion or vector-search command to initialize metadata.",
                        )),
                        Err(error) => checks.push(Check::failed(
                            "embedding_compatibility",
                            "Active embedding provider is incompatible with database metadata",
                            error.to_string(),
                        )),
                    }
                }
            }
            Err(error) => checks.push(Check::failed(
                "embedding_provider",
                "Embedding provider cannot produce a dimension probe",
                error.to_string(),
            )),
        },
        Ok(false) | Err(_) => checks.push(Check::warning(
            "embedding_provider",
            "Embedding provider is unavailable",
            format!(
                "{}; database-only commands remain usable",
                redact_endpoint(&embedding_capabilities.endpoint)
            ),
        )),
    }

    match extractor.health().await {
        Ok(true) => checks.push(Check::healthy(
            "extraction_provider",
            format!(
                "{} / {} healthy",
                extraction_capabilities.provider, extraction_capabilities.model
            ),
        )),
        Ok(false) | Err(_) => checks.push(Check::warning(
            "extraction_provider",
            "Extraction provider is unavailable",
            redact_endpoint(&extraction_capabilities.endpoint),
        )),
    }
}

async fn embedding_integrity_counts(db: &DbConnection) -> graphrag_db::Result<(usize, usize)> {
    let note_count = count_query(
        db,
        "SELECT count() AS count FROM note WHERE embedding IS NONE OR array::len(embedding) != 1024 GROUP ALL",
    )
    .await?;
    let message_count = count_query(
        db,
        "SELECT count() AS count FROM message WHERE embedding IS NONE OR array::len(embedding) != 1024 GROUP ALL",
    )
    .await?;
    let conversation_count = count_query(
        db,
        "SELECT count() AS count FROM conversation WHERE summary IS NOT NONE AND summary != '' AND (summary_embedding IS NONE OR array::len(summary_embedding) != 1024) GROUP ALL",
    )
    .await?;
    let interrupted_migrations = count_query(
        db,
        "SELECT count() AS count FROM schema_migration_attempt GROUP ALL",
    )
    .await?;
    Ok((
        note_count
            .saturating_add(message_count)
            .saturating_add(conversation_count),
        interrupted_migrations,
    ))
}

async fn count_query(db: &DbConnection, query: &str) -> graphrag_db::Result<usize> {
    let records: Vec<serde_json::Value> = db.query(query).await?.take(0)?;
    Ok(records
        .first()
        .and_then(|record| record.get("count"))
        .and_then(serde_json::Value::as_u64)
        .and_then(|count| usize::try_from(count).ok())
        .unwrap_or(0))
}

async fn table_info(
    db: &DbConnection,
    table: &str,
) -> graphrag_db::Result<serde_json::Map<String, serde_json::Value>> {
    let info: Option<serde_json::Value> =
        db.query(format!("INFO FOR TABLE {table}")).await?.take(0)?;
    info.and_then(|value| value.as_object().cloned())
        .ok_or_else(|| {
            graphrag_db::DbError::QueryFailed(format!(
                "INFO FOR TABLE {table} did not return a schema object"
            ))
        })
}

fn redact_endpoint(endpoint: &str) -> String {
    let without_credentials = endpoint
        .split_once("://")
        .map(|(scheme, remainder)| {
            let host = remainder
                .rsplit_once('@')
                .map(|(_, host)| host)
                .unwrap_or(remainder);
            format!("{scheme}://{host}")
        })
        .unwrap_or_else(|| endpoint.to_string());
    without_credentials
        .split_once('?')
        .map(|(value, _)| value.to_string())
        .unwrap_or(without_credentials)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_contract_is_stable_and_redacts_credentials() {
        let report = DoctorReport::from_checks(vec![Check::warning(
            "provider",
            "offline",
            redact_endpoint("https://user:secret@example.test/path?token=secret"),
        )]);
        let json = serde_json::to_value(report).unwrap();
        assert_eq!(json["schema_version"], DOCTOR_SCHEMA_VERSION);
        assert_eq!(json["exit_code"], EXIT_WARNING);
        assert!(!json.to_string().contains("secret"));
        assert_eq!(
            redact_endpoint("https://user:secret@example.test/path?token=secret"),
            "https://example.test/path"
        );
    }

    #[test]
    fn lock_failures_name_the_database_and_competing_process() {
        let check = lock_check(
            Path::new("/tmp/graphrag-test"),
            "IO error: lock held by another process",
        );
        assert_eq!(check.status, Status::Failed);
        assert!(check.detail.unwrap().contains("/tmp/graphrag-test"));
    }

    #[test]
    fn exit_codes_distinguish_healthy_warning_and_failed_reports() {
        assert_eq!(
            DoctorReport::from_checks(vec![Check::healthy("ok", "ok")]).exit_code,
            EXIT_HEALTHY
        );
        assert_eq!(
            DoctorReport::from_checks(vec![Check::warning("warn", "warn", "detail")]).exit_code,
            EXIT_WARNING
        );
        assert_eq!(
            DoctorReport::from_checks(vec![Check::failed("failed", "failed", "detail")]).exit_code,
            EXIT_FAILED
        );
    }

    #[tokio::test]
    async fn initialized_database_reports_required_tables_and_indexes() {
        let db = init_memory().await.unwrap();
        let mut checks = Vec::new();
        inspect_database(&mut checks, &db).await;
        assert_eq!(
            checks
                .iter()
                .find(|check| check.name == "schema_objects")
                .expect("schema object check")
                .status,
            Status::Healthy
        );
    }

    #[tokio::test]
    async fn missing_search_index_is_reported_as_a_failed_schema_check() {
        let db = init_memory().await.unwrap();
        db.query("REMOVE INDEX idx_message_content ON message")
            .await
            .unwrap()
            .check()
            .unwrap();

        let mut checks = Vec::new();
        inspect_database(&mut checks, &db).await;
        let schema = checks
            .iter()
            .find(|check| check.name == "schema_objects")
            .expect("schema object check");
        assert_eq!(schema.status, Status::Failed);
        assert!(schema
            .detail
            .as_deref()
            .is_some_and(|detail| detail.contains("idx_message_content")));
    }

    #[tokio::test]
    async fn empty_conversation_summaries_are_not_missing_embeddings() {
        let db = init_memory().await.unwrap();
        db.query(
            "CREATE conversation CONTENT {
                uuid: 'empty-summary',
                summary: '',
                created_at: time::now(),
                updated_at: time::now()
            }",
        )
        .await
        .unwrap()
        .check()
        .unwrap();
        let (missing_embeddings, _) = embedding_integrity_counts(&db).await.unwrap();
        assert_eq!(missing_embeddings, 0);
    }
}
