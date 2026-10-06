use super::Migration;

/// Schema-version capability fence for durable pre-promotion graph checkpoints.
/// Older workers clear unknown-phase results during failure settlement, so they
/// must reject this schema before claiming a restored policy-migration job.
/// Ordinary upload admissions remain unchanged and carry no contract marker.
pub(super) const MIGRATION: Migration = Migration {
    version: 20,
    name: "source_policy_migration",
    sql: r#"
DEFINE FIELD IF NOT EXISTS remote_migration_contract_version ON processing_job TYPE option<int> ASSERT $value IS NONE OR $value = 1;
"#,
};
