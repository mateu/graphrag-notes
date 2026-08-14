use super::Migration;

/// Immutable baseline for all schema objects that existed before application
/// migration tracking was introduced.
pub(super) const MIGRATION: Migration = Migration {
    version: 1,
    name: "initial_schema",
    sql: crate::schema::BASELINE_SCHEMA,
};
