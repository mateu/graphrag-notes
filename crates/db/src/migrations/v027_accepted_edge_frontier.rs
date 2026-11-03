use super::Migration;

// Separate endpoint indexes also support the planner's union for bidirectional
// traversal. Creating indexes over existing relations backfills their entries;
// no accepted edges or endpoint IDs are rewritten.
pub(super) const MIGRATION: Migration = Migration {
    version: 27,
    name: "accepted_edge_frontier",
    sql: r#"
DEFINE INDEX IF NOT EXISTS idx_supports_in ON supports FIELDS in;
DEFINE INDEX IF NOT EXISTS idx_supports_out ON supports FIELDS out;
DEFINE INDEX IF NOT EXISTS idx_contradicts_in ON contradicts FIELDS in;
DEFINE INDEX IF NOT EXISTS idx_contradicts_out ON contradicts FIELDS out;
DEFINE INDEX IF NOT EXISTS idx_related_to_in ON related_to FIELDS in;
DEFINE INDEX IF NOT EXISTS idx_related_to_out ON related_to FIELDS out;
DEFINE INDEX IF NOT EXISTS idx_derived_from_in ON derived_from FIELDS in;
DEFINE INDEX IF NOT EXISTS idx_derived_from_out ON derived_from FIELDS out;
"#,
};
