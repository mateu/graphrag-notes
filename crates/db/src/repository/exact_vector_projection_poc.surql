-- Unadopted provider-free projection experiment. No production migration.
-- Original embeddings/HNSW retained. Synchronous events, typed mirror keys.

DEFINE TABLE exact_note_vector_probe SCHEMAFULL;

DEFINE FIELD record_id ON exact_note_vector_probe TYPE record<note>;

DEFINE FIELD embedding_cbor ON exact_note_vector_probe TYPE option<bytes>;

DEFINE FIELD embedding_uses_primary ON exact_note_vector_probe TYPE bool;

DEFINE FIELD embedding ON exact_note_vector_probe TYPE option<array<float>> COMPUTED
    IF embedding_uses_primary THEN record_id.embedding
    ELSE IF embedding_cbor IS NONE THEN NONE ELSE encoding::cbor::decode(embedding_cbor) END;

DEFINE FIELD created_at ON exact_note_vector_probe TYPE datetime;

DEFINE FIELD source_id ON exact_note_vector_probe TYPE option<record<source>>;

DEFINE FIELD source_generation ON exact_note_vector_probe TYPE option<int>;

DEFINE FUNCTION fn::materialize_exact_note_vector_probe($row: object) {
    LET $embedding = $row.embedding;
    LET $finite = array::len($embedding ?? []) = 0 OR
        (math::min($embedding) > math::NEG_INFINITY AND math::max($embedding) < math::INFINITY);
    UPSERT type::record('exact_note_vector_probe', record::id($row.id)) CONTENT {
        record_id: $row.id,
        embedding_cbor: IF $embedding IS NONE OR $finite = false THEN NONE ELSE encoding::cbor::encode($embedding) END,
        embedding_uses_primary: $finite = false,
        created_at: $row.created_at,
        source_id: $row.source_id,
        source_generation: $row.source_generation
    };
    RETURN NONE;
};
DEFINE EVENT materialize_exact_note_vector_probe ON note WHEN $event != 'DELETE' THEN fn::materialize_exact_note_vector_probe($after);
DEFINE EVENT delete_exact_note_vector_probe ON note WHEN $event = 'DELETE' THEN (DELETE type::record('exact_note_vector_probe', record::id($before.id)));
FOR $id IN ((SELECT VALUE id FROM note ORDER BY id ASC) ?? []) {
    fn::materialize_exact_note_vector_probe($id.*);
};

DEFINE TABLE exact_message_vector_probe SCHEMAFULL;

DEFINE FIELD record_id ON exact_message_vector_probe TYPE record<message>;

DEFINE FIELD embedding_cbor ON exact_message_vector_probe TYPE option<bytes>;

DEFINE FIELD embedding_uses_primary ON exact_message_vector_probe TYPE bool;

DEFINE FIELD embedding ON exact_message_vector_probe TYPE option<array<float>> COMPUTED
    IF embedding_uses_primary THEN record_id.embedding
    ELSE IF embedding_cbor IS NONE THEN NONE ELSE encoding::cbor::decode(embedding_cbor) END;

DEFINE FIELD created_at ON exact_message_vector_probe TYPE option<datetime>;

DEFINE FIELD conversation_id ON exact_message_vector_probe TYPE record<conversation>;

DEFINE FUNCTION fn::materialize_exact_message_vector_probe($row: object) {
    LET $embedding = $row.embedding;
    LET $finite = array::len($embedding ?? []) = 0 OR
        (math::min($embedding) > math::NEG_INFINITY AND math::max($embedding) < math::INFINITY);
    UPSERT type::record('exact_message_vector_probe', record::id($row.id)) CONTENT {
        record_id: $row.id,
        embedding_cbor: IF $embedding IS NONE OR $finite = false THEN NONE ELSE encoding::cbor::encode($embedding) END,
        embedding_uses_primary: $finite = false,
        created_at: $row.created_at,
        conversation_id: $row.conversation_id
    };
    RETURN NONE;
};
DEFINE EVENT materialize_exact_message_vector_probe ON message WHEN $event != 'DELETE' THEN fn::materialize_exact_message_vector_probe($after);
DEFINE EVENT delete_exact_message_vector_probe ON message WHEN $event = 'DELETE' THEN (DELETE type::record('exact_message_vector_probe', record::id($before.id)));
FOR $id IN ((SELECT VALUE id FROM message ORDER BY id ASC) ?? []) {
    fn::materialize_exact_message_vector_probe($id.*);
};

DEFINE TABLE exact_conversation_vector_probe SCHEMAFULL;

DEFINE FIELD record_id ON exact_conversation_vector_probe TYPE record<conversation>;

DEFINE FIELD embedding_cbor ON exact_conversation_vector_probe TYPE option<bytes>;

DEFINE FIELD embedding_uses_primary ON exact_conversation_vector_probe TYPE bool;

DEFINE FIELD embedding ON exact_conversation_vector_probe TYPE option<array<float>> COMPUTED
    IF embedding_uses_primary THEN record_id.summary_embedding
    ELSE IF embedding_cbor IS NONE THEN NONE ELSE encoding::cbor::decode(embedding_cbor) END;

DEFINE FIELD updated_at ON exact_conversation_vector_probe TYPE datetime;

DEFINE FIELD source_uri ON exact_conversation_vector_probe TYPE option<string>;

DEFINE FUNCTION fn::materialize_exact_conversation_vector_probe($row: object) {
    LET $embedding = $row.summary_embedding;
    LET $finite = array::len($embedding ?? []) = 0 OR
        (math::min($embedding) > math::NEG_INFINITY AND math::max($embedding) < math::INFINITY);
    UPSERT type::record('exact_conversation_vector_probe', record::id($row.id)) CONTENT {
        record_id: $row.id,
        embedding_cbor: IF $embedding IS NONE OR $finite = false THEN NONE ELSE encoding::cbor::encode($embedding) END,
        embedding_uses_primary: $finite = false,
        updated_at: $row.updated_at,
        source_uri: $row.source_uri
    };
    RETURN NONE;
};
DEFINE EVENT materialize_exact_conversation_vector_probe ON conversation WHEN $event != 'DELETE' THEN fn::materialize_exact_conversation_vector_probe($after);
DEFINE EVENT delete_exact_conversation_vector_probe ON conversation WHEN $event = 'DELETE' THEN (DELETE type::record('exact_conversation_vector_probe', record::id($before.id)));
FOR $id IN ((SELECT VALUE id FROM conversation ORDER BY id ASC) ?? []) {
    fn::materialize_exact_conversation_vector_probe($id.*);
};
