-- Run only with issue166_benchmark BACKUP_COPY --sql FILE.
-- These statements execute the real index plans on the copied corpus.
SELECT VALUE entity FROM entity_lexeme WHERE lexeme = 'openclaw' EXPLAIN FULL;
SELECT VALUE entity FROM entity_lexeme WHERE lexeme IN ['how', 'does', 'memory', 'work', 'how does', 'does memory', 'memory work', 'how does memory', 'does memory work', 'how does memory work'] EXPLAIN FULL;
SELECT VALUE entity FROM entity_lexeme WHERE lexeme >= 'opencl' AND lexeme < 'opencl􏿿' EXPLAIN FULL;
LET $entity = (SELECT VALUE entity FROM entity_lexeme WHERE lexeme = 'openclaw' LIMIT 1)[0];
SELECT id, in, out, metadata FROM mentions WHERE out = $entity EXPLAIN FULL;
SELECT mention, entity, note, lexeme FROM mention_lexeme WHERE lexeme = 'openclaw' EXPLAIN FULL;
SELECT mention, entity, note, lexeme FROM mention_lexeme WHERE entity = $entity EXPLAIN FULL;
