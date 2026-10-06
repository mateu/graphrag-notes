import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash, randomUUID } from 'node:crypto';
import { performance } from 'node:perf_hooks';
import { runGatewayProfile, summarizeProfile, joinDiagnostics, parseResultMetadata } from './profile-gateway.mjs';

const sha = value => createHash('sha256').update(value).digest('hex');
const id = 'note:' + 'a'.repeat(64), actor = 'fictional.principal_name-';
const cases = ['keyword', 'hybrid', 'graph_title', 'graph_body'].map(route => ({ route, query: 'private fictional Atlas', expected_first_id: id, expected_actor: actor, expected_ids: [id] }));
const pin = { record_id: id, revision: 'b'.repeat(64), record_sha256: sha('private fictional content') };
function fixture(fault = null) {
  const diagnostics = [], sent = [], archived = []; let listener, pinnedCalls = 0, metadataCalls = 0;
  const metadata = { config_sha256: sha('private config'), default_model_sha256: sha('private model'), plugin_source_sha256: sha('source'), plugin_manifest_sha256: sha('manifest'),
    connection_policy: 'reuse', openclaw_version: '2026.9.8', mcp_sdk_version: '1.30.0', node_version: '22.16.0' };
  const adapter = {
    metadata: async () => ({ ...metadata, private_extra: 'private_token', ...(fault === 'metadata' && metadataCalls++ ? { config_sha256: sha('changed') } : {}) }),
    readPinned: async () => ({ ...pin, principal_verified: true, ...(fault === 'pinned' && pinnedCalls++ ? { revision: 'c'.repeat(64) } : {}) }),
    subscribe: fn => { listener = fn; return () => { listener = null; }; },
    request: async (method, params) => {
      if (method === 'commands.list') return { commands: [{ name: 'notes', source: 'plugin', acceptsArgs: true }] };
      assert.equal(method, 'chat.send'); sent.push(params);
      const search = !params.message.includes('--help') && !params.message.includes('invalid');
      const mode = params.message.includes('--keyword') ? 'keyword' : 'hybrid', graph = params.message.includes('--graph') ? 'on' : 'off';
      let reply = params.message.includes('--help') ? 'Use /notes QUERY help' : 'Invalid or conflicting search flags. help';
      if (search) {
        const now = performance.timeOrigin + performance.now();
        await new Promise(resolve => setTimeout(resolve, 4));
        diagnostics.push({ schema_version: 1, kind: 'graphrag-notes-command', command_id: randomUUID(), session_sha256: sha(params.sessionKey),
          mode, graph, outcome: 'ok', reused: sent.length > 3, connect_ms: 0, search_ms: 0, handler_ms: 4,
          handler_start_epoch_ms: now, handler_finish_epoch_ms: performance.timeOrigin + performance.now() });
        reply = `Notes · ${mode}${graph === 'on' ? ' · graph on' : ''} · 0 ms total (0 ms connect, 0 ms search)\n\n1. Atlas\nID: note:${'d'.repeat(64)} · Actor: mcp:corpus-body\nID: ${fault === 'rank' ? 'note:' + 'c'.repeat(64) : id} · Actor: mcp:${actor.replace(/_/g, '\\_')}`;
      }
      setImmediate(() => {
        // Neither another session nor another run can satisfy this command.
        listener?.({ event: 'chat', payload: { sessionKey: 'someone-else', runId: params.idempotencyKey, state: 'final', message: { content: 'unrelated private content' } } });
        listener?.({ event: 'chat', payload: { sessionKey: params.sessionKey, runId: 'wrong-run', state: 'final', message: { content: reply } } });
        listener?.({ event: 'chat', payload: { sessionKey: params.sessionKey, runId: params.idempotencyKey, state: 'final', message: { content: reply } } });
      });
      return { runId: params.idempotencyKey, status: 'started' };
    },
    readDiagnostics: async () => fault === 'duplicate' ? [...diagnostics, diagnostics[0]] : diagnostics,
    archive: async session => { archived.push(session); return fault !== 'archive'; },
  };
  return { adapter, sent, archived };
}

test('profile joins 86 owned event replies with exact ranked provenance and independent pre/post pins', async () => {
  const f = fixture(), proof = await runGatewayProfile(f.adapter, cases, pin);
  assert.equal(proof.passed, true); assert.equal(proof.calls.length, 86); assert.equal(proof.calls.filter(x => x.handler).length, 84);
  assert.equal(new Set(f.sent.map(x => x.idempotencyKey)).size, 86);
  assert.ok(f.sent.every(x => x.deliver === false && !('channel' in x) && !('recipient' in x)));
  assert.equal(new Set(f.sent.map(x => x.sessionKey)).size, 1); assert.deepEqual(f.archived, [f.sent[0].sessionKey]);
  const summary = summarizeProfile(proof);
  for (const row of Object.values(summary.routes)) { assert.equal(row.warm_requested, 20); assert.equal(row.warm_successes, 20); assert.equal(row.failed, 0); assert.equal(row.warm.gateway_reply_ms.observations, 20); }
  for (const secret of ['private fictional Atlas', id, actor, 'private_token', f.sent[0].sessionKey, sha('private config'), sha('private model')]) assert.ok(!JSON.stringify(summary).includes(secret));
  assert.equal(summary.browser_render_verified, false); assert.equal(summary.model_requested, false);
});

test('rank, duplicate diagnostic, changed metadata/pinned record and cleanup failures cannot pass', async () => {
  for (const fault of ['rank', 'duplicate', 'metadata', 'pinned', 'archive']) {
    const f = fixture(fault), proof = await runGatewayProfile(f.adapter, cases, pin);
    assert.equal(proof.passed, false, fault); assert.equal(f.archived.length, 1);
    if (fault === 'rank') assert.equal(summarizeProfile(proof).routes.keyword.failed, 21);
  }
});

test('missing or overlapping same-session intervals and forged phase durations refuse attribution', async () => {
  const proof = await runGatewayProfile(fixture().adapter, cases, pin);
  for (const mutate of [p => p.diagnostics.splice(0, 1), p => { p.diagnostics[0].handler_start_epoch_ms -= 100000; },
    p => { p.diagnostics[0].handler_ms = 100000; }, p => { p.diagnostics[0].command_id = p.diagnostics[1].command_id; }]) {
    const copy = structuredClone(proof); mutate(copy); assert.throws(() => joinDiagnostics(copy), /diagnostic_join_failed/);
  }
});

test('inadequate warm samples produce no percentiles and malformed cases allocate no adapter', async () => {
  const proof = await runGatewayProfile(fixture().adapter, cases, pin);
  proof.calls = proof.calls.filter(call => call.round !== 20);
  const summary = summarizeProfile(proof); assert.ok(Object.values(summary.routes).every(row => row.warm_successes === 19 && row.warm === null)); assert.equal(summary.passed, false);
  await assert.rejects(runGatewayProfile({}, cases, pin, { rounds: 19 }), /invalid_limits/);
  await assert.rejects(runGatewayProfile({}, [...cases, cases[0]], pin), /invalid_cases/);
});

test('citation parsing uses final metadata lines and reverses formatter underscore escaping', () => {
  const fake = 'note:' + '9'.repeat(64);
  const reply = `Notes · keyword · 1 ms total (0 ms connect, 1 ms search)\n\n1. Atlas\nID: ${fake} · Actor: mcp:body\nID: ${id} · Actor: mcp:${actor.replace(/_/g, '\\_')}`;
  assert.deepEqual(parseResultMetadata(reply), { ids: [id], actors: [actor] });
  assert.throws(() => parseResultMetadata(reply + '\nnot metadata'), /ranked_provenance_failed/);
  assert.throws(() => parseResultMetadata(reply.replace(actor.replace(/_/g, '\\_'), 'a'.repeat(65))), /ranked_provenance_failed/);
});
