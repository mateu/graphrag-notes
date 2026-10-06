import { createHash, randomUUID } from 'node:crypto';
import { performance } from 'node:perf_hooks';
import { isDeepStrictEqual } from 'node:util';

const ROUTES = ['keyword', 'hybrid', 'graph_title', 'graph_body'];
const sha = value => createHash('sha256').update(value).digest('hex');
const digest = value => typeof value === 'string' && /^[a-f0-9]{64}$/.test(value);
const check = (condition, code) => { if (!condition) throw new Error(code); };
const text = message => typeof message?.content === 'string' ? message.content :
  Array.isArray(message?.content) ? message.content.filter(x => x?.type === 'text' && typeof x.text === 'string').map(x => x.text).join('\n') : '';
const epoch = () => performance.timeOrigin + performance.now();
const finite = value => typeof value === 'number' && Number.isFinite(value) && value >= 0;
const policy = route => ({ mode: route === 'keyword' ? 'keyword' : 'hybrid', graph: route.startsWith('graph') ? 'on' : 'off' });

function metadata(value) {
  check(value && typeof value === 'object' && !Array.isArray(value), 'invalid_metadata');
  const fields = ['config_sha256', 'default_model_sha256', 'plugin_source_sha256', 'plugin_manifest_sha256'];
  check(fields.every(key => digest(value[key])), 'invalid_metadata');
  check(['fresh', 'reuse'].includes(value.connection_policy), 'invalid_metadata');
  check(['openclaw_version', 'mcp_sdk_version', 'node_version'].every(key => typeof value[key] === 'string' && /^[a-zA-Z0-9.+_-]{1,64}$/.test(value[key])), 'invalid_metadata');
  return Object.fromEntries([...fields, 'connection_policy', 'openclaw_version', 'mcp_sdk_version', 'node_version'].map(key => [key, value[key]]));
}
function pinned(value, expected) {
  check(value && typeof value === 'object' && digest(value.record_sha256) && value.record_sha256 === expected.record_sha256 &&
    value.record_id === expected.record_id && value.revision === expected.revision && value.principal_verified === true, 'pinned_readback_failed');
  return { record_sha256: value.record_sha256, record_id: value.record_id, revision: value.revision, principal_verified: true };
}
function validateCases(cases, pin) {
  check(Array.isArray(cases) && cases.length === 4 && cases.every((item, i) => item?.route === ROUTES[i] &&
    typeof item.query === 'string' && item.query.length > 0 && Array.from(item.query).length <= 1024 &&
    /^note:[a-f0-9]{64}$/.test(item.expected_first_id) && /^[a-zA-Z0-9_-]{1,96}$/.test(item.expected_actor) &&
    (!item.expected_ids || (Array.isArray(item.expected_ids) && item.expected_ids.length >= 1 && item.expected_ids.length <= 5 &&
      item.expected_ids[0] === item.expected_first_id && new Set(item.expected_ids).size === item.expected_ids.length &&
      item.expected_ids.every(id => /^note:[a-f0-9]{64}$/.test(id))))), 'invalid_cases');
  check(pin && /^note:[a-f0-9]{64}$/.test(pin.record_id) && digest(pin.revision) && digest(pin.record_sha256), 'invalid_pin');
}

// This runner has no gateway credential or SDK ownership. The operator adapter
// uses its supported native client, subscribes before requests, resolves the
// existing approved identity read-only, and implements exact pinned MCP reads.
export async function runGatewayProfile(adapter, cases, pin, { rounds = 20, commandTimeoutMs = 90000, budgetMs = 600000 } = {}) {
  validateCases(cases, pin);
  check(Number.isInteger(rounds) && rounds >= 20 && rounds <= 100 && Number.isInteger(commandTimeoutMs) && commandTimeoutMs >= 250 && commandTimeoutMs <= 120000 &&
    Number.isInteger(budgetMs) && budgetMs >= commandTimeoutMs && budgetMs <= 3600000, 'invalid_limits');
  const sessionKey = `agent:main:notes-profile-108-${randomUUID()}`;
  const sessionHash = sha(sessionKey), started = performance.now();
  const proof = { schema_version: 1, kind: 'openclaw-direct-profile', passed: false, session_sha256: sessionHash,
    delivery: { deliver: false, originating_fields_sent: false }, model_requested: false,
    browser_render_verified: false, rounds, calls: [], errors: [] };
  const waiters = new Map(); let unsubscribe, stage = 'preflight';
  try {
    proof.metadata = metadata(await adapter.metadata());
    proof.pinned_before = pinned(await adapter.readPinned(pin), pin);
    unsubscribe = adapter.subscribe(event => {
      const p = event?.payload;
      if (event?.event !== 'chat' || p?.sessionKey !== sessionKey || !waiters.has(p.runId)) return;
      if (['final', 'error', 'aborted'].includes(p.state)) waiters.get(p.runId)({ state: p.state, reply: text(p.message), received: epoch() });
    });
    const contracts = [{ label: 'help', message: '/notes --help', test: reply => /^Use \/notes QUERY/.test(reply) },
      { label: 'conflicting_flags', message: '/notes --keyword --graph invalid', test: reply => /^Invalid or conflicting search flags\./.test(reply) }];
    async function turn(spec, phase, round) {
      stage = spec.label ?? spec.route;
      const runId = randomUUID(); let timer;
      const row = { route: spec.route ?? null, contract: spec.label ?? null, phase, round, run_id: runId, outcome: 'failed' };
      proof.calls.push(row);
      try {
        const remaining = Math.floor(budgetMs - (performance.now() - started)); check(remaining > 0, 'suite_timeout');
        const timeout = Math.min(commandTimeoutMs, remaining);
        const commands = await adapter.request('commands.list', { agentId: 'main', provider: 'webchat', scope: 'text', includeArgs: false }, { timeoutMs: timeout });
        const notes = (commands?.commands ?? []).filter(c => c?.name === 'notes' || c?.textAliases?.includes('/notes'));
        check(notes.length === 1 && notes[0].source === 'plugin' && notes[0].acceptsArgs === true, 'command_registration_failed');
        const terminal = new Promise((resolve, reject) => { waiters.set(runId, resolve); timer = setTimeout(() => reject(new Error('event_timeout')), timeout); });
        // Observe an early deadline even while awaiting the acknowledgement.
        terminal.catch(() => {});
        row.gateway_start_epoch_ms = epoch();
        const command = spec.message ?? `/notes ${spec.route === 'keyword' ? '--keyword' : spec.route === 'hybrid' ? '--hybrid --no-graph' : '--graph'} -- ${spec.query}`;
        const ack = await adapter.request('chat.send', { sessionKey, agentId: 'main', message: command,
          deliver: false, timeoutMs: timeout, idempotencyKey: runId }, { timeoutMs: timeout });
        check(ack?.runId === runId && ['started', 'ok'].includes(ack.status), 'invalid_ack');
        const final = await terminal; row.gateway_finish_epoch_ms = final.received;
        row.gateway_reply_ms = final.received - row.gateway_start_epoch_ms;
        check(final.state === 'final', 'terminal_failed');
        row.reply_sha256 = sha(final.reply);
        if (spec.test) check(spec.test(final.reply), 'contract_failed');
        else {
          const timing = final.reply.match(/^Notes · (keyword|hybrid)(?: · graph (off|auto|on))? · (\d+) ms total \((\d+) ms connect, (\d+) ms search\)/);
          const desired = policy(spec.route); check(timing && timing[1] === desired.mode && (timing[2] ?? 'off') === desired.graph, 'policy_failed');
          row.mode = desired.mode; row.graph = desired.graph;
          row.record_ids = [...final.reply.matchAll(/ID: (note:[a-f0-9]{64})/g)].map(match => match[1]);
          row.actors = [...final.reply.matchAll(/Actor: mcp:([a-zA-Z0-9_-]+)/g)].map(match => match[1]);
          check(row.record_ids.length >= 1 && row.record_ids.length <= 5 && new Set(row.record_ids).size === row.record_ids.length &&
            row.record_ids[0] === spec.expected_first_id && row.actors.length === row.record_ids.length && row.actors[0] === spec.expected_actor &&
            (!spec.expected_ids || isDeepStrictEqual(row.record_ids, spec.expected_ids)), 'ranked_provenance_failed');
          row.plugin_reported_ms = { total: Number(timing[3]), connect: Number(timing[4]), search: Number(timing[5]) };
        }
        row.outcome = 'ok'; row.event_verified = true;
      } catch (error) {
        row.failure = ['suite_timeout', 'event_timeout', 'command_registration_failed', 'invalid_ack', 'terminal_failed', 'contract_failed', 'policy_failed', 'ranked_provenance_failed'].includes(error.message) ? error.message : 'gateway_unavailable';
      } finally { clearTimeout(timer); waiters.delete(runId); }
    }
    for (const contract of contracts) await turn(contract, 'contract', null);
    for (const item of cases) await turn(item, 'first', null);
    for (let round = 1; round <= rounds; round++) for (const item of cases) await turn(item, 'warm', round);
    stage = 'postflight';
    proof.pinned_after = pinned(await adapter.readPinned(pin), pin);
    check(isDeepStrictEqual(metadata(await adapter.metadata()), proof.metadata), 'metadata_changed');
    proof.diagnostics = await adapter.readDiagnostics(sessionHash);
    joinDiagnostics(proof);
    proof.passed = proof.calls.every(call => call.outcome === 'ok') && proof.calls.length === 2 + 4 * (rounds + 1);
  } catch (error) {
    proof.errors.push({ stage, code: ['invalid_metadata', 'pinned_readback_failed', 'metadata_changed', 'diagnostic_join_failed'].includes(error.message) ? error.message : 'profile_failed' });
  } finally {
    unsubscribe?.();
    try { proof.owned_session_archived = await adapter.archive(sessionKey) === true; } catch { proof.owned_session_archived = false; }
    if (!proof.owned_session_archived) { proof.passed = false; proof.errors.push({ stage: 'cleanup', code: 'archive_failed' }); }
  }
  return proof;
}

export function joinDiagnostics(proof) {
  check(Array.isArray(proof.diagnostics) && proof.diagnostics.length <= proof.calls.length, 'diagnostic_join_failed');
  const used = new Set();
  for (const call of proof.calls.filter(call => call.route && call.outcome === 'ok')) {
    const matches = proof.diagnostics.filter(entry => entry?.schema_version === 1 && entry.kind === 'graphrag-notes-command' &&
      entry.session_sha256 === proof.session_sha256 && entry.mode === call.mode && entry.graph === call.graph && entry.outcome === 'ok' &&
      finite(entry.handler_start_epoch_ms) && finite(entry.handler_finish_epoch_ms) &&
      entry.handler_start_epoch_ms >= call.gateway_start_epoch_ms - 2 && entry.handler_finish_epoch_ms <= call.gateway_finish_epoch_ms + 2 &&
      entry.handler_finish_epoch_ms >= entry.handler_start_epoch_ms);
    check(matches.length === 1, 'diagnostic_join_failed');
    const entry = matches[0];
    check(typeof entry.command_id === 'string' && /^[a-f0-9-]{36}$/.test(entry.command_id) && !used.has(entry.command_id) &&
      ['connect_ms', 'search_ms', 'handler_ms'].every(key => finite(entry[key])) && typeof entry.reused === 'boolean' &&
      entry.handler_ms <= call.gateway_reply_ms + 2 && entry.connect_ms + entry.search_ms <= entry.handler_ms + 2, 'diagnostic_join_failed');
    used.add(entry.command_id);
    call.handler = { command_id: entry.command_id, reused: entry.reused, connect_ms: entry.connect_ms,
      search_ms: entry.search_ms, handler_ms: entry.handler_ms,
      dispatch_before_ms: Math.max(0, entry.handler_start_epoch_ms - call.gateway_start_epoch_ms),
      dispatch_after_ms: Math.max(0, call.gateway_finish_epoch_ms - entry.handler_finish_epoch_ms) };
  }
  // Extra successful entries in the owned session are evidence of unaccounted work.
  check(proof.diagnostics.filter(entry => entry?.outcome === 'ok').length === used.size, 'diagnostic_join_failed');
}

function distribution(values) {
  const sorted = values.toSorted((a, b) => a - b);
  return { observations: sorted.length, median_ms: (sorted[Math.floor((sorted.length - 1) / 2)] + sorted[Math.ceil((sorted.length - 1) / 2)]) / 2,
    p95_ms: sorted[Math.ceil(sorted.length * .95) - 1], min_ms: sorted[0], max_ms: sorted.at(-1) };
}
export function summarizeProfile(proof) {
  check(proof?.schema_version === 1 && proof.kind === 'openclaw-direct-profile' && Array.isArray(proof.calls), 'invalid_proof');
  const result = { schema_version: 1, passed: proof.passed === true, metadata: Object.fromEntries(Object.entries(metadata(proof.metadata)).filter(([key]) => !['config_sha256', 'default_model_sha256'].includes(key))),
    browser_render_verified: proof.browser_render_verified === true, model_requested: false,
    owned_session_archived: proof.owned_session_archived === true, requested: proof.calls.length,
    failed: proof.calls.filter(call => call?.outcome !== 'ok').length,
    diagnostic_joined: proof.calls.filter(call => call?.handler).length,
    contract_requested: proof.calls.filter(call => call?.contract).length,
    contract_failed: proof.calls.filter(call => call?.contract && call.outcome !== 'ok').length, routes: {},
    caveat: 'Event receipt is not browser rendering. Intervals are matched per command; no unrelated percentile subtraction. First observations are not guaranteed cold.' };
  for (const route of ROUTES) {
    const calls = proof.calls.filter(call => call?.route === route), warm = calls.filter(call => call.phase === 'warm'),
      successes = warm.filter(call => call.outcome === 'ok' && call.event_verified && call.handler);
    const row = result.routes[route] = { ...policy(route), requested: calls.length, failed: calls.filter(call => call.outcome !== 'ok').length,
      warm_requested: warm.length, warm_successes: successes.length, first: null, warm: null };
    const first = calls.find(call => call.phase === 'first');
    if (first) row.first = { outcome: first.outcome, gateway_reply_ms: first.gateway_reply_ms ?? null, reused: first.handler?.reused ?? null };
    if (successes.length >= 20) row.warm = Object.fromEntries(['gateway_reply_ms', 'connect_ms', 'search_ms', 'handler_ms', 'dispatch_before_ms', 'dispatch_after_ms'].map(key => [key,
      distribution(successes.map(call => key === 'gateway_reply_ms' ? call[key] : call.handler[key]))]));
  }
  return result;
}
