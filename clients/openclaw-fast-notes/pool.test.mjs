import test from 'node:test';
import assert from 'node:assert/strict';
import plugin, { createNotesCommand } from './index.mjs';

const context = (args = '--keyword Atlas', name = 'graphrag', token = 'fictional-one', url = 'http://127.0.0.1:9011/mcp') => ({
  isAuthorizedSender: true, sessionKey: 'fictional-session', args,
  config: { mcp: { servers: { [name]: { url, headers: { Authorization: `Bearer ${token}` } } } } },
});
const response = query => ({ structuredContent: { schema_version: 1, error: null, data: { records: [{
  id: 'note:fictional', title: query, content: 'Fictional content', provenance: { instance_id: 'fixture-principal' },
}] } } });
const deferred = () => { let resolve; const promise = new Promise(done => { resolve = done; }); return { promise, resolve }; };
const tick = () => new Promise(resolve => setImmediate(resolve));

function fixture(config = {}, options = {}) {
  const events = [], timings = [], env = {}, hold = deferred(), connectHold = deferred(); let clock = 0;
  const state = { pending: false, fail: false, status: 200, stalledConnect: false, suspendConnect: false };
  class Transport {
    constructor(url, init) { this.url = url; this.init = init; events.push(['transport', url.href]); }
    async close() { events.push(['transport-close']); this.onclose?.(); }
  }
  class Client {
    async connect(transport, opts) {
      this.transport = transport; events.push(['connect']);
      if (state.stalledConnect) await new Promise(() => {});
      else await transport.init.fetch(transport.url, { method: 'POST', body: 'initialize' });
      if (state.suspendConnect) await connectHold.promise;
      opts.signal.throwIfAborted();
    }
    async callTool(request, _schema, opts) {
      events.push(['call', request]);
      await this.transport.init.fetch(this.transport.url, { method: 'POST', body: JSON.stringify({ jsonrpc: '2.0', method: 'tools/call', params: request }), headers: { Authorization: 'wrong' } });
      const signal = events.filter(e => e[0] === 'fetch').at(-1)[3];
      if (state.pending) await Promise.race([hold.promise, new Promise((_, reject) => Promise.resolve().then(() => {
        if (signal.aborted) reject(signal.reason); else signal.addEventListener('abort', () => reject(signal.reason), { once: true });
      }))]);
      opts.signal.throwIfAborted(); return response(request.arguments.query);
    }
    async close() { events.push(['client-close']); if (options.stalledClose) await new Promise(() => {}); this.onclose?.(); }
  }
  const command = createNotesCommand(config, { env,
    loadSdk: async () => { events.push(['sdk']); return { Client, StreamableHTTPClientTransport: Transport }; },
    fetch: async (_input, init) => {
      events.push(['fetch', init.headers.get('Authorization'), init.redirect, init.signal]);
      if (state.fail) throw new Error('fictional-one secret private.invalid');
      return new Response('{}', { status: state.status, headers: { 'Content-Type': 'application/json' } });
    },
    ...(options.fakeClock ? { now: () => clock } : {}), onDiagnostics: timing => timings.push(timing),
  });
  return { command, state, env, events, timings, hold, connectHold, advance: ms => { clock += ms; }, count: name => events.filter(x => x[0] === name).length };
}

test('warm searches reuse initialization while each HTTP request authenticates, no credentials in diagnostics', async () => {
  const f = fixture();
  try {
    for (const query of ['Atlas', 'Borealis', 'Cirrus']) assert.match((await f.command.handler(context('--keyword ' + query))).text, new RegExp(query));
    assert.equal(f.count('connect'), 1); assert.equal(f.count('sdk'), 1); assert.equal(f.count('call'), 3);
    assert.equal(f.count('fetch'), 4);
    for (const event of f.events.filter(x => x[0] === 'fetch')) { assert.equal(event[1], 'Bearer fictional-one'); assert.equal(event[2], 'error'); }
    assert.deepEqual(f.timings.map(x => x.reused), [false, true, true]);
    assert.equal(new Set(f.timings.map(x => x.command_id)).size, 3);
    for (const secret of ['fictional-one', 'fictional-session', 'Atlas', 'note:fictional', '127.0.0.1']) assert.ok(!JSON.stringify(f.timings).includes(secret));
  } finally { await f.command.close(); }
  assert.equal(f.count('client-close'), 1);
});

test('simultaneous callers share one initialize, preserve payloads and reject excess work without a queue', async () => {
  const f = fixture({ poolMaxInFlight: 4 }); f.state.pending = true;
  try {
    const pending = ['A', 'B', 'C', 'D'].map(q => f.command.handler(context('--keyword ' + q)));
    await tick(); assert.equal(f.count('connect'), 1); assert.equal(f.count('call'), 4);
    assert.match((await f.command.handler(context('--keyword excess'))).text, /busy/);
    assert.equal(f.count('call'), 4);
    f.hold.resolve(); const results = await Promise.all(pending);
    results.forEach((r, i) => assert.match(r.text, new RegExp(['A', 'B', 'C', 'D'][i])));
    assert.equal(new Set(f.timings.map(x => x.command_id)).size, 5);
  } finally { await f.command.close(); }
});

test('selected credentials and canonical endpoints are isolated; rotation retires old session', async () => {
  const f = fixture();
  try {
    await f.command.handler(context());
    await f.command.handler(context('--keyword Atlas', 'graphrag', 'fictional-two'));
    await f.command.handler(context('--keyword Atlas', 'graphrag', 'fictional-two', 'http://127.0.0.1:9012/mcp'));
    assert.equal(f.count('connect'), 3); assert.equal(f.count('client-close'), 2);
    assert.deepEqual(f.events.filter(x => x[0] === 'fetch').map(x => x[1]), ['Bearer fictional-one', 'Bearer fictional-one', 'Bearer fictional-two', 'Bearer fictional-two', 'Bearer fictional-two', 'Bearer fictional-two']);
  } finally { await f.command.close(); }
});

test('context pool count is bounded, and different authenticated contexts never share a client', async () => {
  const f = fixture({ poolMaxEntries: 2 });
  try {
    for (const name of ['one', 'two']) {
      const c = context('--keyword Atlas', name, 'fictional-' + name); c.config.plugins = { entries: { 'graphrag-fast-notes': { config: { serverName: name, poolMaxEntries: 2 } } } };
      assert.match((await f.command.handler(c)).text, /Notes/);
    }
    const third = context('--keyword Atlas', 'three', 'fictional-three'); third.config.plugins = { entries: { 'graphrag-fast-notes': { config: { serverName: 'three', poolMaxEntries: 2 } } } };
    assert.match((await f.command.handler(third)).text, /busy/); assert.equal(f.count('connect'), 2);
    assert.equal(f.count('call'), 2);
  } finally { await f.command.close(); }
});

test('revoked credentials and tunnel loss fail honestly, retire cache and never replay; next explicit request reconnects', async () => {
  for (const kind of ['revoked', 'lost']) {
    const f = fixture();
    try {
      await f.command.handler(context());
      if (kind === 'revoked') f.state.status = 401; else f.state.fail = true;
      const failed = await f.command.handler(context());
      assert.match(failed.text, kind === 'revoked' ? /rejected/ : /unavailable/);
      assert.ok(!failed.text.includes('fictional-one')); assert.ok(!failed.text.includes('private.invalid'));
      assert.equal(f.count('call'), 2); assert.equal(f.count('connect'), 1);
      f.state.status = 200; f.state.fail = false;
      assert.match((await f.command.handler(context())).text, /Notes/);
      assert.equal(f.count('connect'), 2); assert.equal(f.count('call'), 3);
    } finally { await f.command.close(); }
  }
});

test('deadline bounds a stalled initialization and prevents a search after it', async () => {
  const f = fixture({ timeoutMs: 250 }); f.state.stalledConnect = true;
  const start = performance.now();
  try {
    assert.match((await f.command.handler(context())).text, /timed out after 250/);
    assert.ok(performance.now() - start < 1500); assert.equal(f.count('call'), 0);
    f.state.stalledConnect = false; assert.match((await f.command.handler(context())).text, /Notes/);
    assert.equal(f.count('connect'), 2);
  } finally { await f.command.close(); }
});

test('search timeout retires session without fallback or retry, and cleanup itself is bounded', async () => {
  const f = fixture({ timeoutMs: 250 }, { stalledClose: true }); f.state.pending = true;
  const start = performance.now();
  assert.match((await f.command.handler(context())).text, /timed out after 250/);
  assert.equal(f.count('call'), 1); assert.ok(performance.now() - start < 1500);
  await f.command.close();
});

test('rapid rotation cannot exceed total sessions while SDK close or initialize refuses to settle', async () => {
  for (const stalled of ['close', 'initialize']) {
    const f = fixture({ poolMaxEntries: 2, timeoutMs: 250 }, { stalledClose: stalled === 'close' });
    if (stalled === 'initialize') f.state.stalledConnect = true;
    for (let i = 0; i < 12; i++) await f.command.handler(context('--keyword Atlas', 'graphrag', 'fixture-rotation-' + i));
    assert.equal(f.count('connect'), 2);
    assert.ok(f.timings.some(t => t.outcome === 'busy'));
    await f.command.close(); f.command.start();
    assert.match((await f.command.handler(context())).text, /busy/);
    assert.equal(f.count('connect'), 2);
    await f.command.close();
  }
});

test('credential rotation during asynchronous SDK preparation cannot dispatch with the old credential', async () => {
  const f = fixture(); f.state.suspendConnect = true;
  const ctx = context(); const pending = f.command.handler(ctx);
  try {
    await tick(); ctx.config.mcp.servers.graphrag.headers.Authorization = 'Bearer newly-rotated';
    f.connectHold.resolve(); assert.match((await pending).text, /changed/);
    assert.equal(f.count('call'), 0);
    assert.match((await f.command.handler(ctx)).text, /Notes/);
    assert.equal(f.count('connect'), 2); assert.equal(f.count('call'), 1);
    assert.equal(f.events.filter(e => e[0] === 'fetch').at(-1)[1], 'Bearer newly-rotated');
  } finally { await f.command.close(); }
});

test('idle and absolute TTL are enforced even when timer execution is delayed', async () => {
  const f = fixture({ poolIdleMs: 1000, poolLifetimeMs: 5000 }, { fakeClock: true });
  try {
    await f.command.handler(context()); f.advance(999); await f.command.handler(context());
    assert.equal(f.count('connect'), 1);
    f.advance(1001); await f.command.handler(context()); assert.equal(f.count('connect'), 2);
    for (let i = 0; i < 6; i++) { f.advance(900); await f.command.handler(context()); }
    assert.equal(f.count('connect'), 3);
  } finally { await f.command.close(); }
});

test('shutdown stops admission and restart creates a fresh connection; reconfiguration closes prior pool', async () => {
  const f = fixture();
  await f.command.handler(context()); await f.command.close();
  assert.match((await f.command.handler(context())).text, /unavailable/); assert.equal(f.count('connect'), 1);
  f.command.start(); await f.command.handler(context()); assert.equal(f.count('connect'), 2);
  const c = context(); c.config.plugins = { entries: { 'graphrag-fast-notes': { config: { poolIdleMs: 1000 } } } };
  await f.command.handler(c); assert.equal(f.count('connect'), 3);
  await f.command.close();
});

test('unauthorized/help/conflicting commands allocate no session; explicit graph policy never falls back', async () => {
  const f = fixture();
  try {
    await f.command.handler({ ...context(), isAuthorizedSender: false });
    await f.command.handler(context('--help')); await f.command.handler(context('--graph --no-graph Atlas'));
    assert.equal(f.count('connect'), 0);
    for (const args of ['Atlas', '--hybrid Atlas', '--graph Atlas', '--no-graph Atlas', '--keyword Atlas']) await f.command.handler(context(args));
    assert.deepEqual(f.events.filter(x => x[0] === 'call').map(x => [x[1].arguments.mode, x[1].arguments.graph]), [['hybrid', 'on'], ['hybrid', 'on'], ['hybrid', 'on'], ['hybrid', 'off'], ['keyword', 'off']]);
  } finally { await f.command.close(); }
});

test('public service lifecycle is registered and unsupported lifecycle keeps per-command clients', () => {
  const commands = [], services = [];
  plugin.register({ registerCommand: c => commands.push(c), registerService: s => services.push(s) });
  assert.deepEqual(commands.map(c => c.name), ['notes', 'notesave']); assert.equal(commands[1].requiredScopes[0], 'operator.write');
  assert.deepEqual(services[0].reload.configPrefixes, ['plugins.entries.graphrag-fast-notes.config', 'mcp.servers']);
  assert.equal(typeof services[0].stop, 'function'); assert.equal(typeof services[0].start, 'function');
});

test('one deadline invalidates concurrent reads without replay; longer caller reports unavailable', async () => {
  const f = fixture({ timeoutMs: 250, graphTimeoutMs: 1000 }); f.state.pending = true;
  try {
    const results = await Promise.all([f.command.handler(context('--keyword first')), f.command.handler(context('--graph second'))]);
    assert.match(results[0].text, /timed out after 250/); assert.match(results[1].text, /unavailable/);
    assert.equal(f.count('call'), 2); assert.equal(f.count('connect'), 1);
    assert.deepEqual(f.timings.map(t => t.outcome), ['timeout', 'unavailable']);
  } finally { await f.command.close(); }
});

test('shutdown denies fresh bypass admission too; explicit restart is required', async () => {
  const f = fixture({ reuseConnections: false });
  await f.command.close(); assert.match((await f.command.handler(context())).text, /unavailable/);
  assert.equal(f.count('connect'), 0); f.command.start();
  assert.match((await f.command.handler(context())).text, /^Notes ·/); assert.equal(f.count('connect'), 1);
  await f.command.close();
});

test('shutdown cancels an active fresh search and fences late SDK initialization after restart', async () => {
  for (const phase of ['search', 'initialize']) {
    const f = fixture({ reuseConnections: false, graphTimeoutMs: 120000 });
    if (phase === 'search') f.state.pending = true; else f.state.suspendConnect = true;
    const pending = f.command.handler(context('--graph Atlas'));
    await tick(); const before = f.count('call'), started = performance.now();
    await f.command.close();
    assert.ok(performance.now() - started < 1500);
    assert.match((await pending).text, /unavailable/);
    assert.match((await f.command.handler(context())).text, /unavailable/);
    f.command.start(); f.state.pending = false; f.state.suspendConnect = false; f.connectHold.resolve();
    await tick();
    assert.equal(f.count('call'), before);
    assert.match((await f.command.handler(context('--keyword Borealis'))).text, /^Notes ·/);
    assert.equal(f.count('call'), before + 1);
    await f.command.close();
  }
});

test('fresh work and closing reservations remain bounded when SDK cleanup never settles', async () => {
  const f = fixture({ reuseConnections: false }, { stalledClose: true });
  const pending = Array.from({ length: 4 }, (_, i) => f.command.handler(context('--keyword fixture-' + i)));
  await tick();
  assert.match((await f.command.handler(context())).text, /busy/);
  assert.equal(f.count('connect'), 4);
  assert.ok((await Promise.all(pending)).every(reply => reply.text.startsWith('Notes ·')));
  await f.command.close(); f.command.start();
  assert.match((await f.command.handler(context())).text, /busy/);
  assert.equal(f.count('connect'), 4);
  await f.command.close();
});

test('fresh/reused configuration churn cannot bypass unfinished cleanup reservations', async () => {
  for (const initialFresh of [true, false]) {
    const f = fixture({}, { stalledClose: true });
    const first = context(); first.config.plugins = { entries: { 'graphrag-fast-notes': { config: { reuseConnections: !initialFresh, poolMaxEntries: 1 } } } };
    const pending = f.command.handler(first);
    await tick();
    if (initialFresh) await pending;
    else assert.match((await pending).text, /^Notes ·/);
    const next = context(); next.config.plugins = { entries: { 'graphrag-fast-notes': { config: { reuseConnections: initialFresh, poolMaxEntries: 1 } } } };
    assert.match((await f.command.handler(next)).text, /busy/);
    assert.equal(f.count('connect'), 1);
    await f.command.close();
  }
});
