import test from 'node:test';
import assert from 'node:assert/strict';
import plugin, { createNotesCommand, parseNotesArguments } from './index.mjs';

const authorized = { isAuthorizedSender: true, config: {}, args: 'Atlas' };
const record = { id: 'note:abc123', title: 'Atlas', content: 'Launch Monday at 10:00.', provenance: { instance_id: 'openclaw-shiva' } };
const result = (records = [record]) => ({ structuredContent: { schema_version: 1, error: null, data: { records } } });

function fixture(options = {}) {
  const events = [];
  let transport, fetchOptions;
  class Client {
    async connect(value, opts) {
      transport = value; events.push(['connect', opts]);
      if (options.connectError) throw new Error('Bearer private-fixture-credential ' + 'http://private.invalid');
      if (options.connectPending) await new Promise((_, reject) => opts.signal.addEventListener('abort', () => reject(opts.signal.reason), { once: true }));
    }
    async callTool(value, schema, opts) {
      events.push(['call', value, opts]);
      if (options.pending) await new Promise((_, reject) => opts.signal.addEventListener('abort', () => reject(opts.signal.reason), { once: true }));
      if (options.callError) throw new Error('private-fixture-credential');
      return options.result ?? result();
    }
    async close() { events.push(['close']); }
  }
  class StreamableHTTPClientTransport {
    constructor(url, config) { this.url = url; this.config = config; }
  }
  const command = createNotesCommand({ reuseConnections: false, ...(options.config ?? {}) }, {
    env: options.env ?? { GRAPHRAG_NOTES_TOKEN: 'private-fixture-credential' },
    loadSdk: async () => { events.push(['sdk']); return { Client, StreamableHTTPClientTransport }; },
    fetch: async (_input, init) => { fetchOptions = init; return new Response(null, { status: 202 }); },
  });
  return { command, events, get transport() { return transport; }, get fetchOptions() { return fetchOptions; } };
}

test('registers an async authorized command and help/denied calls never touch MCP', async () => {
  const definitions = [];
  plugin.register({ pluginConfig: {}, registerCommand: value => definitions.push(value) });
  const definition = definitions.find(value => value.name === 'notes');
  assert.ok(definition);
  assert.equal(definition.requireAuth, true); assert.equal(definition.acceptsArgs, true);
  const f = fixture();
  assert.match((await f.command.handler({ ...authorized, isAuthorizedSender: false })).text, /authorization/);
  assert.match((await f.command.handler({ ...authorized, args: '' })).text, /Use \/notes/);
  assert.match((await f.command.handler({ ...authorized, args: '--help' })).text, /keyword/);
  assert.match((await f.command.handler({ ...authorized, args: '--hybrid' })).text, /Use \/notes/);
  assert.deepEqual(f.events, []);
});

test('default search has exact graph-enabled hybrid arguments, header, citation/actor and closes its own client', async () => {
  const f = fixture(); const reply = await f.command.handler(authorized);
  assert.deepEqual(f.events.find(e => e[0] === 'call')[1], { name: 'search_notes', arguments: { query: 'Atlas', mode: 'hybrid', graph: 'on', scope: 'notes', limit: 5 } });
  assert.match(reply.text, /Notes · hybrid · graph on/);
  assert.match(reply.text, /note:abc123/); assert.match(reply.text, /Actor: mcp:openclaw-shiva/); assert.match(reply.text, /ms total .*ms connect, .*ms search/);
  assert.equal(reply.continueAgent, false); assert.equal(f.events.at(-1)[0], 'close');
  assert.equal(f.transport.config.requestInit.headers.Authorization, 'Bearer private-fixture-credential');
  await f.transport.config.fetch(f.transport.url, {});
  assert.equal(f.fetchOptions.redirect, 'error'); assert.equal(f.fetchOptions.signal.aborted, true);
});

test('parser and mock MCP preserve graph defaults, explicit opt-outs, escape, and conflicts', async () => {
  assert.deepEqual(parseNotesArguments('Atlas'), { query: 'Atlas', mode: 'hybrid', graph: 'on' });
  assert.deepEqual(parseNotesArguments('--hybrid Atlas'), { query: 'Atlas', mode: 'hybrid', graph: 'on' });
  assert.deepEqual(parseNotesArguments('--graph Atlas'), { query: 'Atlas', mode: 'hybrid', graph: 'on' });
  assert.deepEqual(parseNotesArguments('--keyword Atlas'), { query: 'Atlas', mode: 'keyword', graph: 'off' });
  assert.deepEqual(parseNotesArguments('--no-graph Atlas'), { query: 'Atlas', mode: 'hybrid', graph: 'off' });
  assert.deepEqual(parseNotesArguments('-- --graph literal'), { query: '--graph literal', mode: 'hybrid', graph: 'on' });
  assert.deepEqual(parseNotesArguments('Atlas', { defaultGraph: 'auto' }), { query: 'Atlas', mode: 'hybrid', graph: 'auto' });
  assert.throws(() => parseNotesArguments('--graph --no-graph Atlas'));
  const f = fixture({ result: result([]) });
  const reply = await f.command.handler({ ...authorized, args: '--no-graph Atlas launch' });
  assert.deepEqual(f.events.find(e => e[0] === 'call')[1].arguments, { query: 'Atlas launch', mode: 'hybrid', graph: 'off', scope: 'notes', limit: 5 });
  assert.match(reply.text, /No matching notes/); assert.equal(reply.continueAgent, false);
});

test('only a resolved selected-server Bearer header is reused, no unresolved placeholders', async () => {
  const f = fixture({ env: { GRAPHRAG_NOTES_TOKEN: 'env-credential' } });
  await f.command.handler({ ...authorized, config: { mcp: { servers: { graphrag: { url: 'http://127.0.0.1:33156/mcp', headers: { Authorization: 'Bearer ${GRAPHRAG_NOTES_TOKEN}' } } } } } });
  assert.equal(f.transport.config.requestInit.headers.Authorization, 'Bearer env-credential');
  const direct = fixture({ env: {} });
  await direct.command.handler({ ...authorized, config: { mcp: { servers: { graphrag: { url: 'http://127.0.0.1:33156/mcp', headers: { Authorization: 'Bearer already-resolved' } } } } } });
  assert.equal(direct.transport.config.requestInit.headers.Authorization, 'Bearer already-resolved');
  const unresolved = fixture({ env: {} });
  const reply = await unresolved.command.handler({ ...authorized, config: { mcp: { servers: { graphrag: { headers: { Authorization: 'Bearer ${MISSING_TOKEN}' } } } } } });
  assert.match(reply.text, /credential is unavailable/); assert.deepEqual(unresolved.events, []);
  const otherEndpoint = fixture({ env: {}, config: { endpoint: 'http://127.0.0.1:99/mcp' } });
  const rejected = await otherEndpoint.command.handler({ ...authorized, config: { mcp: { servers: { graphrag: { url: 'http://127.0.0.1:33156/mcp', headers: { Authorization: 'Bearer already-resolved' } } } } } });
  assert.match(rejected.text, /credential is unavailable/); assert.deepEqual(otherEndpoint.events, []);
});

test('bounds query by Unicode characters before network access and rejects credential-bearing URLs', async () => {
  const f = fixture();
  assert.match((await f.command.handler({ ...authorized, args: '😀'.repeat(1025) })).text, /1024 characters/);
  assert.deepEqual(f.events, []);
  await f.command.handler({ ...authorized, args: '😀'.repeat(1024) });
  assert.equal(f.events.filter(e => e[0] === 'call').length, 1);
  for (const endpoint of ['http://example.org/mcp', 'https://user:password@example.org/mcp', 'https://example.org/mcp?token=secret']) {
    const bad = fixture({ config: { endpoint } });
    assert.match((await bad.command.handler(authorized)).text, /configuration is invalid/);
    assert.deepEqual(bad.events, []);
  }
});

test('untrusted output is bounded, plain, mention-safe and limits to five citations', async () => {
  const records = Array.from({ length: 9 }, (_, i) => ({ ...record, id: `note:${i}`, title: '@everyone [attack](https://example.org)'.repeat(20), content: '```\u0000 @everyone **body** '.repeat(100), provenance: { instance_id: '@everyone' } }));
  const reply = await fixture({ result: result(records) }).command.handler(authorized);
  assert.ok(reply.text.length <= 4000); assert.equal((reply.text.match(/ID: note:/g) ?? []).length, 5);
  assert.ok(!reply.text.includes('@everyone')); assert.ok(!reply.text.includes('\u0000'));
});

test('MCP failure and invalid envelopes disclose no SDK details and always close', async () => {
  for (const options of [{ connectError: true }, { callError: true }, { result: { structuredContent: { schema_version: 2, error: null, data: { records: [] } } } }]) {
    const f = fixture(options); const reply = await f.command.handler(authorized);
    assert.ok(!reply.text.includes('private-fixture-credential')); assert.ok(!reply.text.includes('private.invalid'));
    assert.equal(reply.continueAgent, false); assert.equal(f.events.at(-1)[0], 'close');
  }
});

test('a stalled graph-enabled connect or search hits graph deadline, aborts, closes, and allows the next invocation', async () => {
  for (const stage of ['connectPending', 'pending']) {
    const f = fixture({ [stage]: true, config: { timeoutMs: 250, graphTimeoutMs: 250 } });
    const start = performance.now(); const reply = await f.command.handler(authorized);
    assert.match(reply.text, /Graph-enabled search timed out after 250 ms/);
    assert.match(reply.text, /--keyword QUERY or \/notes --no-graph QUERY/); assert.ok(performance.now() - start < 1500);
    assert.equal(f.events.at(-1)[0], 'close');
    assert.equal(f.events.find(e => e[0] === 'connect')[1].signal.aborted, true);
    if (stage === 'connectPending') assert.ok(!f.events.some(e => e[0] === 'call'));
  }
  assert.match((await fixture().command.handler(authorized)).text, /note:abc123/);
});
