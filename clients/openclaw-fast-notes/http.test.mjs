import test from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { gzipSync } from 'node:zlib';
import { createNotesCommand } from './index.mjs';

async function fixture(config = {}) {
  const calls = [], revoked = new Set();
  const state = { disconnect: false, expire: false, stall: false, sse: false, oversized: false, truncated: false, oversizedInitialize: false, gzip: false };
  const server = createServer(async (req, res) => {
    const token = req.headers.authorization;
    if (!token || revoked.has(token)) { res.writeHead(401); res.end('fictional secret'); return; }
    if (req.method !== 'POST') { res.writeHead(405); res.end(); return; }
    let raw = ''; for await (const chunk of req) raw += chunk;
    const message = JSON.parse(raw); calls.push({ method: message.method, params: message.params, token });
    if (!Object.hasOwn(message, 'id')) { res.writeHead(202); res.end(); return; }
    if ((state.oversized && message.method === 'tools/call') || (state.oversizedInitialize && message.method === 'initialize')) {
      res.writeHead(200, { 'Content-Type': 'application/json' }); res.end('x'.repeat(2 * 1024 * 1024 + 1)); return;
    }
    if (state.truncated && message.method === 'tools/call') {
      res.writeHead(200, { 'Content-Type': 'application/json', 'Content-Length': '100' }); res.end('{}');
      setTimeout(() => req.socket.destroy(), 20); return;
    }
    let result;
    if (message.method === 'initialize') result = { protocolVersion: message.params.protocolVersion,
      capabilities: { tools: {} }, serverInfo: { name: 'fictional-only', version: '1' } };
    else if (message.method === 'tools/call') {
      if (state.disconnect) { req.socket.destroy(); return; }
      if (state.expire) { res.writeHead(404); res.end('fictional expired session'); return; }
      if (state.stall) return;
      if (state.sse) { res.writeHead(200, { 'Content-Type': 'text/event-stream' }); res.write('event: message\ndata: {}\n\n'); return; }
      assert.equal(message.params.name, 'search_notes');
      const envelope = { schema_version: 1, error: null, data: { records: [{
        id: 'note:fictional-http', title: message.params.arguments.query,
        content: 'Fictional body', provenance: { instance_id: token === 'Bearer principal-one' ? 'one' : 'two' },
      }] } };
      result = { content: [{ type: 'text', text: JSON.stringify(envelope) }], structuredContent: envelope };
      // Force overlap across independently correlated SDK requests.
      await new Promise(resolve => setTimeout(resolve, 10));
    } else throw new Error('unexpected fixture method');
    const body = JSON.stringify({ jsonrpc: '2.0', id: message.id, result });
    if (state.gzip) {
      const compressed = gzipSync(body);
      res.writeHead(200, { 'Content-Type': 'application/json', 'Content-Encoding': 'gzip', 'Content-Length': String(compressed.length) });
      res.end(compressed);
    } else {
      res.writeHead(200, { 'Content-Type': 'application/json', 'Mcp-Session-Id': token.replace('Bearer ', '') }); res.end(body);
    }
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const url = `http://127.0.0.1:${server.address().port}/mcp`;
  const command = createNotesCommand({ timeoutMs: 5000, ...config });
  const ctx = (query = 'Atlas', token = 'principal-one') => ({ isAuthorizedSender: true,
    args: '--keyword ' + query, config: { mcp: { servers: { graphrag: { url, headers: { Authorization: 'Bearer ' + token } } } } } });
  return { calls, revoked, state, command, ctx,
    close: async () => { await command.close(); server.closeAllConnections(); await new Promise(resolve => server.close(resolve)); } };
}

test('real SDK initializes once, authenticates concurrent HTTP calls and preserves each caller result', async () => {
  const f = await fixture();
  try {
    assert.match((await f.command.handler(f.ctx())).text, /Actor: mcp:one/);
    const queries = ['Borealis', 'Cirrus', 'Dune', 'Ember'];
    const results = await Promise.all(queries.map(q => f.command.handler(f.ctx(q))));
    results.forEach((r, i) => assert.match(r.text, new RegExp(queries[i])));
    assert.equal(f.calls.filter(c => c.method === 'initialize').length, 1);
    assert.equal(f.calls.filter(c => c.method === 'tools/call').length, 5);
    assert.ok(f.calls.every(c => c.token === 'Bearer principal-one'));
    await f.command.handler(f.ctx('Frost', 'principal-two'));
    assert.equal(f.calls.filter(c => c.method === 'initialize').length, 2);
    assert.equal(f.calls.at(-1).token, 'Bearer principal-two');
  } finally { await f.close(); }
});

test('real SDK credential revocation is enforced on a reused session, without fallback or auth replay', async () => {
  const f = await fixture();
  try {
    await f.command.handler(f.ctx()); f.revoked.add('Bearer principal-one');
    const result = await f.command.handler(f.ctx('Borealis'));
    assert.match(result.text, /credential was rejected/); assert.ok(!result.text.includes('secret'));
    assert.equal(f.calls.filter(c => c.method === 'tools/call').length, 1);
    await f.command.handler(f.ctx('Cirrus', 'principal-two'));
    assert.equal(f.calls.filter(c => c.method === 'initialize').length, 2);
  } finally { await f.close(); }
});

test('real SDK tunnel loss and expired session do not resubmit; explicit next command reconnects', async () => {
  for (const failure of ['disconnect', 'expire']) {
    const f = await fixture();
    try {
      await f.command.handler(f.ctx()); f.state[failure] = true;
      assert.match((await f.command.handler(f.ctx('Borealis'))).text, /unavailable/);
      assert.equal(f.calls.filter(c => c.method === 'tools/call').length, 2);
      f.state[failure] = false; assert.match((await f.command.handler(f.ctx('Cirrus'))).text, /Cirrus/);
      assert.equal(f.calls.filter(c => c.method === 'initialize').length, 2);
      assert.equal(f.calls.filter(c => c.method === 'tools/call').length, 3);
    } finally { await f.close(); }
  }
});

test('real SDK stalled HTTP body is aborted within deadline and cannot replay a capture', async () => {
  const f = await fixture({ timeoutMs: 250 });
  try {
    await f.command.handler(f.ctx()); f.state.stall = true;
    const start = performance.now(); assert.match((await f.command.handler(f.ctx('Borealis'))).text, /timed out/);
    assert.ok(performance.now() - start < 1500);
    f.state.stall = false; assert.match((await f.command.handler(f.ctx('Cirrus'))).text, /Cirrus/);
    const tools = f.calls.filter(c => c.method === 'tools/call');
    assert.equal(tools.length, 3); assert.ok(tools.every(c => c.params.name === 'search_notes'));
  } finally { await f.close(); }
});

test('unexpected SSE tool transport is refused and closed without accumulating live readers', async () => {
  const f = await fixture();
  try {
    await f.command.handler(f.ctx()); f.state.sse = true;
    assert.match((await f.command.handler(f.ctx('Borealis'))).text, /unexpected response/);
    assert.equal(f.calls.filter(c => c.method === 'tools/call').length, 2);
    f.state.sse = false; assert.match((await f.command.handler(f.ctx('Cirrus'))).text, /Cirrus/);
    assert.equal(f.calls.filter(c => c.method === 'initialize').length, 2);
  } finally { await f.close(); }
});

test('initialization and chunked search JSON have a hard size bound; truncated JSON fails without replay', async () => {
  for (const fault of ['oversizedInitialize', 'oversized', 'truncated']) {
    const f = await fixture();
    try {
      if (fault !== 'oversizedInitialize') await f.command.handler(f.ctx());
      f.state[fault] = true;
      assert.ok(!(await f.command.handler(f.ctx('Borealis'))).text.startsWith('Notes ·'));
      const before = f.calls.filter(c => c.method === 'tools/call').length;
      assert.equal(before, fault === 'oversizedInitialize' ? 0 : 2);
      f.state[fault] = false; assert.match((await f.command.handler(f.ctx('Cirrus'))).text, /Cirrus/);
      assert.equal(f.calls.filter(c => c.method === 'initialize').length, 2);
      assert.equal(f.calls.filter(c => c.method === 'tools/call').length, before + 1);
    } finally { await f.close(); }
  }
});

test('valid compressed initialization/search JSON keeps decoded bound without comparing compressed length', async () => {
  const f = await fixture(); f.state.gzip = true;
  try {
    assert.match((await f.command.handler(f.ctx())).text, /Atlas/);
    assert.match((await f.command.handler(f.ctx('Borealis'))).text, /Borealis/);
    assert.equal(f.calls.filter(c => c.method === 'initialize').length, 1);
  } finally { await f.close(); }
});
