import test from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { createNotesSaveCommand } from './index.mjs';

async function fixture() {
  const posts = [], records = new Map(); let loseFirst = true;
  const server = createServer(async (request, reply) => {
    assert.equal(request.headers.authorization, 'Bearer fictional-capture');
    if (request.method !== 'POST') { reply.writeHead(405); reply.end(); return; }
    let raw = ''; for await (const chunk of request) raw += chunk;
    const message = JSON.parse(raw); posts.push(message);
    if (!Object.hasOwn(message, 'id')) { reply.writeHead(202); reply.end(); return; }
    let result;
    if (message.method === 'initialize') result = { protocolVersion: message.params.protocolVersion, capabilities: { tools: {} }, serverInfo: { name: 'fictional', version: '1' } };
    else if (message.params.name === 'capture_note') {
      const p = message.params.arguments, replayed = records.has(p.request_id);
      const record = records.get(p.request_id) ?? { id: 'note:' + 'a'.repeat(64), revision: 'b'.repeat(64), title: p.title, content: p.content,
        tags: p.tags, provenance: { instance_id: 'fictional-captor', source: p.provenance } };
      records.set(p.request_id, record);
      // Deliberately expire after the fictional service commits the capture.
      if (loseFirst) { loseFirst = false; reply.writeHead(404); reply.end('fictional private expired session'); return; }
      const envelope = { schema_version: 1, error: null, data: { request_id: p.request_id, replayed, record } };
      result = { content: [{ type: 'text', text: JSON.stringify(envelope) }], structuredContent: envelope };
    } else {
      assert.equal(message.params.name, 'get_record'); const record = [...records.values()][0];
      assert.deepEqual(message.params.arguments, { id: record.id, revision: record.revision, neighbors: 0 });
      const envelope = { schema_version: 1, error: null, data: { hit_type: 'note', ...record } };
      result = { content: [{ type: 'text', text: JSON.stringify(envelope) }], structuredContent: envelope };
    }
    reply.writeHead(200, { 'Content-Type': 'application/json', 'Mcp-Session-Id': 'fictional-session' });
    reply.end(JSON.stringify({ jsonrpc: '2.0', id: message.id, result }));
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  return { posts, records, ctx: { isAuthorizedSender: true, sessionKey: 'fictional-owned-session', args: 'Atlas | exact fictional body', commandBody: '/notesave Atlas | exact fictional body',
    config: { mcp: { servers: { graphrag: { url: `http://127.0.0.1:${server.address().port}/mcp` } } } } },
    close: async () => { server.closeAllConnections(); await new Promise(resolve => server.close(resolve)); } };
}

test('real SDK expired capture session cannot replay implicitly; explicit original request verifies one stored record', async () => {
  const f = await fixture();
  const command = createNotesSaveCommand({ captureActor: 'fictional-captor' }, { env: { GRAPHRAG_NOTES_TOKEN: 'fictional-capture' } });
  try {
    const first = await command.handler(f.ctx); assert.match(first.text, /outcome is unconfirmed/); assert.ok(!first.text.includes('private expired'));
    const captures = () => f.posts.filter(p => p.method === 'tools/call' && p.params.name === 'capture_note');
    assert.equal(captures().length, 1); assert.equal(f.records.size, 1);
    assert.equal(f.posts.filter(p => p.method === 'tools/call' && p.params.name === 'get_record').length, 0);
    assert.match((await command.handler(f.ctx)).text, /replayed and independently verified/);
    assert.equal(captures().length, 2); assert.deepEqual(captures()[0].params.arguments, captures()[1].params.arguments);
    assert.equal(f.records.size, 1); assert.equal(f.posts.filter(p => p.method === 'tools/call' && p.params.name === 'get_record').length, 1);
  } finally { await f.close(); }
});

test('an SDK attempting a duplicate capture POST is blocked before fetch, and off-target/auth-context changes are fenced', async () => {
  for (const fault of ['duplicate', 'off_target', 'credential']) {
    const fetched = [], env = { GRAPHRAG_NOTES_TOKEN: 'fictional-capture' };
    class Transport { constructor(url, options) { this.url = url; this.options = options; } }
    class Client {
      async connect(transport) { this.transport = transport; } async close() {}
      async callTool(request) {
        const options = { method: 'POST', body: JSON.stringify({ jsonrpc: '2.0', id: 1, method: 'tools/call', params: request }), headers: { Authorization: 'wrong' } };
        if (fault === 'credential') env.GRAPHRAG_NOTES_TOKEN = 'rotated';
        const target = fault === 'off_target' ? new URL('http://127.0.0.1:9999/mcp') : this.transport.url;
        await this.transport.options.fetch(target, options);
        await this.transport.options.fetch(target, options); // synthetic SDK replay independent of maxRetries
      }
    }
    const command = createNotesSaveCommand({ captureActor: 'fictional-captor' }, { env, loadSdk: async () => ({ Client, StreamableHTTPClientTransport: Transport }),
      fetch: async (input, init) => { fetched.push([input.href, init.headers.get('Authorization')]); return new Response('{}', { headers: { 'Content-Type': 'application/json' } }); } });
    const ctx = { isAuthorizedSender: true, sessionKey: 'fictional-owned-session', args: 'Atlas | body', commandBody: '/notesave Atlas | body', config: { mcp: { servers: { graphrag: { url: 'http://127.0.0.1:9991/mcp' } } } } };
    assert.match((await command.handler(ctx)).text, /outcome is unconfirmed/);
    assert.equal(fetched.length, fault === 'duplicate' ? 1 : 0);
    if (fetched.length) assert.deepEqual(fetched[0], ['http://127.0.0.1:9991/mcp', 'Bearer fictional-capture']);
  }
});
