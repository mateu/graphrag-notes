import test from 'node:test';
import assert from 'node:assert/strict';
import plugin, { captureArguments, capturePayload, createNotesSaveCommand, createNotesCommand } from './index.mjs';

test('notesave remains registered, authorization-gated, and validation rejects before MCP work', async () => {
  const definitions = [];
  plugin.register({ pluginConfig: {}, registerCommand: value => definitions.push(value) });
  const save = definitions.find(value => value.name === 'notesave');
  assert.ok(save);
  assert.equal(save.requireAuth, true);
  assert.deepEqual(save.requiredScopes, ['operator.write']);
  const command = createNotesSaveCommand({}, { loadSdk: async () => { throw new Error('must not load SDK'); } });
  assert.match((await command.handler({ isAuthorizedSender: false, args: 'Title | body' })).text, /authorization/);
  assert.match((await command.handler({ isAuthorizedSender: true, args: 'Title only', commandBody: '/notesave Title only' })).text, /Title \| single-line body/);
});

test('notesave keeps exact single-line payload and stable retry receipt input', () => {
  const ctx = { channel: 'webchat', senderId: 'fictional-operator', sessionKey: 'agent:main:test', args: 'Title | body', commandBody: '/notesave Title | body' };
  assert.equal(captureArguments(ctx), 'Title | body');
  const first = capturePayload('Title | body', ctx);
  const second = capturePayload('Title | body', ctx);
  assert.equal(first.request_id, second.request_id);
  assert.equal(first.title, 'Title');
  assert.equal(first.content, 'body');
  assert.deepEqual(first.tags, []);
});

test('capture uses a separate fresh client, verifies receipt/readback and never borrows the warmed read pool', async () => {
  const clients = [], tools = []; let record;
  const data = value => ({ structuredContent: { schema_version: 1, error: null, data: value } });
  class Client {
    constructor() { clients.push(this); this.closed = false; }
    async connect() {}
    async close() { this.closed = true; }
    async callTool(request) {
      tools.push([this, request.name]);
      if (request.name === 'search_notes') return data({ records: [] });
      if (request.name === 'capture_note') {
        const p = request.arguments;
        record = { id: 'note:' + 'a'.repeat(64), revision: 'b'.repeat(64), title: p.title, content: p.content,
          tags: p.tags, provenance: { instance_id: 'fictional-captor', source: p.provenance } };
        return data({ request_id: p.request_id, replayed: false, record });
      }
      assert.equal(request.name, 'get_record'); assert.equal(request.arguments.revision, record.revision);
      return data({ hit_type: 'note', ...record });
    }
  }
  class Transport { async close() {} }
  const overrides = { env: { GRAPHRAG_NOTES_TOKEN: 'fictional-capture' }, loadSdk: async () => ({ Client, StreamableHTTPClientTransport: Transport }) };
  const ctx = { isAuthorizedSender: true, senderId: 'fictional-operator', sessionKey: 'fixture',
    config: { mcp: { servers: { graphrag: { url: 'http://127.0.0.1:9991/mcp' } } } }, args: '--keyword fixture' };
  const read = createNotesCommand({}, overrides);
  try {
    await read.handler(ctx); await read.handler(ctx); assert.equal(clients.length, 1);
    const capture = createNotesSaveCommand({ captureActor: 'fictional-captor' }, overrides);
    const reply = await capture.handler({ ...ctx, args: 'Title | exact body', commandBody: '/notesave Title | exact body' });
    assert.match(reply.text, /saved and independently verified/);
    assert.equal(clients.length, 2); assert.equal(clients[0].closed, false); assert.equal(clients[1].closed, true);
    assert.deepEqual(tools.map(x => x[1]), ['search_notes', 'search_notes', 'capture_note', 'get_record']);
    assert.ok(tools.slice(2).every(x => x[0] === clients[1]));
  } finally { await read.close(); }
});

test('a lost capture acknowledgement is unconfirmed with one dispatch, exact replay requires another explicit command', async () => {
  const requests = []; let replay = false;
  const data = value => ({ structuredContent: { schema_version: 1, error: null, data: value } });
  const record = p => ({ id: 'note:' + 'a'.repeat(64), revision: 'b'.repeat(64), title: p.title, content: p.content,
    tags: p.tags, provenance: { instance_id: 'fictional-captor', source: p.provenance } });
  class Client {
    async connect() {} async close() {}
    async callTool(request) {
      if (request.name === 'capture_note') {
        requests.push(request.arguments);
        if (!replay) throw new Error('fictional committed acknowledgement lost');
        return data({ request_id: request.arguments.request_id, replayed: true, record: record(request.arguments) });
      }
      assert.equal(request.name, 'get_record'); return data({ hit_type: 'note', ...record(requests.at(-1)) });
    }
  }
  class Transport {}
  const command = createNotesSaveCommand({ captureActor: 'fictional-captor' }, { env: { GRAPHRAG_NOTES_TOKEN: 'fixture' }, loadSdk: async () => ({ Client, StreamableHTTPClientTransport: Transport }) });
  const ctx = { isAuthorizedSender: true, sessionKey: 'fixture', args: 'Title | exact body', commandBody: '/notesave Title | exact body', config: { mcp: { servers: { graphrag: { url: 'http://127.0.0.1:9991/mcp' } } } } };
  assert.match((await command.handler(ctx)).text, /outcome is unconfirmed/); assert.equal(requests.length, 1);
  replay = true; assert.match((await command.handler(ctx)).text, /replayed and independently verified/);
  assert.equal(requests.length, 2); assert.deepEqual(requests[0], requests[1]);
});
