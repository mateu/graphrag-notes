import fs from 'node:fs';
import { pathToFileURL } from 'node:url';
import path from 'node:path';

const input = JSON.parse(fs.readFileSync(0, 'utf8'));
const { createSessionMcpRuntime } = await import(pathToFileURL(input.runtime_entry).href);
const version = JSON.parse(fs.readFileSync(path.join(input.install_root, 'package.json'), 'utf8')).version;
const runtimes = [];
function require(condition, message) { if (!condition) throw new Error(message); }
require(typeof createSessionMcpRuntime === 'function', 'Installed OpenClaw runtime entry point changed; inspect its installed version');
function make(name) {
  const workspace = input.directory + '/' + name;
  fs.mkdirSync(workspace, { recursive: true, mode: 0o700 });
  const runtime = createSessionMcpRuntime({ sessionId: name, sessionKey: name,
    workspaceDir: workspace, agentDir: workspace,
    cfg: { mcp: { servers: { graphrag: { url: input.url, transport: 'streamable-http',
      headers: { Authorization: 'Bearer ' + input.tokens[name] }, requestTimeoutMs: 300000,
      supportsParallelToolCalls: true } } } } });
  runtimes.push(runtime);
  return runtime;
}
async function tools(runtime) {
  await runtime.getCatalog();
  const catalog = await runtime.listTools('graphrag');
  return catalog.tools.map(tool => tool.name).sort();
}
function envelope(result) {
  return result.structuredContent;
}
async function call(runtime, name, args) {
  const result = await runtime.callTool('graphrag', name, args);
  const value = envelope(result);
  require(result.isError !== true && value?.schema_version === 1 && value.error === null,
    'Native OpenClaw tool call did not return a successful version-one envelope');
  return value.data;
}
const search = { query: 'nativeclientatlas', mode: 'keyword', scope: 'all', graph: 'off', limit: 10 };
try {
  if (input.phase === 'calls') {
    const runtime = make(input.instance);
    const names = await tools(runtime);
    const results = [];
    for (const request of input.calls) {
      require(names.includes(request.tool), 'Requested native OpenClaw tool was not advertised');
      const result = await runtime.callTool('graphrag', request.tool, request.arguments);
      const value = envelope(result);
      require(value?.schema_version === 1, 'Native OpenClaw result lost its version-one envelope');
      if (request.expect_error) {
        require(result.isError === true && value.data === null && value.error?.code === request.expect_error,
          'Native OpenClaw call did not return the expected categorized service error');
        results.push({ error_code: value.error.code, retryable: value.error.retryable });
      } else {
        require(result.isError !== true && value.error === null, 'Native OpenClaw extended call failed');
        results.push(value.data);
      }
    }
    console.log('HARNESS_RESULT=' + JSON.stringify({ version, node_version: process.version,
      catalog: names, exercised_tools: [...new Set(input.calls.map(request => request.tool))].sort(), results }));
  } else if (input.phase === 'read-only') {
    const observer = make('observer');
    const names = await tools(observer);
    require(names.includes('search_notes') && !names.includes('capture_note'), 'Read-only native catalog did not restrict capture');
    const found = await call(observer, 'search_notes', search);
    require(found.records.length === 3, 'Read-only principal could not read shared captures');
    let denied = false;
    try {
      const result = await observer.callTool('graphrag', 'capture_note', input.capture);
      denied = result.isError === true || envelope(result)?.error != null;
    } catch { denied = true; }
    require(denied, 'Read-only native principal unexpectedly captured a note');
    console.log('HARNESS_RESULT=' + JSON.stringify({ version, node_version: process.version, catalog: names, read_succeeded: true,
      capture_not_advertised: true, native_mutation_denied: true,
      limitation: 'The native runtime may refuse a tool outside its catalog; service-side enforcement is separately tested.' }));
  } else if (input.phase === 'revoked') {
    const control = make('openclaw-a');
    await tools(control);
    require((await call(control, 'search_notes', search)).records.length === 3,
      'Valid native control failed during revoked-credential check');
    const observer = make('observer');
    let denied = false;
    try {
      const catalog = await observer.getCatalog();
      // This installed OpenClaw redacts HTTP status/body in diagnostics, then
      // reports "not connected" on a call. Retain that precise evidence and
      // a same-endpoint valid control instead of claiming an observed 401.
      const failedCatalog = !catalog.servers?.graphrag && catalog.diagnostics?.length > 0;
      try {
        const result = await observer.callTool('graphrag', 'search_notes', search);
        denied = result.isError === true && /unauthori[sz]ed|authenticat|forbidden/i.test(envelope(result)?.error?.code ?? '');
      } catch { denied = failedCatalog; }
    } catch (error) {
      denied = /401|unauthori[sz]ed|authenticat|forbidden/i.test(String(error));
    }
    require(denied, 'Revoked native credential still connected or called the service');
    console.log('HARNESS_RESULT=' + JSON.stringify({ version, node_version: process.version,
      revoked_credential_connection_denied: true, valid_control_read_succeeded: true,
      limitation: 'Installed OpenClaw redacts HTTP status/body; native evidence proves denial against a healthy control, exact HTTP status is separately tested.' }));
  } else {
  const a = make('openclaw-a');
  const names = await tools(a);
  require(['build_context', 'capture_note', 'get_record', 'search_notes'].every(name => names.includes(name)), 'OpenClaw required catalog mismatch');
  if (input.phase === 'replay') {
    const replay = await call(a, 'capture_note', input.capture);
    require(replay.replayed === true && JSON.stringify(replay.record) === JSON.stringify(input.original),
      'Restarted service did not preserve authoritative capture receipt');
    const found = await call(a, 'search_notes', search);
    require(found.records.length === 3, 'Restarted service corpus count changed');
    console.log('HARNESS_RESULT=' + JSON.stringify({ version, node_version: process.version, catalog: names, replay_after_restart: true,
      note_count: found.records.length, record: replay.record }));
  } else {
    const b = make('openclaw-b');
    require((await tools(b)).join(',') === names.join(','), 'Second OpenClaw catalog mismatch');
    const first = await call(a, 'capture_note', input.capture);
    const replay = await call(a, 'capture_note', input.capture);
    require(first.replayed === false && replay.replayed === true, 'OpenClaw same-ID retry failed');
    require(JSON.stringify(first.record) === JSON.stringify(replay.record), 'OpenClaw receipt changed on replay');
    require(first.record.provenance.instance_id === 'openclaw-a', 'OpenClaw A principal was not trusted');
    const found = await call(b, 'search_notes', search);
    require(found.records.some(record => record.id === first.record.id), 'OpenClaw B did not find A capture');
    const inspected = await call(b, 'get_record', { id: first.record.id, revision: first.record.revision, neighbors: 0 });
    require(inspected.content === input.capture.content && inspected.provenance.instance_id === 'openclaw-a',
      'OpenClaw B inspection lost A content or provenance');
    const independent = await call(b, 'capture_note', input.capture);
    require(independent.replayed === false && independent.record.id !== first.record.id &&
      independent.record.provenance.instance_id === 'openclaw-b', 'OpenClaw principal namespaces collided');
    const context = await call(b, 'build_context', { query: 'nativeclientatlas', scope: 'all', graph: 'off',
      max_chunks: 3, max_total_tokens: 1000, max_chunk_tokens: 200 });
    require(context.chunks.length > 0 && context.total_tokens <= 1000 &&
      context.chunks.every(chunk => chunk.id.startsWith('note:') && chunk.citation > 0) &&
      context.rendered_context.includes('[C1]'), 'OpenClaw native context lost citations or budget');
    console.log('HARNESS_RESULT=' + JSON.stringify({ version, node_version: process.version, catalog: names,
      native_runtime_api: 'createSessionMcpRuntime/getCatalog/callTool',
      openclaw_b_read_openclaw_a: true, same_instance_replay: true,
      native_context_citations: true, native_context_tokens: context.total_tokens,
      independent_principals: ['openclaw-a', 'openclaw-b'], record: first.record,
      second_record_id: independent.record.id }));
  }
  }
} finally {
  for (const runtime of runtimes) await runtime.dispose();
}
