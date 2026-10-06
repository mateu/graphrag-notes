import { createRequire } from 'node:module';
import { isAbsolute } from 'node:path';
import { pathToFileURL } from 'node:url';
import { performance } from 'node:perf_hooks';
import { createHash, randomUUID } from 'node:crypto';
import { ReadConnectionPool, PoolError, boundedResponse } from './mcp-read-pool.mjs';
import { isDeepStrictEqual } from 'node:util';

const HELP = 'Use /notes QUERY for hybrid search with graph on. --hybrid and --graph also use hybrid search with graph on. --keyword uses keyword search with graph off; --no-graph uses hybrid search with graph off. --graph and --no-graph conflict; --keyword conflicts with --hybrid or --graph. Use -- before a literal query starting with --. Returns up to five notes with record IDs. Keyword search uses no model. Graph-enabled search has a separate longer deadline and never falls back automatically. Save an explicit note with /notesave Title | body.';
const DEFAULT_ENDPOINT = 'http://127.0.0.1:33156/mcp';
const DEFAULT_TOKEN_ENV = 'GRAPHRAG_NOTES_TOKEN';

class NotesError extends PoolError {
  constructor(code) { super(code); this.code = code; }
}

function boundedText(value, maximum) {
  const plain = String(value ?? '').replace(/[\u0000-\u001f\u007f-\u009f]/g, ' ')
    .replace(/\s+/g, ' ').trim().replace(/@/g, '＠');
  const chars = Array.from(plain);
  return (chars.length > maximum ? chars.slice(0, maximum - 1).join('') + '…' : plain)
    .replace(/[\\`*_{}\[\]<>]/g, '\\$&');
}

function connection(config, ctx, env) {
  const server = ctx.config?.mcp?.servers?.[config.serverName ?? 'graphrag'];
  let endpoint;
  try { endpoint = new URL(config.endpoint ?? server?.url ?? DEFAULT_ENDPOINT); }
  catch { throw new NotesError('configuration'); }
  const loopback = ['127.0.0.1', 'localhost', '[::1]'].includes(endpoint.hostname);
  if (endpoint.username || endpoint.password || endpoint.hash || endpoint.search ||
      !(endpoint.protocol === 'https:' || (endpoint.protocol === 'http:' && loopback))) {
    throw new NotesError('configuration');
  }
  const envName = config.tokenEnv ?? DEFAULT_TOKEN_ENV;
  if (!/^[A-Z_][A-Z0-9_]*$/.test(envName)) throw new NotesError('configuration');
  let token = env[envName];
  // Reuse only the selected MCP server's Bearer credential, never arbitrary headers.
  // An explicit different endpoint must have its own environment credential.
  if (!token && (!config.endpoint || config.endpoint === server?.url)) {
    const authorization = Object.entries(server?.headers ?? {})
      .find(([key]) => key.toLowerCase() === 'authorization')?.[1];
    const bearer = typeof authorization === 'string' && authorization.match(/^Bearer (.+)$/i)?.[1];
    const reference = bearer?.match(/^\$\{([A-Z_][A-Z0-9_]*)\}$/)?.[1];
    token = reference ? env[reference] : bearer;
  }
  if (typeof token !== 'string' || !token.trim() || /[\s\u0000-\u001f\u007f]/.test(token) || token.includes('${')) {
    throw new NotesError('credential');
  }
  return { endpoint, token };
}

export async function loadSdk(config = {}) {
  // Public MCP SDK exports only. A host anchor reuses OpenClaw's installed SDK
  // without depending on version-specific OpenClaw bundle internals.
  const anchor = config.sdkAnchor;
  const require = anchor ? (() => {
    if (!isAbsolute(anchor)) throw new NotesError('configuration');
    return createRequire(anchor);
  })() : createRequire(import.meta.url);
  try {
    const [{ Client }, { StreamableHTTPClientTransport }] = await Promise.all([
      import(pathToFileURL(require.resolve('@modelcontextprotocol/sdk/client/index.js')).href),
      import(pathToFileURL(require.resolve('@modelcontextprotocol/sdk/client/streamableHttp.js')).href),
    ]);
    return { Client, StreamableHTTPClientTransport };
  } catch { throw new NotesError('sdk'); }
}

function envelope(result) {
  const value = result?.structuredContent;
  if (result?.isError || value?.schema_version !== 1 || value?.error || !Array.isArray(value?.data?.records)) {
    const code = value?.error?.code;
    throw new NotesError(['forbidden', 'unauthorized', 'invalid_input', 'provider_unavailable', 'compatibility'].includes(code) ? code : 'response');
  }
  return value.data.records;
}

function formatResults(records, mode, graph, timings) {
  const head = `Notes · ${mode}${graph === 'off' ? '' : ` · graph ${graph}`} · ${Math.round(timings.total)} ms total (${Math.round(timings.connect)} ms connect, ${Math.round(timings.search)} ms search)`;
  if (!records.length) return `${head}\nNo matching notes.`;
  const lines = [head];
  for (const record of records.slice(0, 5)) {
    if (typeof record?.id !== 'string' || !record.id || Array.from(record.id).length > 512) throw new NotesError('response');
    const actor = record.provenance?.instance_id ? `mcp:${record.provenance.instance_id}` : 'unknown';
    const line = `${lines.length}. ${boundedText(record.title || 'Untitled', 80)}\n${boundedText(record.content, 220)}\nID: ${boundedText(record.id, 512)} · Actor: ${boundedText(actor, 96)}`;
    if (lines.join('\n\n').length + line.length + 2 > 4000) break;
    lines.push(line);
  }
  return lines.join('\n\n');
}

function failure(code, elapsed, timeoutMs, graph = 'off') {
  const message = {
    busy: 'Notes search is busy. Retry this explicit command after the current searches finish.',
    reconfigured: 'The notes connection changed during preparation. Retry this command explicitly.',
    credential: 'Notes search credential is unavailable. Ask the operator to load its environment variable.',
    configuration: 'Notes search configuration is invalid.',
    sdk: 'The MCP SDK dependency is unavailable. Ask the operator to check sdkAnchor.',
    timeout: graph !== 'off'
      ? `Graph-enabled search timed out after ${timeoutMs} ms. Try /notes --keyword QUERY or /notes --no-graph QUERY.`
      : `Notes search timed out after ${timeoutMs} ms. Check the service and SSH tunnel.`,
    forbidden: 'This notes credential does not have read access.',
    unauthorized: 'The notes credential was rejected.',
    invalid_input: 'The search query was rejected.',
    provider_unavailable: 'The semantic search provider is unavailable. Try /notes --keyword QUERY for keyword search.',
    compatibility: 'Semantic search configuration differs from the corpus. Try /notes --keyword QUERY for keyword search.',
    response: 'The notes service returned an unexpected response.',
    unavailable: 'Notes search is unavailable. Check the service and SSH tunnel.',
  }[code] ?? 'Notes search is unavailable. Check the service and SSH tunnel.';
  return { text: `${message} (${Math.round(elapsed)} ms)`, continueAgent: false };
}

export function parseNotesArguments(args, config = {}) {
  if (typeof args !== 'string') throw new NotesError('flags');
  let query = args.trim();
  if (!query || ['--help', 'help', '-h'].includes(query)) return null;
  const configuredGraph = config.defaultGraph ?? 'on';
  if (!['off', 'auto', 'on'].includes(configuredGraph)) throw new NotesError('configuration');
  let mode = 'hybrid', graph = configuredGraph;
  const seen = new Set();
  while (query.startsWith('--')) {
    const flag = query.match(/^\S+/)[0];
    query = query.slice(flag.length).trimStart();
    if (flag === '--') break;
    if (!['--hybrid', '--keyword', '--graph', '--no-graph'].includes(flag) || seen.has(flag)) throw new NotesError('flags');
    seen.add(flag);
    if (flag === '--keyword') { mode = 'keyword'; graph = 'off'; }
    else if (flag === '--graph') { mode = 'hybrid'; graph = 'on'; }
    else if (flag === '--no-graph') { mode = 'hybrid'; graph = 'off'; }
    else mode = 'hybrid';
  }
  if (seen.has('--graph') && seen.has('--no-graph')) throw new NotesError('flags');
  if (seen.has('--keyword') && (seen.has('--hybrid') || seen.has('--graph'))) throw new NotesError('flags');
  if (seen.has('--keyword')) { mode = 'keyword'; graph = 'off'; }
  return query ? { query, mode, graph } : null;
}

export function createFreshNotesCommand(config = {}, overrides = {}) {
  const sdkLoader = overrides.loadSdk ?? loadSdk;
  const env = overrides.env ?? process.env;
  const fetchImpl = overrides.fetch ?? globalThis.fetch;
  const clock = overrides.now ?? (() => performance.now());
  return {
    name: 'notes', description: 'Search shared notes directly without an agent model',
    acceptsArgs: true, requireAuth: true,
    handler: async (ctx) => {
      if (ctx.isAuthorizedSender !== true) return { text: 'This command requires authorization.', continueAgent: false };
      const current = { ...(ctx.config?.plugins?.entries?.['graphrag-fast-notes']?.config ?? config) };
      let parsed;
      try { parsed = parseNotesArguments(ctx.args ?? '', current); }
      catch { return { text: `Invalid or conflicting search flags. ${HELP}`, continueAgent: false }; }
      if (!parsed) return { text: HELP, continueAgent: false };
      const { query, mode, graph } = parsed;
      if (Array.from(query).length > 1024) return { text: 'Search query must contain at most 1024 characters.', continueAgent: false };
      const started = clock();
      const timeoutMs = graph !== 'off' ? (current.graphTimeoutMs ?? 60000) : (current.timeoutMs ?? 5000);
      if (!Number.isInteger(timeoutMs) || timeoutMs < 250 || timeoutMs > (graph !== 'off' ? 120000 : 30000)) return failure('configuration', clock() - started, timeoutMs, graph);
      const abort = new AbortController();
      const timer = setTimeout(() => abort.abort(new NotesError('timeout')), timeoutMs);
      let client, connectMs = 0, searchMs = 0, outcome = 'unavailable';
      try {
        const { endpoint, token } = connection(current, ctx, env);
        const revalidate = () => {
          const latest = ctx.config?.plugins?.entries?.['graphrag-fast-notes']?.config ?? config;
          const selected = connection(latest, ctx, env);
          if (selected.endpoint.href !== endpoint.href || selected.token !== token || latest.sdkAnchor !== current.sdkAnchor) throw new NotesError('reconfigured');
          if (ctx.isAuthorizedSender !== true) throw new NotesError('forbidden');
        };
        const deadline = new Promise((_, reject) => abort.signal.addEventListener('abort', () => reject(new NotesError('timeout')), { once: true }));
        const work = async () => {
          const { Client, StreamableHTTPClientTransport } = await sdkLoader(current);
          abort.signal.throwIfAborted();
          revalidate();
          client = new Client({ name: 'graphrag-fast-notes', version: '0.5.0' }, { capabilities: {} });
          // Do not log SDK errors, URLs, headers, queries, or returned corpus text.
          client.onerror = () => {};
          const transport = new StreamableHTTPClientTransport(endpoint, {
            requestInit: { headers: { Authorization: `Bearer ${token}` } },
            reconnectionOptions: { maxRetries: 0 },
            fetch: async (input, init = {}) => {
              revalidate();
              const signal = init.signal ? AbortSignal.any([init.signal, abort.signal]) : abort.signal;
              signal.throwIfAborted();
              const response = await fetchImpl(input, { ...init, redirect: 'error', signal });
              if (response.ok && ![202, 204].includes(response.status) &&
                  response.headers.get('content-type')?.split(';')[0].trim().toLowerCase() !== 'application/json') {
                await response.body?.cancel(); throw new NotesError('response');
              }
              return boundedResponse(response, signal);
            },
          });
          await client.connect(transport, { signal: abort.signal, timeout: timeoutMs });
          abort.signal.throwIfAborted();
          revalidate();
          const connected = clock();
          connectMs = connected - started;
          const result = await client.callTool({ name: 'search_notes', arguments: { query, mode, graph, scope: 'notes', limit: 5 } }, undefined, { signal: abort.signal, timeout: timeoutMs });
          abort.signal.throwIfAborted();
          const finished = clock();
          searchMs = finished - connected; outcome = 'ok';
          return { text: formatResults(envelope(result), mode, graph, { total: finished - started, connect: connected - started, search: finished - connected }), continueAgent: false };
        };
        return await Promise.race([work(), deadline]);
      } catch (error) {
        outcome = abort.signal.aborted ? 'timeout' : error instanceof PoolError ? error.code : 'unavailable';
        return failure(outcome, clock() - started, timeoutMs, graph);
      } finally {
        const cleanupStarted = clock();
        clearTimeout(timer);
        abort.abort();
        if (client) {
          let closeTimer;
          try { await Promise.race([Promise.resolve().then(() => client.close()).catch(() => {}), new Promise(resolve => { closeTimer = setTimeout(resolve, 1000); })]); }
          finally { clearTimeout(closeTimer); }
        }
        reportTiming(overrides, ctx, { mode, graph, outcome, reused: false, connect_ms: connectMs, search_ms: searchMs, cleanup_ms: clock() - cleanupStarted, handler_ms: clock() - started, handler_start_epoch_ms: performance.timeOrigin + started, handler_finish_epoch_ms: performance.timeOrigin + clock() });
      }
    },
  };
}

function reportTiming(overrides, ctx, values) {
  // No URLs, credentials, queries, corpus text or raw sender/session identifiers.
  const timing = { schema_version: 1, kind: 'graphrag-notes-command', command_id: randomUUID(),
    session_sha256: typeof ctx.sessionKey === 'string' ? createHash('sha256').update(ctx.sessionKey).digest('hex') : null,
    ...values };
  try { overrides.onDiagnostics?.(Object.freeze(timing)); } catch { /* diagnostics cannot alter the command */ }
}

export function createNotesCommand(config = {}, overrides = {}) {
  const env = overrides.env ?? process.env;
  const clock = overrides.now ?? (() => performance.now());
  const pool = new ReadConnectionPool({ loadSdk: overrides.loadSdk ?? loadSdk,
    fetch: overrides.fetch ?? globalThis.fetch, now: clock });
  return {
    name: 'notes', description: 'Search shared notes directly without an agent model', acceptsArgs: true, requireAuth: true,
    close: () => pool.close(), start: () => pool.start(),
    handler: async ctx => {
      const current = ctx.config?.plugins?.entries?.['graphrag-fast-notes']?.config ?? config;
      if (ctx.isAuthorizedSender !== true) return { text: 'This command requires authorization.', continueAgent: false };
      let parsed;
      try { parsed = parseNotesArguments(ctx.args ?? '', current); }
      catch { return { text: `Invalid or conflicting search flags. ${HELP}`, continueAgent: false }; }
      if (!parsed) return { text: HELP, continueAgent: false };
      const { query, mode, graph } = parsed;
      if (Array.from(query).length > 1024) return { text: 'Search query must contain at most 1024 characters.', continueAgent: false };
      const started = clock();
      if (pool.closed) return failure('unavailable', clock() - started, current.timeoutMs ?? 5000, graph);
      if (current.reuseConnections === false || overrides.allowReuse === false || env.GRAPHRAG_NOTES_DISABLE_REUSE === '1') {
        await pool.retireAll();
        if (pool.closed) return failure('unavailable', clock() - started, current.timeoutMs ?? 5000, graph);
        return createFreshNotesCommand(current, overrides).handler(ctx);
      }
      const timeoutMs = graph !== 'off' ? (current.graphTimeoutMs ?? 60000) : (current.timeoutMs ?? 5000);
      if (!Number.isInteger(timeoutMs) || timeoutMs < 250 || timeoutMs > (graph !== 'off' ? 120000 : 30000)) return failure('configuration', clock() - started, timeoutMs, graph);
      const abort = new AbortController(); const timer = setTimeout(() => abort.abort(new NotesError('timeout')), timeoutMs);
      let lease, connectMs = 0, searchMs = 0, outcome = 'unavailable', httpRequests = 0;
      try {
        const selected = connection(current, ctx, env);
        lease = await pool.acquire(selected, current, abort.signal, timeoutMs);
        const latest = connection(current, ctx, env);
        if (latest.endpoint.href !== selected.endpoint.href || latest.token !== selected.token) {
          lease.retire(); throw new NotesError('reconfigured');
        }
        const connected = clock(); connectMs = connected - started;
        const cancelled = new Promise((_, reject) => abort.signal.addEventListener('abort', () => reject(new NotesError('timeout')), { once: true }));
        const work = lease.call({ name: 'search_notes', arguments: { query, mode, graph, scope: 'notes', limit: 5 } }, { signal: abort.signal, timeout: timeoutMs });
        const { result, httpRequests: count } = await Promise.race([work, cancelled]);
        abort.signal.throwIfAborted();
        const finished = clock(); searchMs = finished - connected; httpRequests = count;
        const text = formatResults(envelope(result), mode, graph, { total: finished - started, connect: connectMs, search: searchMs });
        outcome = 'ok'; return { text, continueAgent: false };
      } catch (error) {
        outcome = abort.signal.aborted ? 'timeout' : (lease?.failure() ?? (error instanceof PoolError ? error.code : 'unavailable'));
        if (['timeout', 'unauthorized', 'forbidden', 'unavailable', 'response'].includes(outcome)) lease?.retire();
        return failure(outcome, clock() - started, timeoutMs, graph);
      } finally {
        clearTimeout(timer); lease?.release();
        reportTiming(overrides, ctx, { mode, graph, outcome, reused: lease?.reused ?? false,
          connect_ms: connectMs, search_ms: searchMs, handler_ms: clock() - started, search_http_requests: lease?.httpRequests() ?? httpRequests,
          handler_start_epoch_ms: performance.timeOrigin + started, handler_finish_epoch_ms: performance.timeOrigin + clock() });
      }
    },
  };
}

const SAVE_HELP = 'Use /notesave Title | single-line body to save a new note directly. Title and outer body whitespace are trimmed; inner text is preserved. Multiline bodies are refused. Title: 1–512 characters. Keep total arguments within OpenClaw’s 4096 UTF-16 code-unit limit (emoji count as two). Capture uses server embedding/extraction models. Retry the exact same command from the same sender and conversation to reuse its durable receipt.';
const digest = value => createHash('sha256').update(value).digest('hex');
const noteId = value => typeof value === 'string' && /^note:[0-9a-f]{64}$/.test(value);
const revisionId = value => typeof value === 'string' && /^[0-9a-f]{64}$/.test(value);

export function captureArguments(ctx) {
  // The installed dispatcher sanitizes and truncates ctx.args. The full command
  // body lets us reject that change before submitting any shortened note.
  if (/[\r\n\u2028\u2029]/u.test(ctx.commandBody ?? '') || /[\r\n\u2028\u2029]/u.test(ctx.args ?? '')) throw new NotesError('multiline');
  const match = typeof ctx.commandBody === 'string' && ctx.commandBody.trim().match(/^\/\s*notesave(?:\s+([\s\S]*))?$/i);
  const raw = match ? (match[1] ?? '').trim() : null;
  if (raw === null || raw.length > 4096 || raw !== (ctx.args ?? '').trim()) throw new NotesError('command_changed');
  return raw;
}

export function capturePayload(args, ctx, actor = 'openclaw-clawd-daily') {
  if (typeof args !== 'string' || !args.trim() || ['help', '--help', '-h'].includes(args.trim())) return null;
  if (/[\r\n\u2028\u2029]/u.test(args)) throw new NotesError('multiline');
  const separator = args.indexOf('|');
  if (separator < 0) throw new NotesError('save_input');
  const title = args.slice(0, separator).trim();
  const content = args.slice(separator + 1).trim();
  if (!title || Array.from(title).length > 512 || /[\u0000-\u001f\u007f-\u009f]/u.test(title) ||
      !content || Buffer.byteLength(content, 'utf8') > 65536 || content.includes('\0') ||
      /[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/u.test(title + content)) {
    throw new NotesError('save_input');
  }
  if (typeof actor !== 'string' || !actor.trim() || actor.trim() !== actor || Array.from(actor).length > 128 ||
      Buffer.byteLength(actor) > 256 || /[\u0000-\u001f\u007f-\u009f]/u.test(actor)) throw new NotesError('configuration');
  const stable = value => typeof value === 'string' && Boolean(value.trim());
  const senderId = ctx.senderId ?? ctx.from;
  const target = stable(ctx.to) || stable(ctx.threadParentId) || stable(ctx.messageThreadId) || Number.isFinite(ctx.messageThreadId);
  if (!stable(ctx.sessionKey) && !(stable(senderId) && target)) throw new NotesError('identity');
  // Sender context is descriptive only. The service credential sets trusted
  // provenance.instance_id; the configured actor is checked against that value.
  // Do not use ephemeral session IDs: retrying after reload must retain identity.
  const sender = JSON.stringify([ctx.channel ?? null, ctx.channelId ?? null, ctx.accountId ?? null,
    ctx.senderId ?? ctx.from ?? null, ctx.agentId ?? null, ctx.sessionKey ?? null, ctx.to ?? null,
    ctx.messageThreadId ?? null, ctx.threadParentId ?? null]);
  const payload = { content, title, tags: [], provenance: { uri: null, label: 'OpenClaw direct note',
    metadata: { interface: 'notesave', sender_sha256: digest(sender) } } };
  return { ...payload, request_id: 'notesave-' + digest(JSON.stringify(['notesave-v1', actor, sender, payload])) };
}

function captureData(result) {
  const value = result?.structuredContent;
  if (result?.isError || value?.schema_version !== 1 || value?.error || !value?.data || typeof value.data !== 'object') {
    const code = value?.error?.code;
    throw new NotesError(['forbidden', 'unauthorized', 'invalid_input', 'provider_unavailable', 'compatibility'].includes(code) ? code : 'response');
  }
  return value.data;
}

function verifyReceipt(data, payload, actor) {
  const record = data.record;
  if (data.request_id !== payload.request_id || typeof data.replayed !== 'boolean' ||
      !noteId(record?.id) || !revisionId(record?.revision) || record.title !== payload.title ||
      record.content !== payload.content || !isDeepStrictEqual(record.tags, payload.tags) ||
      record.provenance?.instance_id !== actor || !isDeepStrictEqual(record.provenance?.source, payload.provenance)) {
    throw new NotesError('receipt');
  }
  return record;
}

function verifyReadback(data, record) {
  if (data.hit_type !== 'note' || data.id !== record.id || data.revision !== record.revision ||
      data.title !== record.title || data.content !== record.content || !isDeepStrictEqual(data.provenance, record.provenance)) {
    throw new NotesError('readback');
  }
}

export function createNotesSaveCommand(config = {}, overrides = {}) {
  const sdkLoader = overrides.loadSdk ?? loadSdk;
  const env = overrides.env ?? process.env;
  const fetchImpl = overrides.fetch ?? globalThis.fetch;
  const clock = overrides.now ?? (() => performance.now());
  return {
    name: 'notesave', description: 'Save an explicit note directly and verify its durable receipt',
    acceptsArgs: true, requireAuth: true, requiredScopes: ['operator.write'],
    handler: async ctx => {
      if (ctx.isAuthorizedSender !== true) return { text: 'This command requires authorization.', continueAgent: false };
      const current = { ...(ctx.config?.plugins?.entries?.['graphrag-fast-notes']?.config ?? config) };
      let payload;
      const actor = current.captureActor ?? 'openclaw-clawd-daily';
      try { payload = capturePayload(captureArguments(ctx), ctx, actor); }
      catch (error) { return { text: error?.code === 'multiline' ? 'No capture submitted: /notesave accepts a single-line body. OpenClaw normalizes multiline commands before plugins receive them. Use a deliberate single-line note or upload longer Markdown separately.' : error?.code === 'command_changed' ? 'No capture submitted: OpenClaw shortened or altered the command arguments. Use a shorter note (at most 4096 UTF-16 code units, emoji count as two) without control characters.' : error?.code === 'identity' ? 'No capture submitted: a stable sender and conversation identity is unavailable. Retry from an authenticated Control UI conversation.' : SAVE_HELP, continueAgent: false }; }
      if (!payload) return { text: SAVE_HELP, continueAgent: false };
      const started = clock();
      const timeoutMs = current.captureTimeoutMs ?? 30000;
      if (!Number.isInteger(timeoutMs) || timeoutMs < 15000 || timeoutMs > 30000) {
        return { text: 'Notes capture configuration is invalid.', continueAgent: false };
      }
      const abort = new AbortController();
      const timer = setTimeout(() => abort.abort(new NotesError('timeout')), timeoutMs);
      let client, submitted = false, receiptRecord;
      const toolPosts = new Set();
      try {
        if (!current.endpoint && !ctx.config?.mcp?.servers?.[current.serverName ?? 'graphrag']?.url) throw new NotesError('configuration');
        const { endpoint, token } = connection(current, ctx, env);
        const revalidate = () => {
          const latest = ctx.config?.plugins?.entries?.['graphrag-fast-notes']?.config ?? config;
          const selected = connection(latest, ctx, env);
          if (selected.endpoint.href !== endpoint.href || selected.token !== token ||
              (latest.captureActor ?? 'openclaw-clawd-daily') !== actor || latest.sdkAnchor !== current.sdkAnchor) throw new NotesError('reconfigured');
          if (ctx.isAuthorizedSender !== true) throw new NotesError('forbidden');
        };
        const deadline = new Promise((_, reject) => abort.signal.addEventListener('abort', () => reject(new NotesError('timeout')), { once: true }));
        const work = async () => {
          const { Client, StreamableHTTPClientTransport } = await sdkLoader(current);
          abort.signal.throwIfAborted();
          client = new Client({ name: 'graphrag-fast-notes', version: '0.5.0' }, { capabilities: {} });
          client.onerror = () => {};
          const transport = new StreamableHTTPClientTransport(endpoint, {
            requestInit: { headers: { Authorization: `Bearer ${token}` } }, reconnectionOptions: { maxRetries: 0 },
            fetch: async (input, init = {}) => {
              revalidate();
              const target = new URL(input instanceof Request ? input.url : input);
              if (target.href !== endpoint.href) throw new NotesError('response');
              if (init.method === 'POST') {
                let message; try { message = JSON.parse(init.body); } catch { throw new NotesError('response'); }
                if (message.method === 'tools/call') {
                  const name = message.params?.name;
                  const expected = name === 'capture_note' ? payload : name === 'get_record' && receiptRecord
                    ? { id: receiptRecord.id, revision: receiptRecord.revision, neighbors: 0 } : null;
                  if (!expected || toolPosts.has(name) || !isDeepStrictEqual(message.params.arguments, expected)) throw new NotesError('response');
                  // Stream retry options do not fence expired-session SDK replay.
                  // Consume the permission before fetch; never submit it again.
                  toolPosts.add(name);
                }
              }
              const headers = new Headers(init.headers); headers.set('Authorization', `Bearer ${token}`);
              const signal = init.signal ? AbortSignal.any([init.signal, abort.signal]) : abort.signal;
              signal.throwIfAborted();
              const response = await fetchImpl(input, { ...init, headers, redirect: 'error', signal });
              return await boundedResponse(response, signal);
            },
          });
          await client.connect(transport, { signal: abort.signal, timeout: timeoutMs });
          abort.signal.throwIfAborted();
          await ctx.assertOwnerCurrent?.();
          revalidate();
          abort.signal.throwIfAborted();
          submitted = true;
          const receipt = captureData(await client.callTool({ name: 'capture_note', arguments: payload }, undefined, { signal: abort.signal, timeout: timeoutMs }));
          receiptRecord = verifyReceipt(receipt, payload, actor);
          abort.signal.throwIfAborted();
          const readback = captureData(await client.callTool({ name: 'get_record', arguments: { id: receiptRecord.id, revision: receiptRecord.revision, neighbors: 0 } }, undefined, { signal: abort.signal, timeout: timeoutMs }));
          abort.signal.throwIfAborted();
          verifyReadback(readback, receiptRecord);
          return { text: `Note ${receipt.replayed ? 'replayed' : 'saved'} and independently verified · ${Math.round(clock() - started)} ms\n${boundedText(receiptRecord.title, 100)}\nID: ${receiptRecord.id}\nRevision: ${receiptRecord.revision}\nActor: ${boundedText(receiptRecord.provenance.instance_id, 128)}\nRequest: ${payload.request_id}`, continueAgent: false };
        };
        return await Promise.race([work(), deadline]);
      } catch (error) {
        const detail = submitted
          ? `Capture outcome is unconfirmed${receiptRecord ? `; receipt record: ${receiptRecord.id}` : ''}. It may already be saved. Retry the exact same /notesave command from the same sender and conversation to reuse its receipt.`
          : 'Capture was not submitted. Check the MCP service, SSH tunnel, credential, and capture permission, then retry the same command.';
        return { text: `${detail}\nRequest: ${payload.request_id} · ${Math.round(clock() - started)} ms`, continueAgent: false };
      } finally {
        clearTimeout(timer); abort.abort();
        // Session close is best effort and itself bounded, including a broken SDK.
        if (client) {
          let closeTimer;
          try { await Promise.race([Promise.resolve().then(() => client.close()).catch(() => {}), new Promise(resolve => { closeTimer = setTimeout(resolve, 1000); })]); }
          finally { clearTimeout(closeTimer); }
        }
      }
    },
  };
}

export default {
  id: 'graphrag-fast-notes', name: 'GraphRAG Fast Notes',
  description: 'Authorized direct shared-note search and explicit capture using MCP',
  register(api) {
    const command = createNotesCommand(api.pluginConfig ?? {}, {
      allowReuse: typeof api.registerService === 'function',
      onDiagnostics: process.env.GRAPHRAG_NOTES_PROFILE_TIMINGS === '1'
        ? timing => api.logger?.info?.(JSON.stringify(timing)) : undefined,
    });
    const { close, start, ...definition } = command;
    api.registerCommand(definition);
    api.registerService?.({ id: 'graphrag-fast-notes-connections',
      reload: { configPrefixes: ['plugins.entries.graphrag-fast-notes.config', 'mcp.servers'] }, start, stop: close });
    api.registerCommand(createNotesSaveCommand(api.pluginConfig ?? {}));
  },
};
