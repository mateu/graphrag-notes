import { AsyncLocalStorage } from 'node:async_hooks';
import { createHash } from 'node:crypto';

export class PoolError extends Error {
  constructor(code) { super(code); this.code = code; }
}

const hash = value => createHash('sha256').update(value).digest('hex');

const MAX_RESPONSE_BYTES = 2 * 1024 * 1024;
export async function boundedResponse(response, signal) {
  const length = response.headers.get('content-length');
  // Fetch decodes compression while retaining the encoded Content-Length.
  const encoding = response.headers.get('content-encoding')?.trim().toLowerCase();
  const decodedLength = !encoding || encoding === 'identity';
  if (length !== null && (!/^\d+$/.test(length) || (decodedLength && Number(length) > MAX_RESPONSE_BYTES))) {
    await response.body?.cancel(); throw new PoolError('response');
  }
  const expected = length === null || !decodedLength ? null : Number(length);
  if (!response.body) {
    if (expected !== null && expected !== 0) throw new PoolError('response');
    return response;
  }
  const reader = response.body.getReader(), parts = []; let size = 0, abort;
  const cancelled = new Promise((_, reject) => {
    abort = () => { void reader.cancel().catch(() => {}); reject(signal.reason); };
    signal.addEventListener('abort', abort, { once: true });
    if (signal.aborted) abort();
  });
  try {
    while (true) {
      const { done, value } = await Promise.race([reader.read(), cancelled]);
      if (done) break;
      size += value.byteLength;
      if (size > MAX_RESPONSE_BYTES) { await reader.cancel(); throw new PoolError('response'); }
      parts.push(value);
    }
    if (expected !== null && expected !== size) throw new PoolError('response');
    const body = new Uint8Array(size); let offset = 0;
    for (const part of parts) { body.set(part, offset); offset += part.byteLength; }
    return new Response(size ? body : null, { status: response.status, statusText: response.statusText, headers: response.headers });
  } finally { signal.removeEventListener('abort', abort); reader.releaseLock(); }
}

function settings(config) {
  const options = { idle: config.poolIdleMs ?? 30000, lifetime: config.poolLifetimeMs ?? 300000,
    entries: config.poolMaxEntries ?? 4, inFlight: config.poolMaxInFlight ?? 4 };
  for (const [key, minimum, maximum] of [['idle', 1000, 300000], ['lifetime', 5000, 900000], ['entries', 1, 4], ['inFlight', 1, 4]]) {
    if (!Number.isInteger(options[key]) || options[key] < minimum || options[key] > maximum) throw new PoolError('configuration');
  }
  return options;
}

// A factory owns its pool. Nothing is written to disk; raw credentials are never
// used as map keys or diagnostics. Each HTTP request still carries its Bearer.
export class ReadConnectionPool {
  constructor({ loadSdk, fetch, now }) {
    this.loadSdk = loadSdk; this.fetch = fetch; this.now = now;
    this.entries = new Map(); this.closing = new Set(); this.closed = false; this.reconfiguring = false;
  }

  async acquire(connection, config, signal, timeoutMs) {
    if (this.closed) throw new PoolError('unavailable');
    signal.throwIfAborted();
    const limits = settings(config);
    const policy = JSON.stringify(limits);
    if (this.reconfiguring) throw new PoolError('busy');
    if (this.policy && this.policy !== policy) {
      this.reconfiguring = true;
      try { await this.retireAll(); }
      finally { this.reconfiguring = false; }
      if (this.closed) throw new PoolError('unavailable');
      signal.throwIfAborted();
    }
    this.policy = policy;
    const context = hash(JSON.stringify([config.serverName ?? 'graphrag', config.tokenEnv ?? 'GRAPHRAG_NOTES_TOKEN', config.sdkAnchor ?? null]));
    const key = hash(JSON.stringify([context, connection.endpoint.href, hash(connection.token)]));
    for (const entry of this.entries.values()) {
      if ((entry.context === context && entry.key !== key) || this.now() - entry.created >= entry.limits.lifetime ||
          (entry.active === 0 && this.now() - entry.lastUsed >= entry.limits.idle)) this.retire(entry);
    }
    let entry = this.entries.get(key);
    if (entry && entry.token !== connection.token) { this.retire(entry); throw new PoolError('credential'); }
    const reused = Boolean(entry);
    if (!entry) {
      if (this.entries.size + this.closing.size >= limits.entries) throw new PoolError('busy');
      entry = { key, context, limits, created: this.now(), lastUsed: this.now(), active: 0,
        abort: new AbortController(), calls: new AsyncLocalStorage(), retired: false, token: connection.token };
      this.entries.set(key, entry);
      entry.ageTimer = setTimeout(() => this.retire(entry), limits.lifetime);
      entry.ageTimer.unref?.();
      entry.ready = this.connect(entry, connection, config, Math.min(timeoutMs, 10000));
      // A caller can leave before initialization finishes. Always observe failure.
      entry.ready.catch(() => this.retire(entry));
    }
    if (entry.active >= limits.inFlight) { if (!reused) this.retire(entry); throw new PoolError('busy'); }
    entry.active++;
    clearTimeout(entry.idleTimer);
    let released = false, cancel;
    const release = () => {
      if (released) return;
      released = true; signal.removeEventListener('abort', abort);
      if (cancel) signal.removeEventListener('abort', cancel);
      entry.active--; entry.lastUsed = this.now();
      if (!entry.retired && entry.active === 0) {
        entry.idleTimer = setTimeout(() => this.retire(entry), limits.idle);
        entry.idleTimer.unref?.();
      }
    };
    const abort = () => this.retire(entry);
    signal.addEventListener('abort', abort, { once: true });
    if (signal.aborted) abort();
    try {
      const cancelled = new Promise((_, reject) => {
        if (signal.aborted) reject(new PoolError('timeout'));
        else { cancel = () => reject(new PoolError('timeout')); signal.addEventListener('abort', cancel, { once: true }); }
      });
      await Promise.race([entry.ready, cancelled]);
      signal.throwIfAborted();
      if (entry.retired) throw new PoolError('unavailable');
      const call = { signal, requests: 0, sent: false, invoked: false };
      return { reused, release, retire: () => this.retire(entry), failure: () => entry.failure,
        httpRequests: () => call.requests,
        call: (args, options) => entry.calls.run(call, async () => {
          if (args.name !== 'search_notes' || call.invoked) throw new PoolError('response');
          call.invoked = true;
          const current = entry.calls.getStore();
          const result = await entry.client.callTool(args, undefined, options);
          return { result, httpRequests: current.requests };
        }) };
    } catch (error) { release(); throw entry.failure ? new PoolError(entry.failure) : error; }
  }

  async connect(entry, connection, config, timeoutMs) {
    const timer = setTimeout(() => entry.abort.abort(new PoolError('timeout')), timeoutMs);
    let cancel;
    const cancelled = new Promise((_, reject) => {
      cancel = () => reject(entry.abort.signal.reason);
      entry.abort.signal.addEventListener('abort', cancel, { once: true });
    });
    const work = async () => {
      entry.abort.signal.throwIfAborted();
      const { Client, StreamableHTTPClientTransport } = await this.loadSdk(config);
      entry.abort.signal.throwIfAborted();
      entry.client = new Client({ name: 'graphrag-fast-notes', version: '0.5.0' }, { capabilities: {} });
      entry.client.onerror = error => {
        entry.failure ??= error instanceof PoolError ? error.code : 'unavailable';
        this.retire(entry);
      };
      entry.client.onclose = () => this.retire(entry);
      entry.transport = new StreamableHTTPClientTransport(connection.endpoint, {
        requestInit: { headers: { Authorization: `Bearer ${connection.token}` } },
        reconnectionOptions: { maxRetries: 0 },
        fetch: async (input, init = {}) => {
          const current = entry.calls.getStore();
          if (current) {
            // SDK reconnection options govern streams. Independently fence a
            // duplicate tool POST so SDK changes cannot silently replay a call.
            if (init.method === 'POST') {
              let message;
              try { message = JSON.parse(init.body); } catch { throw new PoolError('response'); }
              if (message.method === 'tools/call') {
                if (message.params?.name !== 'search_notes' || current.sent) throw new PoolError('response');
                current.sent = true;
              }
            }
            current.requests++;
          }
          const headers = new Headers(init.headers);
          headers.set('Authorization', `Bearer ${connection.token}`);
          const signal = AbortSignal.any([entry.abort.signal, ...(init.signal ? [init.signal] : []), ...(current ? [current.signal] : [])]);
          signal.throwIfAborted();
          const response = await this.fetch(input, { ...init, headers, redirect: 'error', signal });
          if (response.status === 401 || response.status === 403) {
            entry.failure = response.status === 401 ? 'unauthorized' : 'forbidden';
            throw new PoolError(entry.failure);
          }
          if (response.ok && ![202, 204].includes(response.status) &&
              response.headers.get('content-type')?.split(';')[0].trim().toLowerCase() !== 'application/json') {
            // GraphRAG's owning service uses stateless JSON responses. Refuse
            // a live SSE reader rather than releasing an unbounded stream.
            entry.failure = 'response';
            await response.body?.cancel();
            throw new PoolError('response');
          }
          try { return await boundedResponse(response, signal); }
          catch (error) { entry.failure ??= error instanceof PoolError ? error.code : 'unavailable'; throw error; }
        },
      });
      await entry.client.connect(entry.transport, { signal: entry.abort.signal, timeout: timeoutMs });
      entry.abort.signal.throwIfAborted();
    };
    entry.initializationWork = Promise.resolve().then(work);
    try { return await Promise.race([entry.initializationWork, cancelled]); }
    finally { clearTimeout(timer); entry.abort.signal.removeEventListener('abort', cancel); }
  }

  retire(entry) {
    if (entry.retired) return entry.closing;
    entry.retired = true; entry.abort.abort(new PoolError('unavailable'));
    clearTimeout(entry.ageTimer); clearTimeout(entry.idleTimer);
    this.entries.delete(entry.key);
    this.closing.add(entry);
    // Our fetches have already aborted. Bound SDK cleanup even if it is broken.
    // Keep a reservation until initialization AND cleanup actually settle;
    // a cleanup deadline does not grant permission to create unlimited clients.
    let timer;
    entry.cleanup = Promise.allSettled([
      entry.initializationWork,
      Promise.resolve().then(() => entry.client?.close()),
      Promise.resolve().then(() => entry.transport?.close()),
    ]).finally(() => { this.closing.delete(entry); entry.token = null; entry.calls.disable(); });
    entry.closing = Promise.race([
      entry.cleanup,
      new Promise(resolve => { timer = setTimeout(resolve, 1000); }),
    ]).finally(() => clearTimeout(timer));
    return entry.closing;
  }

  retireAll() { return Promise.all([...this.entries.values(), ...this.closing].map(entry => this.retire(entry))); }
  async close() { this.closed = true; await this.retireAll(); }
  start() { this.closed = false; }
}
