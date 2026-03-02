import { groqServices } from './services/groq';
import { cerebrasServices } from './services/cerebras';
import { openrouterFreeServices, openrouterPaidServices } from './services/openrouter';
import { mistralService } from './services/mistral';
import { codestralService } from './services/codestral';
import { geminiServices } from './services/gemini';
import { cohereService } from './services/cohere';
import { nvidiaService } from './services/nvidia';
import type { AIService, ChatRequest } from './types';

// ─── Service pool ─────────────────────────────────────────────────────────────

/** Free models — included in automatic rotation */
const freeServices: AIService[] = [
  ...groqServices,            // dynamic (7 with tools)
  ...openrouterFreeServices,  // dynamic free (auto-discovered)
  ...cerebrasServices,        // 3 models
  ...geminiServices,          // 7 models (Gemini/Gemma)
  mistralService,             // tools ✅
  codestralService,           // tools ✅
  cohereService,
  nvidiaService,              // tools ✅
];

/** Paid OpenRouter models — only used when explicitly requested by name */
const paidServices: AIService[] = [
  ...openrouterPaidServices,
];

// ─── State tracking ──────────────────────────────────────────────────────────

const MIN_MS = 60 * 1_000;
const HOUR_MS = 60 * MIN_MS;

interface ServiceState {
  service: AIService;
  cooldownUntil: number;
  disabled: boolean;
  /** If true, excluded from auto-routing pools — only accessible by explicit name */
  paidOnly: boolean;
}

const states: ServiceState[] = [
  ...freeServices.map(s => ({ service: s, cooldownUntil: 0, disabled: false, paidOnly: false })),
  ...paidServices.map(s => ({ service: s, cooldownUntil: 0, disabled: false, paidOnly: true })),
];

// ─── Routing pools ──────────────────────────────────────────────────────────

/**
 * AGENT_MODELS: comma-separated list of model names to use exclusively
 * when the request includes tools (agent/agentic mode).
 *
 * Example in .env:
 *   AGENT_MODELS=Groq/llama-3.3-70b-versatile,Mistral (La Plateforme),Cerebras/llama-3.3-70b
 *
 * If not set, the router uses all models that support tools.
 */
const AGENT_MODEL_NAMES: string[] = (process.env.AGENT_MODELS ?? '')
  .split(',')
  .map(s => s.trim())
  .filter(Boolean);

/** Returns the sub-pool for the given mode. Paid-only models are always excluded. */
function getPool(requireTools: boolean): ServiceState[] {
  if (requireTools) {
    // Agent pool: use explicit list if configured, else all tool-supporting models
    return AGENT_MODEL_NAMES.length > 0
      ? states.filter(s => AGENT_MODEL_NAMES.includes(s.service.name) && s.service.supportsTools && !s.disabled && !s.paidOnly)
      : states.filter(s => s.service.supportsTools && !s.disabled && !s.paidOnly);
  }
  // Chat pool: all non-disabled, non-paid models
  return states.filter(s => !s.disabled && !s.paidOnly);
}

/** Separate sticky indices for each pool. */
let preferredChat = 0;
let preferredAgent = 0;

function getService(requireTools: boolean): ServiceState {
  const now = Date.now();
  const pool = getPool(requireTools);

  if (pool.length === 0) {
    // Fallback: any non-disabled service
    return states.find(s => !s.disabled) ?? states[0]!;
  }

  const pref = requireTools ? preferredAgent : preferredChat;

  for (let i = 0; i < pool.length; i++) {
    const s = pool[(pref + i) % pool.length]!;
    if (s.cooldownUntil <= now) {
      const next = (pref + i) % pool.length;
      if (requireTools) preferredAgent = next;
      else preferredChat = next;
      return s;
    }
  }

  // All in cooldown — pick soonest available
  return pool.reduce((a, b) => (a.cooldownUntil < b.cooldownUntil ? a : b))!;
}

function handleServiceError(state: ServiceState, err: any): void {
  const status: number = err?.status ?? err?.statusCode ?? err?.error?.status ?? 0;
  const name = state.service.name;
  const idx = states.indexOf(state);

  if (status === 429) {
    state.cooldownUntil = Date.now() + MIN_MS;
    console.warn(`[${name}] Rate limited → cooldown 1 min`);
  } else if (status === 402) {
    state.cooldownUntil = Date.now() + HOUR_MS;
    console.warn(`[${name}] Quota exceeded → cooldown 1 h`);
  } else if (status === 413) {
    // Payload too large — model can't handle this context size.
    // Cool down for 1 hour; payload may be smaller later.
    state.cooldownUntil = Date.now() + HOUR_MS;
    console.warn(`[${name}] Payload too large → cooldown 1 h`);
  } else if (status === 401) {
    state.disabled = true;
    console.warn(`[${name}] Unauthorized → permanently disabled`);
  } else if (status === 404) {
    state.disabled = true;
    console.warn(`[${name}] Model not found → permanently disabled`);
  } else {
    state.cooldownUntil = Date.now() + 10_000;
    console.warn(`[${name}] Error ${status || 'unknown'} → cooldown 10 s`);
  }

  // Advance the preferred index away from the failed service
  const pool = getPool(state.service.supportsTools);
  const idxInPool = pool.indexOf(state);
  if (idxInPool >= 0) {
    const next = (idxInPool + 1) % pool.length;
    if (state.service.supportsTools) preferredAgent = next;
    else preferredChat = next;
  }
}

// ─── CORS ────────────────────────────────────────────────────────────────────

const CORS: Record<string, string> = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type, Authorization',
};

function withCors(headers: Record<string, string> = {}): Record<string, string> {
  return { ...CORS, ...headers };
}

// ─── Auth ─────────────────────────────────────────────────────────────────────

const API_SECRET = process.env.API_SECRET;

function isAuthorized(req: Request): boolean {
  if (!API_SECRET) return true; // auth disabled
  const header = req.headers.get('Authorization') ?? '';
  return header === `Bearer ${API_SECRET}`;
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

const startTime = Date.now();

function genId(): string {
  return `chatcmpl-${Math.random().toString(36).slice(2, 11)}`;
}

/** Wraps a stream to catch mid-stream errors and emit a proper SSE error before [DONE]. */
async function* withErrorBoundary(
  source: AsyncIterable<string>,
  serviceName: string
): AsyncGenerator<string> {
  try {
    yield* source;
  } catch (err: any) {
    const msg = err?.message ?? String(err);
    console.error(`[${serviceName}] Mid-stream error: ${msg}`);
    yield `data: ${JSON.stringify({
      error: { message: msg, type: 'stream_error' }
    })}

`;
    yield 'data: [DONE]\n\n';
  }
}

/**
 * Collects a stream of SSE lines and assembles a complete message
 * (including tool_calls for agent use cases).
 */
async function collectSSE(source: AsyncIterable<string>): Promise<{
  content: string;
  tool_calls: any[];
  finish_reason: string;
}> {
  let content = '';
  const toolCallMap: Record<number, any> = {};
  let finish_reason = 'stop';

  for await (const line of source) {
    const trimmed = line.trim();
    if (!trimmed.startsWith('data: ') || trimmed === 'data: [DONE]') continue;

    try {
      const chunk = JSON.parse(trimmed.slice(6));
      const choice = chunk.choices?.[0];
      if (!choice) continue;

      if (choice.finish_reason) finish_reason = choice.finish_reason;

      const delta = choice.delta;
      if (delta?.content) content += delta.content;

      if (delta?.tool_calls) {
        for (const tc of delta.tool_calls) {
          const idx = tc.index ?? 0;
          if (!toolCallMap[idx]) {
            toolCallMap[idx] = {
              id: tc.id ?? '',
              type: 'function',
              function: { name: '', arguments: '' },
            };
          }
          if (tc.id) toolCallMap[idx].id = tc.id;
          if (tc.function?.name) toolCallMap[idx].function.name += tc.function.name;
          if (tc.function?.arguments) toolCallMap[idx].function.arguments += tc.function.arguments;
        }
      }
    } catch { /* non-JSON SSE lines, skip */ }
  }

  const tool_calls = Object.values(toolCallMap);
  return { content, tool_calls, finish_reason };
}

// ─── Service dispatchers ──────────────────────────────────────────────────────

const MAX_RETRIES = 3;

async function tryServices(
  request: ChatRequest,
  id: string
): Promise<{ stream: AsyncIterable<string>; serviceName: string }> {
  const requireTools = !!(request.tools?.length);
  const errors: string[] = [];

  for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
    const state = getService(requireTools);
    console.log(`[Attempt ${attempt}/${MAX_RETRIES}] Using ${state.service.name}`);

    try {
      const stream = await state.service.chat(request, id);
      return { stream, serviceName: state.service.name };
    } catch (err: any) {
      const msg = err?.message ?? String(err);
      console.error(`[${state.service.name}] Error (attempt ${attempt}): ${msg}`);
      errors.push(`${state.service.name}: ${msg}`);
      handleServiceError(state, err);
    }
  }

  throw Object.assign(new Error('All retries failed'), { details: errors });
}

async function trySpecificService(
  modelName: string,
  request: ChatRequest,
  id: string
): Promise<{ stream: AsyncIterable<string>; serviceName: string }> {
  const state = states.find(s => s.service.name === modelName);

  if (!state) {
    throw Object.assign(
      new Error(`Model '${modelName}' not found. Use GET /v1/models to list available models.`),
      { code: 'model_not_found', httpStatus: 404 }
    );
  }
  if (state.disabled) {
    throw Object.assign(
      new Error(`Model '${modelName}' is permanently disabled.`),
      { code: 'model_disabled', httpStatus: 503 }
    );
  }
  const now = Date.now();
  if (state.cooldownUntil > now) {
    const retryAfter = Math.ceil((state.cooldownUntil - now) / 1000);
    throw Object.assign(
      new Error(`Model '${modelName}' is rate-limited. Retry after ${retryAfter}s.`),
      { code: 'rate_limit_exceeded', httpStatus: 429, retryAfter }
    );
  }

  console.log(`[Specific] Using ${modelName}`);
  try {
    const stream = await state.service.chat(request, id);
    return { stream, serviceName: state.service.name };
  } catch (err: any) {
    handleServiceError(state, err);
    throw err;
  }
}

// ─── HTTP Server ─────────────────────────────────────────────────────────────

const server = Bun.serve({
  port: process.env.PORT ?? 3000,
  hostname: '0.0.0.0',

  async fetch(req) {
    const { pathname } = new URL(req.url);

    // ── CORS preflight ────────────────────────────────────────────────────────
    if (req.method === 'OPTIONS') {
      return new Response(null, { status: 204, headers: withCors() });
    }

    // ── Auth check (skip /health) ─────────────────────────────────────────────
    if (pathname !== '/health' && !isAuthorized(req)) {
      return new Response(
        JSON.stringify({ error: { message: 'Unauthorized', type: 'auth_error' } }),
        { status: 401, headers: withCors({ 'Content-Type': 'application/json' }) }
      );
    }

    // ── Health ───────────────────────────────────────────────────────────────
    if (req.method === 'GET' && pathname === '/health') {
      const now = Date.now();
      const available = states.filter(s => !s.disabled && s.cooldownUntil <= now).length;
      return new Response(JSON.stringify({
        status: 'ok',
        uptime_seconds: Math.floor((now - startTime) / 1000),
        services: { total: states.length, available, disabled: states.filter(s => s.disabled).length },
      }), { headers: withCors({ 'Content-Type': 'application/json' }) });
    }

    // ── Admin reset ─────────────────────────────────────────────────────────
    if (req.method === 'POST' && pathname.startsWith('/admin/reset')) {
      const modelName = decodeURIComponent(pathname.replace('/admin/reset/', '').replace('/admin/reset', '').trim());
      if (modelName) {
        const state = states.find(s => s.service.name === modelName);
        if (!state) {
          return new Response(JSON.stringify({ error: `Model '${modelName}' not found` }),
            { status: 404, headers: withCors({ 'Content-Type': 'application/json' }) });
        }
        state.cooldownUntil = 0;
        state.disabled = false;
        console.log(`[Admin] Reset ${modelName}`);
        return new Response(JSON.stringify({ ok: true, reset: modelName }),
          { headers: withCors({ 'Content-Type': 'application/json' }) });
      }
      // Reset all
      states.forEach(s => { s.cooldownUntil = 0; s.disabled = false; });
      preferredChat = 0; preferredAgent = 0;
      console.log('[Admin] Reset ALL services');
      return new Response(JSON.stringify({ ok: true, reset: 'all', count: states.length }),
        { headers: withCors({ 'Content-Type': 'application/json' }) });
    }

    // ── Status ───────────────────────────────────────────────────────────────
    if (req.method === 'GET' && pathname === '/status') {
      const now = Date.now();
      const report = states.map(s => ({
        name: s.service.name,
        supportsTools: s.service.supportsTools,
        status: s.disabled
          ? 'disabled'
          : s.cooldownUntil > now
            ? `cooldown ${Math.ceil((s.cooldownUntil - now) / 1000)}s`
            : 'available',
      }));
      return new Response(JSON.stringify(report, null, 2), {
        headers: withCors({ 'Content-Type': 'application/json' }),
      });
    }

    // ── Models list ──────────────────────────────────────────────────────────
    if (req.method === 'GET' && pathname === '/v1/models') {
      const now = Date.now();
      const data = states.filter(s => !s.disabled).map(s => ({
        id: s.service.name,
        object: 'model',
        created: Math.floor(now / 1000),
        owned_by: s.service.name.split('/')[0],
        supports_tools: s.service.supportsTools,
        status: s.cooldownUntil > now ? 'cooldown' : 'available',
      }));
      return new Response(JSON.stringify({ object: 'list', data }, null, 2), {
        headers: withCors({ 'Content-Type': 'application/json' }),
      });
    }

    // ── Chat completions ─────────────────────────────────────────────────────
    if (req.method === 'POST' &&
      (pathname === '/v1/chat/completions' || pathname === '/chat')) {

      let body: ChatRequest & { stream?: boolean; model?: string; stream_options?: any };
      try {
        body = await req.json() as ChatRequest & { stream?: boolean; model?: string; stream_options?: any };
      } catch {
        return new Response(
          JSON.stringify({ error: { message: 'Invalid JSON body', type: 'invalid_request_error' } }),
          { status: 400, headers: withCors({ 'Content-Type': 'application/json' }) }
        );
      }

      // eslint-disable-next-line @typescript-eslint/no-unused-vars
      const { stream: wantsStream = true, model = 'auto', stream_options: _so, ...chatRequest } = body;
      const id = genId();
      const useAuto = !model || model === 'auto';

      const hasTools = !!(chatRequest.tools?.length);
      if (hasTools) {
        console.log(`[Tools] Request includes ${chatRequest.tools!.length} tool(s)`);
      }

      try {
        const { stream, serviceName } = useAuto
          ? await tryServices(chatRequest, id)
          : await trySpecificService(model, chatRequest, id);

        // ── Streaming ──────────────────────────────────────────────────────
        if (wantsStream) {
          return new Response(withErrorBoundary(stream, serviceName), {
            headers: withCors({
              'Content-Type': 'text/event-stream',
              'Cache-Control': 'no-cache',
              'Connection': 'keep-alive',
              'X-Service': serviceName,
            }),
          });
        }

        // ── Non-streaming ──────────────────────────────────────────────────
        const { content, tool_calls, finish_reason } = await collectSSE(stream);
        const created = Math.floor(Date.now() / 1000);

        const message: any = { role: 'assistant', content: content || null };
        if (tool_calls.length) message.tool_calls = tool_calls;

        return new Response(
          JSON.stringify({
            id,
            object: 'chat.completion',
            created,
            model: serviceName,
            choices: [{ index: 0, message, finish_reason }],
            usage: { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
          }),
          { headers: withCors({ 'Content-Type': 'application/json', 'X-Service': serviceName }) }
        );

      } catch (err: any) {
        const httpStatus = err?.httpStatus ?? 502;
        const headers = withCors({ 'Content-Type': 'application/json' });
        if (err?.retryAfter) headers['Retry-After'] = String(err.retryAfter);

        return new Response(
          JSON.stringify({
            error: {
              message: err.message ?? 'Request failed',
              type: err.code ?? 'service_unavailable',
              details: err.details ?? [],
            },
          }),
          { status: httpStatus, headers }
        );
      }
    }

    return new Response(JSON.stringify({ error: 'Not found' }), {
      status: 404,
      headers: withCors({ 'Content-Type': 'application/json' }),
    });
  },
});

console.log(`Server running on ${server.url}`);
console.log(`Services: ${states.length} total | ${states.filter(s => !s.disabled).length} enabled | ${states.filter(s => s.service.supportsTools).length} support tools`);
console.log(`OpenAI-compatible: ${server.url}v1/chat/completions`);
