import { groqServices } from './services/groq';
import { cerebrasServices } from './services/cerebras';
import { openrouterServices } from './services/openrouter';
import { mistralService } from './services/mistral';
import { codestralService } from './services/codestral';
import { geminiService } from './services/gemini';
import { cohereService } from './services/cohere';
import { nvidiaService } from './services/nvidia';
import type { AIService, ChatMessage } from './types';

// ─── Service pool ────────────────────────────────────────────────────────────

const allServices: AIService[] = [
  ...groqServices,
  ...openrouterServices,
  ...cerebrasServices,
  mistralService,
  codestralService,
  geminiService,
  cohereService,
  nvidiaService,
];

// ─── State tracking ──────────────────────────────────────────────────────────

const MIN_MS = 60 * 1_000;
const HOUR_MS = 60 * MIN_MS;

interface ServiceState {
  service: AIService;
  cooldownUntil: number;
  disabled: boolean;
}

const states: ServiceState[] = allServices.map(s => ({
  service: s,
  cooldownUntil: 0,
  disabled: false,
}));

let preferred = 0;

function getService(): ServiceState {
  const now = Date.now();
  for (let i = 0; i < states.length; i++) {
    const s = states[(preferred + i) % states.length]!;
    if (!s.disabled && s.cooldownUntil <= now) {
      preferred = (preferred + i) % states.length;
      return s;
    }
  }
  // All in cooldown — pick soonest
  return states
    .filter(s => !s.disabled)
    .reduce((a, b) => (a.cooldownUntil < b.cooldownUntil ? a : b)) as ServiceState;
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

  preferred = (idx + 1) % states.length;
}

// ─── OpenAI-compatible helpers ────────────────────────────────────────────────

function genId(): string {
  return `chatcmpl-${Math.random().toString(36).slice(2, 11)}`;
}

/**
 * Wraps our raw text stream into OpenAI SSE format:
 *   data: { ...chunk }\n\n
 *   data: [DONE]\n\n
 */
async function* toOpenAISSE(
  source: AsyncIterable<string>,
  id: string,
  model: string
): AsyncGenerator<string> {
  const created = Math.floor(Date.now() / 1000);
  const base = { id, object: 'chat.completion.chunk', created, model };

  // Opening delta with role
  yield `data: ${JSON.stringify({
    ...base,
    choices: [{ index: 0, delta: { role: 'assistant', content: '' }, finish_reason: null }],
  })}\n\n`;

  for await (const text of source) {
    if (!text) continue;
    yield `data: ${JSON.stringify({
      ...base,
      choices: [{ index: 0, delta: { content: text }, finish_reason: null }],
    })}\n\n`;
  }

  // Closing delta
  yield `data: ${JSON.stringify({
    ...base,
    choices: [{ index: 0, delta: {}, finish_reason: 'stop' }],
  })}\n\n`;

  yield 'data: [DONE]\n\n';
}

/** Collects all chunks into a single string (for non-streaming responses). */
async function collectStream(source: AsyncIterable<string>): Promise<string> {
  let out = '';
  for await (const chunk of source) out += chunk;
  return out;
}

/**
 * Attempts to get a response from up to MAX_RETRIES services.
 * Returns the raw text stream and the chosen service name.
 */
const MAX_RETRIES = 3;

async function tryServices(
  messages: ChatMessage[]
): Promise<{ stream: AsyncIterable<string>; serviceName: string }> {
  const errors: string[] = [];

  for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
    const state = getService();
    console.log(`[Attempt ${attempt}/${MAX_RETRIES}] Using ${state.service.name}`);

    try {
      const stream = await state.service.chat(messages);
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

/**
 * Uses one specific service by name, respecting its current cooldown/disabled state.
 */
async function trySpecificService(
  modelName: string,
  messages: ChatMessage[]
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
      new Error(`Model '${modelName}' is permanently disabled (auth error or model removed).`),
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
    const stream = await state.service.chat(messages);
    return { stream, serviceName: state.service.name };
  } catch (err: any) {
    const msg = err?.message ?? String(err);
    console.error(`[${modelName}] Error: ${msg}`);
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

    // ── Status ────────────────────────────────────────────────────────────────
    if (req.method === 'GET' && pathname === '/status') {
      const now = Date.now();
      const report = states.map(s => ({
        name: s.service.name,
        status: s.disabled
          ? 'disabled'
          : s.cooldownUntil > now
            ? `cooldown ${Math.ceil((s.cooldownUntil - now) / 1000)}s`
            : 'available',
      }));
      return new Response(JSON.stringify(report, null, 2), {
        headers: { 'Content-Type': 'application/json' },
      });
    }

    // ── Models list (OpenAI-compatible) ───────────────────────────────────────
    if (req.method === 'GET' && pathname === '/v1/models') {
      const now = Date.now();
      const data = states
        .filter(s => !s.disabled)
        .map(s => ({
          id: s.service.name,
          object: 'model',
          created: Math.floor(now / 1000),
          owned_by: s.service.name.split('/')[0],
          status: s.cooldownUntil > now ? 'cooldown' : 'available',
        }));
      return new Response(JSON.stringify({ object: 'list', data }, null, 2), {
        headers: { 'Content-Type': 'application/json' },
      });
    }

    // ── OpenAI-compatible chat completions ────────────────────────────────────
    if (req.method === 'POST' &&
      (pathname === '/v1/chat/completions' || pathname === '/chat')) {

      let body: { messages: ChatMessage[]; stream?: boolean; model?: string };
      try {
        body = await req.json() as { messages: ChatMessage[]; stream?: boolean; model?: string };
      } catch {
        return new Response(JSON.stringify({ error: { message: 'Invalid JSON body', type: 'invalid_request_error' } }), {
          status: 400,
          headers: { 'Content-Type': 'application/json' },
        });
      }

      const { messages, stream: wantsStream = true, model = 'auto' } = body;
      const id = genId();
      // 'auto' (or empty/null) → sticky router; anything else → specific model
      const useAuto = !model || model === 'auto';

      try {
        const { stream, serviceName } = useAuto
          ? await tryServices(messages)
          : await trySpecificService(model, messages);

        // ── Streaming response ─────────────────────────────────────────────
        if (wantsStream) {
          return new Response(toOpenAISSE(stream, id, serviceName), {
            headers: {
              'Content-Type': 'text/event-stream',
              'Cache-Control': 'no-cache',
              'Connection': 'keep-alive',
              'X-Service': serviceName,
            },
          });
        }

        // ── Non-streaming response ─────────────────────────────────────────
        const content = await collectStream(stream);
        const created = Math.floor(Date.now() / 1000);
        return new Response(
          JSON.stringify({
            id,
            object: 'chat.completion',
            created,
            model: serviceName,
            choices: [{
              index: 0,
              message: { role: 'assistant', content },
              finish_reason: 'stop',
            }],
            usage: { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
          }),
          {
            headers: {
              'Content-Type': 'application/json',
              'X-Service': serviceName,
            },
          }
        );

      } catch (err: any) {
        const httpStatus = err?.httpStatus ?? 502;
        const headers: Record<string, string> = { 'Content-Type': 'application/json' };
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

    return new Response('Not found', { status: 404 });
  },
});

console.log(`Server running on ${server.url}`);
console.log(`Services: ${states.length} total, ${states.filter(s => !s.disabled).length} enabled`);
console.log(`OpenAI-compatible endpoint: ${server.url}v1/chat/completions`);
