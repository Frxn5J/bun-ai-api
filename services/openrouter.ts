import OpenAI from 'openai';
import type { AIService, ChatRequest } from '../types';

const client = new OpenAI({
    baseURL: 'https://openrouter.ai/api/v1',
    apiKey: process.env.OPENROUTER_API_KEY,
});

// ─── OpenRouter model schema ──────────────────────────────────────────────────

interface ORModel {
    id: string;
    name: string;
    pricing: {
        prompt: string | number;
        completion: string | number;
        image?: string | number;
        request?: string | number;
    };
    context_length?: number;
    architecture?: {
        modality?: string;     // e.g. "text->text", "text+image->text"
        tokenizer?: string;
        instruct_type?: string;
    };
    supported_parameters?: string[];
    capabilities?: {
        tools?: boolean;
        function_calling?: boolean;
    };
}

// ─── Filters ──────────────────────────────────────────────────────────────────

function isFree(m: ORModel): boolean {
    return Number(m.pricing.prompt) === 0 && Number(m.pricing.completion) === 0;
}

/** Exclude embedding, image-gen, TTS, speech-to-text models */
function isChatCapable(m: ORModel): boolean {
    const id = m.id.toLowerCase();
    const skip = ['embed', 'dall-e', 'tts', 'whisper', 'ocr'];
    if (skip.some(p => id.includes(p))) return false;

    // If architecture.modality is present and doesn't include text output, skip
    const modality = m.architecture?.modality?.toLowerCase() ?? '';
    if (modality && !modality.includes('text')) return false;

    return true;
}

// ─── Tool support detection ───────────────────────────────────────────────────

function heuristicToolSupport(id: string): boolean {
    const l = id.toLowerCase();

    const NO = [
        '1.2b', ':1b', '-1b:', '2b:', ':2b', '3b:', ':3b',
        '-nano', '-mini',
        '-vl', 'vision', '-ocr',
        'safeguard', 'guard', 'embedding',
        ':thinking', '-thinking',
        'orpheus', 'tts', 'dall-e', 'whisper',
        'dolphin-mistral', 'trinity-mini', 'step-3.5',
    ];
    if (NO.some(p => l.includes(p))) return false;

    const YES = [
        'llama-3', 'llama-4',
        'mistral', 'mixtral',
        'qwen3', 'qwen-2.5',
        'hermes', 'gemma-3-12b', 'gemma-3-27b',
        'solar-pro', 'nemotron-3-nano-30b',
        'command-r', 'gpt-4', 'gpt-3.5',
        'claude', 'deepseek-v', 'deepseek-chat',
        'phi-4', 'glm-4', 'trinity-large',
        'kimi', 'moonshot', 'minimax',
    ];
    if (YES.some(p => l.includes(p))) return true;

    return false;
}

function detectTools(m: ORModel): boolean {
    if (Array.isArray(m.supported_parameters) && m.supported_parameters.length > 0) {
        return m.supported_parameters.some(p => p === 'tools' || p === 'tool_choice');
    }
    if (m.capabilities !== undefined) {
        return !!(m.capabilities.tools ?? m.capabilities.function_calling);
    }
    return heuristicToolSupport(m.id);
}

// ─── Fetch from OpenRouter ────────────────────────────────────────────────────

async function fetchModels(): Promise<{
    free: { id: string; supportsTools: boolean }[];
    paid: { id: string; supportsTools: boolean }[];
}> {
    if (!process.env.OPENROUTER_API_KEY) {
        console.warn('[OpenRouter] OPENROUTER_API_KEY not set — skipping fetch');
        return { free: [], paid: [] };
    }

    const resp = await fetch('https://openrouter.ai/api/v1/models', {
        headers: { 'Authorization': `Bearer ${process.env.OPENROUTER_API_KEY}` },
        signal: AbortSignal.timeout(10_000),
    });

    if (!resp.ok) throw new Error(`HTTP ${resp.status} ${resp.statusText}`);

    const body = await resp.json() as { data: ORModel[] };
    const chatModels = body.data.filter(isChatCapable);

    const free = chatModels
        .filter(isFree)
        .map(m => ({ id: m.id, supportsTools: detectTools(m) }));

    const paid = chatModels
        .filter(m => !isFree(m))
        .map(m => ({ id: m.id, supportsTools: detectTools(m) }));

    return { free, paid };
}

// ─── Service factory ──────────────────────────────────────────────────────────

function createOpenRouterService(
    { id: model, supportsTools }: { id: string; supportsTools: boolean }
): AIService {
    return {
        name: `OpenRouter/${model}`,
        supportsTools,
        async chat(request: ChatRequest, id: string) {
            const {
                messages, tools, tool_choice,
                temperature = 0.6, max_tokens = 4096,
            } = request;

            const stream = await client.chat.completions.create({
                model,
                messages: messages as any,
                stream: true,
                temperature,
                max_tokens,
                ...(supportsTools && tools?.length && { tools }),
                ...(supportsTools && tool_choice !== undefined && { tool_choice }),
            });

            return (async function* () {
                for await (const chunk of stream) {
                    yield `data: ${JSON.stringify({ ...chunk, id, model: `OpenRouter/${model}` })}\n\n`;
                }
                yield 'data: [DONE]\n\n';
            })();
        },
    };
}

// ─── Initialization ───────────────────────────────────────────────────────────

let openrouterFreeServices: AIService[] = [];
let openrouterPaidServices: AIService[] = [];

try {
    console.log('[OpenRouter] Fetching model list...');
    const { free, paid } = await fetchModels();

    openrouterFreeServices = free.map(createOpenRouterService);
    openrouterPaidServices = paid.map(createOpenRouterService);

    const fTools = free.filter(m => m.supportsTools).length;
    const pTools = paid.filter(m => m.supportsTools).length;

    console.log(`[OpenRouter] ✅ Free:  ${free.length} models (${fTools} with tools)`);
    console.log(`[OpenRouter] 💰 Paid:  ${paid.length} models (${pTools} with tools) — only used when requested by name`);
} catch (err: any) {
    console.error(`[OpenRouter] ❌ Failed to fetch models: ${err.message}`);
}

export { openrouterFreeServices, openrouterPaidServices };

/** @deprecated use openrouterFreeServices */
export const openrouterServices = openrouterFreeServices;
