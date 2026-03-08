import OpenAI from 'openai';
import type { AIService, ChatRequest } from '../types';

const client = new OpenAI({
    baseURL: 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1',
    apiKey: process.env.ALIBABA_API_KEY,
});

const ALIBABA_MODELS = [
    'qvq-max-2025-03-25', 'qwen3.5-122b-a10b', 'qwen-vl-ocr-2025-11-20', 'qwen-vl-max-2025-04-08',
    'qwen3-vl-235b-a22b-thinking', 'qwen-vl-plus-2025-05-07', 'qwen2.5-vl-72b-instruct',
    'qwen2.5-7b-instruct-1m', 'qwen-plus-2025-07-28', 'qwen3-max', 'qwen2.5-vl-3b-instruct',
    'qwen-vl-plus-latest', 'qwen3.5-plus-2026-02-15', 'qwen-max', 'qwen2.5-14b-instruct',
    'qwen-mt-flash', 'qwen3-235b-a22b-thinking-2507', 'qwen3-vl-30b-a3b-thinking',
    'qwen2.5-vl-7b-instruct', 'qwen2.5-7b-instruct', 'qwen3-32b', 'qwen-vl-max-2025-08-13',
    'qwen3.5-397b-a17b', 'qwen3-vl-plus-2025-09-23', 'qwen-vl-plus', 'qwen3-coder-next',
    'qwen3.5-flash', 'qwen3.5-35b-a3b', 'qwen3-30b-a3b-thinking-2507', 'qwen-max-2025-01-25',
    'qwen2.5-72b-instruct', 'qwen3-coder-plus-2025-09-23', 'qwen-plus-latest', 'qwen3-max-2026-01-23',
    'qwen3-coder-480b-a35b-instruct', 'qwen3-coder-plus', 'qwen3-vl-8b-thinking',
    'qwen-plus-2025-09-11', 'wan2.2-kf2v-flash', 'qwen3-vl-flash-2026-01-22',
    'qwen3.5-flash-2026-02-23', 'qwen3-vl-flash-2025-10-15', 'qwen3-max-preview', 'qwen-vl-max',
    'qwen3-vl-30b-a3b-instruct', 'qwen3-vl-235b-a22b-instruct', 'qwen3-8b', 'qwen2.5-14b-instruct-1m',
    'qwen3-coder-30b-a3b-instruct', 'qwen3-4b', 'qwen3-235b-a22b', 'qwen-plus', 'qwen-turbo',
    'qwen-mt-lite', 'qwen3-0.6b', 'qwen3-1.7b', 'qwen-vl-plus-2025-01-25', 'qvq-max',
    'qwen3-coder-flash', 'qwen-vl-plus-2025-08-15', 'qwen3-vl-plus', 'qwen-vl-max-latest',
    'qwen3-next-80b-a3b-thinking', 'qwen3.5-27b', 'qwen-turbo-2025-04-28', 'qwen2.5-vl-32b-instruct',
    'qwen3-30b-a3b', 'qwen-mt-plus', 'qwen3-vl-flash', 'qwen3-14b', 'qwen2.5-32b-instruct',
    'qwen-turbo-latest', 'qwen3-vl-8b-instruct', 'qwen3-max-2025-09-23', 'qwen-plus-character',
    'qwen3-coder-flash-2025-07-28', 'qwen-flash-character', 'qwen3-vl-plus-2025-12-19',
    'qwen-plus-2025-04-28', 'qwen-mt-turbo', 'qwen3-30b-a3b-instruct-2507', 'qwen3.5-plus',
    'qvq-max-latest', 'qwen-flash', 'qwen-flash-2025-07-28', 'qwen-plus-2025-07-14',
    'qwen3-235b-a22b-instruct-2507', 'qwq-plus', 'qwen3-coder-plus-2025-07-22', 'qwen-vl-ocr',
    'qwen3-next-80b-a3b-instruct'
];

const MODELS: { id: string; supportsTools: boolean }[] = ALIBABA_MODELS.map(id => ({
    id,
    supportsTools: !id.includes('-vl-') && !id.includes('-mt-') && !id.startsWith('wan') && !id.startsWith('qvq'),
}));

function createAlibabaService({ id: model, supportsTools }: { id: string; supportsTools: boolean }): AIService {
    return {
        name: `Alibaba/${model}`,
        supportsTools,
        async chat(request: ChatRequest, id: string) {
            const {
                messages, tools, tool_choice,
                temperature = 0.6, max_tokens = 4096, top_p = 0.95,
            } = request;

            const stream = await client.chat.completions.create({
                messages: messages as any,
                model,
                stream: true,
                temperature,
                max_tokens,
                top_p,
                ...(supportsTools && tools?.length && { tools }),
                ...(supportsTools && tool_choice !== undefined && { tool_choice }),
            });

            return (async function* () {
                for await (const chunk of stream) {
                    yield `data: ${JSON.stringify({ ...chunk, id, model: `Alibaba/${model}` })}\n\n`;
                }
                yield 'data: [DONE]\n\n';
            })();
        },
    };
}

export const alibabaServices: AIService[] = MODELS.map(createAlibabaService);
