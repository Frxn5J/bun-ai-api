import { Groq } from 'groq-sdk';
import type { AIService, ChatMessage } from '../types';

const groq = new Groq();

// Free chat-capable models on Groq (excludes guard/prompt-guard classifiers and TTS models)
const MODELS = [
  'moonshotai/kimi-k2-instruct',
  'moonshotai/kimi-k2-instruct-0905',
  'groq/compound',
  'groq/compound-mini',
  'llama-3.3-70b-versatile',
  'meta-llama/llama-4-maverick-17b-128e-instruct',
  'meta-llama/llama-4-scout-17b-16e-instruct',
  'openai/gpt-oss-120b',
  'openai/gpt-oss-20b',
  'qwen/qwen3-32b',
  // Legacy models (still available on Groq)
  'mixtral-8x7b-32768',
  'deepseek-r1-distill-llama-70b',
];

function createGroqService(model: string): AIService {
  return {
    name: `Groq/${model}`,
    async chat(messages: ChatMessage[]) {
      const stream = await groq.chat.completions.create({
        messages,
        model,
        temperature: 0.6,
        max_completion_tokens: 4096,
        top_p: 1,
        stream: true,
        stop: null,
      });

      return (async function* () {
        for await (const chunk of stream) {
          yield chunk.choices[0]?.delta?.content || '';
        }
      })();
    },
  };
}

export const groqServices: AIService[] = MODELS.map(createGroqService);
