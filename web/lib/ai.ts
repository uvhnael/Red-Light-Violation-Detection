// Types for AI assistant
export interface AIQueryResult {
  question: string;
  sql: string;
  /** Câu trả lời dạng văn báo cáo (Gemini lần 2 — narration sau khi SQL chạy) */
  answer?: string;
  columns: string[];
  rows: Record<string, unknown>[];
  count: number;
  chartType: 'table' | 'bar';
  error?: string;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'result';
  content: string;
  result?: AIQueryResult;
  timestamp: number;
}
