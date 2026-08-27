'use client';

import { useState, useRef, useEffect } from 'react';
import { DataTable } from '@/components/DataTable';
import { BarChart } from '@/components/BarChart';
import { ChatMessage, AIQueryResult } from '@/lib/ai';

const SUGGESTIONS = [
  'Có bao nhiêu vi phạm hôm nay?',
  'Top 5 node nhiều vi phạm nhất',
  'Tỉ lệ duyệt hồ sơ là bao nhiêu?',
  'Vi phạm theo trạng thái đèn',
  'Biển số nào vi phạm nhiều nhất?',
];

let chatIdCounter = 0;
function createUniqueChatId(prefix: string): string {
  chatIdCounter += 1;
  return `${prefix}-${chatIdCounter}`;
}

export default function AIChat() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSubmit = async (question: string) => {
    if (!question.trim() || loading) return;

    setError(null);
    setLoading(true);

    const timestamp = 1700000000000;
    const userMsg: ChatMessage = {
      id: createUniqueChatId('user'),
      role: 'user',
      content: question,
      timestamp,
    };

    setMessages(prev => [...prev, userMsg]);
    setInput('');

    try {
      const response = await fetch('/api/ai-query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question }),
      });

      const data: AIQueryResult = await response.json();

      if (data.error) {
        const errMsg: ChatMessage = {
          id: createUniqueChatId('err'),
          role: 'assistant',
          content: data.error,
          timestamp,
        };
        setMessages(prev => [...prev, errMsg]);
        return;
      }

      // SQL explanation
      const explainMsg: ChatMessage = {
        id: createUniqueChatId('sql'),
        role: 'assistant',
        content: `Truy vấn: ${data.sql}\n\nKết quả: ${data.count} dòng`,
        timestamp,
      };
      setMessages(prev => [...prev, explainMsg]);

      // Result message
      const resultMsg: ChatMessage = {
        id: createUniqueChatId('res'),
        role: 'result',
        content: '',
        result: data,
        timestamp,
      };
      setMessages(prev => [...prev, resultMsg]);

    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Lỗi kết nối';
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(input);
    }
  };

  return (
    <div className="glass-card border border-border rounded-2xl overflow-hidden flex flex-col" style={{ height: messages.length > 0 ? '65vh' : 'Auto' }}>
      {/* Header */}
      <div className="flex items-center gap-2 px-4 py-3 border-b border-border bg-gradient-to-r from-primary-500/5 to-blue-500/5">
        <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-primary-500 to-purple-500 flex items-center justify-center">
          <svg className="w-4 h-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09zM18.25 10.5h-2.25M18.25 7.5V5.5M18.25 13.5v1.5M12.75 3.75h1.5M9.75 3.75h1.5" />
          </svg>
        </div>
        <span className="font-semibold text-sm text-text-primary">Trợ lý AI</span>
        <span className="text-[10px] text-text-muted ml-auto">Text-to-SQL</span>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-4 py-3 space-y-3">
        {messages.length === 0 && (
          <div className="space-y-2">
            <p className="text-sm text-text-secondary mb-3 text-center">
              Hỏi tôi bất cứ câu hỏi nào về dữ liệu vi phạm giao thông bằng tiếng Việt.
            </p>
            <div className="flex flex-wrap gap-2">
              {SUGGESTIONS.map((s) => (
                <button
                  key={s}
                  onClick={() => handleSubmit(s)}
                  className="text-xs px-3 py-1.5 rounded-full border border-border bg-surface-2 text-text-secondary hover:border-primary-500/30 hover:text-primary-400 transition-colors"
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((msg) => (
          <div key={msg.id} className="space-y-2">
            {/* User message */}
            {msg.role === 'user' && (
              <div className="flex justify-end">
                <div className="max-w-[85%] bg-primary-600/20 border border-primary-500/20 rounded-2xl rounded-br-md px-4 py-2.5">
                  <p className="text-sm text-text-primary">{msg.content}</p>
                </div>
              </div>
            )}

            {/* Assistant bubble */}
            {msg.role === 'assistant' && (
              <div className="flex justify-start">
                <div className="max-w-[90%] bg-surface-2 border border-border rounded-2xl rounded-bl-md px-4 py-2.5">
                  <p className="text-xs text-text-secondary font-mono whitespace-pre-wrap">
                    {msg.content}
                  </p>
                </div>
              </div>
            )}

            {/* Data result */}
            {msg.role === 'result' && msg.result && (
              <div className="flex justify-start">
                <div className="w-full bg-surface-2/70 border border-primary-500/20 rounded-2xl overflow-hidden">
                  {msg.result.chartType === 'bar' && (
                    <BarChart columns={msg.result.columns} rows={msg.result.rows} />
                  )}
                  <DataTable columns={msg.result.columns} rows={msg.result.rows} maxHeight={350} />
                </div>
              </div>
            )}
          </div>
        ))}

        {/* Loading indicator */}
        {loading && (
          <div className="flex justify-start">
            <div className="bg-surface-2 border border-border rounded-2xl rounded-bl-md px-5 py-3">
              <div className="flex items-center gap-1.5">
                <div className="w-2 h-2 rounded-full bg-primary-500 animate-bounce" style={{ animationDelay: '0ms' }} />
                <div className="w-2 h-2 rounded-full bg-primary-500 animate-bounce" style={{ animationDelay: '150ms' }} />
                <div className="w-2 h-2 rounded-full bg-primary-500 animate-bounce" style={{ animationDelay: '300ms' }} />
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="border-t border-border px-4 py-3 bg-surface-2/50">
        {error && (
          <div className="mb-2 px-3 py-2 bg-red-500/10 border border-red-500/20 rounded-xl">
            <p className="text-xs text-red-400">{error}</p>
          </div>
        )}
        <div className="flex gap-2">
          <input
            ref={inputRef}
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Hỏi bất cứ điều gì về dữ liệu..."
            className="flex-1 bg-surface-3 border border-border rounded-xl px-4 py-2.5 text-sm text-text-primary placeholder:text-text-pmuted focus:outline-none focus:border-primary-500/40 transition-colors"
            disabled={loading}
          />
          <button
            onClick={() => handleSubmit(input)}
            disabled={loading || !input.trim()}
            className="p-2.5 bg-primary-500/20 border border-primary-500/30 rounded-xl text-primary-400 hover:bg-primary-500/30 disabled:opacity-30 disabled:cursor-not-allowed transition-all"
          >
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M6 12L3.269 3.126A59.768 59.768 0 0121.485 12 59.77 59.77 0 013.269 20.876L6 12zm0 0h7.5" />
            </svg>
          </button>
        </div>
      </div>
    </div>
  );
}