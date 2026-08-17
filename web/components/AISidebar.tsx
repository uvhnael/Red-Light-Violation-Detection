"use client";

import { useState, useRef, useEffect } from "react";
import { Sparkles, X, Send, Code, CheckCircle, BarChart3 } from "lucide-react";
import { DataTable } from "@/components/DataTable";
import { BarChart } from "@/components/BarChart";
import { ChatMessage, AIQueryResult } from "@/lib/ai";

const SUGGESTIONS = [
  "Có bao nhiêu vi phạm hôm nay?",
  "Top 5 node nhiều vi phạm nhất",
  "Tỉ lệ duyệt hồ sơ là bao nhiêu?",
  "Vi phạm theo trạng thái đèn",
  "Biển số nào vi phạm nhiều nhất?",
];

interface AISidebarProps {
  isOpen: boolean;
  onToggle: () => void;
}

export default function AISidebar({ isOpen, onToggle }: AISidebarProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    if (!isOpen) {
      setMessages([]);
      setError(null);
      setInput("");
    }
  }, [isOpen]);

  const handleSubmit = async (question: string) => {
    if (!question.trim() || loading) return;

    setError(null);
    setLoading(true);

    const userMsg: ChatMessage = {
      id: Date.now().toString(),
      role: "user",
      content: question,
      timestamp: Date.now(),
    };

    setMessages((prev) => [...prev, userMsg]);
    setInput("");

    try {
      const response = await fetch("/api/ai-query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });

      const data: AIQueryResult = await response.json();

      if (data.error) {
        const errMsg: ChatMessage = {
          id: (Date.now() + 1).toString(),
          role: "assistant",
          content: data.error,
          timestamp: Date.now(),
        };
        setMessages((prev) => [...prev, errMsg]);
        return;
      }

      // SQL explanation
      const explainMsg: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: data.sql,
        timestamp: Date.now(),
      };
      setMessages((prev) => [...prev, explainMsg]);

      // Result
      const resultMsg: ChatMessage = {
        id: (Date.now() + 2).toString(),
        role: "result",
        content: "",
        result: data,
        timestamp: Date.now(),
      };
      setMessages((prev) => [...prev, resultMsg]);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Connection error";
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(input);
    }
  };

  return (
    <aside
      className={`fixed right-0 top-0 h-screen w-[440px] max-w-[95vw] bg-zinc-950/98 backdrop-blur-2xl border-l border-white/5 flex flex-col z-50 transition-transform duration-400 ease-in-out shadow-2xl shadow-black/40 ${
        isOpen ? "translate-x-0" : "translate-x-full"
      }`}
    >
      {/* Header */}
      <div className="flex items-center justify-between px-5 py-4 border-b border-white/5 bg-gradient-to-r from-indigo-500/5 via-violet-500/5 to-purple-500/5 shrink-0 h-[65px]">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-indigo-500 to-violet-600 flex items-center justify-center shadow-lg shadow-indigo-500/20">
            <Sparkles className="w-4 h-4 text-white" />
          </div>
          <div>
            <span className="font-semibold text-sm text-zinc-100">AI Assistant</span>
            <p className="text-[10px] text-zinc-500">Text-to-SQL · Gemini</p>
          </div>
        </div>
        <button
          onClick={onToggle}
          className="p-1.5 rounded-lg hover:bg-zinc-800/50 text-zinc-400 hover:text-zinc-200 transition-colors"
        >
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-4 py-4 space-y-4">
        {messages.length === 0 && (
          <div className="h-full flex flex-col items-center justify-center gap-6">
            <div className="w-20 h-20 rounded-2xl bg-gradient-to-br from-indigo-500/15 to-violet-500/15 flex items-center justify-center">
              <Sparkles className="w-10 h-10 text-indigo-400" />
            </div>
            <div className="text-center space-y-2">
              <h3 className="text-sm font-semibold text-zinc-100">Ask about traffic data</h3>
              <p className="text-xs text-zinc-500 max-w-[280px]">
                I translate your natural language questions into SQL and return structured results.
              </p>
            </div>
            <div className="flex flex-wrap gap-2 justify-center max-w-[340px]">
              {SUGGESTIONS.map((s) => (
                <button
                  key={s}
                  onClick={() => handleSubmit(s)}
                  className="text-xs px-3 py-1.5 rounded-full border border-indigo-500/20 bg-indigo-500/5 text-indigo-300 hover:bg-indigo-500/15 hover:border-indigo-500/30 transition-all"
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((msg) => (
          <div key={msg.id} className="space-y-2">
            {/* User bubble */}
            {msg.role === "user" && (
              <div className="flex justify-end">
                <div className="max-w-[88%] bg-indigo-600/15 border border-indigo-500/20 rounded-2xl rounded-br-md px-3.5 py-2">
                  <p className="text-sm text-zinc-100">{msg.content}</p>
                </div>
              </div>
            )}

            {/* Assistant SQL bubble */}
            {msg.role === "assistant" && (
              <div className="flex justify-start">
                <div className="max-w-[96%] w-full bg-zinc-900/80 border border-indigo-500/15 rounded-xl overflow-hidden">
                  {/* SQL header */}
                  <div className="flex items-center gap-1.5 px-3 py-1.5 bg-zinc-800/80 border-b border-white/5">
                    <Code className="w-3.5 h-3.5 text-indigo-400" />
                    <span className="text-[10px] text-zinc-500 font-semibold uppercase tracking-wider">
                      Generated SQL
                    </span>
                  </div>
                  <pre className="px-3 py-2.5 text-xs text-indigo-300/80 font-mono overflow-x-auto whitespace-pre-wrap leading-relaxed">
                    {msg.content}
                  </pre>
                </div>
              </div>
            )}

            {/* Result */}
            {msg.role === "result" && msg.result && (
              <div className="w-full bg-zinc-900/60 border border-indigo-500/15 rounded-xl overflow-hidden">
                {/* Result header */}
                <div className="flex items-center justify-between px-3 py-2 bg-zinc-800/60 border-b border-white/5">
                  <div className="flex items-center gap-2">
                    <CheckCircle className="w-3.5 h-3.5 text-emerald-400" />
                    <span className="text-[10px] text-zinc-500 font-semibold uppercase tracking-wider">
                      {msg.result.count} results
                    </span>
                  </div>
                  <div className="flex items-center gap-1.5">
                    {msg.result.chartType === "bar" && (
                      <span className="flex items-center gap-1 text-[10px] text-amber-400 bg-amber-500/10 px-2 py-0.5 rounded-full">
                        <BarChart3 className="w-3 h-3" />
                        Bar
                      </span>
                    )}
                    <span className="text-[10px] text-zinc-500 bg-zinc-700/50 px-2 py-0.5 rounded-full">
                      {msg.result.chartType === "bar" ? "Chart" : "Table"}
                    </span>
                  </div>
                </div>

                {/* Chart or Table */}
                {msg.result.chartType === "bar" && (
                  <BarChart columns={msg.result.columns} rows={msg.result.rows} />
                )}
                <DataTable columns={msg.result.columns} rows={msg.result.rows} maxHeight={300} />
              </div>
            )}
          </div>
        ))}

        {/* Loading */}
        {loading && (
          <div className="flex justify-start">
            <div className="bg-zinc-900/80 border border-white/5 rounded-xl px-4 py-3">
              <div className="flex items-center gap-2">
                <div className="flex items-center gap-1">
                  <div className="w-1.5 h-1.5 rounded-full bg-indigo-500 animate-pulse" />
                  <div className="w-1.5 h-1.5 rounded-full bg-violet-500 animate-pulse" style={{ animationDelay: "0.15s" }} />
                  <div className="w-1.5 h-1.5 rounded-full bg-purple-500 animate-pulse" style={{ animationDelay: "0.3s" }} />
                </div>
                <span className="text-xs text-zinc-500 ml-1">Processing...</span>
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="border-t border-white/5 px-4 py-3 bg-zinc-950/80 shrink-0">
        {error && (
          <div className="mb-2 px-3 py-2 bg-rose-500/10 border border-rose-500/20 rounded-lg">
            <div className="flex items-start gap-2">
              <AlertTriangle className="w-3.5 h-3.5 text-rose-400 mt-0.5 shrink-0" />
              <p className="text-xs text-rose-400">{error}</p>
            </div>
          </div>
        )}
        <div className="flex gap-2">
          <input
            ref={inputRef}
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask about violations data..."
            className="flex-1 bg-zinc-800/80 border border-white/5 rounded-xl px-4 py-2.5 text-sm text-zinc-100 placeholder:text-zinc-500 focus:outline-none focus:border-indigo-500/40 focus:ring-1 focus:ring-indigo-500/20 transition-all"
            disabled={loading}
          />
          <button
            onClick={() => handleSubmit(input)}
            disabled={loading || !input.trim()}
            className="p-2.5 bg-indigo-600/80 hover:bg-indigo-500/90 border border-indigo-500/30 rounded-xl text-white disabled:opacity-30 disabled:cursor-not-allowed transition-all shadow-lg shadow-indigo-500/10"
          >
            <Send className="w-4.5 h-4.5" />
          </button>
        </div>
      </div>
    </aside>
  );
}

// Helper for missing icon
function AlertTriangle({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z" />
    </svg>
  );
}