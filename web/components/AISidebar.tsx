"use client";

import { useState, useRef, useEffect } from "react";
import { Sparkles, Send, Code, CheckCircle, BarChart3, AlertTriangle, X } from "lucide-react";
import { DataTable } from "@/components/DataTable";
import { BarChart } from "@/components/BarChart";
import { ChatMessage, AIQueryResult } from "@/lib/ai";
import { getSession } from "@/lib/auth";

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

let idCounter = 0;
function createUniqueId(prefix: string): string {
  idCounter += 1;
  return `${prefix}-${idCounter}`;
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

  // Focus vào ô nhập khi mở panel
  useEffect(() => {
    if (isOpen) inputRef.current?.focus();
  }, [isOpen]);

  const handleClose = () => {
    setMessages([]);
    setError(null);
    setInput("");
    onToggle();
  };

  const handleSubmit = async (question: string) => {
    if (!question.trim() || loading) return;

    setError(null);
    setLoading(true);

    const timestamp = 1700000000000;
    const userMsg: ChatMessage = {
      id: createUniqueId("user"),
      role: "user",
      content: question,
      timestamp,
    };

    setMessages((prev) => [...prev, userMsg]);
    setInput("");

    try {
      const response = await fetch("/api/ai-query", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          // Central yêu cầu JWT cho /api/ai/** — gắn token từ session
          ...(getSession()?.token
            ? { Authorization: `Bearer ${getSession()!.token}` }
            : {}),
        },
        body: JSON.stringify({ question }),
      });

      const data: AIQueryResult = await response.json();

      if (data.error) {
        const errMsg: ChatMessage = {
          id: createUniqueId("err"),
          role: "assistant",
          content: data.error,
          timestamp,
        };
        setMessages((prev) => [...prev, errMsg]);
        return;
      }

      // SQL explanation
      const explainMsg: ChatMessage = {
        id: createUniqueId("sql"),
        role: "assistant",
        content: data.sql,
        timestamp,
      };
      setMessages((prev) => [...prev, explainMsg]);

      // Result
      const resultMsg: ChatMessage = {
        id: createUniqueId("res"),
        role: "result",
        content: "",
        result: data,
        timestamp,
      };
      setMessages((prev) => [...prev, resultMsg]);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Lỗi kết nối — thử lại sau";
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
      className={`fixed right-0 top-0 h-screen w-[440px] max-w-[95vw] bg-surface/95 backdrop-blur-2xl border-l border-border flex flex-col z-50 transition-transform duration-400 ease-in-out shadow-2xl ${
        isOpen ? "translate-x-0" : "translate-x-full"
      }`}
    >
      {/* Header */}
      <div className="flex items-center justify-between px-5 py-4 border-b border-border bg-gradient-to-r from-indigo-500/5 via-violet-500/5 to-purple-500/5 shrink-0 h-[65px]">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-indigo-500 to-violet-600 flex items-center justify-center shadow-lg shadow-indigo-500/20">
            <Sparkles className="w-4 h-4 text-white" />
          </div>
          <div>
            <span className="font-semibold text-sm text-text-primary">Trợ lý AI Phân tích</span>
            <p className="text-[10px] text-text-muted">Truy vấn Text-to-SQL · Gemini</p>
          </div>
        </div>
        <button
          onClick={handleClose}
          className="p-1.5 rounded-lg hover:bg-surface-3/50 text-text-muted hover:text-text-primary transition-colors cursor-pointer"
          aria-label="Đóng trợ lý AI"
        >
          <X className="w-5 h-5" />
        </button>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-4 py-4 space-y-4">
        {messages.length === 0 && (
          <div className="h-full flex flex-col items-center justify-center gap-6">
            <div className="w-20 h-20 rounded-2xl bg-gradient-to-br from-indigo-500/15 to-violet-500/15 flex items-center justify-center">
              <Sparkles className="w-10 h-10 text-indigo-500" />
            </div>
            <div className="text-center space-y-2">
              <h3 className="text-sm font-semibold text-text-primary">Tra cứu dữ liệu giao thông</h3>
              <p className="text-xs text-text-muted max-w-[280px]">
                Đặt câu hỏi bằng tiếng Việt, AI sẽ tự sinh truy vấn SQL và trực quan hóa kết quả dạng bảng hoặc biểu đồ.
              </p>
            </div>
            <div className="flex flex-wrap gap-2 justify-center max-w-[340px]">
              {SUGGESTIONS.map((s) => (
                <button
                  key={s}
                  onClick={() => handleSubmit(s)}
                  className="text-xs px-3 py-1.5 rounded-full border border-indigo-500/20 bg-indigo-500/10 text-indigo-500 font-medium hover:bg-indigo-500/20 transition-all cursor-pointer"
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
                  <p className="text-sm text-text-primary font-medium">{msg.content}</p>
                </div>
              </div>
            )}

            {/* Assistant SQL bubble */}
            {msg.role === "assistant" && (
              <div className="flex justify-start">
                <div className="max-w-[96%] w-full bg-surface-3/60 border border-border rounded-xl overflow-hidden">
                  {/* SQL header */}
                  <div className="flex items-center gap-1.5 px-3 py-1.5 bg-surface-3 border-b border-border">
                    <Code className="w-3.5 h-3.5 text-indigo-500" />
                    <span className="text-[10px] text-text-muted font-semibold uppercase tracking-wider">
                      Truy vấn SQL tự động
                    </span>
                  </div>
                  <pre className="px-3 py-2.5 text-xs text-indigo-500 font-mono overflow-x-auto whitespace-pre-wrap leading-relaxed">
                    {msg.content}
                  </pre>
                </div>
              </div>
            )}

            {/* Result */}
            {msg.role === "result" && msg.result && (
              <div className="w-full bg-surface-3/40 border border-border rounded-xl overflow-hidden">
                {/* Result header */}
                <div className="flex items-center justify-between px-3 py-2 bg-surface-3 border-b border-border">
                  <div className="flex items-center gap-2">
                    <CheckCircle className="w-3.5 h-3.5 text-emerald-500" />
                    <span className="text-[10px] text-text-muted font-semibold uppercase tracking-wider">
                      {msg.result.count} kết quả
                    </span>
                  </div>
                  <div className="flex items-center gap-1.5">
                    {msg.result.chartType === "bar" && (
                      <span className="flex items-center gap-1 text-[10px] text-amber-500 bg-amber-500/10 px-2 py-0.5 rounded-full font-medium">
                        <BarChart3 className="w-3 h-3" />
                        Cột
                      </span>
                    )}
                    <span className="text-[10px] text-text-muted bg-surface-4/40 px-2 py-0.5 rounded-full">
                      {msg.result.chartType === "bar" ? "Biểu đồ" : "Bảng dữ liệu"}
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
            <div className="bg-surface-3 border border-border rounded-xl px-4 py-3">
              <div className="flex items-center gap-2">
                <div className="flex items-center gap-1">
                  <div className="w-1.5 h-1.5 rounded-full bg-indigo-500 animate-pulse" />
                  <div className="w-1.5 h-1.5 rounded-full bg-violet-500 animate-pulse" style={{ animationDelay: "0.15s" }} />
                  <div className="w-1.5 h-1.5 rounded-full bg-purple-500 animate-pulse" style={{ animationDelay: "0.3s" }} />
                </div>
                <span className="text-xs text-text-muted ml-1">Đang phân tích dữ liệu...</span>
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="border-t border-border px-4 py-3 bg-surface shrink-0">
        {error && (
          <div className="mb-2 px-3 py-2 bg-rose-500/10 border border-rose-500/20 rounded-lg">
            <div className="flex items-start gap-2">
              <AlertTriangle className="w-3.5 h-3.5 text-rose-500 mt-0.5 shrink-0" />
              <p className="text-xs text-rose-500">{error}</p>
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
            placeholder="Hỏi về dữ liệu vi phạm..."
            className="input-field flex-1"
            disabled={loading}
          />
          <button
            onClick={() => handleSubmit(input)}
            disabled={loading || !input.trim()}
            className="btn-primary p-2.5 rounded-xl disabled:opacity-30 disabled:cursor-not-allowed cursor-pointer"
            aria-label="Gửi câu hỏi"
          >
            <Send className="w-4 h-4" />
          </button>
        </div>
      </div>
    </aside>
  );
}
