"use client";

import { useEffect, useState, useCallback, useRef } from "react";
import { useRouter } from "next/navigation";
import {
  Search,
  LayoutDashboard,
  AlertTriangle,
  Server,
  Camera,
  ShieldCheck,
  Sparkles,
  ArrowRight,
  X,
  FileText,
  Settings,
} from "lucide-react";
import { getViolationsPage, getEdgeNodes } from "@/lib/api";
import { ViolationResponse, EdgeNodeResponse } from "@/lib/types";

interface CommandPaletteProps {
  isOpen: boolean;
  onClose: () => void;
  onOpenAI?: () => void;
}

export function CommandPalette({
  isOpen,
  onClose,
  onOpenAI,
}: CommandPaletteProps) {
  const [query, setQuery] = useState("");
  const [violations, setViolations] = useState<ViolationResponse[]>([]);
  const [nodes, setNodes] = useState<EdgeNodeResponse[]>([]);
  const [loading, setLoading] = useState(false);
  const router = useRouter();
  const inputRef = useRef<HTMLInputElement>(null);

  // Đóng + reset từ khóa cho lần mở sau
  const close = useCallback(() => {
    setQuery("");
    onClose();
  }, [onClose]);

  // Esc đóng; Ctrl/Cmd+K toggle đã xử lý ở AppLayout (cửa sổ chỉ mở khi isOpen)
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape" && isOpen) {
        close();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isOpen, close]);

  // Focus input mỗi khi mở + khóa scroll body
  useEffect(() => {
    if (isOpen) {
      inputRef.current?.focus();
      document.body.style.overflow = "hidden";
    } else {
      document.body.style.overflow = "";
    }
    return () => {
      document.body.style.overflow = "";
    };
  }, [isOpen]);

  useEffect(() => {
    let active = true;
    if (!isOpen || !query.trim()) {
      // Dọn kết quả cũ khi đóng/hủy từ khóa — gói trong microtask để không
      // setState sync ngay trong thân effect.
      const t = setTimeout(() => {
        setViolations([]);
        setNodes([]);
      }, 0);
      return () => {
        active = false;
        clearTimeout(t);
      };
    }

    const timer = setTimeout(() => {
      if (active) setLoading(true);
      Promise.all([
        getViolationsPage({ plateText: query, size: 5 })
          .then((p) => p.content)
          .catch(() => [] as ViolationResponse[]),
        getEdgeNodes().catch(() => []),
      ]).then(([vData, nData]) => {
        if (active) {
          setViolations(vData.slice(0, 5));
          setNodes(
            nData.filter(
              (n) =>
                n.name.toLowerCase().includes(query.toLowerCase()) ||
                n.node_id.toLowerCase().includes(query.toLowerCase())
            ).slice(0, 4)
          );
          setLoading(false);
        }
      });
    }, 200);

    return () => {
      active = false;
      clearTimeout(timer);
    };
  }, [query, isOpen]);

  const navigateTo = useCallback(
    (path: string) => {
      router.push(path);
      close();
    },
    [router, close]
  );

  if (!isOpen) return null;

  const NAV_PAGES = [
    { label: "Bảng điều khiển", path: "/", icon: LayoutDashboard },
    { label: "Danh sách vi phạm", path: "/violations", icon: AlertTriangle },
    { label: "Duyệt hồ sơ (human-in-the-loop)", path: "/review", icon: ShieldCheck },
    { label: "Edge Nodes", path: "/nodes", icon: Server },
    { label: "Cameras trực tiếp", path: "/cameras", icon: Camera },
    { label: "Cài đặt hệ thống", path: "/settings", icon: Settings },
  ];

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center pt-20 px-4 bg-black/50 backdrop-blur-md animate-in fade-in duration-150"
      onClick={close}
      role="presentation"
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-label="Tìm kiếm toàn hệ thống"
        className="w-full max-w-2xl bg-surface border border-border rounded-2xl shadow-2xl overflow-hidden flex flex-col animate-in fade-in zoom-in-95 duration-200"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Search Header */}
        <div className="flex items-center px-4 py-3.5 border-b border-border gap-3 bg-surface-3/40">
          <Search className="w-5 h-5 text-indigo-500 shrink-0" />
          <input
            ref={inputRef}
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Tìm biển số (VD: 29-H12345), node, hoặc nhảy trang..."
            className="flex-1 bg-transparent text-sm text-text-primary placeholder:text-text-muted focus:outline-none"
            aria-label="Từ khóa tìm kiếm"
          />
          {query && (
            <button
              onClick={() => setQuery("")}
              className="p-1 rounded-md text-text-muted hover:text-text-primary text-xs cursor-pointer"
              aria-label="Xóa từ khóa"
            >
              <X className="w-4 h-4" />
            </button>
          )}
          <kbd className="hidden sm:inline-block text-[10px] text-text-muted bg-surface-4/40 px-2 py-0.5 rounded border border-border font-mono">
            ESC
          </kbd>
        </div>

        {/* Content Body */}
        <div className="max-h-[60vh] overflow-y-auto p-3 space-y-4">
          {/* Quick AI query suggestion */}
          {onOpenAI && (
            <button
              onClick={() => {
                close();
                onOpenAI();
              }}
              className="w-full flex items-center justify-between p-3 rounded-xl bg-gradient-to-r from-indigo-500/10 via-violet-500/10 to-purple-500/10 border border-indigo-500/20 hover:border-indigo-500/40 text-left transition-all group cursor-pointer"
            >
              <div className="flex items-center gap-3 min-w-0">
                <div className="w-8 h-8 rounded-lg bg-indigo-500/20 flex items-center justify-center shrink-0">
                  <Sparkles className="w-4 h-4 text-indigo-500" />
                </div>
                <div className="min-w-0">
                  <p className="text-xs font-semibold text-text-primary">
                    Hỏi Trợ lý AI
                  </p>
                  <p className="text-[11px] text-text-muted truncate">
                    Truy vấn thống kê vi phạm bằng ngôn ngữ tự nhiên
                  </p>
                </div>
              </div>
              <ArrowRight className="w-4 h-4 text-indigo-500 group-hover:translate-x-1 transition-transform shrink-0" />
            </button>
          )}

          {/* Navigation Pages */}
          {!query && (
            <div>
              <p className="text-[10px] font-semibold text-text-muted uppercase tracking-wider px-3 mb-2">
                Điều hướng nhanh
              </p>
              <div className="space-y-1">
                {NAV_PAGES.map((page) => {
                  const Icon = page.icon;
                  return (
                    <button
                      key={page.path}
                      onClick={() => navigateTo(page.path)}
                      className="w-full flex items-center justify-between px-3 py-2.5 rounded-xl hover:bg-surface-3/60 text-text-secondary hover:text-text-primary text-xs font-medium transition-colors text-left cursor-pointer"
                    >
                      <div className="flex items-center gap-2.5">
                        <Icon className="w-4 h-4 text-text-muted" />
                        <span>{page.label}</span>
                      </div>
                      <span className="text-[10px] text-text-muted font-mono">
                        Tới trang →
                      </span>
                    </button>
                  );
                })}
              </div>
            </div>
          )}

          {/* Search Results */}
          {query && (
            <div className="space-y-4">
              {loading && (
                <div className="py-8 text-center text-xs text-text-muted animate-pulse">
                  Đang tìm trong cơ sở dữ liệu &quot;{query}&quot;...
                </div>
              )}

              {/* License Plate Matches */}
              {!loading && violations.length > 0 && (
                <div>
                  <p className="text-[10px] font-semibold text-text-muted uppercase tracking-wider px-3 mb-2">
                    Hồ sơ khớp biển số ({violations.length})
                  </p>
                  <div className="space-y-1">
                    {violations.map((v) => (
                      <button
                        key={v.id}
                        onClick={() => navigateTo(`/violations/${v.id}`)}
                        className="w-full flex items-center justify-between p-3 rounded-xl hover:bg-surface-3/60 border border-transparent hover:border-border text-left transition-all cursor-pointer"
                      >
                        <div className="flex items-center gap-3 min-w-0">
                          <span className="plate-badge text-xs shrink-0">
                            {v.plate_text || "NO-PLATE"}
                          </span>
                          <div className="min-w-0">
                            <p className="text-xs font-bold text-text-primary">
                              Hồ sơ #{v.id} · Track #{v.track_id}
                            </p>
                            <p className="text-[11px] text-text-muted font-mono truncate">
                              Node: {v.node_id}
                            </p>
                          </div>
                        </div>
                        <span
                          className={`text-[10px] px-2 py-0.5 rounded-full font-semibold shrink-0 ${
                            v.status === "approved"
                              ? "status-approved"
                              : v.status === "rejected"
                                ? "status-rejected"
                                : "status-pending"
                          }`}
                        >
                          {v.status}
                        </span>
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {/* Edge Node Matches */}
              {!loading && nodes.length > 0 && (
                <div>
                  <p className="text-[10px] font-semibold text-text-muted uppercase tracking-wider px-3 mb-2">
                    Edge node khớp ({nodes.length})
                  </p>
                  <div className="space-y-1">
                    {nodes.map((n) => (
                      <button
                        key={n.node_id}
                        onClick={() => navigateTo(`/nodes/${n.node_id}`)}
                        className="w-full flex items-center justify-between p-3 rounded-xl hover:bg-surface-3/60 border border-transparent hover:border-border text-left transition-all cursor-pointer"
                      >
                        <div className="flex items-center gap-3 min-w-0">
                          <div className="w-8 h-8 rounded-lg bg-indigo-500/10 flex items-center justify-center shrink-0">
                            <Server className="w-4 h-4 text-indigo-500" />
                          </div>
                          <div className="min-w-0">
                            <p className="text-xs font-bold text-text-primary">{n.name}</p>
                            <p className="text-[11px] text-text-muted font-mono truncate">
                              {n.node_id} · {n.ip_address || "Local Edge"}
                            </p>
                          </div>
                        </div>
                        <span className="text-[10px] text-emerald-500 font-semibold font-mono shrink-0">
                          {n.status}
                        </span>
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {!loading && violations.length === 0 && nodes.length === 0 && (
                <div className="py-8 text-center">
                  <FileText className="w-8 h-8 text-text-muted mx-auto mb-2" />
                  <p className="text-xs text-text-muted font-medium">
                    Không tìm thấy kết quả cho &quot;{query}&quot;
                  </p>
                  <p className="text-[11px] text-text-muted mt-1">
                    Thử tìm theo biển số (VD: 29-H) hoặc node ID (VD: edge-node-01)
                  </p>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
