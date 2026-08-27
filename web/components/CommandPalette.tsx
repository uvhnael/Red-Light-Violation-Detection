"use client";

import { useEffect, useState, useCallback } from "react";
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
import { getViolations, getEdgeNodes } from "@/lib/api";
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

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        if (isOpen) {
          onClose();
        } else {
          setQuery("");
        }
      }
      if (e.key === "Escape" && isOpen) {
        onClose();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isOpen, onClose]);

  useEffect(() => {
    let active = true;
    if (!isOpen || !query.trim()) {
      return;
    }

    const timer = setTimeout(() => {
      if (active) setLoading(true);
      Promise.all([
        getViolations({ plateText: query }).catch(() => []),
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
      onClose();
    },
    [router, onClose]
  );

  if (!isOpen) return null;

  const NAV_PAGES = [
    { label: "Dashboard Overview", path: "/", icon: LayoutDashboard },
    { label: "Violations List", path: "/violations", icon: AlertTriangle },
    { label: "Review Queue (Human-in-the-Loop)", path: "/review", icon: ShieldCheck },
    { label: "Edge Nodes Registry", path: "/nodes", icon: Server },
    { label: "Live Cameras", path: "/cameras", icon: Camera },
    { label: "System & Theme Settings", path: "/settings", icon: Settings },
  ];

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center pt-20 px-4 bg-black/50 backdrop-blur-md transition-all">
      <div className="w-full max-w-2xl bg-surface border border-border rounded-2xl shadow-2xl overflow-hidden flex flex-col animate-in fade-in zoom-in-95 duration-200">
        {/* Search Header */}
        <div className="flex items-center px-4 py-3.5 border-b border-border gap-3 bg-surface-3/40">
          <Search className="w-5 h-5 text-indigo-500 shrink-0" />
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search plate numbers (e.g. 29A-12345), nodes, or quick jump..."
            className="flex-1 bg-transparent text-sm text-text-primary placeholder:text-text-muted focus:outline-none"
            autoFocus
          />
          {query && (
            <button
              onClick={() => setQuery("")}
              className="p-1 rounded-md text-text-muted hover:text-text-primary text-xs cursor-pointer"
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
                onClose();
                onOpenAI();
              }}
              className="w-full flex items-center justify-between p-3 rounded-xl bg-gradient-to-r from-indigo-500/10 via-violet-500/10 to-purple-500/10 border border-indigo-500/20 hover:border-indigo-500/40 text-left transition-all group cursor-pointer"
            >
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 rounded-lg bg-indigo-500/20 flex items-center justify-center shrink-0">
                  <Sparkles className="w-4 h-4 text-indigo-500" />
                </div>
                <div>
                  <p className="text-xs font-semibold text-text-primary">
                    Ask Traffic AI Assistant
                  </p>
                  <p className="text-[11px] text-text-muted">
                    Query violation stats &amp; trends with natural language
                  </p>
                </div>
              </div>
              <ArrowRight className="w-4 h-4 text-indigo-500 group-hover:translate-x-1 transition-transform" />
            </button>
          )}

          {/* Navigation Pages */}
          {!query && (
            <div>
              <p className="text-[10px] font-semibold text-text-muted uppercase tracking-wider px-3 mb-2">
                Quick Navigation
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
                        Jump to →
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
                  Searching database for &quot;{query}&quot;...
                </div>
              )}

              {/* License Plate Matches */}
              {!loading && violations.length > 0 && (
                <div>
                  <p className="text-[10px] font-semibold text-text-muted uppercase tracking-wider px-3 mb-2">
                    Plate Match Results ({violations.length})
                  </p>
                  <div className="space-y-1">
                    {violations.map((v) => (
                      <button
                        key={v.id}
                        onClick={() => navigateTo(`/violations/${v.id}`)}
                        className="w-full flex items-center justify-between p-3 rounded-xl hover:bg-surface-3/60 border border-transparent hover:border-border text-left transition-all cursor-pointer"
                      >
                        <div className="flex items-center gap-3">
                          <span className="plate-badge text-xs">
                            {v.plate_text || "NO-PLATE"}
                          </span>
                          <div>
                            <p className="text-xs font-bold text-text-primary">
                              Record #{v.id} · Track #{v.track_id}
                            </p>
                            <p className="text-[11px] text-text-muted font-mono">
                              Node: {v.node_id}
                            </p>
                          </div>
                        </div>
                        <span
                          className={`text-[10px] px-2 py-0.5 rounded-full font-semibold ${
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
                    Edge Node Matches ({nodes.length})
                  </p>
                  <div className="space-y-1">
                    {nodes.map((n) => (
                      <button
                        key={n.node_id}
                        onClick={() => navigateTo(`/nodes/${n.node_id}`)}
                        className="w-full flex items-center justify-between p-3 rounded-xl hover:bg-surface-3/60 border border-transparent hover:border-border text-left transition-all cursor-pointer"
                      >
                        <div className="flex items-center gap-3">
                          <div className="w-8 h-8 rounded-lg bg-indigo-500/10 flex items-center justify-center shrink-0">
                            <Server className="w-4 h-4 text-indigo-500" />
                          </div>
                          <div>
                            <p className="text-xs font-bold text-text-primary">{n.name}</p>
                            <p className="text-[11px] text-text-muted font-mono">
                              {n.node_id} · {n.ip_address || "Local Edge"}
                            </p>
                          </div>
                        </div>
                        <span className="text-[10px] text-emerald-500 font-semibold font-mono">
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
                  <p className="text-xs text-text-muted font-medium">No matches found for &quot;{query}&quot;</p>
                  <p className="text-[11px] text-text-muted mt-1">
                    Try searching by plate number (e.g. 29A) or node ID (e.g. edge-node-01)
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
