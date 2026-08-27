"use client";

import { useState, useEffect } from "react";
import {
  LayoutDashboard,
  AlertTriangle,
  Server,
  Camera,
  ShieldCheck,
  MessageSquare,
  ChevronLeft,
  ChevronRight,
  Bell,
  Search,
  Command,
  Activity,
  CheckCircle,
  X,
  Settings,
} from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { AISidebarProvider } from "@/components/AISidebarProvider";
import AISidebar from "@/components/AISidebar";
import { CommandPalette } from "@/components/CommandPalette";
import { getViolations } from "@/lib/api";

const NAV_ITEMS = [
  {
    href: "/",
    label: "Dashboard",
    icon: LayoutDashboard,
  },
  {
    href: "/violations",
    label: "Violations",
    icon: AlertTriangle,
  },
  {
    href: "/nodes",
    label: "Edge Nodes",
    icon: Server,
  },
  {
    href: "/cameras",
    label: "Cameras",
    icon: Camera,
  },
  {
    href: "/review",
    label: "Review Queue",
    icon: ShieldCheck,
    badgeKey: "pendingCount",
  },
  {
    href: "/settings",
    label: "Settings",
    icon: Settings,
  },
];

export function AppLayout({ children }: { children: React.ReactNode }) {
  const [collapsed, setCollapsed] = useState(false);
  const [aiOpen, setAiOpen] = useState(false);
  const [paletteOpen, setPaletteOpen] = useState(false);
  const [notifOpen, setNotifOpen] = useState(false);
  const [pendingCount, setPendingCount] = useState(0);
  const pathname = usePathname();

  useEffect(() => {
    let active = true;
    getViolations({ status: "pending" })
      .then((data) => {
        if (active) setPendingCount(data.length);
      })
      .catch(() => {
        if (active) setPendingCount(0);
      });
    return () => {
      active = false;
    };
  }, [pathname]);

  return (
    <AISidebarProvider>
      <div className="min-h-screen bg-surface-0 text-text-primary flex transition-colors duration-200">
        {/* ── SIDEBAR (Left) ── */}
        <aside
          className={`fixed left-0 top-0 h-screen z-40 flex flex-col bg-surface-2/95 backdrop-blur-xl border-r border-border transition-all duration-300 ${
            collapsed ? "w-[72px]" : "w-[260px]"
          }`}
        >
          {/* Logo */}
          <div className="px-5 py-5 border-b border-border flex items-center gap-3 h-[65px] shrink-0">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-indigo-500 to-violet-600 flex items-center justify-center shadow-lg shadow-indigo-500/20 shrink-0">
              <ShieldCheck className="w-5 h-5 text-white" />
            </div>
            {!collapsed && (
              <div className="overflow-hidden whitespace-nowrap">
                <span className="font-bold text-base text-text-primary tracking-tight">
                  Traffic<span className="text-indigo-500">AI</span>
                </span>
                <p className="text-[11px] text-text-muted mt-0.5 font-medium">
                  Central Control Server
                </p>
              </div>
            )}
          </div>

          {/* Nav */}
          <nav className="flex-1 px-3 py-4 space-y-1 overflow-y-auto">
            {!collapsed && (
              <p className="text-[11px] font-semibold text-text-muted uppercase tracking-[0.15em] mb-3 px-3">
                Navigation
              </p>
            )}
            {NAV_ITEMS.map((item) => {
              const isActive =
                item.href === "/"
                  ? pathname === "/"
                  : pathname.startsWith(item.href);
              const Icon = item.icon;

              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={`flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition-all duration-200 group relative ${
                    isActive
                      ? "bg-indigo-600/15 text-indigo-500 font-bold border border-indigo-500/20 shadow-sm shadow-indigo-500/5"
                      : "text-text-secondary hover:text-text-primary hover:bg-surface-3/60"
                  } ${collapsed ? "justify-center" : ""}`}
                >
                  <Icon
                    className={`w-5 h-5 shrink-0 ${
                      isActive
                        ? "text-indigo-500"
                        : "text-text-muted group-hover:text-text-primary"
                    }`}
                  />
                  {!collapsed && (
                    <span className="flex-1 truncate">{item.label}</span>
                  )}
                  {!collapsed && item.badgeKey === "pendingCount" && pendingCount > 0 && (
                    <span className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-amber-500/20 text-amber-500 border border-amber-500/30">
                      {pendingCount}
                    </span>
                  )}
                </Link>
              );
            })}
          </nav>

          {/* System status widget in sidebar */}
          {!collapsed && (
            <div className="mx-3 mb-3 p-3 rounded-xl bg-surface-3/50 border border-border space-y-1">
              <div className="flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse shadow-[0_0_6px_#10b981]" />
                <span className="text-xs font-semibold text-emerald-500">
                  System Normal
                </span>
              </div>
              <p className="text-[10px] text-text-muted">AI Inference Engine: Online</p>
            </div>
          )}

          {/* Bottom: Collapse toggle */}
          <div className="p-3 border-t border-border">
            <button
              onClick={() => setCollapsed(!collapsed)}
              className="w-full flex items-center justify-center p-2 rounded-xl text-text-muted hover:text-text-primary hover:bg-surface-3/50 transition-colors"
              title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
            >
              {collapsed ? (
                <ChevronRight className="w-4 h-4" />
              ) : (
                <ChevronLeft className="w-4 h-4" />
              )}
            </button>
          </div>
        </aside>

        {/* ── MAIN AREA ── */}
        <div
          className={`flex-1 flex flex-col min-h-screen transition-all duration-300 ${
            collapsed ? "ml-[72px]" : "ml-[260px]"
          }`}
        >
          {/* ── TOP HEADER ── */}
          <header className="sticky top-0 z-30 h-16 bg-surface/80 backdrop-blur-xl border-b border-border flex items-center px-6 gap-4">
            {/* Breadcrumb */}
            <div className="flex items-center gap-2 text-sm">
              <span className="text-text-muted">Home</span>
              <ChevronRight className="w-3.5 h-3.5 text-text-muted" />
              <span className="text-text-primary font-medium capitalize">
                {pathname === "/"
                  ? "Dashboard"
                  : pathname.slice(1).replace(/-/g, " ")}
              </span>
            </div>

            {/* Search bar trigger */}
            <div className="flex-1 max-w-md ml-auto mr-4">
              <button
                onClick={() => setPaletteOpen(true)}
                className="w-full flex items-center justify-between bg-surface-3/60 hover:bg-surface-3 border border-border hover:border-indigo-500/30 rounded-xl px-3.5 py-2 text-sm text-text-muted transition-all text-left group"
              >
                <div className="flex items-center gap-2.5">
                  <Search className="w-4 h-4 text-text-muted group-hover:text-indigo-500 transition-colors" />
                  <span className="text-xs">Search plate numbers, nodes, or quick jump...</span>
                </div>
                <kbd className="flex items-center gap-0.5 text-[10px] text-text-muted bg-surface-4/40 px-2 py-0.5 rounded-md border border-border font-mono">
                  <Command className="w-2.5 h-2.5" />K
                </kbd>
              </button>
            </div>

            {/* Right actions */}
            <div className="flex items-center gap-3">
              {/* Notification bell */}
              <button
                onClick={() => setNotifOpen(!notifOpen)}
                className="relative p-2.5 rounded-xl text-text-secondary hover:text-text-primary hover:bg-surface-3/50 transition-colors"
                title="System Notifications"
              >
                <Bell className="w-5 h-5" />
                {pendingCount > 0 && (
                  <span className="absolute top-1.5 right-1.5 w-2 h-2 rounded-full bg-amber-500 ring-2 ring-surface-0 animate-pulse" />
                )}
              </button>

              {/* User avatar */}
              <div className="flex items-center gap-3 pl-3 border-l border-border">
                <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-indigo-500 to-violet-600 flex items-center justify-center text-white text-xs font-bold shadow-md shadow-indigo-500/20">
                  CS
                </div>
                <div className="hidden sm:block">
                  <p className="text-xs font-semibold text-text-primary">CSGT Operator</p>
                  <p className="text-[10px] text-text-muted">Traffic Police Control</p>
                </div>
              </div>
            </div>
          </header>

          {/* Quick Notification Dropdown */}
          {notifOpen && (
            <div className="fixed right-6 top-20 z-40 w-80 bg-surface border border-border rounded-2xl shadow-2xl overflow-hidden p-4 space-y-3 animate-in fade-in zoom-in-95 duration-150">
              <div className="flex items-center justify-between border-b border-border pb-2">
                <h4 className="text-xs font-semibold text-text-primary flex items-center gap-2">
                  <Bell className="w-4 h-4 text-indigo-500" /> Notifications
                </h4>
                <button
                  onClick={() => setNotifOpen(false)}
                  className="text-text-muted hover:text-text-primary"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
              <div className="space-y-2 text-xs">
                {pendingCount > 0 ? (
                  <Link
                    href="/review"
                    onClick={() => setNotifOpen(false)}
                    className="block p-3 rounded-xl bg-amber-500/10 border border-amber-500/20 hover:border-amber-500/40 transition-all"
                  >
                    <p className="font-semibold text-amber-500 flex items-center gap-1.5">
                      <AlertTriangle className="w-3.5 h-3.5" /> {pendingCount} Pending Reviews
                    </p>
                    <p className="text-text-muted mt-1 text-[11px]">
                      Traffic violations require operator human-in-the-loop review.
                    </p>
                  </Link>
                ) : (
                  <div className="p-4 text-center text-text-muted">
                    <CheckCircle className="w-6 h-6 mx-auto mb-1 text-emerald-500" />
                    No pending items. System clear!
                  </div>
                )}
                <div className="p-3 rounded-xl bg-surface-3/50 border border-border">
                  <p className="font-medium text-text-primary flex items-center gap-1.5">
                    <Activity className="w-3.5 h-3.5 text-indigo-500" /> Edge Node Heartbeat
                  </p>
                  <p className="text-text-muted mt-0.5 text-[11px]">
                    All registered camera streams are connected.
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Page Content */}
          <main className="flex-1 p-6">{children}</main>
        </div>

        {/* ── AI SIDEBAR (Right) ── */}
        <AISidebar isOpen={aiOpen} onToggle={() => setAiOpen(!aiOpen)} />

        {/* ── COMMAND PALETTE MODAL (Cmd+K) ── */}
        <CommandPalette
          isOpen={paletteOpen}
          onClose={() => setPaletteOpen(false)}
          onOpenAI={() => setAiOpen(true)}
        />

        {/* ── Floating AI bubble (bottom-right) ── */}
        {!aiOpen && (
          <button
            onClick={() => setAiOpen(true)}
            className="fixed bottom-6 right-6 z-40 w-14 h-14 rounded-2xl bg-gradient-to-br from-indigo-600 to-violet-700 flex items-center justify-center shadow-xl shadow-indigo-500/30 hover:shadow-2xl hover:shadow-indigo-500/40 hover:scale-105 active:scale-95 transition-all duration-200 group cursor-pointer"
            title="AI Assistant"
          >
            <MessageSquare className="w-6 h-6 text-white group-hover:scale-110 transition-transform" />
            <span className="absolute top-0 right-0 w-3.5 h-3.5 rounded-full bg-rose-500 border-2 border-surface-0 animate-pulse" />
          </button>
        )}
      </div>
    </AISidebarProvider>
  );
}