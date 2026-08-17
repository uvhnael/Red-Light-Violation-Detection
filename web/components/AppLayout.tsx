"use client";

import { useState } from "react";
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
} from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { AISidebarProvider } from "@/components/AISidebarProvider";
import AISidebar from "@/components/AISidebar";

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
  },
];

export function AppLayout({ children }: { children: React.ReactNode }) {
  const [collapsed, setCollapsed] = useState(false);
  const [aiOpen, setAiOpen] = useState(false);
  const pathname = usePathname();

  return (
    <AISidebarProvider>
      <div className="flex min-h-screen bg-surface-0">
        {/* ── LEFT SIDEBAR ── */}
        <aside
          className={`fixed left-0 top-0 h-screen z-40 flex flex-col bg-zinc-950/95 backdrop-blur-xl border-r border-white/5 transition-all duration-300 ${
            collapsed ? "w-[72px]" : "w-[260px]"
          }`}
        >
          {/* Logo */}
          <div className="px-5 py-5 border-b border-white/5 flex items-center gap-3 h-[65px] shrink-0">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-indigo-500 to-violet-600 flex items-center justify-center shadow-lg shadow-indigo-500/20 shrink-0">
              <ShieldCheck className="w-5 h-5 text-white" />
            </div>
            {!collapsed && (
              <div className="overflow-hidden whitespace-nowrap">
                <span className="font-bold text-base text-white tracking-tight">
                  Traffic<span className="text-indigo-400">AI</span>
                </span>
                <p className="text-[11px] text-zinc-400 mt-0.5 font-medium">
                  Central Server
                </p>
              </div>
            )}
          </div>

          {/* Nav */}
          <nav className="flex-1 px-3 py-4 space-y-1 overflow-y-auto">
            {!collapsed && (
              <p className="text-[11px] font-semibold text-zinc-400 uppercase tracking-[0.15em] mb-3 px-3">
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
                  className={`flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition-all duration-200 group ${
                    isActive
                      ? "bg-indigo-600/15 text-indigo-400 border border-indigo-500/15 shadow-sm shadow-indigo-500/5"
                      : "text-zinc-400 hover:text-zinc-100 hover:bg-zinc-800/60"
                  } ${collapsed ? "justify-center" : ""}`}
                >
                  <Icon
                    className={`w-5 h-5 shrink-0 ${
                      isActive
                        ? "text-indigo-400"
                        : "text-zinc-500 group-hover:text-zinc-300"
                    }`}
                  />
                  {!collapsed && <span>{item.label}</span>}
                </Link>
              );
            })}
          </nav>

          {/* Bottom: Collapse toggle */}
          <div className="p-3 border-t border-white/5">
            {/* Collapse toggle */}
            <button
              onClick={() => setCollapsed(!collapsed)}
              className="w-full flex items-center justify-center p-2 rounded-xl text-zinc-500 hover:text-zinc-300 hover:bg-zinc-800/50 transition-colors"
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
          <header className="sticky top-0 z-40 h-16 bg-surface-0/80 backdrop-blur-xl border-b border-white/5 flex items-center px-6 gap-4">
            {/* Breadcrumb */}
            <div className="flex items-center gap-2 text-sm">
              <span className="text-zinc-400">Home</span>
              <ChevronRight className="w-3.5 h-3.5 text-zinc-600" />
              <span className="text-zinc-200 font-medium">
                {pathname === "/"
                  ? "Dashboard"
                  : pathname.slice(1).charAt(0).toUpperCase() + pathname.slice(2)}
              </span>
            </div>

            {/* Search bar */}
            <div className="flex-1 max-w-md ml-auto mr-4">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-zinc-500" />
                <input
                  type="text"
                  placeholder="Search violations, nodes..."
                  className="w-full bg-zinc-900/80 border border-white/5 rounded-xl pl-10 pr-16 py-2 text-sm text-zinc-100 placeholder:text-zinc-500 focus:outline-none focus:border-indigo-500/40 focus:ring-2 focus:ring-indigo-500/10 transition-all"
                />
                <kbd className="absolute right-3 top-1/2 -translate-y-1/2 flex items-center gap-0.5 text-[10px] text-zinc-500 bg-zinc-700/50 px-2 py-0.5 rounded-md border border-zinc-600/50">
                  <Command className="w-2.5 h-2.5" />K
                </kbd>
              </div>
            </div>

            {/* Right actions */}
            <div className="flex items-center gap-3">
              {/* Notification bell */}
              <button className="relative p-2 rounded-xl text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/50 transition-colors">
                <Bell className="w-5 h-5" />
                <span className="absolute top-1.5 right-1.5 w-2 h-2 rounded-full bg-rose-500 ring-2 ring-zinc-950 animate-pulse" />
              </button>

              {/* User avatar */}
              <div className="flex items-center gap-3 pl-3 border-l border-white/5">
                <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-indigo-500 to-violet-600 flex items-center justify-center text-white text-xs font-bold shadow-md shadow-indigo-500/20">
                  AD
                </div>
                <div className="hidden sm:block">
                  <p className="text-sm font-medium text-zinc-100">Admin</p>
                  <p className="text-[11px] text-zinc-500">Operator</p>
                </div>
              </div>
            </div>
          </header>

          {/* Page Content */}
          <main className="flex-1 p-6">{children}</main>
        </div>

        {/* ── AI SIDEBAR (Right) ── */}
        <AISidebar isOpen={aiOpen} onToggle={() => setAiOpen(!aiOpen)} />

        {/* ── Floating AI bubble (bottom-right) ── */}
        {!aiOpen && (
          <button
            onClick={() => setAiOpen(true)}
            className="fixed bottom-6 right-6 z-50 w-14 h-14 rounded-2xl bg-gradient-to-br from-indigo-600 to-violet-700 flex items-center justify-center shadow-xl shadow-indigo-500/30 hover:shadow-2xl hover:shadow-indigo-500/40 hover:scale-105 active:scale-95 transition-all duration-200 group"
            title="AI Assistant"
          >
            <MessageSquare className="w-6 h-6 text-white group-hover:scale-110 transition-transform" />
            <span className="absolute top-0 right-0 w-3.5 h-3.5 rounded-full bg-rose-500 border-2 border-zinc-950 animate-pulse" />
          </button>
        )}
      </div>
    </AISidebarProvider>
  );
}