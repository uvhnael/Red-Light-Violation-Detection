"use client";

import { useState, useEffect, useRef, useCallback } from "react";
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
  LogOut,
  Menu,
} from "lucide-react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { AISidebarProvider } from "@/components/AISidebarProvider";
import AISidebar from "@/components/AISidebar";
import { CommandPalette } from "@/components/CommandPalette";
import { ToastProvider } from "@/components/Toast";
import { getViolationCounts } from "@/lib/api";
import { useSession, logout } from "@/lib/auth";

const ROLE_LABELS: Record<string, string> = {
  ADMIN: "Quản trị viên",
  OPERATOR: "Kỹ thuật vận hành",
  OFFICER: "Cán bộ xử lý vi phạm",
};

const NAV_ITEMS = [
  {
    href: "/",
    label: "Bảng điều khiển",
    icon: LayoutDashboard,
  },
  {
    href: "/violations",
    label: "Vi phạm",
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
    label: "Duyệt hồ sơ",
    icon: ShieldCheck,
    badgeKey: "pendingCount",
  },
  {
    href: "/settings",
    label: "Cài đặt",
    icon: Settings,
  },
];

/** Nhãn breadcrumb tiếng Việt cho từng route. */
const CRUMB_LABELS: Record<string, string> = {
  "": "Bảng điều khiển",
  violations: "Vi phạm",
  nodes: "Edge Nodes",
  cameras: "Cameras",
  review: "Duyệt hồ sơ",
  settings: "Cài đặt",
};

export function AppLayout({ children }: { children: React.ReactNode }) {
  const [collapsed, setCollapsed] = useState(false);
  const [mobileNavOpen, setMobileNavOpen] = useState(false);
  const [aiOpen, setAiOpen] = useState(false);
  const [paletteOpen, setPaletteOpen] = useState(false);
  const [notifOpen, setNotifOpen] = useState(false);
  const [userMenuOpen, setUserMenuOpen] = useState(false);
  const [pendingCount, setPendingCount] = useState(0);
  const pathname = usePathname();
  const router = useRouter();
  const session = useSession();

  // Click-outside cho dropdown user + notification
  const userMenuRef = useRef<HTMLDivElement>(null);
  const notifRef = useRef<HTMLDivElement>(null);
  useClickOutside(userMenuRef, () => setUserMenuOpen(false));
  useClickOutside(notifRef, () => setNotifOpen(false));

  // Đóng mobile nav khi chuyển route — pattern "adjust state khi props
  // đổi": so sánh với giá trị render trước đó thay vì setState trong effect.
  const [lastPathname, setLastPathname] = useState(pathname);
  if (pathname !== lastPathname) {
    setLastPathname(pathname);
    if (mobileNavOpen) setMobileNavOpen(false);
  }

  // ESC đóng overlay mobile nav
  useEffect(() => {
    if (!mobileNavOpen) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setMobileNavOpen(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [mobileNavOpen]);

  const handleLogout = async () => {
    // Revoke refresh token phía server (idempotent) trước khi clear local.
    await logout(session);
    router.replace("/login");
  };

  useEffect(() => {
    let active = true;
    getViolationCounts()
      .then((counts) => {
        if (active) setPendingCount(counts.pending);
      })
      .catch(() => {
        if (active) setPendingCount(0);
      });
    return () => {
      active = false;
    };
  }, [pathname]);

  const crumb =
    CRUMB_LABELS[pathname.replace(/^\/+/, "").split("/")[0] ?? ""] ??
    pathname.slice(1);

  const sidebarContent = (
    <>
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
      <nav className="flex-1 px-3 py-4 space-y-1 overflow-y-auto" aria-label="Điều hướng chính">
        {!collapsed && (
          <p className="text-[11px] font-semibold text-text-muted uppercase tracking-[0.15em] mb-3 px-3">
            Điều hướng
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
              aria-current={isActive ? "page" : undefined}
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
              Hệ thống hoạt động
            </span>
          </div>
          <p className="text-[10px] text-text-muted">AI Inference Engine: Online</p>
        </div>
      )}

      {/* Bottom: Collapse toggle — desktop only */}
      <div className="p-3 border-t border-border hidden lg:block">
        <button
          onClick={() => setCollapsed(!collapsed)}
          className="w-full flex items-center justify-center p-2 rounded-xl text-text-muted hover:text-text-primary hover:bg-surface-3/50 transition-colors"
          aria-label={collapsed ? "Mở rộng thanh bên" : "Thu gọn thanh bên"}
          aria-expanded={!collapsed}
        >
          {collapsed ? (
            <ChevronRight className="w-4 h-4" />
          ) : (
            <ChevronLeft className="w-4 h-4" />
          )}
        </button>
      </div>
    </>
  );

  return (
    <ToastProvider>
      <AISidebarProvider>
        <div className="min-h-screen bg-surface-0 text-text-primary flex transition-colors duration-200">
          {/* ── SIDEBAR (Left, desktop) ── */}
          <aside
            className={`fixed left-0 top-0 h-screen z-40 hidden lg:flex flex-col bg-surface-2/95 backdrop-blur-xl border-r border-border transition-all duration-300 ${
              collapsed ? "w-[72px]" : "w-[260px]"
            }`}
          >
            {sidebarContent}
          </aside>

          {/* ── MOBILE NAV OVERLAY ── */}
          {mobileNavOpen && (
            <div
              className="fixed inset-0 z-[55] bg-black/60 backdrop-blur-sm lg:hidden animate-in fade-in duration-150"
              onClick={() => setMobileNavOpen(false)}
              role="presentation"
            >
              <aside
                className="fixed left-0 top-0 h-full w-[270px] max-w-[85vw] flex flex-col bg-surface-2 border-r border-border shadow-2xl animate-in slide-in-from-left duration-200"
                role="dialog"
                aria-modal="true"
                aria-label="Menu điều hướng"
              >
                <button
                  onClick={() => setMobileNavOpen(false)}
                  className="absolute top-4 right-3 p-2 rounded-lg text-text-muted hover:text-text-primary hover:bg-surface-3/60 transition-colors"
                  aria-label="Đóng menu"
                >
                  <X className="w-4 h-4" />
                </button>
                {sidebarContent}
              </aside>
            </div>
          )}

          {/* ── MAIN AREA ── */}
          <div
            className={`flex-1 flex flex-col min-h-screen transition-all duration-300 ${
              collapsed ? "lg:ml-[72px]" : "lg:ml-[260px]"
            }`}
          >
            {/* ── TOP HEADER ── */}
            <header className="sticky top-0 z-30 h-16 bg-surface/80 backdrop-blur-xl border-b border-border flex items-center px-4 sm:px-6 gap-3 sm:gap-4">
              {/* Mobile menu trigger */}
              <button
                onClick={() => setMobileNavOpen(true)}
                className="lg:hidden p-2 rounded-xl text-text-secondary hover:text-text-primary hover:bg-surface-3/50 transition-colors"
                aria-label="Mở menu điều hướng"
              >
                <Menu className="w-5 h-5" />
              </button>

              {/* Breadcrumb */}
              <div className="flex items-center gap-2 text-sm min-w-0">
                <span className="text-text-muted hidden sm:inline">Trang chủ</span>
                <ChevronRight className="w-3.5 h-3.5 text-text-muted hidden sm:inline" />
                <span className="text-text-primary font-medium capitalize truncate">
                  {crumb}
                </span>
              </div>

              {/* Search bar trigger */}
              <div className="flex-1 max-w-md ml-auto mr-0 sm:mr-4">
                <button
                  onClick={() => setPaletteOpen(true)}
                  className="w-full flex items-center justify-between bg-surface-3/60 hover:bg-surface-3 border border-border hover:border-indigo-500/30 rounded-xl px-3.5 py-2 text-sm text-text-muted transition-all text-left group"
                  aria-label="Mở thanh tìm kiếm (phím tắt Ctrl+K)"
                >
                  <div className="flex items-center gap-2.5 min-w-0">
                    <Search className="w-4 h-4 text-text-muted group-hover:text-indigo-500 transition-colors shrink-0" />
                    <span className="text-xs truncate">
                      Tìm biển số, node, trang...
                    </span>
                  </div>
                  <kbd className="hidden sm:flex items-center gap-0.5 text-[10px] text-text-muted bg-surface-4/40 px-2 py-0.5 rounded-md border border-border font-mono shrink-0">
                    <Command className="w-2.5 h-2.5" />K
                  </kbd>
                </button>
              </div>

              {/* Right actions */}
              <div className="flex items-center gap-2 sm:gap-3">
                {/* Notification bell */}
                <div className="relative" ref={notifRef}>
                  <button
                    onClick={() => setNotifOpen(!notifOpen)}
                    className="relative p-2.5 rounded-xl text-text-secondary hover:text-text-primary hover:bg-surface-3/50 transition-colors"
                    aria-label="Thông báo hệ thống"
                    aria-expanded={notifOpen}
                    aria-haspopup="true"
                  >
                    <Bell className="w-5 h-5" />
                    {pendingCount > 0 && (
                      <span className="absolute top-1.5 right-1.5 w-2 h-2 rounded-full bg-amber-500 ring-2 ring-surface-0 animate-pulse" />
                    )}
                  </button>

                  {/* Notification Dropdown */}
                  {notifOpen && (
                    <div className="absolute right-0 top-12 z-50 w-80 max-w-[calc(100vw-2rem)] bg-surface border border-border rounded-2xl shadow-2xl overflow-hidden p-4 space-y-3 animate-in fade-in zoom-in-95 duration-150">
                      <div className="flex items-center justify-between border-b border-border pb-2">
                        <h4 className="text-xs font-semibold text-text-primary flex items-center gap-2">
                          <Bell className="w-4 h-4 text-indigo-500" /> Thông báo
                        </h4>
                        <button
                          onClick={() => setNotifOpen(false)}
                          className="text-text-muted hover:text-text-primary"
                          aria-label="Đóng thông báo"
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
                              <AlertTriangle className="w-3.5 h-3.5" /> {pendingCount} hồ sơ chờ duyệt
                            </p>
                            <p className="text-text-muted mt-1 text-[11px]">
                              Các vi phạm cần cán bộ xác nhận trước khi lập biên bản.
                            </p>
                          </Link>
                        ) : (
                          <div className="p-4 text-center text-text-muted">
                            <CheckCircle className="w-6 h-6 mx-auto mb-1 text-emerald-500" />
                            Không có mục nào chờ xử lý.
                          </div>
                        )}
                        <div className="p-3 rounded-xl bg-surface-3/50 border border-border">
                          <p className="font-medium text-text-primary flex items-center gap-1.5">
                            <Activity className="w-3.5 h-3.5 text-indigo-500" /> Edge Node Heartbeat
                          </p>
                          <p className="text-text-muted mt-0.5 text-[11px]">
                            Toàn bộ luồng camera đã đăng ký đang kết nối.
                          </p>
                        </div>
                      </div>
                    </div>
                  )}
                </div>

                {/* User avatar + menu */}
                <div className="relative flex items-center gap-3 pl-2 sm:pl-3 border-l border-border" ref={userMenuRef}>
                  <button
                    onClick={() => setUserMenuOpen(!userMenuOpen)}
                    className="flex items-center gap-3 group"
                    aria-label="Menu tài khoản"
                    aria-expanded={userMenuOpen}
                    aria-haspopup="true"
                  >
                    <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-indigo-500 to-violet-600 flex items-center justify-center text-white text-xs font-bold shadow-md shadow-indigo-500/20">
                      {session?.fullName?.charAt(0).toUpperCase() || session?.username?.charAt(0).toUpperCase() || "U"}
                    </div>
                    <div className="hidden sm:block text-left">
                      <p className="text-xs font-semibold text-text-primary">
                        {session?.fullName || session?.username || "Chưa đăng nhập"}
                      </p>
                      <p className="text-[10px] text-text-muted">
                        {session?.role ? ROLE_LABELS[session.role] : ""}
                      </p>
                    </div>
                  </button>
                  {userMenuOpen && (
                    <div className="absolute right-0 top-10 z-50 w-44 bg-surface border border-border rounded-xl shadow-2xl p-2 animate-in fade-in zoom-in-95 duration-150">
                      <button
                        onClick={handleLogout}
                        className="w-full flex items-center gap-2 px-3 py-2 text-xs text-rose-400 hover:bg-rose-500/10 rounded-lg transition-colors"
                      >
                        <LogOut className="w-3.5 h-3.5" /> Đăng xuất
                      </button>
                    </div>
                  )}
                </div>
              </div>
            </header>

            {/* Page Content */}
            <main id="main-content" className="flex-1 p-4 sm:p-6">{children}</main>
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
              aria-label="Mở trợ lý AI"
            >
              <MessageSquare className="w-6 h-6 text-white group-hover:scale-110 transition-transform" />
              <span className="absolute top-0 right-0 w-3.5 h-3.5 rounded-full bg-rose-500 border-2 border-surface-0 animate-pulse" />
            </button>
          )}
        </div>
      </AISidebarProvider>
    </ToastProvider>
  );
}

/** Hook: đóng dropdown khi click ra ngoài phần tử ref. */
function useClickOutside(
  ref: React.RefObject<HTMLElement | null>,
  onClose: () => void
) {
  const handler = useCallback(
    (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        onClose();
      }
    },
    [ref, onClose]
  );
  useEffect(() => {
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [handler]);
}
