"use client";

import { useCallback, useEffect, useSyncExternalStore } from "react";
import { usePathname, useRouter } from "next/navigation";
import {
  getSession,
  clearSession,
  UNAUTHORIZED_EVENT,
  type AuthSession,
} from "@/lib/auth";

/**
 * Cổng phiên đăng nhập phía client cho toàn bộ nhóm route (dashboard).
 *
 * Proxy đã chặn ở tầng server cho mỗi request HTML, nhưng SPA vẫn còn ba
 * lỗ hổng mà proxy không thấy được:
 *  1. điều hướng client-side (Link/router.push) không đi qua proxy;
 *  2. phiên hết hạn ngay khi người dùng đang ngồi trên trang;
 *  3. API trả 401 và refresh token cũng chết → session bị clear.
 *
 * Cả ba đều được đưa về /login kèm `next` để quay lại đúng trang cũ.
 */

/** Snapshot phải so sánh được bằng Object.is — getSession() trả object mới
 * mỗi lần gọi, nên giữ tham chiếu cuối cùng và chỉ đổi khi nội dung đổi. */
let lastSnapshot: AuthSession | null = null;
function getSnapshot(): AuthSession | null {
  const next = getSession();
  const same =
    next === lastSnapshot ||
    (next !== null &&
      lastSnapshot !== null &&
      next.token === lastSnapshot.token &&
      next.expiresAt === lastSnapshot.expiresAt &&
      next.role === lastSnapshot.role &&
      next.username === lastSnapshot.username);
  if (!same) lastSnapshot = next;
  return lastSnapshot;
}

/** SSR/hydration không có localStorage → coi như chưa xác nhận được phiên. */
function getServerSnapshot(): AuthSession | null {
  return null;
}

function subscribe(onStoreChange: () => void): () => void {
  const onFocus = () => {
    // Tab mở lại sau thời gian dài: token có thể đã hết hạn.
    getSnapshot();
    onStoreChange();
  };
  const onTick = () => {
    if (getSession() === null && lastSnapshot !== null) onStoreChange();
  };
  window.addEventListener("storage", onFocus);
  window.addEventListener("focus", onFocus);
  window.addEventListener(UNAUTHORIZED_EVENT, onStoreChange);
  // Phiên hết hạn đúng lúc người dùng đang ngồi trên trang.
  const timer = window.setInterval(onTick, 15_000);
  return () => {
    window.removeEventListener("storage", onFocus);
    window.removeEventListener("focus", onFocus);
    window.removeEventListener(UNAUTHORIZED_EVENT, onStoreChange);
    window.clearInterval(timer);
  };
}

export function RequireSession({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  const pathname = usePathname();
  const session = useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot);

  const goLogin = useCallback(() => {
    router.replace(`/login?next=${encodeURIComponent(pathname)}`);
  }, [router, pathname]);

  useEffect(() => {
    if (session) return;
    // Không có session phía client: xoá luôn cookie rlvd_token (có thể còn
    // sót lại sau khi localStorage bị xoá) — nếu không proxy lại cho qua
    // và hai bên redirect nhau thành vòng lặp.
    clearSession();
    goLogin();
  }, [session, goLogin]);

  // 401 từ tầng API (refresh cũng thất bại) → lib/api.ts bắn sự kiện.
  useEffect(() => {
    const handler = () => {
      clearSession();
      goLogin();
    };
    window.addEventListener(UNAUTHORIZED_EVENT, handler);
    return () => window.removeEventListener(UNAUTHORIZED_EVENT, handler);
  }, [goLogin]);

  if (!session) {
    // Chưa xác nhận được phiên: không render nội dung dashboard ra DOM
    // (tránh lộ dữ liệu/nhấp nháy) — hiện nền trống có nhãn cho screen reader.
    return (
      <div className="min-h-screen bg-surface-0" aria-busy="true" aria-live="polite">
        <span className="sr-only">Đang kiểm tra phiên đăng nhập…</span>
      </div>
    );
  }

  return <>{children}</>;
}
