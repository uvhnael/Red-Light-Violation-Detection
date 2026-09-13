/**Quản lý phiên đăng nhập JWT trên client.

Token + thông tin người dùng lưu tại localStorage (key "rlvd_auth") và
ghi đồng bộ sang cookie rlvd_token để proxy.ts (Next.js) đọc được khi
kiểm tra ở tầng server (localStorage không tới được từ proxy).

Refresh token model:
* accessToken — JWT, TTL 12h, dùng cho Authorization header.
* refreshToken — raw token từ backend (SHA-256 hash trong DB), TTL 7d.
  Lưu localStorage, dùng để đổi access token mới khi hết hạn.
  Không lưu vào cookie (HttpOnly không khả thi vì client cần đọc để gọi
  /api/auth/refresh) — chấp nhận rủi ro XSS để có rotation liền mạch.
*/
import { useEffect, useState } from "react";

export type Role = "ADMIN" | "OPERATOR" | "OFFICER";

export interface AuthSession {
  token: string;        // access JWT
  refreshToken: string; // raw refresh token (rotation qua /api/auth/refresh)
  username: string;
  fullName: string;
  role: Role;
  expiresAt: number;    // epoch ms — access token hết hạn
}

const STORAGE_KEY = "rlvd_auth";
const COOKIE_NAME = "rlvd_token";

/**
 * Tên sự kiện bắn ra khi phiên bị mất/hết hạn mà app không tự đăng xuất
 * (ví dụ 401 sau khi refresh thất bại). RequireSession lắng nghe để đưa
 * người dùng về /login ngay cả khi họ đang ngồi trên trang dashboard.
 */
const UNAUTHORIZED_EVENT = "rlvd:unauthorized";

export function notifyUnauthorized(): void {
  if (typeof window === "undefined") return;
  window.dispatchEvent(new Event(UNAUTHORIZED_EVENT));
}

export { UNAUTHORIZED_EVENT };

function writeCookie(token: string | null) {
  if (typeof document === "undefined") return;
  if (token) {
    document.cookie = `${COOKIE_NAME}=${encodeURIComponent(token)}; path=/; SameSite=Lax`;
  } else {
    document.cookie = `${COOKIE_NAME}=; path=/; Max-Age=0; SameSite=Lax`;
  }
}

export function saveSession(session: AuthSession) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(session));
  writeCookie(session.token);
}

export function clearSession() {
  localStorage.removeItem(STORAGE_KEY);
  writeCookie(null);
}

export function getSession(): AuthSession | null {
  if (typeof window === "undefined") return null;
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const session: AuthSession = JSON.parse(raw);
    if (session.expiresAt && Date.now() > session.expiresAt) {
      clearSession();
      return null;
    }
    return session;
  } catch {
    return null;
  }
}

/** Hook React lấy session — null khi chưa đăng nhập/hết hạn.

Lazy-init đọc localStorage ngay trong lần render đầu (client-side),
không cần setState trong effect.
*/
export function useSession(): AuthSession | null {
  // useState initializer chỉ chạy client-side vì hook này luôn nằm trong
  // client component ("use client").
  const [session, setSession] = useState<AuthSession | null>(() => getSession());
  useEffect(() => {
    const onStorage = () => setSession(getSession());
    const onUnauthorized = () => setSession(null);
    window.addEventListener("storage", onStorage);
    window.addEventListener(UNAUTHORIZED_EVENT, onUnauthorized);
    // Tab được mở lại sau thời gian dài: token có thể đã hết hạn.
    const onFocus = () => setSession(getSession());
    window.addEventListener("focus", onFocus);
    return () => {
      window.removeEventListener("storage", onStorage);
      window.removeEventListener(UNAUTHORIZED_EVENT, onUnauthorized);
      window.removeEventListener("focus", onFocus);
    };
  }, []);
  return session;
}

/**
 * Kiểm tra vai trò có được phép thực hiện hành động.
 * ADMIN > OPERATOR > OFFICER (thứ tự quyền giảm dần).
 */
const ROLE_LEVEL: Record<Role, number> = {
  ADMIN: 3,
  OPERATOR: 2,
  OFFICER: 1,
};

export function hasRole(session: AuthSession | null, minRole: Role): boolean {
  if (!session) return false;
  return ROLE_LEVEL[session.role] >= ROLE_LEVEL[minRole];
}

/** Đăng nhập qua central — trả session hoặc ném lỗi tiếng Việt. */
export async function login(
  username: string,
  password: string
): Promise<AuthSession> {
  const res = await fetch("/api/auth/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    const msg =
      body.detail || body.message || "Đăng nhập thất bại — kiểm tra lại thông tin";
    throw new Error(msg);
  }
  const data = await res.json();
  const session: AuthSession = {
    token: data.access_token ?? data.token,
    refreshToken: data.refresh_token,
    username: data.username,
    fullName: data.full_name ?? data.fullName ?? "",
    role: data.role,
    expiresAt: Date.now() + (data.expires_in ?? 43200) * 1000,
  };
  saveSession(session);
  return session;
}

/**
 * Đổi refresh token lấy access token mới — gọi khi access JWT hết hạn.
 *
 * Backend xử lý rotation: token cũ bị revoke, cấp cặp mới.
 * Trả về session mới (đã lưu localStorage + cookie) hoặc throw nếu
 * refresh token đã hết hạn/bị revoke → caller phải logout.
 */
export async function refreshAccessToken(session: AuthSession): Promise<AuthSession> {
  const res = await fetch("/api/auth/refresh", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ refreshToken: session.refreshToken }),
  });
  if (!res.ok) {
    // Refresh token hết hạn hoặc bị revoke → xoá session, caller xử lý.
    clearSession();
    throw new Error("Phiên đăng nhập đã hết hạn — đăng nhập lại");
  }
  const data = await res.json();
  const newSession: AuthSession = {
    token: data.access_token ?? data.token,
    refreshToken: data.refresh_token,  // token mới sau rotation
    username: data.username,
    fullName: data.full_name ?? data.fullName ?? "",
    role: data.role,
    expiresAt: Date.now() + (data.expires_in ?? 43200) * 1000,
  };
  saveSession(newSession);
  return newSession;
}

/**
 * Đăng xuất: revoke refresh token phía server + clear local session.
 * Endpoint idempotent — luôn clear dù server có thành công hay không.
 */
export async function logout(session: AuthSession | null): Promise<void> {
  if (session?.refreshToken) {
    try {
      await fetch("/api/auth/logout", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ refreshToken: session.refreshToken }),
      });
    } catch {
      // Network error → vẫn clear local để UX không kẹt.
    }
  }
  clearSession();
}