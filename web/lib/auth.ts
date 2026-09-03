"use client";

/**
 * Quản lý phiên đăng nhập JWT trên client.
 *
 * Token + thông tin người dùng lưu tại localStorage (key "rlvd_auth") và
 * ghi đồng bộ sang cookie rlvd_token để middleware Next.js đọc được khi
 * SSR/redirect (localStorage không tới được từ middleware).
 */
import { useEffect, useState } from "react";

export type Role = "ADMIN" | "OPERATOR" | "OFFICER";

export interface AuthSession {
  token: string;
  username: string;
  fullName: string;
  role: Role;
  expiresAt: number; // epoch ms
}

const STORAGE_KEY = "rlvd_auth";
const COOKIE_NAME = "rlvd_token";

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
 *
 * Lazy-init đọc localStorage ngay trong lần render đầu (client-side),
 * không cần setState trong effect.
 */
export function useSession(): AuthSession | null {
  // useState initializer chỉ chạy client-side vì hook này luôn nằm trong
  // client component ("use client").
  const [session, setSession] = useState<AuthSession | null>(() => getSession());
  useEffect(() => {
    const onStorage = () => setSession(getSession());
    window.addEventListener("storage", onStorage);
    return () => window.removeEventListener("storage", onStorage);
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
    token: data.token,
    username: data.username,
    fullName: data.full_name ?? data.fullName ?? "",
    role: data.role,
    expiresAt: Date.now() + (data.expires_in ?? 43200) * 1000,
  };
  saveSession(session);
  return session;
}
