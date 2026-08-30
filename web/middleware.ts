import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";

/**
 * Middleware bảo vệ route phía server.
 *
 * Kiểm tra cookie rlvd_token (được lib/auth.ts ghi khi đăng nhập).
 * Chưa có token → redirect sang /login kèm đường dẫn gốc để quay lại
 * sau khi đăng nhập.
 */
export function middleware(request: NextRequest) {
  const { pathname } = request.nextUrl;
  const token = request.cookies.get("rlvd_token")?.value;

  // Trang login + tài nguyên tĩnh không cần token
  if (
    pathname.startsWith("/login") ||
    pathname.startsWith("/_next") ||
    pathname === "/favicon.ico"
  ) {
    return NextResponse.next();
  }

  if (!token) {
    const url = request.nextUrl.clone();
    url.pathname = "/login";
    url.search = `?next=${encodeURIComponent(pathname)}`;
    return NextResponse.redirect(url);
  }
  return NextResponse.next();
}

export const config = {
  matcher: ["/((?!api|_next/static|_next/image|favicon.ico).*)"],
};
