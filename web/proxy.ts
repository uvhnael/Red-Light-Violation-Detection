import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";
import { AUTH_COOKIE, verifyAccessToken } from "@/lib/session-guard";

/**
 * Proxy bảo vệ route phía server (Next.js 16: `middleware.ts` đã bị
 * khai tử và đổi tên thành `proxy.ts`).
 *
 * Chạy TRƯỚC khi route render, nên người chưa đăng nhập không bao giờ
 * thấy HTML của trang dashboard — bị 307 thẳng về /login kèm `next`
 * để quay lại đúng chỗ sau khi đăng nhập.
 *
 * Kiểm tra hai lớp, không chỉ "có cookie":
 *  1. cookie rlvd_token phải tồn tại;
 *  2. JWT phải còn hạn (`exp`) và đúng chữ ký HS256 khi JWT_SECRET được
 *     cấu hình — cookie cũ/rác không đủ để lọt qua.
 *
 * Lưu ý: đây là cổng UX. Biên bảo mật thật vẫn là Central Server
 * (JwtAuthFilter trả 401/403) — xem docs proxy: "Always verify
 * authentication and authorization inside each route rather than
 * relying on Proxy alone".
 */
export async function proxy(request: NextRequest) {
  const { pathname } = request.nextUrl;

  // Trang đăng nhập không cần token; đã đăng nhập rồi thì đẩy về trang chủ
  // để không kẹt lại ở form login.
  if (pathname.startsWith("/login")) {
    const claims = await verifyAccessToken(
      request.cookies.get(AUTH_COOKIE)?.value
    );
    if (claims) {
      const url = request.nextUrl.clone();
      url.pathname = "/";
      url.search = "";
      return NextResponse.redirect(url);
    }
    return NextResponse.next();
  }

  const token = request.cookies.get(AUTH_COOKIE)?.value;
  const claims = await verifyAccessToken(token);
  if (claims) {
    return NextResponse.next();
  }

  // Chưa đăng nhập / token hết hạn: xoá cookie chết rồi redirect.
  const url = request.nextUrl.clone();
  url.pathname = "/login";
  url.search = `?next=${encodeURIComponent(pathname)}`;
  const response = NextResponse.redirect(url);
  if (token) {
    response.cookies.set({
      name: AUTH_COOKIE,
      value: "",
      path: "/",
      maxAge: 0,
      sameSite: "lax",
    });
  }
  return response;
}

export const config = {
  // Bỏ qua API route (đã có Central/edge tự xác thực qua Bearer token),
  // tài nguyên tĩnh và tối ưu ảnh.
  matcher: [
    "/((?!api|edge-api|_next/static|_next/image|favicon.ico).*)",
  ],
};
