/** Kiểm tra phiên đăng nhập ở tầng server/proxy.

File này được proxy.ts import và chạy trong bundle proxy — KHÔNG được
import React hay bất cứ thứ gì kéo theo runtime browser (proxy chạy trước
khi route render và không chia sẻ module với app).
*/

/** Tên cookie chứa access JWT (do lib/auth.ts ghi phía client). */
export const AUTH_COOKIE = "rlvd_token";

/**
 * Secret ký JWT — phải trùng `JWT_SECRET` của Central Server (HS256,
 * jjwt Keys.hmacShaKeyFor dùng raw UTF-8 bytes).
 *
 * Không có mặc định: thiếu secret thì KHÔNG xác thực chữ ký (chỉ còn hạn
 * dùng), thay vì âm thầm khoá hết người dùng ra ngoài. Central mới là
 * biên bảo mật thật — nó fail-fast khi thiếu JWT_SECRET.
 */
function jwtSecret(): string | null {
  const secret = process.env["JWT_SECRET"];
  return secret && secret.length >= 32 ? secret : null;
}

function base64UrlDecode(input: string): string {
  const padded = input.replace(/-/g, "+").replace(/_/g, "/");
  return Buffer.from(padded, "base64").toString("utf8");
}

/** So sánh hằng số, chống timing attack khi đối chiếu chữ ký. */
function timingSafeEqual(a: string, b: string): boolean {
  if (a.length !== b.length) return false;
  let diff = 0;
  for (let i = 0; i < a.length; i++) diff |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return diff === 0;
}

async function hmacSha256Base64Url(data: string, secret: string): Promise<string> {
  const crypto = await import("node:crypto");
  return crypto.createHmac("sha256", secret).update(data).digest("base64url");
}

/** Payload tối thiểu của access JWT do Central cấp. */
export interface JwtClaims {
  sub?: string;
  /** Epoch giây (chuẩn JWT); dung nạp epoch ms. */
  exp?: number;
  role?: string;
  /** True khi chữ ký HS256 đã được đối chiếu với JWT_SECRET. */
  signatureVerified: boolean;
}

/**
 * Giải mã token, kiểm tra hạn dùng (`exp`) và — khi có JWT_SECRET —
 * cả chữ ký HS256. Trả về claims khi dùng được, `null` khi rác/hết hạn/sai chữ ký.
 */
export async function verifyAccessToken(
  token: string | undefined | null
): Promise<JwtClaims | null> {
  if (!token) return null;
  const parts = token.split(".");
  if (parts.length !== 3) return null;
  const [headerB64, payloadB64, signatureB64] = parts;

  try {
    const header = JSON.parse(base64UrlDecode(headerB64)) as { alg?: string };
    // Chỉ chấp nhận đúng thuật toán ta ký — chặn "alg: none" và tráo RS↔HS.
    if (header.alg !== "HS256") return null;

    const secret = jwtSecret();
    if (secret) {
      const expected = await hmacSha256Base64Url(
        `${headerB64}.${payloadB64}`,
        secret
      );
      if (!timingSafeEqual(expected, signatureB64)) return null;
    }

    const parsed = JSON.parse(base64UrlDecode(payloadB64)) as Omit<JwtClaims, "signatureVerified">;
    if (typeof parsed.exp === "number") {
      // Chuẩn JWT là epoch giây; 13 chữ số → epoch ms.
      const expMs = parsed.exp > 1e11 ? parsed.exp : parsed.exp * 1000;
      if (Date.now() >= expMs) return null;
    }
    return { ...parsed, signatureVerified: Boolean(secret) };
  } catch {
    return null;
  }
}
