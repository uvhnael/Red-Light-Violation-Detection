package com.rlvd.centralserver.dto;

import lombok.Getter;
import lombok.Setter;

/**
 * Phản hồi sau khi đăng nhập thành công (hoặc rotate refresh token).
 *
 * Trả về CẢ access token + refresh token:
 * - accessToken: JWT, TTL ngắn (12h), stateless, gắn vào Authorization header.
 * - refreshToken: raw token (32 bytes URL-safe base64), TTL dài (7d), stateful
 *   (hash SHA-256 lưu DB), dùng để đổi access token mới khi hết hạn.
 *
 * Client PHẢI lưu cả hai và xoá refreshToken khi logout.
 */
@Getter
@Setter
public class LoginResponse {

    private String token;          // alias của accessToken, giữ để tương thích code cũ
    private String accessToken;    // JWT Bearer
    private String tokenType = "Bearer";
    private long expiresIn;        // access token TTL (giây)

    private String refreshToken;   // raw refresh token (KHÔNG phải JWT)
    private long refreshExpiresIn; // refresh token TTL (giây)

    private String username;
    private String fullName;
    private String role;
}