package com.rlvd.centralserver.entity;

import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import java.time.Instant;

/**
 * Refresh token cho JWT authentication.
 *
 * Bảo mật:
 * - Token field lưu SHA-256 hash của refresh token thô, KHÔNG lưu raw value.
 *   Nếu DB leak, attacker không replay được token (cần raw value).
 * - TTL mặc định 7 ngày (REFRESH_TOKEN_TTL_SECONDS, cấu hình qua env).
 * - Rotation: mỗi lần refresh, token cũ bị revoke + cấp token mới.
 *   Nếu token đã revoke được dùng → dấu hiệu đánh cắp → revoke TẤT CẢ
 *   token của user (TODO khi cần).
 *
 * Tách khỏi session JWT (stateless) vì:
 * - Access token ngắn hạn (12h), dùng nhiều → phải stateless để scale.
 * - Refresh token dài hạn (7d), dùng ít → cần stateful để revoke khi mập.
 */
@Entity
@Table(name = "refresh_tokens", indexes = {
        @Index(name = "idx_refresh_tokens_user", columnList = "user_id"),
        @Index(name = "idx_refresh_tokens_hash", columnList = "token_hash", unique = true),
})
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class RefreshToken {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(name = "user_id", nullable = false)
    private Long userId;

    /** SHA-256 hex digest của raw refresh token. Không lưu raw. */
    @Column(name = "token_hash", nullable = false, length = 64, unique = true)
    private String tokenHash;

    @Column(name = "expires_at", nullable = false)
    private Instant expiresAt;

    /** Null = còn hiệu lực. Set = đã revoke (logout hoặc rotate). */
    @Column(name = "revoked_at")
    private Instant revokedAt;

    @Column(name = "created_at", nullable = false)
    private Instant createdAt;

    /** User-agent / IP của client lúc cấp — hỗ trợ audit khi cần. */
    @Column(name = "user_agent", length = 255)
    private String userAgent;
}