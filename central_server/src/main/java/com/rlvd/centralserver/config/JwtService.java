package com.rlvd.centralserver.config;

import io.jsonwebtoken.Claims;
import io.jsonwebtoken.JwtException;
import io.jsonwebtoken.Jwts;
import io.jsonwebtoken.security.Keys;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import javax.crypto.SecretKey;
import java.nio.charset.StandardCharsets;
import java.time.Instant;
import java.util.Date;

/**
 * Sinh và xác thực JWT (HS256).
 *
 * Secret đọc từ biến môi trường JWT_SECRET — bắt buộc ≥ 32 ký tự.
 * Nếu thiếu, server ném lỗi khi khởi động (fail-fast) thay vì chạy
 * với secret yếu.
 */
@Service
public class JwtService {

    private static final Logger log = LoggerFactory.getLogger(JwtService.class);

    /** Thời gian token sống mặc định: 12 giờ (đúng một ca trực). */
    public static final long DEFAULT_TTL_SECONDS = 12 * 3600;

    private final SecretKey key;
    private final long ttlSeconds;

    public JwtService(
            @Value("${jwt.secret:${JWT_SECRET:}}") String secret,
            @Value("${jwt.ttl-seconds:${JWT_TTL_SECONDS:43200}}") long ttlSeconds) {
        if (secret == null || secret.length() < 32) {
            throw new IllegalStateException(
                    "JWT_SECRET phải được cấu hình và dài tối thiểu 32 ký tự.");
        }
        this.key = Keys.hmacShaKeyFor(secret.getBytes(StandardCharsets.UTF_8));
        this.ttlSeconds = ttlSeconds > 0 ? ttlSeconds : DEFAULT_TTL_SECONDS;
        log.info("JWT service sẵn sàng (ttl={}s)", this.ttlSeconds);
    }

    /** Sinh access token cho người dùng. */
    public String generateToken(String username, String role) {
        Instant now = Instant.now();
        return Jwts.builder()
                .subject(username)
                .claim("role", role)
                .issuedAt(Date.from(now))
                .expiration(Date.from(now.plusSeconds(ttlSeconds)))
                .signWith(key)
                .compact();
    }

    /** Sinh token với TTL tuỳ chỉnh (dùng cho refresh / remember-me nếu cần). */
    public String generateToken(String username, String role, long ttlSecondsOverride) {
        Instant now = Instant.now();
        return Jwts.builder()
                .subject(username)
                .claim("role", role)
                .issuedAt(Date.from(now))
                .expiration(Date.from(now.plusSeconds(ttlSecondsOverride)))
                .signWith(key)
                .compact();
    }

    /**
     * Xác thực và trích claims. Trả về null nếu token sai/hết hạn —
     * caller xử lý 401.
     */
    public Claims validate(String token) {
        try {
            return Jwts.parser().verifyWith(key).build()
                    .parseSignedClaims(token).getPayload();
        } catch (JwtException | IllegalArgumentException e) {
            return null;
        }
    }

    public long getTtlSeconds() {
        return ttlSeconds;
    }
}
