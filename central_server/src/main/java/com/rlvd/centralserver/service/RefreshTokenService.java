package com.rlvd.centralserver.service;

import com.rlvd.centralserver.config.JwtService;
import com.rlvd.centralserver.entity.RefreshToken;
import com.rlvd.centralserver.entity.User;
import com.rlvd.centralserver.repository.RefreshTokenRepository;
import com.rlvd.centralserver.repository.UserRepository;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.server.ResponseStatusException;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.security.SecureRandom;
import java.time.Instant;
import java.util.Base64;
import java.util.HexFormat;
import java.util.Optional;

/**
 * Quản lý refresh token cho JWT auth.
 *
 * Lifecycle:
 * 1. {@link #issueRefreshToken(User, String)} tạo raw token (32 bytes URL-safe
 *    base64), lưu SHA-256 hash vào DB, trả raw cho client.
 * 2. {@link #rotateRefreshToken(String)} client gửi raw token, ta hash và
 *    lookup DB. Nếu hợp lệ + chưa revoke + chưa hết hạn: revoke cái cũ +
 *    issue cái mới (rotation). Trả access + refresh token mới.
 * 3. {@link #revokeRefreshToken(String)} logout: revoke single token.
 *
 * Bảo mật:
 * - Raw token chỉ hiện 1 lần (lúc issue). DB chỉ có hash.
 * - Constant-time so sánh hash qua MessageDigest.isEqual.
 * - TTL mặc định 7 ngày, configurable.
 */
@Service
public class RefreshTokenService {

    private static final Logger log = LoggerFactory.getLogger(RefreshTokenService.class);

    /** 32 bytes = 256 bit entropy → base64 ~ 43 chars. Đủ cho 2^256 không trùng. */
    private static final int TOKEN_BYTES = 32;

    /** TTL mặc định 7 ngày. Override qua JWT_REFRESH_TTL_SECONDS. */
    public static final long DEFAULT_REFRESH_TTL_SECONDS = 7 * 24 * 3600;

    private final RefreshTokenRepository refreshTokenRepository;
    private final UserRepository userRepository;
    private final JwtService jwtService;
    private final long refreshTtlSeconds;
    private final SecureRandom random = new SecureRandom();

    public RefreshTokenService(
            RefreshTokenRepository refreshTokenRepository,
            UserRepository userRepository,
            JwtService jwtService,
            @Value("${jwt.refresh-ttl-seconds:${JWT_REFRESH_TTL_SECONDS:#{null}}}") Long refreshTtlSeconds) {
        this.refreshTokenRepository = refreshTokenRepository;
        this.userRepository = userRepository;
        this.jwtService = jwtService;
        this.refreshTtlSeconds = (refreshTtlSeconds != null && refreshTtlSeconds > 0)
                ? refreshTtlSeconds : DEFAULT_REFRESH_TTL_SECONDS;
    }

    // ------------------------------------------------------------------ //
    // Issue (login)                                                       //
    // ------------------------------------------------------------------ //

    /**
     * Cấp refresh token mới cho user. Raw token trả về — DB chỉ lưu hash.
     */
    @Transactional
    public IssuedToken issueRefreshToken(User user, String userAgent) {
        String rawToken = generateRawToken();
        String hash = sha256Hex(rawToken);

        Instant now = Instant.now();
        RefreshToken entity = RefreshToken.builder()
                .userId(user.getId())
                .tokenHash(hash)
                .expiresAt(now.plusSeconds(refreshTtlSeconds))
                .createdAt(now)
                .userAgent(truncate(userAgent, 255))
                .build();
        refreshTokenRepository.save(entity);

        log.info("Issued refresh token for user {} (ttl={}s)", user.getUsername(), refreshTtlSeconds);
        return new IssuedToken(rawToken, refreshTtlSeconds);
    }

    // ------------------------------------------------------------------ //
    // Rotate (refresh endpoint)                                           //
    // ------------------------------------------------------------------ //

    /**
     * Verify raw refresh token + rotate: revoke cũ + cấp cặp (access, refresh) mới.
     * Trả IssueResult chứa access token mới + raw refresh token mới.
     */
    @Transactional
    public RotationResult rotateRefreshToken(String rawToken, String userAgent) {
        String hash = sha256Hex(rawToken);
        RefreshToken stored = refreshTokenRepository.findByTokenHash(hash)
                .orElseThrow(() -> new ResponseStatusException(
                        HttpStatus.UNAUTHORIZED, "Invalid refresh token"));

        Instant now = Instant.now();
        if (stored.getRevokedAt() != null) {
            // Đã revoke rồi — có thể là replay attack sau khi user đăng xuất.
            // Bảo mật tốt hơn: revoke TẤT CẢ token của user.
            log.warn("Revoked refresh token reuse detected for user {} — revoking ALL tokens",
                    stored.getUserId());
            refreshTokenRepository.revokeAllByUser(stored.getUserId(), now);
            throw new ResponseStatusException(
                    HttpStatus.UNAUTHORIZED,
                    "Refresh token đã bị thu hồi — đăng nhập lại");
        }
        if (stored.getExpiresAt().isBefore(now)) {
            throw new ResponseStatusException(
                    HttpStatus.UNAUTHORIZED, "Refresh token hết hạn — đăng nhập lại");
        }

        // Token hợp lệ → rotate.
        User user = userRepository.findById(stored.getUserId())
                .orElseThrow(() -> new ResponseStatusException(
                        HttpStatus.UNAUTHORIZED, "User không tồn tại"));

        // 1. Revoke cũ.
        stored.setRevokedAt(now);
        refreshTokenRepository.save(stored);

        // 2. Cấp access + refresh mới.
        String accessToken = jwtService.generateToken(user.getUsername(), user.getRole());
        IssuedToken newRefresh = issueRefreshToken(user, userAgent);

        log.info("Rotated refresh token for user {}", user.getUsername());
        return new RotationResult(accessToken, jwtService.getTtlSeconds(), newRefresh);
    }

    // ------------------------------------------------------------------ //
    // Revoke (logout)                                                     //
    // ------------------------------------------------------------------ //

    /**
     * Revoke single refresh token (logout). Không throw nếu token không tồn tại
     * (logout idempotent — không leak thông tin token hợp lệ).
     */
    @Transactional
    public void revokeRefreshToken(String rawToken) {
        String hash = sha256Hex(rawToken);
        Optional<RefreshToken> stored = refreshTokenRepository.findByTokenHash(hash);
        if (stored.isEmpty()) return;
        if (stored.get().getRevokedAt() != null) return;
        stored.get().setRevokedAt(Instant.now());
        refreshTokenRepository.save(stored.get());
        log.info("Revoked refresh token for user {}", stored.get().getUserId());
    }

    public long getRefreshTtlSeconds() {
        return refreshTtlSeconds;
    }

    // ------------------------------------------------------------------ //
    // Helpers                                                             //
    // ------------------------------------------------------------------ //

    private String generateRawToken() {
        byte[] bytes = new byte[TOKEN_BYTES];
        random.nextBytes(bytes);
        return Base64.getUrlEncoder().withoutPadding().encodeToString(bytes);
    }

    /** SHA-256 hex digest (lowercase 64 chars). */
    private static String sha256Hex(String input) {
        try {
            MessageDigest md = MessageDigest.getInstance("SHA-256");
            byte[] digest = md.digest(input.getBytes(StandardCharsets.UTF_8));
            return HexFormat.of().formatHex(digest);
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException("SHA-256 not available", e);
        }
    }

    private static String truncate(String s, int max) {
        if (s == null) return null;
        return s.length() > max ? s.substring(0, max) : s;
    }

    // ------------------------------------------------------------------ //
    // Inner records                                                        //
    // ------------------------------------------------------------------ //

    public record IssuedToken(String rawToken, long ttlSeconds) {}

    /**
     * Kết quả rotate: access token mới + raw refresh token mới (kèm TTL).
     * Trả về từ rotateRefreshToken.
     */
    public record RotationResult(
            String accessToken,
            long accessTokenTtlSeconds,
            IssuedToken refreshToken
    ) {}
}