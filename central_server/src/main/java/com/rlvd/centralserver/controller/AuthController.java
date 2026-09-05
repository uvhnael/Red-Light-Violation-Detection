package com.rlvd.centralserver.controller;

import com.rlvd.centralserver.config.JwtService;
import com.rlvd.centralserver.dto.LoginRequest;
import com.rlvd.centralserver.dto.LoginResponse;
import com.rlvd.centralserver.dto.LogoutRequest;
import com.rlvd.centralserver.dto.RefreshRequest;
import com.rlvd.centralserver.entity.User;
import com.rlvd.centralserver.repository.UserRepository;
import com.rlvd.centralserver.service.RefreshTokenService;
import com.rlvd.centralserver.service.RefreshTokenService.IssuedToken;
import com.rlvd.centralserver.service.RefreshTokenService.RotationResult;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.validation.Valid;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.server.ResponseStatusException;

import java.util.Map;

/**
 * Xác thực người dùng dashboard:
 * - POST /api/auth/login    {username, password} → access + refresh tokens
 * - POST /api/auth/refresh  {refreshToken}      → rotate (access + refresh mới)
 * - POST /api/auth/logout   {refreshToken}      → revoke refresh (idempotent)
 * - GET  /api/auth/me       (Bearer access)     → thông tin user hiện tại
 *
 * Refresh token model:
 * - Access JWT TTL 12h — mỗi request API đính kèm.
 * - Refresh token TTL 7d — opaque (SHA-256 hash lưu DB), chỉ dùng khi
 *   access hết hạn để rotate. Mỗi lần refresh → revoke cũ + cấp cặp mới.
 */
@RestController
@RequestMapping("/api/auth")
public class AuthController {

    private static final Logger log = LoggerFactory.getLogger(AuthController.class);

    private final UserRepository userRepository;
    private final PasswordEncoder passwordEncoder;
    private final JwtService jwtService;
    private final RefreshTokenService refreshTokenService;

    public AuthController(
            UserRepository userRepository,
            PasswordEncoder passwordEncoder,
            JwtService jwtService,
            RefreshTokenService refreshTokenService) {
        this.userRepository = userRepository;
        this.passwordEncoder = passwordEncoder;
        this.jwtService = jwtService;
        this.refreshTokenService = refreshTokenService;
    }

    @PostMapping("/login")
    public ResponseEntity<LoginResponse> login(
            @Valid @RequestBody LoginRequest request,
            HttpServletRequest httpRequest) {
        User user = authenticate(request);
        return ResponseEntity.ok(buildLoginResponse(user, httpRequest));
    }

    @PostMapping("/refresh")
    public ResponseEntity<LoginResponse> refresh(
            @Valid @RequestBody RefreshRequest body,
            HttpServletRequest httpRequest) {
        RotationResult result = refreshTokenService.rotateRefreshToken(
                body.getRefreshToken(),
                httpRequest.getHeader("User-Agent"));
        // Lấy username từ access token mới (đã verify chữ ký).
        String username = jwtService.validate(result.accessToken()).getSubject();
        User user = userRepository.findByUsername(username)
                .orElseThrow(() -> new ResponseStatusException(
                        HttpStatus.UNAUTHORIZED, "Token rotation thất bại"));
        LoginResponse resp = new LoginResponse();
        resp.setAccessToken(result.accessToken());
        resp.setToken(result.accessToken());
        resp.setExpiresIn(result.accessTokenTtlSeconds());
        resp.setRefreshToken(result.refreshToken().rawToken());
        resp.setRefreshExpiresIn(result.refreshToken().ttlSeconds());
        resp.setUsername(user.getUsername());
        resp.setFullName(user.getFullName());
        resp.setRole(user.getRole());
        return ResponseEntity.ok(resp);
    }

    @PostMapping("/logout")
    public ResponseEntity<Map<String, String>> logout(@Valid @RequestBody LogoutRequest body) {
        refreshTokenService.revokeRefreshToken(body.getRefreshToken());
        return ResponseEntity.ok(Map.of("message", "Đã đăng xuất"));
    }

    @GetMapping("/me")
    public ResponseEntity<Map<String, Object>> me() {
        Authentication auth = SecurityContextHolder.getContext().getAuthentication();
        if (auth == null || !auth.isAuthenticated()
                || !(auth.getPrincipal() instanceof String username)) {
            throw new ResponseStatusException(HttpStatus.UNAUTHORIZED, "Chưa đăng nhập");
        }
        User user = userRepository.findByUsername(username)
                .orElseThrow(() -> new ResponseStatusException(
                        HttpStatus.UNAUTHORIZED, "Tài khoản không tồn tại"));
        return ResponseEntity.ok(Map.of(
                "username", user.getUsername(),
                "full_name", user.getFullName() == null ? "" : user.getFullName(),
                "role", user.getRole()));
    }

    // ------------------------------------------------------------------ //
    // Helpers                                                             //
    // ------------------------------------------------------------------ //

    private User authenticate(LoginRequest request) {
        User user = userRepository
                .findByUsernameIgnoreCase(request.getUsername())
                .orElseThrow(() -> new ResponseStatusException(
                        HttpStatus.UNAUTHORIZED, "Sai tên đăng nhập hoặc mật khẩu"));
        if (!Boolean.TRUE.equals(user.getEnabled())) {
            throw new ResponseStatusException(HttpStatus.UNAUTHORIZED, "Tài khoản đã bị khoá");
        }
        if (!passwordEncoder.matches(request.getPassword(), user.getPasswordHash())) {
            log.warn("Đăng nhập sai mật khẩu: username={}", request.getUsername());
            throw new ResponseStatusException(HttpStatus.UNAUTHORIZED,
                    "Sai tên đăng nhập hoặc mật khẩu");
        }
        return user;
    }

    private LoginResponse buildLoginResponse(User user, HttpServletRequest httpRequest) {
        IssuedToken refresh = refreshTokenService.issueRefreshToken(
                user, httpRequest.getHeader("User-Agent"));

        LoginResponse resp = new LoginResponse();
        String accessToken = jwtService.generateToken(user.getUsername(), user.getRole());
        resp.setAccessToken(accessToken);
        resp.setToken(accessToken);  // alias để code cũ dùng .token không vỡ
        resp.setExpiresIn(jwtService.getTtlSeconds());
        resp.setRefreshToken(refresh.rawToken());
        resp.setRefreshExpiresIn(refresh.ttlSeconds());
        resp.setUsername(user.getUsername());
        resp.setFullName(user.getFullName());
        resp.setRole(user.getRole());
        log.info("Đăng nhập thành công: username={}, role={}", user.getUsername(), user.getRole());
        return resp;
    }
}