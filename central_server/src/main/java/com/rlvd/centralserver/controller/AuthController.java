package com.rlvd.centralserver.controller;

import com.rlvd.centralserver.config.JwtService;
import com.rlvd.centralserver.dto.LoginRequest;
import com.rlvd.centralserver.dto.LoginResponse;
import com.rlvd.centralserver.entity.User;
import com.rlvd.centralserver.repository.UserRepository;
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
 * Xác thực người dùng dashboard: đăng nhập lấy JWT, xem thông tin
 * tài khoản hiện tại.
 *
 * POST /api/auth/login  {username, password} → {token, role, ...}
 * GET  /api/auth/me    (Bearer token)      → {username, role, ...}
 */
@RestController
@RequestMapping("/api/auth")
public class AuthController {

    private static final Logger log = LoggerFactory.getLogger(AuthController.class);

    private final UserRepository userRepository;
    private final PasswordEncoder passwordEncoder;
    private final JwtService jwtService;

    public AuthController(UserRepository userRepository,
                          PasswordEncoder passwordEncoder,
                          JwtService jwtService) {
        this.userRepository = userRepository;
        this.passwordEncoder = passwordEncoder;
        this.jwtService = jwtService;
    }

    @PostMapping("/login")
    public ResponseEntity<LoginResponse> login(@Valid @RequestBody LoginRequest request) {
        User user = userRepository
                .findByUsernameIgnoreCase(request.getUsername())
                .orElseThrow(() -> new ResponseStatusException(
                        HttpStatus.UNAUTHORIZED, "Sai tên đăng nhập hoặc mật khẩu"));

        if (!Boolean.TRUE.equals(user.getEnabled())) {
            throw new ResponseStatusException(HttpStatus.UNAUTHORIZED, "Tài khoản đã bị khóa");
        }
        if (!passwordEncoder.matches(request.getPassword(), user.getPasswordHash())) {
            log.warn("Đăng nhập sai mật khẩu: username={}", request.getUsername());
            throw new ResponseStatusException(HttpStatus.UNAUTHORIZED,
                    "Sai tên đăng nhập hoặc mật khẩu");
        }

        LoginResponse resp = new LoginResponse();
        resp.setToken(jwtService.generateToken(user.getUsername(), user.getRole()));
        resp.setExpiresIn(jwtService.getTtlSeconds());
        resp.setUsername(user.getUsername());
        resp.setFullName(user.getFullName());
        resp.setRole(user.getRole());
        log.info("Đăng nhập thành công: username={}, role={}", user.getUsername(), user.getRole());
        return ResponseEntity.ok(resp);
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
}
