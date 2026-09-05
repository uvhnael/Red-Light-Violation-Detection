package com.rlvd.centralserver.config;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.HttpMethod;
import jakarta.servlet.http.HttpServletResponse;
import org.springframework.security.config.annotation.method.configuration.EnableMethodSecurity;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.http.SessionCreationPolicy;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.security.web.SecurityFilterChain;
import org.springframework.security.web.authentication.UsernamePasswordAuthenticationFilter;
import org.springframework.web.cors.CorsConfiguration;
import org.springframework.web.cors.CorsConfigurationSource;
import org.springframework.web.cors.UrlBasedCorsConfigurationSource;

import java.util.List;

/**
 * Cấu hình Spring Security cho Central Server.
 *
 * Mô hình bảo mật hai luồng:
 * <ul>
 *   <li><b>Web users</b> — JWT Bearer (Authorization header) do Spring Security
 *   xử lý qua {@link JwtAuthFilter}; phân quyền theo vai trò ADMIN/OPERATOR/OFFICER.</li>
 *   <li><b>Edge nodes</b> — ingest token tĩnh (X-Ingest-Token) cho các endpoint
 *   nhận dữ liệu từ node biên (POST violations, register). Token này tách khỏi
 *   JWT người dùng vì node biên là máy chủ không phải người dùng cuối.</li>
 * </ul>
 *
 * Đường dẫn công khai: /api/auth/login, health, snapshot phải token edge
 * (đi qua proxy đã xác thực) — mọi đường dẫn còn lại yêu cầu JWT hợp lệ.
 */
@Configuration
@EnableMethodSecurity
public class SecurityConfig {

    private static final Logger log = LoggerFactory.getLogger(SecurityConfig.class);

    private final JwtService jwtService;
    private final IngestTokenFilter ingestTokenFilter;

    public SecurityConfig(JwtService jwtService, IngestTokenFilter ingestTokenFilter) {
        this.jwtService = jwtService;
        this.ingestTokenFilter = ingestTokenFilter;
    }

    @Bean
    public PasswordEncoder passwordEncoder() {
        return new BCryptPasswordEncoder();
    }

    @Bean
    public JwtAuthFilter jwtAuthFilter() {
        return new JwtAuthFilter(jwtService);
    }

    @Bean
    public SecurityFilterChain filterChain(HttpSecurity http) throws Exception {
        http
                .csrf(csrf -> csrf.disable()) // REST API stateless + JWT, không cookie session
                .cors(cors -> cors.configurationSource(corsConfigurationSource()))
                .sessionManagement(
                        session -> session.sessionCreationPolicy(SessionCreationPolicy.STATELESS))
                .authorizeHttpRequests(auth -> auth
                        // ---------- Public ----------
                        .requestMatchers("/api/auth/login").permitAll()
                        .requestMatchers("/api/health").permitAll()
                        .requestMatchers("/actuator/health", "/actuator/info").permitAll()
                        // Spring Boot forward exception tới /error — phải mở,
                        // không thì status thật (401/404) bị che thành 403.
                        .requestMatchers("/error").permitAll()
                        // ---------- Edge ingest (token tĩnh X-Ingest-Token) ----------
                        // Đọng list path đồng bộ với IngestTokenFilter qua IngestPaths.
                        // Khi thêm endpoint ingest mới: sửa IngestPaths, KHÔNG sửa tại đây.
                        .requestMatchers(
                                IngestPaths.INGEST_RULES.stream()
                                        .map(r -> r.method().name() + " " + r.path())
                                        .toArray(String[]::new))
                        .permitAll()
                        // Path chỉ cần permit (không cần token) — proxy đã authenticate.
                        .requestMatchers(
                                IngestPaths.PERMIT_ONLY_RULES.stream()
                                        .map(r -> r.method().name() + " " + r.path())
                                        .toArray(String[]::new))
                        .permitAll()
                        // ---------- Phân quyền theo vai trò ----------
                        // AI: mọi vai trò đã đăng nhập đều hỏi được (Officer dùng nhiều nhất)
                        .requestMatchers("/api/ai/**").hasAnyRole("ADMIN", "OPERATOR", "OFFICER")
                        // Quản lý node + hiệu chuẩn: OPERATOR trở lên
                        .requestMatchers("/api/v1/edge-nodes/**").hasAnyRole("ADMIN", "OPERATOR")
                        // Duyệt / cập nhật trạng thái hồ sơ: OFFICER trở lên (ADMIN gồm cả)
                        .requestMatchers(HttpMethod.PATCH, "/api/violations/*/status").hasAnyRole("ADMIN", "OFFICER")
                        .requestMatchers(HttpMethod.PUT, "/api/violations/*/status").hasAnyRole("ADMIN", "OFFICER")
                        .requestMatchers(HttpMethod.DELETE, "/api/violations/*").hasRole("ADMIN")
                        // Mọi thứ còn lại (GET violations, stats, health chi tiết): đã đăng nhập
                        .anyRequest().authenticated()
                )
                // Mặc định Spring Security trả 403 cho request chưa xác thực
                // khi không cấu hình entry point — ép về 401 chuẩn REST.
                .exceptionHandling(ex -> ex.authenticationEntryPoint(
                        (request, response, authException) ->
                                response.sendError(HttpServletResponse.SC_UNAUTHORIZED)))
                .addFilterBefore(ingestTokenFilter, UsernamePasswordAuthenticationFilter.class)
                .addFilterBefore(jwtAuthFilter(), UsernamePasswordAuthenticationFilter.class);

        log.info("Spring Security: JWT + RBAC (ADMIN/OPERATOR/OFFICER) đã bật");
        return http.build();
    }

    @Bean
    public CorsConfigurationSource corsConfigurationSource() {
        CorsConfiguration config = new CorsConfiguration();
        // Next.js proxy mọi request nên browser không cần gọi trực tiếp;
        // đặt "*" cho phép dev truy cập thẳng khi cần debug.
        config.setAllowedOriginPatterns(List.of("*"));
        config.setAllowedMethods(List.of("GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"));
        config.setAllowedHeaders(List.of("*"));
        config.setAllowCredentials(false);
        UrlBasedCorsConfigurationSource source = new UrlBasedCorsConfigurationSource();
        source.registerCorsConfiguration("/**", config);
        return source;
    }
}
