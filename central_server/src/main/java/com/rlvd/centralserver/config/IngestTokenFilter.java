package com.rlvd.centralserver.config;

import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;
import org.springframework.web.filter.OncePerRequestFilter;

import java.io.IOException;
import java.util.regex.Pattern;

/**
 * Filter ingest token cho node biên: các endpoint ingest (đọng list qua
 * {@link IngestPaths#INGEST_RULES}) yêu cầu header X-Ingest-Token khớp
 * INGEST_TOKEN.
 *
 * Nếu INGEST_TOKEN rỗng (chế độ dev) mọi ingest được chấp nhận và log
 * cảnh báo một lần — giống hành vi EDGE_API_TOKEN của node biên.
 */
@Component
public class IngestTokenFilter extends OncePerRequestFilter {

    private static final Logger log = LoggerFactory.getLogger(IngestTokenFilter.class);

    private final String ingestToken;
    private boolean warned = false;

    public IngestTokenFilter(
            @Value("${security.ingest-token:${INGEST_TOKEN:}}") String ingestToken) {
        this.ingestToken = ingestToken == null ? "" : ingestToken.trim();
    }

    @Override
    protected boolean shouldNotFilter(HttpServletRequest request) {
        // Chỉ áp cho các endpoint ingest từ edge node. Danh sách path/method
        // được định nghĩa TẬP TRUNG tại IngestPaths — SecurityConfig cũng
        // đọc từ đây nên không thể drift.
        String path = request.getRequestURI();
        String method = request.getMethod();
        return IngestPaths.INGEST_RULES.stream().noneMatch(
                rule -> matchPath(rule.method().name(), rule.path(), method, path));
    }

    private boolean matchPath(String expectedMethod, String pattern,
                              String actualMethod, String actualPath) {
        if (!expectedMethod.equals(actualMethod)) return false;
        // Spring style "/api/v1/violations/*/media" → regex "^/api/v1/violations/[^/]+/media$"
        String regex = "^" + pattern.replace("*", "[^/]+") + "$";
        return Pattern.compile(regex).matcher(actualPath).matches();
    }

    @Override
    protected void doFilterInternal(
            HttpServletRequest request,
            HttpServletResponse response,
            FilterChain filterChain) throws ServletException, IOException {

        if (ingestToken.isEmpty()) {
            if (!warned) {
                log.warn("INGEST_TOKEN chưa cấu hình — ingest từ node biên đang MỞ "
                        + "không xác thực. Đặt INGEST_TOKEN trong production.");
                warned = true;
            }
            filterChain.doFilter(request, response);
            return;
        }

        String provided = request.getHeader("X-Ingest-Token");
        if (provided == null || provided.isBlank()) {
            response.sendError(HttpServletResponse.SC_UNAUTHORIZED,
                    "Missing X-Ingest-Token header");
            return;
        }
        boolean match = java.security.MessageDigest.isEqual(
                provided.getBytes(java.nio.charset.StandardCharsets.UTF_8),
                ingestToken.getBytes(java.nio.charset.StandardCharsets.UTF_8));
        if (!match) {
            response.sendError(HttpServletResponse.SC_UNAUTHORIZED, "Invalid ingest token");
            return;
        }
        filterChain.doFilter(request, response);
    }
}
