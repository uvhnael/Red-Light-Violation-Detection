package com.rlvd.centralserver.config;

import org.springframework.http.HttpMethod;

import java.util.List;
import java.util.Set;

/**
 * Single source of truth cho đường dẫn ingest từ edge node.
 *
 * Hai nơi phải đồng bộ (đã từng drift, mất 263 evidence rows):
 * 1. {@link SecurityConfig} permitAll() — để Spring Security không chặn.
 * 2. {@link IngestTokenFilter} shouldNotFilter() — để filter kiểm token.
 *
 * Khi thêm endpoint ingest mới:
 * 1. Thêm path vào {@HERING_ED_ED_PATH} bên dưới.
 * 2. Không cần sửa SecurityConfig / IngestTokenFilter — cả hai đọc từ đây.
 *
 * Lưu ý: những path KHÔNG cần ingest token (chỉ cần permit cho Spring Security)
 * — VD GET media blob proxy, GET calibration snapshot proxy — vẫn nằm trong
 * SecurityConfig vì chúng không cần token check.
 */
public final class IngestPaths {

    private IngestPaths() {}

    /**
     * Đường dẫn yêu cầu X-Ingest-Token hợp lệ (cùng token với JSON batch).
     * Cả SecurityConfig và IngestTokenFilter đọc từ đây.
     */
    public static final List<PathRule> INGEST_RULES = List.of(
            new PathRule(HttpMethod.POST, "/api/violations"),
            new PathRule(HttpMethod.POST, "/api/v1/violations"),
            new PathRule(HttpMethod.POST, "/api/violations/batch"),
            new PathRule(HttpMethod.POST, "/api/v1/violations/batch"),
            new PathRule(HttpMethod.POST, "/api/v1/edge-nodes/register"),
            new PathRule(HttpMethod.POST, "/api/v1/violations/*/media")
    );

    /**
     * Path chỉ cần permit (không cần token check) — thường là GET proxy từ
     * web đã authenticate. IngestTokenFilter BỎ QUA các path này.
     * SecurityConfig thêm permit cho các path này qua additionalPermitOnly.
     */
    public static final List<PathRule> PERMIT_ONLY_RULES = List.of(
            new PathRule(HttpMethod.GET, "/api/v1/violations/*/media/blob"),
            new PathRule(HttpMethod.GET, "/api/v1/edge-nodes/*/calibration/snapshot")
    );

    public static Set<String> allIngestPaths() {
        return INGEST_RULES.stream().map(PathRule::path).collect(java.util.stream.Collectors.toSet());
    }

    /**
     * Rule đơn giản: HTTP method + path pattern.
     */
    public record PathRule(HttpMethod method, String path) {}
}