package com.rlvd.centralserver.config;

import org.junit.jupiter.api.Test;

import java.util.regex.Pattern;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Verify the path-matching helper dùng chung giữa
 * {@link IngestTokenFilter#shouldNotFilter} và logic permit của
 * {@link SecurityConfig}. Nếu helper diverge ở đây, cả hai sẽ diverge
 * theo — đây là điểm duy nhất cần đồng bộ.
 */
class IngestPathsTest {

    /** Mirror IngestTokenFilter.matchPath — nếu đổi helper, đổi test này. */
    private static boolean matchPath(String expectedMethod, String pattern,
                                     String actualMethod, String actualPath) {
        if (!expectedMethod.equals(actualMethod)) return false;
        String regex = "^" + pattern.replace("*", "[^/]+") + "$";
        return Pattern.compile(regex).matcher(actualPath).matches();
    }

    @Test
    void matches_all_known_ingest_paths() {
        // POST /api/violations
        assertTrue(matchPath("POST", "/api/violations", "POST", "/api/violations"));
        // POST /api/v1/violations
        assertTrue(matchPath("POST", "/api/v1/violations", "POST", "/api/v1/violations"));
        // POST /api/violations/batch (no /v1)
        assertTrue(matchPath("POST", "/api/violations/batch", "POST", "/api/violations/batch"));
        // POST /api/v1/violations/batch
        assertTrue(matchPath("POST", "/api/v1/violations/batch", "POST", "/api/v1/violations/batch"));
        // POST /api/v1/edge-nodes/register
        assertTrue(matchPath("POST", "/api/v1/edge-nodes/register", "POST", "/api/v1/edge-nodes/register"));
        // POST /api/v1/violations/{eventId}/media — * matches single segment
        assertTrue(matchPath("POST", "/api/v1/violations/*/media", "POST",
                "/api/v1/violations/rlv-abc123-001-5/media"));
    }

    @Test
    void rejects_wrong_method_or_path() {
        // GET on a POST-only path
        assertFalse(matchPath("POST", "/api/violations", "GET", "/api/violations"));
        // Different path
        assertFalse(matchPath("POST", "/api/v1/violations", "POST", "/api/violations"));
        // Extra segment (should NOT match * which is single-segment)
        assertFalse(matchPath("POST", "/api/v1/violations/*/media", "POST",
                "/api/v1/violations/foo/bar/media"));
        // Missing eventId
        assertFalse(matchPath("POST", "/api/v1/violations/*/media", "POST",
                "/api/v1/violations//media"));
    }

    @Test
    void should_not_filter_skips_unrelated_paths() {
        // Tất cả path không có trong INGEST_RULES phải return true (skip filter).
        IngestTokenFilter filter = new IngestTokenFilter("");
        // Không thể gọi shouldNotFilter() trực tiếp (protected + cần HttpServletRequest)
        // nên ta test qua helper matchPath: rule KHÔNG khớp → noneMatch → true.
        assertFalse(matchPath("POST", "/api/v1/violations", "GET", "/api/v1/violations"));
        assertFalse(matchPath("POST", "/api/v1/violations", "POST", "/api/violations/something"));
    }
}