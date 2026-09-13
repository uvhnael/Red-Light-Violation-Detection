package com.rlvd.centralserver.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.rlvd.centralserver.dto.AiQueryRequest;
import com.rlvd.centralserver.dto.AiQueryResponse;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Service;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.*;
import java.util.regex.Pattern;

/**
 * AI Text-to-SQL service: translates Vietnamese natural language
 * into SQL via Gemini API, executes against PostgreSQL, returns results.
 */
@Service
public class AiQueryService {

    private static final Logger log = LoggerFactory.getLogger(AiQueryService.class);

    private final JdbcTemplate jdbc;

    private final HttpClient httpClient;

    @Value("${ai.gemini.api-key:}")
    private String geminiApiKey;

    @Value("${ai.gemini.model:gemini-3.5-flash-lite}")
    private String geminiModel;

    private static final String GEMINI_URL =
            "https://generativelanguage.googleapis.com/v1beta/models/%s:generateContent?key=%s";

    private static String getSystemPrompt() {
        return """
            You are a PostgreSQL expert. Return ONLY raw PostgreSQL SELECT query.
            Rules:
            - The user asks questions in Vietnamese about a traffic violation database.
            - Translate to a single SELECT statement.
            - Return ONLY the SQL, no markdown, no backticks, no explanation.
            - Always add LIMIT 50 if not specified.
            - Use CURRENT_DATE for "today" queries (e.g., created_at::date = CURRENT_DATE).
            - Count queries: SELECT COUNT(*) as total FROM ...
            - Do NOT use DELETE, INSERT, UPDATE, DROP, CREATE, ALTER, TRUNCATE.

            Schema:
            - violations: id, event_id, node_id, track_id, frame_index, timestamp_ms,
              light_state (red/yellow/green/unknown), light_confidence,
              plate_text, plate_confidence, status (pending/approved/rejected),
              created_at, updated_at
            - edge_nodes: id, node_id, name, ip_address,
              status (online/offline/maintenance), online, last_ping, created_at

            Current date: %s
            """.formatted(java.time.LocalDate.now().toString());
    }

    private static final Pattern UNSAFE_SQL =
            Pattern.compile(
                    "\\b(DROP\\s+TABLE|DELETE\\s+FROM|INSERT\\s+INTO|UPDATE\\s+\\w+\\s+SET|" +
                    "ALTER\\s+(TABLE|DATABASE)|CREATE\\s+(TABLE|INDEX|DATABASE|SCHEMA)|" +
                    "TRUNCATE|GRANT|REVOKE)\\b",
                    Pattern.CASE_INSENSITIVE);

    private final ObjectMapper objectMapper = new ObjectMapper();

    public AiQueryService(JdbcTemplate jdbc) {
        this.jdbc = jdbc;
        this.httpClient = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(10))
                .build();
    }

    // ------------------------------------------------------------------ //
    // Public API                                                           //
    // ------------------------------------------------------------------ //

    public AiQueryResponse handleQuery(AiQueryRequest request) {
        AiQueryResponse resp = new AiQueryResponse();
        resp.setQuestion(request.getQuestion());

        // 1. Generate SQL via Gemini
        String sql;
        try {
            sql = generateSQL(request.getQuestion());
        } catch (Exception e) {
            log.error("Gemini call failed", e);
            resp.setError("Không thể kết nối AI. Vui lòng kiểm tra API key.");
            return resp;
        }

        if (sql == null || sql.isBlank()) {
            resp.setError("AI không tạo được truy vấn. Thử câu hỏi khác.");
            return resp;
        }

        resp.setSql(sql);

        // 2. Safety check
        if (UNSAFE_SQL.matcher(sql).find()) {
            resp.setError("Chỉ được phép SELECT (không được sửa/xóa dữ liệu).");
            return resp;
        }

        String safeSql = sql.trim().replaceAll(";\\s*$", "");
        if (!safeSql.toUpperCase().startsWith("SELECT") &&
            !safeSql.toUpperCase().startsWith("WITH")) {
            resp.setError("Chỉ cho phép câu SELECT.");
            return resp;
        }

        // 3. Run SQL
        List<Map<String, Object>> rows;
        try {
            rows = jdbc.queryForList(safeSql);
        } catch (Exception e) {
            log.error("SQL execution failed: {}", e.getMessage());
            resp.setError("Lỗi truy vấn: " + e.getMessage().split("\n")[0]);
            return resp;
        }

        resp.setRows(rows);
        resp.setCount(rows.size());

        if (!rows.isEmpty()) {
            resp.setColumns(new ArrayList<>(rows.get(0).keySet()));
        } else {
            resp.setColumns(List.of());
        }

        // 4. Decide chart type
        resp.setChartType(determineChartType(resp.getColumns(), rows.size()));

        // 5. Narrate the result as a staff report (second Gemini call).
        //    Nhẹ nhất có thể: thất bại thì trả bảng không lời — KHÔNG làm
        //    hỏng cả lượt hỏi chỉ vì phần tường thuật lỗi.
        try {
            resp.setAnswer(generateAnswer(request.getQuestion(), rows));
        } catch (Exception e) {
            log.warn("Answer narration failed (non-fatal): {}", e.getMessage());
        }

        return resp;
    }

    // ------------------------------------------------------------------ //
    // Gemini REST client                                                   //
    // ------------------------------------------------------------------ //

    private String generateSQL(String question) throws Exception {
        String prompt = getSystemPrompt() + "\n\nQuestion: " + question + "\nSQL:";
        return cleanSQL(callGemini(prompt, 500, 0.1));
    }

    @SuppressWarnings("unchecked")
    private String extractGeminiText(String jsonBody) throws Exception {
        Map<String, Object> root = objectMapper.readValue(jsonBody, Map.class);
        List<Map<String, Object>> candidates = (List<Map<String, Object>>) root.get("candidates");
        if (candidates == null || candidates.isEmpty()) return "";

        Map<String, Object> content = (Map<String, Object>) candidates.get(0).get("content");
        if (content == null) return "";

        List<Map<String, Object>> parts = (List<Map<String, Object>>) content.get("parts");
        if (parts == null || parts.isEmpty()) return "";

        return (String) parts.get(0).get("text");
    }

    private String cleanSQL(String raw) {
        if (raw == null) return "";
        return raw
                .replaceAll("```sql\\s*\\n?", "")
                .replaceAll("```", "")
                .replaceAll(";\\s*$", "")
                .trim();
    }

    // ------------------------------------------------------------------ //
    // Shared Gemini REST call                                              //
    // ------------------------------------------------------------------ //

    private String callGemini(String prompt, int maxTokens, double temperature) throws Exception {
        String url = String.format(GEMINI_URL, geminiModel, geminiApiKey);

        Map<String, Object> body = Map.of(
                "contents", List.of(
                        Map.of("parts", List.of(Map.of("text", prompt)))
                ),
                "generationConfig", Map.of(
                        "temperature", temperature,
                        "maxOutputTokens", maxTokens
                )
        );

        String jsonBody = objectMapper.writeValueAsString(body);

        HttpRequest httpReq = HttpRequest.newBuilder()
                .uri(URI.create(url))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(jsonBody))
                .timeout(Duration.ofSeconds(30))
                .build();

        HttpResponse<String> httpResp = httpClient.send(httpReq, HttpResponse.BodyHandlers.ofString());

        if (httpResp.statusCode() != 200) {
            throw new RuntimeException("Gemini API error " + httpResp.statusCode() + ": " +
                    httpResp.body().substring(0, Math.min(200, httpResp.body().length())));
        }

        return extractGeminiText(httpResp.body());
    }

    // ------------------------------------------------------------------ //
    // Answer narration — staff-report style                                //
    // ------------------------------------------------------------------ //

    /**
     * Bước "truy vấn xong gọi tiếp API": sau khi SQL đã chạy, gửi kết quả
     * (tối đa 20 dòng) + câu hỏi gốc cho Gemini để tường thuật lại như một
     * cán bộ giám sát giao thông đang báo cáo cho cấp trên — không lặp lại
     * bảng số liệu thô, không tiết lộ SQL.
     */
    private String generateAnswer(String question, List<Map<String, Object>> rows) throws Exception {
        String rowsJson;
        if (rows.isEmpty()) {
            rowsJson = "(không có dòng dữ liệu nào)";
        } else {
            List<Map<String, Object>> sample = rows.subList(0, Math.min(20, rows.size()));
            rowsJson = objectMapper.writeValueAsString(sample);
            if (rows.size() > 20) {
                rowsJson += "\n(... cộng " + (rows.size() - 20) + " dòng nữa)";
            }
        }

        String prompt = """
                Bạn là một cán bộ giám sát giao thông đang báo cáo kết quả truy vấn \
                cho cấp trên qua chat. Dựa trên dữ liệu thực tế dưới đây, viết câu \
                trả lời cho câu hỏi của người dùng.

                Yêu cầu về giọng điệu:
                - Ngắn gọn 2-5 câu, như một nhân viên đang báo cáo: nêu con số \
                chính, điểm nổi bật, và (nếu hợp lý) một nhận xét ngắn.
                - KHÔNG dùng ngôn ngữ kỹ thuật: tránh "truy vấn", "bảng", "cột", \
                "dữ liệu", "SQL", "hệ thống ghi nhận" — chỉ nói về nghiệp vụ \
                (vi phạm, hồ sơ, node, biển số, đèn, ca giám sát).
                - KHÔNG lặp lại toàn bộ bảng — người dùng đã thấy bảng bên dưới.
                - KHÔNG bịa số: chỉ dùng con số có trong kết quả. Nếu kết quả \
                rỗng, nói rõ hiện chưa có hồ sơ phù hợp.
                - Nếu kết quả ít hơn số lượng câu hỏi yêu cầu (ví dụ hỏi top 5 mà \
                chỉ có 1), báo đúng thực tế đó một cách tự nhiên.
                - Nếu câu hỏi không liên quan nghiệp vụ giao thông của hệ thống, \
                lịch sự từ chối và gợi ý hỏi về vi phạm giao thông.

                Current date: %s

                Câu hỏi của người dùng: %s

                Dữ liệu truy vấn được (JSON, tối đa 20 dòng đầu):
                %s
                """.formatted(
                java.time.LocalDate.now().toString(),
                question,
                rowsJson
        );

        // maxTokens 2000: gemini-2.5-flash có thinking tokens ăn vào budget —
        // 400 bị cắt giữa câu (finishReason MAX_TOKENS).
        String answer = callGemini(prompt, 2000, 0.4);
        if (answer != null) {
            answer = answer.replaceAll("```.*?```", "").trim();
        }
        return answer;
    }

    // ------------------------------------------------------------------
    // Chart type heuristic
    // ------------------------------------------------------------------

    private String determineChartType(List<String> columns, int rowCount) {
        if (rowCount == 0) return "table";

        boolean hasNumeric = columns.stream()
                .map(String::toLowerCase)
                .anyMatch(c -> c.matches("^(total|count|so_luong|avg|sum|max|min)$"));

        boolean hasCategory = columns.stream()
                .map(String::toLowerCase)
                .anyMatch(c -> c.matches(
                    "^(hour|status|light_state|node_id|name|date|month|label)$"));

        return (hasNumeric && hasCategory && rowCount <= 20) ? "bar" : "table";
    }
}