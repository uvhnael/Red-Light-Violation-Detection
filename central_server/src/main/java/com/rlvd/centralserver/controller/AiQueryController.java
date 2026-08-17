package com.rlvd.centralserver.controller;

import com.rlvd.centralserver.dto.AiQueryRequest;
import com.rlvd.centralserver.dto.AiQueryResponse;
import com.rlvd.centralserver.service.AiQueryService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

/**
 * REST controller for AI Text-to-SQL assistant.
 * Receives natural language questions, returns SQL + query results + viz hint.
 */
@RestController
@RequestMapping("/api/ai")
public class AiQueryController {

    private static final Logger log = LoggerFactory.getLogger(AiQueryController.class);

    private final AiQueryService aiQueryService;

    public AiQueryController(AiQueryService aiQueryService) {
        this.aiQueryService = aiQueryService;
    }

    /**
     * POST /api/ai/query — translate natural language to SQL, execute, return results.
     */
    @PostMapping("/query")
    public ResponseEntity<AiQueryResponse> handleQuery(@RequestBody AiQueryRequest request) {
        if (request.getQuestion() == null || request.getQuestion().isBlank()) {
            return ResponseEntity.badRequest().body(errorResponse("Vui lòng nhập câu hỏi."));
        }

        if (request.getQuestion().length() > 500) {
            return ResponseEntity.badRequest().body(errorResponse("Câu hỏi quá dài (tối đa 500 ký tự)."));
        }

        log.info("AI query: {}", request.getQuestion().substring(0, Math.min(80, request.getQuestion().length())));

        AiQueryResponse response = aiQueryService.handleQuery(request);
        return ResponseEntity.ok(response);
    }

    private AiQueryResponse errorResponse(String msg) {
        AiQueryResponse r = new AiQueryResponse();
        r.setError(msg);
        return r;
    }
}