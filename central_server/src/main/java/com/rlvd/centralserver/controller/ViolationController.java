package com.rlvd.centralserver.controller;

import com.rlvd.centralserver.dto.ViolationBatchRequest;
import com.rlvd.centralserver.dto.ViolationBatchResponse;
import com.rlvd.centralserver.dto.ViolationRequest;
import com.rlvd.centralserver.dto.ViolationResponse;
import com.rlvd.centralserver.service.ViolationService;
import jakarta.validation.Valid;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.time.ZonedDateTime;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;
import java.util.List;
import java.util.Map;

/**
 * REST controller — the "API Gateway" that receives violations from Edge Nodes
 * and exposes query/management endpoints for the frontend dashboard.
 */
@Slf4j
@RestController
@RequestMapping("/api")
public class ViolationController {

    private final ViolationService violationService;

    public ViolationController(ViolationService violationService) {
        this.violationService = violationService;
    }

    // ------------------------------------------------------------------ //
    // Ingest endpoints (Edge Node → Central Server)                       //
    // ------------------------------------------------------------------ //

    /**
     * POST /api/violations — primary endpoint for edge nodes.
     * This matches the edge node default: CENTRAL_SERVER_URL=http://central-server:8000/api/violations
     */
    @PostMapping("/violations")
    public ResponseEntity<ViolationResponse> receiveViolation(
            @Valid @RequestBody ViolationRequest request,
            @RequestHeader(value = "X-Node-ID", defaultValue = "unknown") String nodeId) {

        log.info("Received violation from node '{}': event_id={}, light_state={}",
                nodeId, request.getEventId(), request.getLightState());

        ViolationResponse response = violationService.createViolation(request, nodeId);
        return ResponseEntity.status(HttpStatus.CREATED).body(response);
    }

    /**
     * POST /api/v1/violations — alias endpoint for versioned API compatibility.
     */
    @PostMapping("/v1/violations")
    public ResponseEntity<ViolationResponse> receiveViolationV1(
            @Valid @RequestBody ViolationRequest request,
            @RequestHeader(value = "X-Node-ID", defaultValue = "unknown") String nodeId) {

        return receiveViolation(request, nodeId);
    }

    /**
     * POST /api/violations/batch — batch ingest from an edge node outbox flush.
     * Idempotent: duplicate event_ids are skipped and reported, not errors.
     * Body shape: {"violations": [ {...}, {...} ]}
     */
    @PostMapping("/violations/batch")
    public ResponseEntity<ViolationBatchResponse> receiveViolationBatch(
            @Valid @RequestBody ViolationBatchRequest request,
            @RequestHeader(value = "X-Node-ID", defaultValue = "unknown") String nodeId) {

        List<ViolationRequest> violations = request.getViolations();
        if (violations == null) {
            violations = List.of();
        }
        log.info("Received violation batch from node '{}': {} item(s)",
                nodeId, violations.size());

        ViolationBatchResponse response =
                violationService.createViolationBatch(violations, nodeId);
        return ResponseEntity.status(HttpStatus.CREATED).body(response);
    }

    /**
     * POST /api/v1/violations/batch — versioned alias for the batch endpoint.
     */
    @PostMapping("/v1/violations/batch")
    public ResponseEntity<ViolationBatchResponse> receiveViolationBatchV1(
            @Valid @RequestBody ViolationBatchRequest request,
            @RequestHeader(value = "X-Node-ID", defaultValue = "unknown") String nodeId) {

        return receiveViolationBatch(request, nodeId);
    }

    // ------------------------------------------------------------------ //
    // Query endpoints (Frontend Dashboard)                                //
    // ------------------------------------------------------------------ //

    /**
     * GET /api/violations — list violations with optional filters.
     */
    @GetMapping("/violations")
    public ResponseEntity<List<ViolationResponse>> listViolations(
            @RequestParam(required = false) String status,
            @RequestParam(required = false) String nodeId,
            @RequestParam(required = false) String plateText) {

        List<ViolationResponse> results;

        if (status != null && !status.isBlank()) {
            results = violationService.getViolationsByStatus(status);
        } else if (nodeId != null && !nodeId.isBlank()) {
            results = violationService.getViolationsByNodeId(nodeId);
        } else if (plateText != null && !plateText.isBlank()) {
            results = violationService.searchByPlateText(plateText);
        } else {
            results = violationService.getAllViolations();
        }

        return ResponseEntity.ok(results);
    }

    /**
     * GET /api/violations/{id} — get a single violation by database ID.
     */
    @GetMapping("/violations/{id}")
    public ResponseEntity<ViolationResponse> getViolation(@PathVariable Long id) {
        return ResponseEntity.ok(violationService.getViolation(id));
    }

    /**
     * GET /api/violations/event/{eventId} — get a violation by event ID.
     */
    @GetMapping("/violations/event/{eventId}")
    public ResponseEntity<ViolationResponse> getViolationByEventId(
            @PathVariable String eventId) {
        return ResponseEntity.ok(violationService.getViolationByEventId(eventId));
    }

    // ------------------------------------------------------------------ //
    // Management endpoints (Human-in-the-loop review)                     //
    // ------------------------------------------------------------------ //

    /**
     * PATCH /api/violations/{id}/status — update review status.
     * Body: {"status": "confirmed"} or {"status": "rejected"}
     */
    @PatchMapping("/violations/{id}/status")
    public ResponseEntity<ViolationResponse> updateStatus(
            @PathVariable Long id,
            @RequestBody Map<String, String> body) {

        String newStatus = body.get("status");
        if (newStatus == null || newStatus.isBlank()) {
            return ResponseEntity.badRequest().build();
        }

        log.info("Updating violation {} status to '{}'", id, newStatus);
        return ResponseEntity.ok(violationService.updateViolationStatus(id, newStatus));
    }

    /**
     * PUT /api/violations/{id}/status — alias for frontend compatibility.
     */
    @PutMapping("/violations/{id}/status")
    public ResponseEntity<ViolationResponse> updateStatusViaPut(
            @PathVariable Long id,
            @RequestBody Map<String, String> body) {

        return updateStatus(id, body);
    }

    /**
     * DELETE /api/violations/{id} — delete a violation record.
     */
    @DeleteMapping("/violations/{id}")
    public ResponseEntity<Void> deleteViolation(@PathVariable Long id) {
        violationService.deleteViolation(id);
        return ResponseEntity.noContent().build();
    }

    // ------------------------------------------------------------------ //
    // Statistics & Health                                                  //
    // ------------------------------------------------------------------ //

    /**
     * GET /api/stats — aggregate violation statistics.
     */
    @GetMapping("/stats")
    public ResponseEntity<Map<String, Object>> getStats() {
        return ResponseEntity.ok(violationService.getStats());
    }

    /**
     * GET /api/health — simple health check endpoint.
     */
    @GetMapping("/health")
    public ResponseEntity<Map<String, Object>> health() {
        return ResponseEntity.ok(Map.of(
                "status", "ok",
                "service", "central-server",
                "timestamp", ZonedDateTime.now(ZoneId.of("Asia/Ho_Chi_Minh")).format(DateTimeFormatter.ISO_OFFSET_DATE_TIME)
        ));
    }
}
