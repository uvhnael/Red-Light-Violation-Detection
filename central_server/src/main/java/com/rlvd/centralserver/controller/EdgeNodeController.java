package com.rlvd.centralserver.controller;

import com.rlvd.centralserver.dto.EdgeNodeRegisterRequest;
import com.rlvd.centralserver.dto.EdgeNodeResponse;
import com.rlvd.centralserver.dto.EdgeNodeUpdateRequest;
import com.rlvd.centralserver.service.EdgeNodeService;
import com.rlvd.centralserver.service.EdgeProxyService;
import jakarta.validation.Valid;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/v1/edge-nodes")
public class EdgeNodeController {

    private final EdgeNodeService edgeNodeService;
    private final EdgeProxyService edgeProxyService;

    public EdgeNodeController(EdgeNodeService edgeNodeService, EdgeProxyService edgeProxyService) {
        this.edgeNodeService = edgeNodeService;
        this.edgeProxyService = edgeProxyService;
    }

    @PostMapping("/register")
    public ResponseEntity<EdgeNodeResponse> register(@Valid @RequestBody EdgeNodeRegisterRequest request) {
        return ResponseEntity.status(HttpStatus.CREATED).body(edgeNodeService.registerOrUpdate(request));
    }

    @GetMapping
    public ResponseEntity<List<EdgeNodeResponse>> list() {
        return ResponseEntity.ok(edgeNodeService.getAllNodes());
    }

    @GetMapping("/{nodeId}")
    public ResponseEntity<EdgeNodeResponse> get(@PathVariable String nodeId) {
        return ResponseEntity.ok(edgeNodeService.getNode(nodeId));
    }

    @PutMapping("/{nodeId}/settings")
    public ResponseEntity<EdgeNodeResponse> updateSettings(
            @PathVariable String nodeId,
            @RequestBody EdgeNodeUpdateRequest request) {
        return ResponseEntity.ok(edgeNodeService.updateSettings(nodeId, request));
    }

    // ------------------------------------------------------------------ //
    // Calibration: manual stop line / light ROI (drawn on the web UI)     //
    // ------------------------------------------------------------------ //

    /** Get the current calibration state (stop line + light ROI) from the edge node. */
    @GetMapping("/{nodeId}/calibration")
    public ResponseEntity<Map<String, Object>> getCalibration(@PathVariable String nodeId) {
        return ResponseEntity.ok(edgeProxyService.getFromEdge(nodeId, "/api/calibration"));
    }

    /** Get the calibration frame (JPEG) from the edge node for overlay drawing. */
    @GetMapping(value = "/{nodeId}/calibration/snapshot", produces = MediaType.IMAGE_JPEG_VALUE)
    public ResponseEntity<byte[]> getCalibrationSnapshot(@PathVariable String nodeId) {
        byte[] jpeg = edgeProxyService.getBytesFromEdge(nodeId, "/api/calibration/snapshot");
        return ResponseEntity.ok()
                .contentType(MediaType.IMAGE_JPEG)
                .header("Cache-Control", "no-cache")
                .body(jpeg);
    }

    /** Manually set the stop line (drawn on the web UI). */
    @PostMapping("/{nodeId}/calibration/stop-line")
    public ResponseEntity<Map<String, Object>> setStopLine(
            @PathVariable String nodeId,
            @RequestBody Map<String, Object> body) {
        return ResponseEntity.ok(edgeProxyService.postToEdge(nodeId, "/action/stop-line", body));
    }

    /** Manually set the traffic-light ROI box (drawn on the web UI). */
    @PostMapping("/{nodeId}/calibration/light-roi")
    public ResponseEntity<Map<String, Object>> setLightRoi(
            @PathVariable String nodeId,
            @RequestBody Map<String, Object> body) {
        return ResponseEntity.ok(edgeProxyService.postToEdge(nodeId, "/action/light-roi", body));
    }
}