package com.rlvd.centralserver.controller;

import com.rlvd.centralserver.dto.EdgeNodeRegisterRequest;
import com.rlvd.centralserver.dto.EdgeNodeResponse;
import com.rlvd.centralserver.dto.EdgeNodeUpdateRequest;
import com.rlvd.centralserver.service.EdgeNodeService;
import jakarta.validation.Valid;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/v1/edge-nodes")
public class EdgeNodeController {

    private final EdgeNodeService edgeNodeService;

    public EdgeNodeController(EdgeNodeService edgeNodeService) {
        this.edgeNodeService = edgeNodeService;
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
}