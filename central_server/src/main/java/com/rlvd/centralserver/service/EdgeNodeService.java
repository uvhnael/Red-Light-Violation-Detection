package com.rlvd.centralserver.service;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.rlvd.centralserver.dto.EdgeNodeRegisterRequest;
import com.rlvd.centralserver.dto.EdgeNodeResponse;
import com.rlvd.centralserver.dto.EdgeNodeUpdateRequest;
import com.rlvd.centralserver.entity.EdgeNode;
import com.rlvd.centralserver.repository.EdgeNodeRepository;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.server.ResponseStatusException;

import java.time.LocalDateTime;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

@Slf4j
@Service
public class EdgeNodeService {

    private static final int ONLINE_WINDOW_MINUTES = 2;

    private final EdgeNodeRepository repository;
    private final ObjectMapper objectMapper;

    public EdgeNodeService(EdgeNodeRepository repository, ObjectMapper objectMapper) {
        this.repository = repository;
        this.objectMapper = objectMapper;
    }

    @Transactional
    public EdgeNodeResponse registerOrUpdate(EdgeNodeRegisterRequest request) {
        EdgeNode node = repository.findByNodeId(request.getNodeId()).orElseGet(EdgeNode::new);

        node.setNodeId(request.getNodeId());
        node.setName(defaultString(request.getName(), request.getNodeId()));
        node.setIpAddress(request.getIpAddress());
        node.setStatus(normalizeStatus(request.getStatus(), "online"));
        node.setLastPing(LocalDateTime.now());
        node.setSettingsJson(serializeSettings(request.getSettings()));

        EdgeNode saved = repository.save(node);
        log.info("Registered/updated edge node {} ({})", saved.getNodeId(), saved.getStatus());
        return toResponse(saved);
    }

    public List<EdgeNodeResponse> getAllNodes() {
        return repository.findAll().stream().map(this::toResponse).toList();
    }

    public EdgeNodeResponse getNode(String nodeId) {
        return toResponse(repository.findByNodeId(nodeId)
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND, "Edge node not found: " + nodeId)));
    }

    @Transactional
    public EdgeNodeResponse updateSettings(String nodeId, EdgeNodeUpdateRequest request) {
        EdgeNode node = repository.findByNodeId(nodeId)
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND, "Edge node not found: " + nodeId));

        if (request.getName() != null && !request.getName().isBlank()) {
            node.setName(request.getName());
        }
        if (request.getIpAddress() != null) {
            node.setIpAddress(request.getIpAddress());
        }
        if (request.getStatus() != null && !request.getStatus().isBlank()) {
            node.setStatus(normalizeStatus(request.getStatus(), node.getStatus()));
        }
        if (request.getSettings() != null) {
            node.setSettingsJson(serializeSettings(request.getSettings()));
        }

        EdgeNode saved = repository.save(node);
        log.info("Updated edge node {} settings", saved.getNodeId());
        return toResponse(saved);
    }

    private String normalizeStatus(String status, String fallback) {
        String value = status == null || status.isBlank() ? fallback : status.trim().toLowerCase(Locale.ROOT);
        return switch (value) {
            case "online", "offline", "degraded", "maintenance" -> value;
            default -> fallback;
        };
    }

    private String defaultString(String value, String fallback) {
        return value == null || value.isBlank() ? fallback : value;
    }

    private String serializeSettings(Map<String, Object> settings) {
        try {
            return objectMapper.writeValueAsString(settings != null ? settings : Collections.emptyMap());
        } catch (JsonProcessingException e) {
            log.warn("Failed to serialize edge node settings: {}", e.getMessage());
            return "{}";
        }
    }

    private Map<String, Object> deserializeSettings(String settingsJson) {
        if (settingsJson == null || settingsJson.isBlank()) {
            return new LinkedHashMap<>();
        }

        try {
            return objectMapper.readValue(settingsJson, new TypeReference<>() {});
        } catch (JsonProcessingException e) {
            log.warn("Failed to parse edge node settings JSON: {}", e.getMessage());
            return new LinkedHashMap<>();
        }
    }

    private EdgeNodeResponse toResponse(EdgeNode node) {
        LocalDateTime now = LocalDateTime.now();
        boolean online = node.getLastPing() != null && node.getLastPing().isAfter(now.minusMinutes(ONLINE_WINDOW_MINUTES));

        return EdgeNodeResponse.builder()
                .id(node.getId())
                .nodeId(node.getNodeId())
                .name(node.getName())
                .ipAddress(node.getIpAddress())
                .status(node.getStatus())
                .lastPing(node.getLastPing())
                .settings(deserializeSettings(node.getSettingsJson()))
                .online(online)
                .createdAt(node.getCreatedAt())
                .updatedAt(node.getUpdatedAt())
                .build();
    }
}