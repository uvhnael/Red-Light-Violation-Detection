package com.rlvd.centralserver.dto;

import lombok.*;

import java.time.LocalDateTime;
import java.util.Map;
import java.util.UUID;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class EdgeNodeResponse {

    private UUID id;
    private String nodeId;
    private String name;
    private String ipAddress;
    private String status;
    private LocalDateTime lastPing;
    private Map<String, Object> settings;
    private boolean online;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
}