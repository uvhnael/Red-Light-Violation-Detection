package com.rlvd.centralserver.dto;

import jakarta.validation.constraints.NotBlank;
import lombok.*;

import java.util.Map;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class EdgeNodeRegisterRequest {

    @NotBlank(message = "node_id is required")
    private String nodeId;

    @NotBlank(message = "name is required")
    private String name;

    private String ipAddress;

    private String status;

    private Map<String, Object> settings;
}