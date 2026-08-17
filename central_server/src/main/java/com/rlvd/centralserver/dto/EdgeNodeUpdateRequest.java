package com.rlvd.centralserver.dto;

import lombok.*;

import java.util.Map;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class EdgeNodeUpdateRequest {

    private String name;
    private String ipAddress;
    private String status;
    private Map<String, Object> settings;
}