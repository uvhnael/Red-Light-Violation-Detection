package com.rlvd.centralserver.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import jakarta.validation.Valid;
import lombok.*;

import java.util.List;

/**
 * Inbound batch payload from an Edge Node outbox flush.
 * Shape: {"violations": [ {...}, {...} ]}
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class ViolationBatchRequest {

    @Valid
    @JsonProperty("violations")
    private List<ViolationRequest> violations;
}
