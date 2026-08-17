package com.rlvd.centralserver.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.*;

import java.time.LocalDateTime;

/**
 * Outbound violation response DTO.
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class ViolationResponse {

    private Long id;

    @JsonProperty("event_id")
    private String eventId;

    @JsonProperty("node_id")
    private String nodeId;

    @JsonProperty("track_id")
    private Integer trackId;

    @JsonProperty("frame_index")
    private Integer frameIndex;

    @JsonProperty("timestamp_ms")
    private Double timestampMs;

    @JsonProperty("crossing_point")
    private PointDto crossingPoint;

    @JsonProperty("previous_point")
    private PointDto previousPoint;

    @JsonProperty("bbox_xyxy")
    private double[] bboxXyxy;

    @JsonProperty("light_state")
    private String lightState;

    @JsonProperty("light_confidence")
    private Double lightConfidence;

    @JsonProperty("previous_side")
    private Integer previousSide;

    @JsonProperty("current_side")
    private Integer currentSide;

    @JsonProperty("plate_text")
    private String plateText;

    @JsonProperty("plate_confidence")
    private Double plateConfidence;

    private String status;

    @JsonProperty("media_url")
    private String mediaUrl;

    private String metadata;

    @JsonProperty("created_at")
    private LocalDateTime createdAt;

    @JsonProperty("updated_at")
    private LocalDateTime updatedAt;
}
