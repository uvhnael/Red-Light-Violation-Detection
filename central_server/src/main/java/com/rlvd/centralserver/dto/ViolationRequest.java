package com.rlvd.centralserver.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import jakarta.validation.constraints.NotBlank;
import lombok.*;

import java.util.List;
import java.util.Map;

/**
 * Inbound violation payload from Edge Node.
 * Maps the snake_case JSON fields from the Python edge node to Java camelCase.
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class ViolationRequest {

    @NotBlank(message = "event_id is required")
    @JsonProperty("event_id")
    private String eventId;

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
    private List<Double> bboxXyxy;

    @NotBlank(message = "light_state is required")
    @JsonProperty("light_state")
    private String lightState;

    @JsonProperty("light_confidence")
    private Double lightConfidence;

    @JsonProperty("previous_side")
    private Integer previousSide;

    @JsonProperty("current_side")
    private Integer currentSide;

    private PlateDto plate;

    private Map<String, Object> metadata;
}
