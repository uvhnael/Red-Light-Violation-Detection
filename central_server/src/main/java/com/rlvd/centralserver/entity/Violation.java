package com.rlvd.centralserver.entity;

import jakarta.persistence.*;
import lombok.*;

import java.time.LocalDateTime;

/**
 * JPA entity representing a red-light violation record.
 */
@Entity
@Table(name = "violations", indexes = {
        @Index(name = "idx_event_id", columnList = "eventId", unique = true),
        @Index(name = "idx_node_id", columnList = "nodeId"),
        @Index(name = "idx_status", columnList = "status"),
        @Index(name = "idx_created_at", columnList = "createdAt")
})
@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class Violation {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(nullable = false, unique = true)
    private String eventId;

    @Column(nullable = false)
    private String nodeId;

    private Integer trackId;
    private Integer frameIndex;
    private Double timestampMs;

    // Crossing point coordinates
    private Double crossingPointX;
    private Double crossingPointY;

    // Previous position coordinates
    private Double previousPointX;
    private Double previousPointY;

    // Bounding box (xyxy format)
    private Double bboxX1;
    private Double bboxY1;
    private Double bboxX2;
    private Double bboxY2;

    // Traffic light state
    private String lightState;
    private Double lightConfidence;

    // Side transition info
    private Integer previousSide;
    private Integer currentSide;

    // License plate OCR results (nullable)
    private String plateText;
    private Double plateConfidence;

    // Human review workflow
    @Builder.Default
    @Column(nullable = false)
    private String status = "pending";

    // Media storage URL in MinIO (nullable)
    @Column(length = 1024)
    private String mediaUrl;

    // Metadata stored as JSON text
    @Column(columnDefinition = "TEXT")
    private String metadata;

    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;

    @PrePersist
    protected void onCreate() {
        createdAt = LocalDateTime.now();
        updatedAt = LocalDateTime.now();
    }

    @PreUpdate
    protected void onUpdate() {
        updatedAt = LocalDateTime.now();
    }
}
