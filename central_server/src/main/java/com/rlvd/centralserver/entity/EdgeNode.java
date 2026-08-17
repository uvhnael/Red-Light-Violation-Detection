package com.rlvd.centralserver.entity;

import jakarta.persistence.*;
import lombok.*;

import java.time.LocalDateTime;
import java.util.UUID;

/**
 * Registry record for an Edge Node connected to the central server.
 */
@Entity
@Table(name = "edge_nodes", indexes = {
        @Index(name = "idx_edge_node_node_id", columnList = "nodeId", unique = true),
        @Index(name = "idx_edge_node_status", columnList = "status"),
        @Index(name = "idx_edge_node_last_ping", columnList = "lastPing")
})
@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class EdgeNode {

    @Id
    @GeneratedValue(strategy = GenerationType.UUID)
    private UUID id;

    @Column(nullable = false, unique = true)
    private String nodeId;

    @Column(nullable = false)
    private String name;

    private String ipAddress;

    @Column(nullable = false)
    private String status;

    private LocalDateTime lastPing;

    @Column(columnDefinition = "TEXT")
    private String settingsJson;

    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;

    @PrePersist
    protected void onCreate() {
        createdAt = LocalDateTime.now();
        updatedAt = LocalDateTime.now();
        if (lastPing == null) {
            lastPing = LocalDateTime.now();
        }
        if (status == null || status.isBlank()) {
            status = "online";
        }
        if (name == null || name.isBlank()) {
            name = nodeId;
        }
    }

    @PreUpdate
    protected void onUpdate() {
        updatedAt = LocalDateTime.now();
    }
}