package com.rlvd.centralserver.service;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.rlvd.centralserver.dto.*;
import com.rlvd.centralserver.entity.Violation;
import com.rlvd.centralserver.repository.ViolationRepository;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.server.ResponseStatusException;

import java.time.LocalDate;
import java.time.LocalDateTime;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Business logic for violation management.
 */
@Slf4j
@Service
public class ViolationService {

    private final ViolationRepository repository;
    private final ObjectMapper objectMapper;

    public ViolationService(ViolationRepository repository, ObjectMapper objectMapper) {
        this.repository = repository;
        this.objectMapper = objectMapper;
    }

    /**
     * Create a new violation record from an edge node payload.
     */
    @Transactional
    public ViolationResponse createViolation(ViolationRequest request, String nodeId) {
        // Check for duplicate event_id
        if (repository.findByEventId(request.getEventId()).isPresent()) {
            log.warn("Duplicate violation event_id: {}", request.getEventId());
            throw new ResponseStatusException(HttpStatus.CONFLICT,
                    "Violation with event_id '" + request.getEventId() + "' already exists");
        }

        Violation violation = Violation.builder()
                .eventId(request.getEventId())
                .nodeId(nodeId != null ? nodeId : "unknown")
                .trackId(request.getTrackId())
                .frameIndex(request.getFrameIndex())
                .timestampMs(request.getTimestampMs())
                .lightState(request.getLightState())
                .lightConfidence(request.getLightConfidence())
                .previousSide(request.getPreviousSide())
                .currentSide(request.getCurrentSide())
                .status("pending")
                .build();

        // Map crossing point
        if (request.getCrossingPoint() != null) {
            violation.setCrossingPointX(request.getCrossingPoint().getX());
            violation.setCrossingPointY(request.getCrossingPoint().getY());
        }

        // Map previous point
        if (request.getPreviousPoint() != null) {
            violation.setPreviousPointX(request.getPreviousPoint().getX());
            violation.setPreviousPointY(request.getPreviousPoint().getY());
        }

        // Map bounding box
        if (request.getBboxXyxy() != null && request.getBboxXyxy().size() == 4) {
            violation.setBboxX1(request.getBboxXyxy().get(0));
            violation.setBboxY1(request.getBboxXyxy().get(1));
            violation.setBboxX2(request.getBboxXyxy().get(2));
            violation.setBboxY2(request.getBboxXyxy().get(3));
        }

        // Map plate
        if (request.getPlate() != null) {
            violation.setPlateText(request.getPlate().getText());
            violation.setPlateConfidence(request.getPlate().getConfidence());
        }

        // Serialize metadata to JSON string
        if (request.getMetadata() != null && !request.getMetadata().isEmpty()) {
            try {
                violation.setMetadata(objectMapper.writeValueAsString(request.getMetadata()));
            } catch (JsonProcessingException e) {
                log.warn("Failed to serialize metadata for event {}: {}",
                        request.getEventId(), e.getMessage());
            }
        }

        Violation saved = repository.save(violation);
        log.info("Created violation record: id={}, eventId={}, nodeId={}",
                saved.getId(), saved.getEventId(), saved.getNodeId());

        return toResponse(saved);
    }

    /**
     * Get a single violation by database ID.
     */
    public ViolationResponse getViolation(Long id) {
        Violation violation = repository.findById(id)
                .orElseThrow(() -> new ResponseStatusException(
                        HttpStatus.NOT_FOUND, "Violation not found: " + id));
        return toResponse(violation);
    }

    /**
     * Get a single violation by event ID.
     */
    public ViolationResponse getViolationByEventId(String eventId) {
        Violation violation = repository.findByEventId(eventId)
                .orElseThrow(() -> new ResponseStatusException(
                        HttpStatus.NOT_FOUND, "Violation not found: " + eventId));
        return toResponse(violation);
    }

    /**
     * Get all violations, ordered by creation time (newest first).
     */
    public List<ViolationResponse> getAllViolations() {
        return repository.findAllByOrderByCreatedAtDesc().stream()
                .map(this::toResponse)
                .collect(Collectors.toList());
    }

    /**
     * Filter violations by status.
     */
    public List<ViolationResponse> getViolationsByStatus(String status) {
        String normalized = normalizeStatus(status);
        return repository.findAll().stream()
            .filter(violation -> normalizeStatus(violation.getStatus()).equals(normalized))
                .map(this::toResponse)
                .collect(Collectors.toList());
    }

    /**
     * Filter violations by node ID.
     */
    public List<ViolationResponse> getViolationsByNodeId(String nodeId) {
        return repository.findByNodeId(nodeId).stream()
                .map(this::toResponse)
                .collect(Collectors.toList());
    }

    /**
     * Search violations by plate text (partial, case-insensitive).
     */
    public List<ViolationResponse> searchByPlateText(String plateText) {
        return repository.findByPlateTextContainingIgnoreCase(plateText).stream()
                .map(this::toResponse)
                .collect(Collectors.toList());
    }

    /**
     * Update the review status of a violation (human-in-the-loop workflow).
     */
    @Transactional
    public ViolationResponse updateViolationStatus(Long id, String status) {
        Violation violation = repository.findById(id)
                .orElseThrow(() -> new ResponseStatusException(
                        HttpStatus.NOT_FOUND, "Violation not found: " + id));
        violation.setStatus(normalizeStatus(status));
        Violation saved = repository.save(violation);
        log.info("Updated violation {} status to '{}'", id, saved.getStatus());
        return toResponse(saved);
    }

    /**
     * Delete a violation record.
     */
    @Transactional
    public void deleteViolation(Long id) {
        if (!repository.existsById(id)) {
            throw new ResponseStatusException(
                    HttpStatus.NOT_FOUND, "Violation not found: " + id);
        }
        repository.deleteById(id);
        log.info("Deleted violation: {}", id);
    }

    /**
     * Get aggregate statistics about violations.
     */
    public Map<String, Object> getStats() {
        Map<String, Object> stats = new LinkedHashMap<>();
        List<Violation> all = repository.findAll();
        LocalDate today = LocalDate.now();
        LocalDateTime activeThreshold = LocalDateTime.now().minusMinutes(15);

        long total = all.size();
        long todayTotal = all.stream()
            .filter(v -> v.getCreatedAt() != null && v.getCreatedAt().toLocalDate().equals(today))
            .count();
        long pending = all.stream().filter(v -> "pending".equals(normalizeStatus(v.getStatus()))).count();
        long approved = all.stream().filter(v -> "approved".equals(normalizeStatus(v.getStatus()))).count();
        long rejected = all.stream().filter(v -> "rejected".equals(normalizeStatus(v.getStatus()))).count();
        long reviewTotal = approved + rejected;
        double approvalRate = reviewTotal > 0 ? Math.round((approved * 1000.0 / reviewTotal)) / 10.0 : 0.0;

        Map<String, Long> perNode = all.stream()
            .collect(Collectors.groupingBy(Violation::getNodeId, LinkedHashMap::new, Collectors.counting()));
        Map<String, Long> perState = all.stream()
            .filter(v -> v.getLightState() != null)
            .collect(Collectors.groupingBy(
                v -> v.getLightState().toLowerCase(Locale.ROOT),
                LinkedHashMap::new,
                Collectors.counting()));

        // Hourly trend with red/yellow light state breakdown (today only)
        Map<String, Long> hourlyRed = new LinkedHashMap<>();
        Map<String, Long> hourlyYellow = new LinkedHashMap<>();
        for (int hour = 0; hour < 24; hour++) {
            String bucket = String.format("%02d", hour);
            hourlyRed.put(bucket, 0L);
            hourlyYellow.put(bucket, 0L);
        }
        for (Violation violation : all) {
            if (violation.getCreatedAt() == null) continue;
            if (!violation.getCreatedAt().toLocalDate().equals(today)) continue;
            String bucket = String.format("%02d", violation.getCreatedAt().getHour());
            String state = violation.getLightState() != null
                    ? violation.getLightState().toLowerCase(Locale.ROOT) : "unknown";
            if ("red".equals(state)) {
                hourlyRed.put(bucket, hourlyRed.getOrDefault(bucket, 0L) + 1L);
            } else if ("yellow".equals(state)) {
                hourlyYellow.put(bucket, hourlyYellow.getOrDefault(bucket, 0L) + 1L);
            }
        }

        // Detect active nodes from violations in last 15 minutes
        Map<String, LocalDateTime> lastSeenByNode = all.stream()
                .filter(v -> v.getNodeId() != null && v.getCreatedAt() != null)
                .collect(Collectors.toMap(
                        Violation::getNodeId,
                        Violation::getCreatedAt,
                        (existing, replacement) -> existing.isAfter(replacement) ? existing : replacement,
                        LinkedHashMap::new));

        // Also track active nodes via edge node registry (more reliable than violation count)
        long activeNodes = 0;
        try {
            activeNodes = lastSeenByNode.values().stream()
                    .filter(timestamp -> timestamp.isAfter(activeThreshold))
                    .count();
        } catch (Exception ignored) {}
        long offlineNodes = Math.max(0, lastSeenByNode.size() - activeNodes);

        // Build hourly trend: merge red + yellow per hour
        List<Map<String, Object>> hourlyTrend = new ArrayList<>();
        for (int hour = 0; hour < 24; hour++) {
            String bucket = String.format("%02d:00", hour);
            Map<String, Object> point = new LinkedHashMap<>();
            point.put("hour", bucket);
            point.put("red", hourlyRed.getOrDefault(String.format("%02d", hour), 0L));
            point.put("yellow", hourlyYellow.getOrDefault(String.format("%02d", hour), 0L));
            hourlyTrend.add(point);
        }

        List<ViolationResponse> recentPending = all.stream()
            .filter(v -> "pending".equals(normalizeStatus(v.getStatus())))
            .sorted(Comparator.comparing(Violation::getCreatedAt, Comparator.nullsLast(Comparator.reverseOrder())))
            .limit(10)
            .map(this::toResponse)
            .collect(Collectors.toList());

        stats.put("total", total);
        stats.put("today_total", todayTotal);
        stats.put("pending", pending);
        stats.put("approved", approved);
        stats.put("rejected", rejected);
        stats.put("approval_rate", approvalRate);
        stats.put("active_nodes", activeNodes);
        stats.put("offline_nodes", offlineNodes);
        stats.put("violations_per_node", perNode);
        stats.put("violations_per_light_state", perState);
        stats.put("hourly_trend", hourlyTrend);
        stats.put("recent_pending", recentPending);

        return stats;
    }

    /**
     * Convert entity to response DTO.
     */
    private ViolationResponse toResponse(Violation entity) {
        return ViolationResponse.builder()
                .id(entity.getId())
                .eventId(entity.getEventId())
                .nodeId(entity.getNodeId())
                .trackId(entity.getTrackId())
                .frameIndex(entity.getFrameIndex())
                .timestampMs(entity.getTimestampMs())
                .crossingPoint(entity.getCrossingPointX() != null
                        ? new PointDto(entity.getCrossingPointX(), entity.getCrossingPointY())
                        : null)
                .previousPoint(entity.getPreviousPointX() != null
                        ? new PointDto(entity.getPreviousPointX(), entity.getPreviousPointY())
                        : null)
                .bboxXyxy(entity.getBboxX1() != null
                        ? new double[]{entity.getBboxX1(), entity.getBboxY1(),
                                       entity.getBboxX2(), entity.getBboxY2()}
                        : null)
                .lightState(entity.getLightState())
                .lightConfidence(entity.getLightConfidence())
                .previousSide(entity.getPreviousSide())
                .currentSide(entity.getCurrentSide())
                .plateText(entity.getPlateText())
                .plateConfidence(entity.getPlateConfidence())
                .status(normalizeStatus(entity.getStatus()))
                .mediaUrl(entity.getMediaUrl())
                .metadata(entity.getMetadata())
                .createdAt(entity.getCreatedAt())
                .updatedAt(entity.getUpdatedAt())
                .build();
    }

    private String normalizeStatus(String status) {
        if (status == null || status.isBlank()) {
            return "pending";
        }

        String normalized = status.trim().toLowerCase(Locale.ROOT);
        return switch (normalized) {
            case "confirmed", "approved" -> "approved";
            case "rejected" -> "rejected";
            case "pending" -> "pending";
            default -> normalized;
        };
    }
}
