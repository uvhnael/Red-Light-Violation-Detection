package com.rlvd.centralserver.service;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.rlvd.centralserver.dto.*;
import com.rlvd.centralserver.entity.Violation;
import com.rlvd.centralserver.repository.ViolationRepository;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.server.ResponseStatusException;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Business logic cho violation lifecycle: CRUD + ingest (single/batch).
 *
 * Stats (KPI cards, chart, recent pending) đã tách sang
 * {@link ViolationStatsService} để service này chỉ lo data thuần.
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
     * Returns the response, or null if the event_id already exists
     * (idempotent ingest — duplicates are skipped, not errors).
     */
    @Transactional
    public ViolationResponse createViolation(ViolationRequest request, String nodeId) {
        // Check for duplicate event_id
        if (repository.findByEventId(request.getEventId()).isPresent()) {
            log.warn("Duplicate violation event_id: {}", request.getEventId());
            throw new ResponseStatusException(HttpStatus.CONFLICT,
                    "Violation with event_id '" + request.getEventId() + "' already exists");
        }

        Violation saved = repository.save(buildEntity(request, nodeId));
        log.info("Created violation record: id={}, eventId={}, nodeId={}",
                saved.getId(), saved.getEventId(), saved.getNodeId());

        return toResponse(saved);
    }

    /**
     * Batch ingest from an edge node outbox flush. Idempotent: duplicate
     * event_ids are counted and skipped instead of failing the whole batch.
     */
    @Transactional
    public ViolationBatchResponse createViolationBatch(
            List<ViolationRequest> requests, String nodeId) {

        List<String> acceptedIds = new ArrayList<>();
        List<String> duplicateIds = new ArrayList<>();
        int failed = 0;

        for (ViolationRequest request : requests) {
            if (request == null || request.getEventId() == null
                    || request.getEventId().isBlank()) {
                failed++;
                continue;
            }
            // Exists-check boolean nhanh hơn findByEventId (không load entity)
            if (repository.existsByEventId(request.getEventId())) {
                duplicateIds.add(request.getEventId());
                continue;
            }
            try {
                Violation saved = repository.save(buildEntity(request, nodeId));
                acceptedIds.add(saved.getEventId());
            } catch (Exception e) {
                log.error("Failed to persist violation {}: {}",
                        request.getEventId(), e.getMessage());
                failed++;
            }
        }

        log.info("Batch ingest from node '{}': accepted={}, duplicates={}, failed={}",
                nodeId, acceptedIds.size(), duplicateIds.size(), failed);

        return ViolationBatchResponse.builder()
                .accepted(acceptedIds.size())
                .duplicates(duplicateIds.size())
                .failed(failed)
                .acceptedEventIds(acceptedIds)
                .duplicateEventIds(duplicateIds)
                .build();
    }

    /**
     * Map an inbound request onto a fresh entity (shared by single + batch).
     */
    private Violation buildEntity(ViolationRequest request, String nodeId) {
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

        return violation;
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
     * Kept for backward compatibility — prefer {@link #getViolationsPage}
     * for anything user-facing (this loads the whole table).
     */
    public List<ViolationResponse> getAllViolations() {
        return repository.findAllByOrderByCreatedAtDesc().stream()
                .map(this::toResponse)
                .collect(Collectors.toList());
    }

    /**
     * Paged violation query with optional filters (status / nodeId / plateText).
     * The frontend loads one page at a time — never the whole table.
     * Status accepts a single value or comma-separated values ("pending,approved").
     */
    public ViolationPageResponse getViolationsPage(
            String status, String nodeId, String plateText, int page, int size) {

        int safePage = Math.max(0, page);
        int safeSize = Math.min(Math.max(1, size), 200);
        Pageable pageable = PageRequest.of(safePage, safeSize);

        Page<Violation> result;
        if (status != null && !status.isBlank()) {
            List<String> statuses = Arrays.stream(status.split(","))
                    .map(this::normalizeStatus)
                    .distinct()
                    .collect(Collectors.toList());
            if (statuses.size() == 1) {
                result = repository.findByStatusOrderByCreatedAtDesc(statuses.get(0), pageable);
            } else {
                result = repository.findByStatusInOrderByCreatedAtDesc(statuses, pageable);
            }
        } else if (nodeId != null && !nodeId.isBlank()) {
            result = repository.findByNodeIdOrderByCreatedAtDesc(nodeId.trim(), pageable);
        } else if (plateText != null && !plateText.isBlank()) {
            result = repository.findByPlateTextContainingIgnoreCaseOrderByCreatedAtDesc(
                    plateText.trim(), pageable);
        } else {
            result = repository.findAllByOrderByCreatedAtDesc(pageable);
        }

        return ViolationPageResponse.builder()
                .content(result.getContent().stream()
                        .map(this::toResponse)
                        .collect(Collectors.toList()))
                .page(result.getNumber())
                .size(result.getSize())
                .totalElements(result.getTotalElements())
                .totalPages(result.getTotalPages())
                .first(result.isFirst())
                .last(result.isLast())
                .build();
    }

    /**
     * Filter violations by status.
     *
     * Dùng query có index (idx_status) thay vì findAll() + filter trong
     * bộ nhớ — bảng lớn sẽ kéo toàn bộ dòng về RAM trước khi lọc.
     */
    public List<ViolationResponse> getViolationsByStatus(String status) {
        String normalized = normalizeStatus(status);
        return repository.findByStatus(normalized).stream()
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
                .mediaUrl(mediaUrlFor(entity))
                .metadata(entity.getMetadata())
                .createdAt(entity.getCreatedAt())
                .updatedAt(entity.getUpdatedAt())
                .build();
    }

    /**
     * Return a browser-facing media URL instead of the raw MinIO object key.
     * The stored value is an object key like "violations/<eventId>/<uuid>.jpg";
     * the frontend needs the proxy endpoint that streams it.
     */
    private String mediaUrlFor(Violation entity) {
        String objectName = entity.getMediaUrl();
        if (objectName == null || objectName.isBlank()) {
            return null;
        }
        // Already a usable URL (legacy rows) — pass through
        if (objectName.startsWith("http") || objectName.startsWith("/")) {
            return objectName;
        }
        return "/api/v1/violations/" + entity.getEventId() + "/media/blob";
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
            // Không cho chuỗi tuỳ ý đi vào DB — status lạ quy về pending
            // thay vì ghi đè giá trị ngoài bộ {pending, approved, rejected}.
            default -> "pending";
        };
    }
}
