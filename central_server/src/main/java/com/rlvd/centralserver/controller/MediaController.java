package com.rlvd.centralserver.controller;

import com.rlvd.centralserver.entity.Violation;
import com.rlvd.centralserver.repository.ViolationRepository;
import com.rlvd.centralserver.service.MinioStorageService;
import jakarta.servlet.http.HttpServletResponse;
import lombok.extern.slf4j.Slf4j;
import org.springframework.core.io.InputStreamResource;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.server.ResponseStatusException;

import java.io.IOException;
import java.io.InputStream;
import java.util.Map;
import java.util.UUID;

/**
 * REST controller for uploading and serving violation media (images, video clips).
 *
 * Media URLs are stored as MinIO object keys (e.g. "violations/<eventId>/<uuid>.jpg")
 * and served via a proxy endpoint that streams the object directly from MinIO.
 */
@Slf4j
@RestController
@RequestMapping("/api")
public class MediaController {

    private final MinioStorageService minioService;
    private final ViolationRepository violationRepository;

    public MediaController(MinioStorageService minioService, ViolationRepository violationRepository) {
        this.minioService = minioService;
        this.violationRepository = violationRepository;
    }

    /**
     * POST /api/v1/violations/{eventId}/media
     * Upload a media file (image or video clip) for a violation.
     */
    @PostMapping("/v1/violations/{eventId}/media")
    public ResponseEntity<Map<String, String>> uploadMedia(
            @PathVariable String eventId,
            @RequestParam("file") MultipartFile file) {

        Violation violation = violationRepository.findByEventId(eventId)
                .orElseThrow(() -> new ResponseStatusException(
                        HttpStatus.NOT_FOUND, "Violation not found: " + eventId));

        if (file.isEmpty()) {
            return ResponseEntity.badRequest().body(Map.of("error", "File is empty"));
        }

        String originalName = file.getOriginalFilename();
        String ext = originalName != null && originalName.contains(".")
                ? originalName.substring(originalName.lastIndexOf("."))
                : ".jpg";

        // Store object key: violations/{eventId}/{uuid}{ext}
        String objectName = String.format("violations/%s/%s%s",
                violation.getEventId(), uuid(), ext);

        try (InputStream stream = file.getInputStream()) {
            String contentType = file.getContentType();
            if (contentType == null) {
                contentType = ext.matches("\\.(mp4|webm|mkv|avi)") ? "video/mp4" : "image/jpeg";
            }
            minioService.uploadFile(objectName, stream, contentType, file.getSize());
        } catch (IOException e) {
            log.error("Failed to upload media for event_id={}: {}", eventId, e.getMessage());
            throw new ResponseStatusException(HttpStatus.INTERNAL_SERVER_ERROR, "Upload failed");
        }

        // Store the object key (not a full URL) so the proxy endpoint resolves it at runtime
        violation.setMediaUrl(objectName);
        violationRepository.save(violation);

        log.info("Media uploaded for event_id={}: object={}", eventId, objectName);

        return ResponseEntity.status(HttpStatus.CREATED).body(Map.of(
                "object_name", objectName,
                "url", "/api/v1/violations/" + eventId + "/media/blob",
                "message", "Media uploaded successfully"
        ));
    }

    /**
     * GET /api/v1/violations/{eventId}/media/blob
     * Proxy endpoint that streams the media file from MinIO.
     * This is the URL the frontend uses — browser‑facing and always accessible.
     */
    @GetMapping("/v1/violations/{eventId}/media/blob")
    public ResponseEntity<InputStreamResource> streamMedia(@PathVariable String eventId) {
        Violation violation = violationRepository.findByEventId(eventId)
                .orElseThrow(() -> new ResponseStatusException(
                        HttpStatus.NOT_FOUND, "Violation not found: " + eventId));

        String objectName = violation.getMediaUrl();
        if (objectName == null || objectName.isBlank()) {
            throw new ResponseStatusException(HttpStatus.NOT_FOUND, "No media for this violation");
        }

        try {
            InputStream stream = minioService.getObjectStream(objectName);
            if (stream == null) {
                throw new ResponseStatusException(HttpStatus.NOT_FOUND, "Media file not found in storage");
            }

            String contentType = objectName.endsWith(".mp4") ? "video/mp4"
                    : objectName.endsWith(".png") ? "image/png"
                    : "image/jpeg";

            return ResponseEntity.ok()
                    .contentType(MediaType.parseMediaType(contentType))
                    .header(HttpHeaders.CACHE_CONTROL, "public, max-age=3600")
                    .body(new InputStreamResource(stream));
        } catch (IOException e) {
            log.error("Failed to stream media {}: {}", objectName, e.getMessage());
            throw new ResponseStatusException(HttpStatus.INTERNAL_SERVER_ERROR, "Stream failed");
        }
    }

    /**
     * GET /api/v1/violations/{eventId}/media
     * Returns the objectName metadata (for debugging / mobile clients).
     */
    @GetMapping("/v1/violations/{eventId}/media")
    public ResponseEntity<Map<String, String>> getMediaInfo(@PathVariable String eventId) {
        Violation violation = violationRepository.findByEventId(eventId)
                .orElseThrow(() -> new ResponseStatusException(
                        HttpStatus.NOT_FOUND, "Violation not found: " + eventId));

        String objectName = violation.getMediaUrl();
        if (objectName == null || objectName.isBlank()) {
            return ResponseEntity.status(HttpStatus.NOT_FOUND)
                    .body(Map.of("error", "No media uploaded"));
        }

        return ResponseEntity.ok(Map.of(
                "object_name", objectName,
                "blob_url", "/api/v1/violations/" + eventId + "/media/blob"
        ));
    }

    /**
     * DELETE /api/v1/violations/{eventId}/media
     * Remove the media file from MinIO and clear the violation's media_url.
     */
    @DeleteMapping("/v1/violations/{eventId}/media")
    public ResponseEntity<Map<String, String>> deleteMedia(@PathVariable String eventId) {
        Violation violation = violationRepository.findByEventId(eventId)
                .orElseThrow(() -> new ResponseStatusException(
                        HttpStatus.NOT_FOUND, "Violation not found: " + eventId));

        String objectName = violation.getMediaUrl();
        if (objectName != null && !objectName.isBlank()) {
            minioService.deleteFile(objectName);
        }

        violation.setMediaUrl(null);
        violationRepository.save(violation);
        return ResponseEntity.ok(Map.of("message", "Media deleted"));
    }

    private static String uuid() {
        return UUID.randomUUID().toString().substring(0, 8);
    }
}