package com.rlvd.centralserver.service;

import io.minio.*;
import io.minio.http.Method;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import jakarta.annotation.PostConstruct;
import java.io.IOException;
import java.io.InputStream;
import java.util.concurrent.TimeUnit;

/**
 * Service for MinIO object storage operations.
 * Handles upload/download/delete of violation media files (images, video clips).
 */
@Slf4j
@Service
public class MinioStorageService {

    private final MinioClient minioClient;

    @Value("${minio.bucket}")
    private String bucketName;

    public String getBucketName() {
        return bucketName;
    }

    public MinioStorageService(MinioClient minioClient) {
        this.minioClient = minioClient;
    }

    /**
     * Ensure the configured bucket exists on startup.
     */
    @PostConstruct
    public void init() {
        try {
            boolean exists = minioClient.bucketExists(
                    BucketExistsArgs.builder().bucket(bucketName).build()
            );
            if (!exists) {
                minioClient.makeBucket(
                        MakeBucketArgs.builder().bucket(bucketName).build()
                );
                log.info("Created MinIO bucket: {}", bucketName);
            } else {
                log.info("MinIO bucket already exists: {}", bucketName);
            }
        } catch (Exception e) {
            log.warn("Could not initialize MinIO bucket '{}': {}. "
                    + "Storage features will be unavailable until MinIO is reachable.",
                    bucketName, e.getMessage());
        }
    }

    /**
     * Upload a file to MinIO and return the object name.
     */
    public String uploadFile(String objectName, InputStream stream,
                             String contentType, long size) {
        try {
            minioClient.putObject(
                    PutObjectArgs.builder()
                            .bucket(bucketName)
                            .object(objectName)
                            .stream(stream, size, -1)
                            .contentType(contentType)
                            .build()
            );
            log.info("Uploaded file to MinIO: {}/{}", bucketName, objectName);
            return objectName;
        } catch (Exception e) {
            log.error("Failed to upload file to MinIO: {}", e.getMessage(), e);
            throw new RuntimeException("MinIO upload failed", e);
        }
    }

    /**
     * Generate a presigned URL for downloading a file (valid for 1 hour).
     */
    public String getFileUrl(String objectName) {
        try {
            return minioClient.getPresignedObjectUrl(
                    GetPresignedObjectUrlArgs.builder()
                            .bucket(bucketName)
                            .object(objectName)
                            .method(Method.GET)
                            .expiry(1, TimeUnit.HOURS)
                            .build()
            );
        } catch (Exception e) {
            log.error("Failed to generate presigned URL: {}", e.getMessage(), e);
            throw new RuntimeException("MinIO URL generation failed", e);
        }
    }

    /**
     * Stream an object directly from MinIO.
     * Returns null if the object doesn't exist.
     */
    public InputStream getObjectStream(String objectName) throws IOException {
        try {
            return minioClient.getObject(
                    GetObjectArgs.builder()
                            .bucket(bucketName)
                            .object(objectName)
                            .build()
            );
        } catch (Exception e) {
            log.error("Failed to get object from MinIO: {}", e.getMessage());
            return null;
        }
    }

    /**
     * Delete a file from MinIO.
     */
    public void deleteFile(String objectName) {
        try {
            minioClient.removeObject(
                    RemoveObjectArgs.builder()
                            .bucket(bucketName)
                            .object(objectName)
                            .build()
            );
            log.info("Deleted file from MinIO: {}/{}", bucketName, objectName);
        } catch (Exception e) {
            log.error("Failed to delete file from MinIO: {}", e.getMessage(), e);
        }
    }
}
