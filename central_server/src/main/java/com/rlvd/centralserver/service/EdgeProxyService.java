package com.rlvd.centralserver.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.rlvd.centralserver.entity.EdgeNode;
import com.rlvd.centralserver.repository.EdgeNodeRepository;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.web.server.ResponseStatusException;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.Map;

/**
 * Forwards calibration commands from the web dashboard to an edge node.
 *
 * The web UI only talks to the central server; this service resolves the
 * edge node's address (ip_address + settings.api_port from registration)
 * and proxies the HTTP call to the edge node's FastAPI control plane.
 */
@Slf4j
@Service
public class EdgeProxyService {

    private static final int DEFAULT_EDGE_API_PORT = 8080;

    private final EdgeNodeRepository repository;
    private final ObjectMapper objectMapper;
    private final HttpClient httpClient;

    public EdgeProxyService(EdgeNodeRepository repository, ObjectMapper objectMapper) {
        this.repository = repository;
        this.objectMapper = objectMapper;
        this.httpClient = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(5))
                .build();
    }

    /** POST a JSON body to an edge node endpoint, return the parsed JSON response. */
    public Map<String, Object> postToEdge(String nodeId, String path, Map<String, Object> body) {
        String url = edgeUrl(nodeId, path);
        try {
            String jsonBody = objectMapper.writeValueAsString(body != null ? body : Map.of());
            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(url))
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(jsonBody))
                    .timeout(Duration.ofSeconds(60)) // calibration loads YOLO + reads a frame
                    .build();
            HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());
            return handleResponse(nodeId, url, response);
        } catch (ResponseStatusException e) {
            throw e;
        } catch (Exception e) {
            log.error("Edge call failed: {} -> {}", url, e.getMessage());
            throw new ResponseStatusException(HttpStatus.BAD_GATEWAY,
                    "Không gọi được edge node: " + e.getMessage());
        }
    }

    /** GET a JSON endpoint on the edge node, return the parsed JSON response. */
    public Map<String, Object> getFromEdge(String nodeId, String path) {
        String url = edgeUrl(nodeId, path);
        try {
            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(url))
                    .GET()
                    .timeout(Duration.ofSeconds(30))
                    .build();
            HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());
            return handleResponse(nodeId, url, response);
        } catch (ResponseStatusException e) {
            throw e;
        } catch (Exception e) {
            log.error("Edge call failed: {} -> {}", url, e.getMessage());
            throw new ResponseStatusException(HttpStatus.BAD_GATEWAY,
                    "Không gọi được edge node: " + e.getMessage());
        }
    }

    /** GET a binary endpoint (JPEG snapshot) on the edge node. */
    public byte[] getBytesFromEdge(String nodeId, String path) {
        String url = edgeUrl(nodeId, path);
        try {
            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(url))
                    .GET()
                    .timeout(Duration.ofSeconds(30))
                    .build();
            HttpResponse<byte[]> response = httpClient.send(request, HttpResponse.BodyHandlers.ofByteArray());
            if (response.statusCode() != 200) {
                throw new ResponseStatusException(HttpStatus.BAD_GATEWAY,
                        "Edge node trả về lỗi " + response.statusCode());
            }
            return response.body();
        } catch (ResponseStatusException e) {
            throw e;
        } catch (Exception e) {
            log.error("Edge call failed: {} -> {}", url, e.getMessage());
            throw new ResponseStatusException(HttpStatus.BAD_GATEWAY,
                    "Không gọi được edge node: " + e.getMessage());
        }
    }

    // ------------------------------------------------------------------ //

    @SuppressWarnings("unchecked")
    private Map<String, Object> handleResponse(String nodeId, String url, HttpResponse<String> response) {
        if (response.statusCode() >= 400) {
            log.warn("Edge node {} returned {} for {}: {}", nodeId, response.statusCode(), url,
                    truncate(response.body()));
            throw new ResponseStatusException(HttpStatus.BAD_GATEWAY,
                    "Edge node trả về lỗi " + response.statusCode() + ": " + truncate(response.body()));
        }
        try {
            return objectMapper.readValue(response.body(), Map.class);
        } catch (Exception e) {
            throw new ResponseStatusException(HttpStatus.BAD_GATEWAY,
                    "Edge node trả về dữ liệu không hợp lệ");
        }
    }

    /** Build http://<ip>:<port><path> from the node's registration data. */
    private String edgeUrl(String nodeId, String path) {
        EdgeNode node = repository.findByNodeId(nodeId)
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND,
                        "Edge node không tồn tại: " + nodeId));

        String ip = node.getIpAddress();
        if (ip == null || ip.isBlank() || "unknown".equalsIgnoreCase(ip)) {
            throw new ResponseStatusException(HttpStatus.BAD_GATEWAY,
                    "Edge node chưa đăng ký địa chỉ IP");
        }

        int port = DEFAULT_EDGE_API_PORT;
        Map<String, Object> settings = deserializeSettings(node.getSettingsJson());
        Object apiPort = settings.get("api_port");
        if (apiPort instanceof Number number) {
            port = number.intValue();
        }

        // Strip an accidental scheme/port the node may have registered with
        String host = ip.replaceFirst("^https?://", "").replaceFirst("[:/].*$", "");
        return "http://" + host + ":" + port + path;
    }

    @SuppressWarnings("unchecked")
    private Map<String, Object> deserializeSettings(String settingsJson) {
        if (settingsJson == null || settingsJson.isBlank()) {
            return Map.of();
        }
        try {
            return objectMapper.readValue(settingsJson, Map.class);
        } catch (Exception e) {
            return Map.of();
        }
    }

    private String truncate(String s) {
        if (s == null) return "";
        return s.length() > 200 ? s.substring(0, 200) : s;
    }
}
