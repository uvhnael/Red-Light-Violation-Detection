package com.rlvd.centralserver.service;

import com.rlvd.centralserver.dto.ViolationResponse;
import com.rlvd.centralserver.repository.ViolationRepository;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.domain.PageRequest;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDate;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Aggregate statistics for the violation table.
 *
 * Tách riêng khỏi {@link ViolationService} để:
 *  * {@link ViolationService} chỉ lo CRUD/ingest/query — đơn giản, dễ test.
 *  * Stats thường xuyên đổi shape (KPI mới, chart mới) — độc lập để không
 *    phải động vào code ingest.
 *
 * Tất cả query đều dùng COUNT/GROUP BY aggregate ở DB — KHÔNG load entity vào RAM.
 */
@Slf4j
@Service
public class ViolationStatsService {

    private final ViolationRepository repository;

    public ViolationStatsService(ViolationRepository repository) {
        this.repository = repository;
    }

    // ------------------------------------------------------------------ //
    // Cheap status counts — dùng cho badge/header                         //
    // ------------------------------------------------------------------ //

    /**
     * Cheap status counts for badges/headers (COUNT queries, no entity load).
     */
    public Map<String, Long> getStatusCounts() {
        Map<String, Long> counts = new LinkedHashMap<>();
        counts.put("total", repository.count());
        counts.put("pending", repository.countByStatus("pending"));
        counts.put("approved", repository.countByStatus("approved"));
        counts.put("rejected", repository.countByStatus("rejected"));
        return counts;
    }

    // ------------------------------------------------------------------ //
    // Big stats blob — KPI cards + chart + recent pending                  //
    // ------------------------------------------------------------------ //

    /**
     * Aggregate statistics about violations.
     * Uses COUNT/GROUP BY aggregate queries — never loads the whole table.
     */
    public Map<String, Object> getStats() {
        Map<String, Object> stats = new LinkedHashMap<>();
        LocalDate today = LocalDate.now();
        LocalDateTime todayStart = today.atStartOfDay();
        LocalDateTime activeThreshold = LocalDateTime.now().minusMinutes(15);

        Map<String, Long> statusCounts = getStatusCounts();
        long total = statusCounts.get("total");
        long pending = statusCounts.get("pending");
        long approved = statusCounts.get("approved");
        long rejected = statusCounts.get("rejected");
        long todayTotal = repository.countByCreatedAtGreaterThanEqual(todayStart);
        long reviewTotal = approved + rejected;
        double approvalRate = reviewTotal > 0
                ? Math.round((approved * 1000.0 / reviewTotal)) / 10.0
                : 0.0;

        Map<String, Long> perNode = aggregatePerNode();
        Map<String, Long> perState = aggregatePerLightState();
        List<Map<String, Object>> hourlyTrend = buildHourlyTrend(todayStart);
        long activeNodes = repository.countDistinctNodesSince(activeThreshold);
        long distinctNodes = repository.countDistinctNodes();
        long offlineNodes = Math.max(0, distinctNodes - activeNodes);
        List<ViolationResponse> recentPending = repository
                .findByStatusOrderByCreatedAtDesc("pending", PageRequest.of(0, 10))
                .getContent().stream()
                .map(this::toResponse)
                .toList();

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

    // ------------------------------------------------------------------ //
    // Internal helpers                                                     //
    // ------------------------------------------------------------------ //

    /** Đếm số vi phạm theo nodeId, sắp xếp giảm dần. */
    private Map<String, Long> aggregatePerNode() {
        Map<String, Long> perNode = new LinkedHashMap<>();
        for (Object[] row : repository.countGroupByNode()) {
            perNode.put((String) row[0], ((Number) row[1]).longValue());
        }
        return perNode;
    }

    /** Đếm số vi phạm theo light state (lowercase). */
    private Map<String, Long> aggregatePerLightState() {
        Map<String, Long> perState = new LinkedHashMap<>();
        for (Object[] row : repository.countGroupByLightState()) {
            perState.put(row[0].toString(), ((Number) row[1]).longValue());
        }
        return perState;
    }

    /**
     * Build hourly trend với red/yellow breakdown, scope hôm nay (00:00 → 23:59).
     * Bucket không có dữ liệu = 0. Filter today-only là bắt buộc — chart subtitle
     * nói "Last 24 hours" nhưng KPI card nói "Violations Today" nên data phải
     * cùng scope.
     */
    private List<Map<String, Object>> buildHourlyTrend(LocalDateTime todayStart) {
        Map<String, Long> hourlyRed = new LinkedHashMap<>();
        Map<String, Long> hourlyYellow = new LinkedHashMap<>();
        for (int hour = 0; hour < 24; hour++) {
            String bucket = String.format("%02d", hour);
            hourlyRed.put(bucket, 0L);
            hourlyYellow.put(bucket, 0L);
        }
        for (Object[] row : repository.hourlyTrendSince(todayStart)) {
            int hour = ((Number) row[0]).intValue();
            String state = row[1] != null ? row[1].toString() : "unknown";
            long cnt = ((Number) row[2]).longValue();
            String bucket = String.format("%02d", hour);
            if ("red".equals(state)) {
                hourlyRed.put(bucket, cnt);
            } else if ("yellow".equals(state)) {
                hourlyYellow.put(bucket, cnt);
            }
        }
        List<Map<String, Object>> trend = new ArrayList<>();
        for (int hour = 0; hour < 24; hour++) {
            Map<String, Object> point = new LinkedHashMap<>();
            point.put("hour", String.format("%02d:00", hour));
            point.put("red", hourlyRed.get(String.format("%02d", hour)));
            point.put("yellow", hourlyYellow.get(String.format("%02d", hour)));
            trend.add(point);
        }
        return trend;
    }

    /**
     * Map entity → ViolationResponse cho recent_pending list.
     * Tách ra khỏi {@link ViolationService#toResponse} (private) để tránh
     * duplicate logic; tái sử dụng pattern đó khi cần expose entity ra ngoài.
     */
    private ViolationResponse toResponse(com.rlvd.centralserver.entity.Violation v) {
        return ViolationResponse.builder()
                .id(v.getId())
                .eventId(v.getEventId())
                .nodeId(v.getNodeId())
                .lightState(v.getLightState())
                .plateText(v.getPlateText())
                .status(v.getStatus())
                .createdAt(v.getCreatedAt())
                .build();
    }
}