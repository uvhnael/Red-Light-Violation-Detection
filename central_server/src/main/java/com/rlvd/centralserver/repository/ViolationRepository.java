package com.rlvd.centralserver.repository;

import com.rlvd.centralserver.entity.Violation;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.JpaSpecificationExecutor;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;

/**
 * Spring Data JPA repository for Violation entities.
 */
@Repository
public interface ViolationRepository extends JpaRepository<Violation, Long>,
        JpaSpecificationExecutor<Violation> {

    Optional<Violation> findByEventId(String eventId);

    /** Exists-check chỉ trả boolean — batch ingest dùng thay vì fetch entity. */
    boolean existsByEventId(String eventId);

    List<Violation> findByNodeId(String nodeId);

    List<Violation> findByStatus(String status);

    List<Violation> findByLightState(String lightState);

    List<Violation> findByPlateTextContainingIgnoreCase(String plateText);

    List<Violation> findAllByOrderByCreatedAtDesc();

    long countByStatus(String status);

    long countByNodeId(String nodeId);

    // ------------------------------------------------------------------ //
    // Paged queries (frontend lazy loading — never load the whole table)   //
    // ------------------------------------------------------------------ //

    Page<Violation> findAllByOrderByCreatedAtDesc(Pageable pageable);

    Page<Violation> findByStatusOrderByCreatedAtDesc(String status, Pageable pageable);

    Page<Violation> findByStatusInOrderByCreatedAtDesc(List<String> statuses, Pageable pageable);

    Page<Violation> findByNodeIdOrderByCreatedAtDesc(String nodeId, Pageable pageable);

    Page<Violation> findByPlateTextContainingIgnoreCaseOrderByCreatedAtDesc(
            String plateText, Pageable pageable);

    // ------------------------------------------------------------------ //
    // Aggregate queries (stats without loading entities into memory)       //
    // ------------------------------------------------------------------ //

    long countByCreatedAtGreaterThanEqual(LocalDateTime threshold);

    @Query("select count(distinct v.nodeId) from Violation v")
    long countDistinctNodes();

    @Query("select count(distinct v.nodeId) from Violation v where v.createdAt >= :threshold")
    long countDistinctNodesSince(@Param("threshold") LocalDateTime threshold);

    @Query("select v.nodeId as nodeId, count(v) as cnt from Violation v "
            + "group by v.nodeId order by count(v) desc")
    List<Object[]> countGroupByNode();

    @Query("select lower(v.lightState) as state, count(v) as cnt from Violation v "
            + "where v.lightState is not null "
            + "group by lower(v.lightState) order by count(v) desc")
    List<Object[]> countGroupByLightState();

    @Query(value = "select extract(hour from created_at) as h, lower(light_state) as s, count(*) as cnt "
            + "from violations where created_at >= :start group by h, s",
            nativeQuery = true)
    List<Object[]> hourlyTrendSince(@Param("start") LocalDateTime start);
}
