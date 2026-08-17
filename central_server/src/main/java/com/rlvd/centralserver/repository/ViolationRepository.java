package com.rlvd.centralserver.repository;

import com.rlvd.centralserver.entity.Violation;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.JpaSpecificationExecutor;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

/**
 * Spring Data JPA repository for Violation entities.
 */
@Repository
public interface ViolationRepository extends JpaRepository<Violation, Long>,
        JpaSpecificationExecutor<Violation> {

    Optional<Violation> findByEventId(String eventId);

    List<Violation> findByNodeId(String nodeId);

    List<Violation> findByStatus(String status);

    List<Violation> findByLightState(String lightState);

    List<Violation> findByPlateTextContainingIgnoreCase(String plateText);

    List<Violation> findAllByOrderByCreatedAtDesc();

    long countByStatus(String status);

    long countByNodeId(String nodeId);
}
