package com.rlvd.centralserver.repository;

import com.rlvd.centralserver.entity.EdgeNode;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.Optional;

@Repository
public interface EdgeNodeRepository extends JpaRepository<EdgeNode, java.util.UUID> {

    Optional<EdgeNode> findByNodeId(String nodeId);
}