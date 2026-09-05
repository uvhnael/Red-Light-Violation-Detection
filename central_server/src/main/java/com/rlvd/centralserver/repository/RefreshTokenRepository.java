package com.rlvd.centralserver.repository;

import com.rlvd.centralserver.entity.RefreshToken;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Modifying;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.time.Instant;
import java.util.List;
import java.util.Optional;

@Repository
public interface RefreshTokenRepository extends JpaRepository<RefreshToken, Long> {

    Optional<RefreshToken> findByTokenHash(String tokenHash);

    List<RefreshToken> findAllByUserIdAndRevokedAtIsNull(Long userId);

    /** Revoke mọi refresh token còn hiệu lực của user (logout-all-devices). */
    @Modifying
    @Query("update RefreshToken rt set rt.revokedAt = :now "
            + "where rt.userId = :userId and rt.revokedAt is null")
    int revokeAllByUser(@Param("userId") Long userId, @Param("now") Instant now);

    /** Dọn token hết hạn hoặc đã revoke > 30 ngày — gọi từ cron job (TODO). */
    @Modifying
    @Query("delete from RefreshToken rt where rt.expiresAt < :threshold")
    int deleteExpiredBefore(@Param("threshold") Instant threshold);
}