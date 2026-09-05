package com.rlvd.centralserver.dto;

import jakarta.validation.constraints.NotBlank;
import lombok.Getter;
import lombok.Setter;

/**
 * Request body cho POST /api/auth/logout — revoke refresh token hiện tại.
 * Endpoint idempotent: nếu token không tồn tại/đã revoke vẫn trả 200.
 */
@Getter
@Setter
public class LogoutRequest {

    @NotBlank(message = "Refresh token không được để trống")
    private String refreshToken;
}