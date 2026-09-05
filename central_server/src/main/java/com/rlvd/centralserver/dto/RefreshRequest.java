package com.rlvd.centralserver.dto;

import jakarta.validation.constraints.NotBlank;
import lombok.Getter;
import lombok.Setter;

/**
 * Request body cho POST /api/auth/refresh — client gửi raw refresh token
 * (nhận từ /login) để đổi lấy access token mới + refresh token mới.
 *
 * Raw token (không phải JWT) — backend hash SHA-256 rồi lookup DB.
 */
@Getter
@Setter
public class RefreshRequest {

    @NotBlank(message = "Refresh token không được để trống")
    private String refreshToken;
}