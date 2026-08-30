package com.rlvd.centralserver.dto;

import lombok.Getter;
import lombok.Setter;

/** Phản hồi sau khi đăng nhập thành công — gắn vào Authorization của web. */
@Getter
@Setter
public class LoginResponse {

    private String token;
    private String tokenType = "Bearer";
    private long expiresIn;
    private String username;
    private String fullName;
    private String role;
}
