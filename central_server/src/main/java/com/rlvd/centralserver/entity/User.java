package com.rlvd.centralserver.entity;

import jakarta.persistence.*;
import lombok.Getter;
import lombok.Setter;
import org.hibernate.annotations.CreationTimestamp;
import org.hibernate.annotations.UpdateTimestamp;

import java.time.LocalDateTime;

/**
 * Tài khoản người dùng dashboard — phục vụ xác thực JWT và phân quyền.
 *
 * Vai trò (role):
 * <ul>
 *   <li>ADMIN — toàn quyền: quản trị người dùng, node, duyệt vi phạm, hiệu chuẩn.</li>
 *   <li>OPERATOR — kỹ thuật vận hành: hiệu chuẩn node/camera, xem dữ liệu, không duyệt.</li>
 *   <li>OFFICER — cán bộ nghiệp vụ: tra cứu, duyệt/từ chối hồ sơ vi phạm, hỏi AI.</li>
 * </ul>
 */
@Getter
@Setter
@Entity
@Table(name = "users", indexes = {
        @Index(name = "idx_users_username", columnList = "username", unique = true)
})
public class User {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    /** Tên đăng nhập — duy nhất, không phân biệt hoa/thường khi tìm. */
    @Column(nullable = false, unique = true, length = 64)
    private String username;

    /** Mật khẩu đã băm BCrypt — KHÔNG bao giờ lưu plain-text. */
    @Column(nullable = false, length = 128)
    private String passwordHash;

    /** Họ tên hiển thị trên dashboard. */
    @Column(length = 128)
    private String fullName;

    /** ADMIN | OPERATOR | OFFICER. */
    @Column(nullable = false, length = 16)
    private String role;

    /** Tài khoản còn hoạt động (khóa tài khoản = false). */
    @Column(nullable = false)
    private Boolean enabled = true;

    @CreationTimestamp
    private LocalDateTime createdAt;

    @UpdateTimestamp
    private LocalDateTime updatedAt;
}
