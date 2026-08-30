package com.rlvd.centralserver.config;

import com.rlvd.centralserver.entity.User;
import com.rlvd.centralserver.repository.UserRepository;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.ApplicationRunner;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.security.crypto.password.PasswordEncoder;

/**
 * Seed dữ liệu người dùng khi khởi động.
 *
 * Tài khoản admin mặc định đọc từ env:
 *   ADMIN_USERNAME (mặc định "admin"), ADMIN_PASSWORD (mặc định "admin123").
 *
 * ⚠️ Trong production PHẢI đổi ADMIN_PASSWORD qua biến môi trường.
 * Nếu username đã tồn tại thì bỏ qua (không ghi đè mật khẩu).
 */
@Configuration
public class UserSeeder {

    private static final Logger log = LoggerFactory.getLogger(UserSeeder.class);

    @Bean
    public ApplicationRunner seedAdminUser(
            UserRepository userRepository,
            PasswordEncoder passwordEncoder,
            @Value("${security.admin-username:${ADMIN_USERNAME:admin}}") String adminUsername,
            @Value("${security.admin-password:${ADMIN_PASSWORD:admin123}}") String adminPassword) {
        return args -> {
            if (userRepository.findByUsernameIgnoreCase(adminUsername).isPresent()) {
                log.info("Tài khoản admin '{}' đã tồn tại — bỏ qua seed.", adminUsername);
                return;
            }
            User admin = new User();
            admin.setUsername(adminUsername);
            admin.setPasswordHash(passwordEncoder.encode(adminPassword));
            admin.setFullName("Quản trị viên");
            admin.setRole("ADMIN");
            admin.setEnabled(true);
            userRepository.save(admin);
            log.warn("Đã tạo tài khoản admin mặc định '{}' — ĐỔI MẬT KHẨU qua biến môi trường "
                    + "ADMIN_PASSWORD trước khi triển khai thực tế!", adminUsername);
        };
    }
}
