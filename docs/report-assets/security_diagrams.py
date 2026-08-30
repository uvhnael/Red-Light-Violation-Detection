# -*- coding: utf-8 -*-
"""Diagram bảo mật: sequence đăng nhập JWT + RBAC + edge hardening."""

DIAGRAMS_SECURITY = {}

# Sequence: đăng nhập + truy cập API có phân quyền
DIAGRAMS_SECURITY["sequence-auth"] = """@startuml
skinparam backgroundColor #FFFFFF
skinparam shadowing false
autonumber "<b>0."
title Chuỗi xác thực: đăng nhập JWT và truy cập API theo vai trò

actor "Người dùng\\n(Admin / Operator / Officer)" as User
box "Web (Next.js)" #FEF3C7
  participant "Trang /login" as Login
  participant "middleware.ts" as MW
  participant "api client\\n(lib/api.ts)" as Client
end box
box "Máy chủ trung tâm (Spring Boot)" #ECFDF5
  participant "SecurityFilterChain" as Sec
  participant "AuthController" as Auth
  participant "JwtService" as Jwt
  database "PostgreSQL\\n(bảng users)" as DB
end box

User -> Login : nhập username + password
Login -> Auth : POST /api/auth/login
Auth -> DB : tra cứu user (BCrypt match)
DB --> Auth : hợp lệ — role
Auth -> Jwt : generateToken(username, role)
Jwt --> Auth : JWT (HS256, TTL 12h)
Auth --> Login : {token, role, expires_in}
Login -> Login : lưu localStorage +\\ncookie rlvd_token

User -> MW : truy cập trang /violations
MW -> MW : kiểm tra cookie rlvd_token\\n(thiếu → redirect /login)
MW --> Client : render trang

Client -> Sec : GET /api/violations/page\\nAuthorization: Bearer (JWT)
Sec -> Sec : JwtAuthFilter xác thực\\nchữ ký + TTL → gắn ROLE_(role)
Sec -> Sec : rule phân quyền\\n(edge-nodes → ADMIN/OPERATOR,\\nstatus → ADMIN/OFFICER…)
alt Token hợp lệ + đủ quyền
  Sec --> Client : 200 + dữ liệu
else Token sai / hết hạn
  Sec --> Client : 401 → web redirect /login
else Thiếu quyền (vd Officer gọi /nodes)
  Sec --> Client : 403 Forbidden
end
@enduml
"""

# Sequence: edge → central ingest có token
DIAGRAMS_SECURITY["sequence-ingest"] = """@startuml
skinparam backgroundColor #FFFFFF
skinparam shadowing false
autonumber "<b>0."
title Chuỗi bảo vệ đường ingest: node biên đẩy hồ sơ lên trung tâm

box "Node biên" #EFF6FF
  participant "ViolationSender" as Sender
end box
box "Máy chủ trung tâm" #ECFDF5
  participant "IngestTokenFilter" as Ingest
  participant "SecurityFilterChain" as Sec
  participant "ViolationController" as Ctrl
end box

Sender -> Ingest : POST /api/violations/batch\\nX-Node-ID + X-Ingest-Token
Ingest -> Ingest : so sánh constant-time\\nvới INGEST_TOKEN
alt Token đúng
  Ingest -> Sec : cho qua, không cần JWT\\n(máy chủ, không phải user)
  Sec -> Ctrl : permitAll cho POST ingest
  Ctrl --> Sender : 201 + accepted/duplicates
else Token sai hoặc thiếu
  Ingest --> Sender : 401 Unauthorized
  note right of Sender
    Outbox giữ pending,
    sender thử lại theo chu kỳ,
    không mất hồ sơ.
    INGEST_TOKEN tách khỏi JWT:
    node biên không nắm secret
    người dùng và ngược lại.
  end note
end
@enduml
"""

# Component: tổng quan lớp bảo mật 3 tầng
DIAGRAMS_SECURITY["security-layers"] = """@startuml
skinparam backgroundColor #FFFFFF
skinparam shadowing false
skinparam componentStyle rectangle
title Các lớp bảo mật qua ba tầng hệ thống

package "Web (Next.js)" #FEF3C7 {
  [middleware.ts — redirect /login\\nkhi thiếu cookie] as MW
  [localStorage JWT +\\nhasRole(minRole) guard] as GUARD
  [Next.js rewrite —\\ngiấu địa chỉ backend] as RW
}

package "Trung tâm (Spring Boot)" #ECFDF5 {
  [Spring Security +\\nSecurityFilterChain] as SEC
  [JWT HS256\\n(TTL 12h, secret env)] as JWT
  [RBAC: ADMIN > OPERATOR > OFFICER] as RBAC
  [BCrypt password hash] as CRYPT
  [IngestTokenFilter\\n(X-Ingest-Token)] as ING
  [SQL guard —\\nchỉ SELECT + LIMIT] as SQLG
}

package "Node biên (FastAPI)" #EFF6FF {
  [X-Edge-Token constant-time\\ncho POST /action/*] as ETOK
  [Rate-limit 30 req/phút/IP\\ncửa sổ trượt] as RATE
  [CORS whitelist\\n(EDGE_ALLOWED_ORIGINS)] as CORS
  [Security headers\\nnosniff/DENY/no-store] as HDR
  [EDGE_REQUIRE_TOKEN —\\nkhoá ghi khi thiếu token] as REQ
}

GUARD --> MW
MW --> RW
RW --> SEC
SEC --> JWT
SEC --> RBAC
SEC --> ING
JWT --> CRYPT
ING --> RBAC
ETOK --> RATE
ETOK --> CORS
ETOK --> HDR
REQ --> ETOK
@enduml
"""
