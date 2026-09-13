# -*- coding: utf-8 -*-
"""Sinh các diagram BỔ SUNG cho đồ án bản cấu trúc mới (7 chương).

Tái dùng cơ chế render đã kiểm chứng của gen_diagrams_vn.py
(PlantUML server -> SVG -> cairosvg + font Noto Sans để giữ dấu tiếng Việt).

Output: docs/report-assets/thesis/<name>.png

Chạy:  myenv/bin/python docs/report-assets/gen_diagrams_thesis.py [name ...]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import gen_diagrams_vn as base  # noqa: E402

OUT = HERE / "thesis"
OUT.mkdir(parents=True, exist_ok=True)

D: dict[str, str] = {}

# ---------------------------------------------------------------- 1. HIỆN TRẠNG
# Quy trình xử lý vi phạm vượt đèn đỏ hiện nay (as-is) — dùng ở 2.1.
D["asis-process"] = """@startuml
skinparam backgroundColor #FFFFFF
skinparam shadowing false
title Quy trình xử lý vi phạm vượt đèn đỏ hiện nay (as-is)

|#FEF2F2|Cảnh sát giao thông tại nút giao|
start
:Trực tiếp quan sát tại chốt;
:Phát hiện phương tiện vượt đèn đỏ;
if (Chặn dừng được phương tiện?) then (Được)
  :Yêu cầu dừng xe, kiểm tra giấy tờ;
  :Lập biên bản tại chỗ;
  :Ra quyết định xử phạt;
  stop
else (Không — xe đã đi khuất)
endif

|#FFF7ED|Tổ xử lý camera / trung tâm|
:Trích xuất video từ camera giám sát;
:Xem lại thủ công, tua từng đoạn;
:Dừng hình, phóng to để đọc biển số;
if (Đọc được biển số rõ?) then (Có)
  :Tra cứu chủ phương tiện;
  :Gửi thông báo vi phạm (phạt nguội);
else (Không — hình mờ, biển khuất)
  :Không đủ căn cứ;
  :Huỷ hồ sơ;
  stop
endif

|#EFF6FF|Chủ phương tiện|
:Nhận thông báo;
if (Đồng ý vi phạm?) then (Đồng ý)
  :Nộp phạt;
  stop
else (Khiếu nại)
  :Yêu cầu xem lại bằng chứng video;
  :Tổ xử lý phải tìm lại đoạn video gốc;
  stop
endif
@enduml
"""

# ---------------------------------------------------------- 2. TO-BE (sau đồ án)
D["tobe-process"] = """@startuml
skinparam backgroundColor #FFFFFF
skinparam shadowing false
title Quy trình sau khi có hệ thống RLVD (to-be)

|#EFF6FF|Node biên tại camera|
start
:Phân tích video ngay tại camera\\n(phát hiện phương tiện + trạng thái đèn);
if (Xe cắt vạch dừng khi đèn đỏ\\nvà đúng hướng giám sát?) then (Đúng)
  :Tạo hồ sơ vi phạm\\n(ảnh bằng chứng + toạ độ + thời điểm);
  :Đọc và chuẩn hoá biển số\\ntheo TT 24/2023/TT-BGTVT;
  :Ghi vào hàng đợi bền vững (outbox);
else (Không)
  :Tiếp tục giám sát;
  stop
endif

|#ECFDF5|Máy chủ trung tâm|
:Nhận hồ sơ theo lô, chống trùng lặp;
:Lưu hồ sơ vào CSDL,\\nảnh bằng chứng vào kho đối tượng;

|#FEF3C7|Web dashboard — cán bộ nghiệp vụ|
:Hồ sơ xuất hiện trong danh sách chờ duyệt;
:Xem ảnh bằng chứng + biển số + thông tin;
if (Đủ căn cứ vi phạm?) then (Đạt)
  :Phê duyệt hồ sơ;
  :Phục vụ lập biên bản / phạt nguội;
else (Không đạt)
  :Từ chối hồ sơ, ghi nhận lý do;
endif
stop
@enduml
"""

# ------------------------------------------------------- 3. USE CASE (có Admin)
D["use-case-full"] = """@startuml
skinparam backgroundColor #FFFFFF
skinparam shadowing false
skinparam actorStyle awesome
left to right direction
skinparam packageStyle rectangle
skinparam usecase {
  BackgroundColor #EFF6FF
  BorderColor #3B82F6
}
actor "Quản trị viên\\n(Administrator)" as Admin
actor "Kỹ thuật viên\\n(Operator)" as Operator
actor "Cán bộ nghiệp vụ\\n(Officer)" as Officer
actor "Node biên\\n(Edge Node)" as Edge

rectangle "Hệ thống phát hiện vi phạm vượt đèn đỏ (RLVD)" {
  usecase "UC-01 Đăng ký node và\\ngửi tín hiệu hoạt động định kỳ" as UC1
  usecase "UC-02 Hiệu chuẩn vạch dừng,\\nvùng đèn, hướng giám sát" as UC2
  usecase "UC-03 Phát hiện phương tiện\\nvượt đèn đỏ" as UC3
  usecase "UC-04 Đọc và chuẩn hoá\\nbiển số phương tiện" as UC4
  usecase "UC-05 Gửi hồ sơ vi phạm\\nvề máy chủ trung tâm" as UC5
  usecase "UC-06 Xem camera trực tuyến\\nvà ảnh chụp nhanh" as UC6
  usecase "UC-07 Duyệt hoặc từ chối\\nhồ sơ vi phạm" as UC7
  usecase "UC-08 Tra cứu vi phạm\\nvà xem thống kê" as UC8
  usecase "UC-09 Hỏi dữ liệu bằng\\ntrợ lý ngôn ngữ tự nhiên" as UC9
  usecase "UC-10 Quản lý node biên\\n(trạng thái, cấu hình)" as UC10
  usecase "UC-11 Quản lý người dùng\\nvà phân quyền" as UC11
}

Admin --> UC11
Admin --> UC2
Admin --> UC7
Admin --> UC10

Operator --> UC1
Operator --> UC2
Operator --> UC6
Operator --> UC10
Operator --> UC8

Officer --> UC7
Officer --> UC8
Officer --> UC9

Edge ..> UC3 : <<thực hiện>>
Edge ..> UC4 : <<thực hiện>>
Edge ..> UC5 : <<thực hiện>>

UC1 ..> UC2 : <<tiền đề>>
UC3 ..> UC4 : <<include>>
UC3 ..> UC5 : <<include>>
UC5 ..> UC7 : <<phân phối>>
@enduml
"""

# ------------------------------------------------- 4. USE CASE theo nhóm vai trò
D["use-case-admin"] = """@startuml
skinparam backgroundColor #FFFFFF
skinparam shadowing false
skinparam actorStyle awesome
left to right direction
actor "Quản trị viên\\n(Administrator)" as Admin
rectangle "Nghiệp vụ quản trị hệ thống" {
  usecase "Khởi tạo tài khoản quản trị\\n(từ biến môi trường lúc khởi động)" as A
  usecase "Cấp / thu hồi quyền theo\\nba vai trò ADMIN, OPERATOR, OFFICER" as B
  usecase "Thu hồi toàn bộ phiên đăng nhập\\nkhi phát hiện token bị dùng lại" as C
  usecase "Xem nhật ký kiểm định:\\nđăng nhập, duyệt hồ sơ, ingest" as D
  usecase "Xoá hồ sơ vi phạm sai lệch" as E
  usecase "Thực hiện sao lưu và\\nkhôi phục dữ liệu" as F
}
Admin --> A
Admin --> B
Admin --> C
Admin --> D
Admin --> E
Admin --> F
B ..> C : <<extend>>
D ..> C : <<phát hiện>>
@enduml
"""

# ---------------------------------------------------------- 5. CLASS DIAGRAM v2
D["class-diagram-v2"] = """@startuml
skinparam backgroundColor #FFFFFF
skinparam shadowing false
skinparam classAttributeIconSize 0
skinparam maxMessageSize 260
skinparam class {
  BackgroundColor #EFF6FF
  BorderColor #3B82F6
}

package "Node biên — tầng thị giác máy tính (Python)" #EFF6FF {
  class RedLightViolationPipeline {
    -detector: ObjectDetector
    -tracker: MultiObjectTracker
    -light: TrafficLightClassifier
    -plates: PlateRecognizer
    -violation: ViolationDetector
    -outbox: ViolationOutbox
    +process(packet: FramePacket): PipelineResult
    -_render_evidence(frame, event): bytes
  }
  class YoloDetector {
    -model_path: str
    -confidence: float
    -vehicle_only: bool
    +detect(frame, index, ts): list[Detection]
  }
  class PlateDetector {
    -confidence = 0.25
    +detect(frame, index, ts): list[Detection]
  }
  class TunedByteTrack {
    -track_activation_threshold = 0.10
    -minimum_matching_threshold = 0.85
    +update(detections, tracks): list[Track]
  }
  class TrackMotion {
    -window: deque[TrackSample]
    +observe(sample: TrackSample)
    +estimate(): MotionState
  }
  class Track <<frozen dataclass>> {
    +track_id: int
    +bbox: BoundingBox
    +label: str
    +confidence: float (EMA)
    +time_since_update: int
    +motion: MotionState
    +trajectory: tuple[TrackSample, ...]
    +crossing_point: Point (bottom_center)
  }
  class MotionState <<frozen dataclass>> {
    +vx, vy: float
    +speed: float (pixel/giây)
    +heading_deg: float
    +direction: str (8 hướng)
    +is_moving: bool
  }
  class YoloLightClassifier {
    +classify(frame, index, ts): LightObservation
  }
  class HsvLightClassifier {
    +classify(roi): LightObservation
  }
  class RedLightStabilizer {
    -required_consecutive_frames: int
    -switch_consecutive_frames: int
    -unknown_tolerance_frames: int
    +update(obs, frame): StableSignal
  }
  class Tripwire {
    +side(point: Point): int
    +is_allowed_transition(prev, cur): bool
    +crossed(p0, p1, s0, s1): bool
  }
  class ViolationDetector {
    -tripwire: Tripwire | None
    -states: dict[int, _TrackCrossingState]
    -run_id: str
    +update(tracks, signal, frame): list[ViolationEvent]
  }
  class PlateAssociator {
    -cache: dict[int, PlateObservation]
    +update(frame, tracks, plates, ocr): tuple[dict, list]
  }
  class FastPlateOCR {
    +recognize_bbox(frame, bbox): PlateObservation
  }
  class vn_plate <<module>> {
    +validate_and_format(text): ValidationResult
    +repair_plate_text(raw): str | None
  }
  class ViolationOutbox {
    -db_path: str
    +enqueue(event, image_bytes)
    +fetch_pending(limit): list[OutboxItem]
    +mark_sent(ids) / bump_attempts(ids)
    +pending_count(): int
  }
  class ViolationSender {
    -batch_size: int = 20
    -flush_interval: int = 5
    +start() / stop()
    -_send_batch(items): bool
    -_upload_media(item): bool
  }
  class EdgeControlAPI {
    +GET /health (metrics, 503 khi stale)
    +GET /api/calibration | /api/light-state
    +POST /action/stop-line | /action/light-roi
    +GET /api/cameras/{id}/stream (HLS)
  }
  class PipelineMetrics {
    +frames_processed: int
    +violations_detected: int
    +fps: float
    +is_stale(): bool
  }
}

package "Máy chủ trung tâm — Spring Boot (Java 17)" #ECFDF5 {
  class ViolationController {
    +POST /api/violations/batch
    +GET /api/violations/page
    +PATCH /api/violations/{id}/status
    +GET /api/stats
  }
  class AuthController {
    +POST /api/auth/login | /refresh | /logout
    +GET /api/auth/me
  }
  class EdgeNodeController {
    +POST /api/v1/edge-nodes/register
    +POST /{nodeId}/calibration/stop-line
  }
  class MediaController {
    +POST /api/v1/violations/{eventId}/media
    +GET /api/v1/violations/{eventId}/media/blob
  }
  class AiQueryController {
    +POST /api/ai/query
  }
  class ViolationService {
    +ingestBatch(items): BatchResponse
    +getPage(filters): ViolationPageResponse
    +updateStatus(id, status, user)
  }
  class ViolationStatsService {
    +getStats(): Stats
    +countByHour / countByNode
  }
  class EdgeNodeService {
    +register(node): EdgeNode
    +isOnline(node): boolean (ping < 2 phút)
  }
  class EdgeProxyService {
    -client: HttpClient (HTTP/1.1)
    +forward(nodeId, method, path, body)
  }
  class RefreshTokenService {
    +issue(user): String (32 byte entropy)
    +rotate(token): TokenPair
    +detectReuse(token): revokeAll(user)
  }
  class JwtService {
    +generateToken(user, role): String (HS256)
    +parse(token): Claims
  }
  class AiQueryService {
    +process(question): AiQueryResponse
    -isSafeSelect(sql): boolean
  }
  class MinioStorageService {
    +put(eventId, bytes): objectKey
    +stream(objectKey): InputStream
  }
  class IngestPaths <<single source of truth>> {
    +INGEST_RULES: List<PathRule>
    +PERMIT_ONLY_RULES: List<PathRule>
  }
  class IngestTokenFilter {
    +doFilter(): so khớp hằng thời gian
  }
  class JwtAuthFilter {
    +doFilter(): gắn ROLE_ vào SecurityContext
  }
  class Violation <<JPA Entity>> {
    eventId: String @unique
    nodeId, trackId, frameIndex
    lightState, lightConfidence
    plateText, plateConfidence
    status: pending|approved|rejected
    mediaUrl, metadata
  }
  class EdgeNode <<JPA Entity>> {
    nodeId @unique, ipAddress
    status, lastPing, settingsJson
  }
  class User <<JPA Entity>> {
    username @unique, passwordHash (BCrypt)
    role: ADMIN|OPERATOR|OFFICER, enabled
  }
  class RefreshToken <<JPA Entity>> {
    tokenHash @unique (SHA-256)
    expiresAt, revokedAt, userAgent
  }
}

package "Web dashboard — Next.js 16 (TypeScript)" #FEF3C7 {
  class apiClient <<lib/api.ts>> {
    -fetchAPI(path, opts): auto refresh 401
    +getViolationsPage / getStats
    +updateViolationStatus
    +setStopLine / setLightRoi
  }
  class authSession <<lib/auth.ts>> {
    +getSession / setSession / clearSession
    +refreshAccessToken(session)
  }
  class VideoPlayer <<rAF loop>> {
    +renderOverlay(ctx) mỗi khung hình
  }
  class NodeCalibrationPanel {
    +chế độ vẽ: vạch / hộp đèn / hướng
    +toạ độ pixel gốc của video
  }
  class AISidebar {
    +ask(question) -> SQL + bảng/biểu đồ
  }
}

RedLightViolationPipeline *-- YoloDetector
RedLightViolationPipeline *-- TunedByteTrack
RedLightViolationPipeline *-- ViolationDetector
RedLightViolationPipeline o-- YoloLightClassifier
RedLightViolationPipeline o-- HsvLightClassifier
RedLightViolationPipeline o-- PlateAssociator
RedLightViolationPipeline o-- ViolationOutbox
YoloLightClassifier ..> HsvLightClassifier : fusion khi lệch
YoloDetector ..> PipelineMetrics : ghi nhận frame
TunedByteTrack ..> Track : sinh
Track *-- MotionState
Track *-- "0..*" TrackSample
TrackMotion ..> TrackSample : tích luỹ
TrackMotion ..> MotionState : ước lượng hồi quy
PlateDetector <|-- YoloDetector
PlateAssociator --> PlateDetector : phát hiện biển
PlateAssociator --> FastPlateOCR : đọc ký tự
FastPlateOCR --> vn_plate : kiểm tra cấu trúc
ViolationDetector --> Tripwire : đọc vạch active mỗi frame
ViolationDetector --> RedLightStabilizer : tín hiệu ổn định
ViolationDetector ..> ViolationOutbox : enqueue hồ sơ
ViolationOutbox <.. ViolationSender : fetch_pending
ViolationSender --> ViolationController : HTTP batch (X-Ingest-Token)
ViolationSender --> MediaController : upload ảnh bằng chứng
EdgeControlAPI ..> ViolationDetector : cập nhật vạch LIVE
EdgeControlAPI ..> HsvLightClassifier : cập nhật vùng đèn LIVE
EdgeControlAPI --> PipelineMetrics : đọc /health

ViolationController --> ViolationService
AuthController --> JwtService
AuthController --> RefreshTokenService
EdgeNodeController --> EdgeNodeService
EdgeNodeController --> EdgeProxyService
MediaController --> MinioStorageService
AiQueryController --> AiQueryService
ViolationService --> Violation
ViolationService --> MinioStorageService
ViolationStatsService --> Violation
EdgeNodeService --> EdgeNode
RefreshTokenService --> RefreshToken
RefreshTokenService --> User
AiQueryService ..> Violation : SELECT có LIMIT
IngestTokenFilter ..> IngestPaths : đọc quy tắc
JwtAuthFilter ..> User : nạp vai trò
EdgeProxyService --> EdgeControlAPI : HTTP/1.1 + X-Edge-Token

apiClient --> authSession
apiClient --> ViolationController : rewrite /api
apiClient --> EdgeNodeController : hiệu chuẩn qua proxy
NodeCalibrationPanel --> VideoPlayer : vẽ trên khung hình live
NodeCalibrationPanel --> apiClient
AISidebar --> AiQueryController
@enduml
"""

# ------------------------------------------------ 6. ACTIVITY: luồng xét vi phạm
D["activity-detection-v2"] = """@startuml
skinparam backgroundColor #FFFFFF
skinparam shadowing false
title Hoạt động xét vi phạm tại node biên (mỗi khung hình)

start
:Đọc khung hình từ nguồn video\\n(tệp MP4 phát lặp / luồng RTSP);
:Ghi nhận metrics: số khung hình, fps;

partition "Nhận biết tín hiệu đèn" {
  if (Đã hiệu chuẩn vùng đèn?) then (Chưa)
    :Tín hiệu = UNKNOWN;
  else (Rồi)
    :Phân loại đèn trong vùng ROI\\n(YOLO26n-cls là chính);
    if (Kết quả lệch với bằng chứng HSV?) then (Lệch)
      :Fusion: kết hợp HSV +\\nvị trí bóng đèn theo chiều dọc;
    else (Đồng thuận)
      :Giữ kết quả của mô hình;
    endif
    :Bộ ổn định trạng thái (debounce +\\nhysteresis, ngưỡng tính theo giây thực);
  endif
}

partition "Theo dõi phương tiện" {
  :YOLO26m fine-tune phát hiện 4 lớp\\n(ngưỡng tin cậy 0,30);
  :ByteTrack gán mã theo dõi,\\ncập nhật quỹ đạo và vận tốc;
  :Điểm xét = bottom_center của hộp giới hạn;
}

if (Tín hiệu ĐỎ đã ổn định\\nvà đã hiệu chuẩn vạch dừng?) then (Không)
  :Không xét vi phạm\\n(vẫn tiếp tục ghi nhận giám sát);
  stop
else (Có)
endif

partition "Xét vi phạm" {
  :Với mỗi track: tính phía của điểm xét\\nso với vạch dừng (có vùng chết deadband);
  if (Chuyển phía hợp lệ theo hướng giám sát\\nvà đoạn di chuyển CẮT vạch?) then (Không)
    stop
  else (Có)
    if (Track này đã có hồ sơ vi phạm?) then (Rồi)
      :Bỏ qua (một track một hồ sơ);
      stop
    else (Chưa)
      :Tạo hồ sơ: event_id =\\nrlv-{run_id}-{frame}-{track};
    endif
  endif
}

partition "Hoàn thiện hồ sơ" {
  :Đọc biển số nếu track có biển\\n(OCR + kiểm tra cấu trúc TT 24/2023);
  if (Biển số hợp lệ?) then (Có)
    :Gắn biển + độ tin cậy vào hồ sơ;
  else (Không)
    :plate_text = null\\n(hồ sơ vẫn được gửi);
  endif
  :Vẽ ảnh bằng chứng: vạch dừng, hộp xe,\\nquỹ đạo, trạng thái đèn;
  :Ghi hồ sơ + ảnh vào outbox SQLite\\n(TRƯỚC mọi thao tác mạng);
}

:Luồng gửi dữ liệu (thread riêng)\\nđẩy lô 20 hồ sơ mỗi 5 giây;
stop
@enduml
"""

# ------------------------------------------------- 7. SEQUENCE: gửi hồ sơ + outbox
D["sequence-delivery"] = """@startuml
skinparam backgroundColor #FFFFFF
skinparam shadowing false
autonumber "<b>0."
title Chuỗi hoạt động: phân phối hồ sơ bền vững (outbox + lô idempotent)

box "Node biên" #EFF6FF
  participant "Pipeline" as P
  database "Outbox\\n(SQLite)" as OB
  participant "ViolationSender\\n(thread riêng)" as S
end box
box "Máy chủ trung tâm" #ECFDF5
  participant "IngestTokenFilter" as F
  participant "ViolationService" as VS
  database "PostgreSQL" as DB
  participant "MinioStorageService" as M
end box

P -> OB : enqueue(hồ sơ, ảnh JPEG)\\ngiao dịch bền vững
note right of OB
  event_id có ràng buộc UNIQUE
  -> ghi trùng bị chặn ngay tại node
end note
P --> P : tiếp tục xử lý khung hình kế\\n(không chờ mạng)

loop mỗi 5 giây (flush_interval)
  S -> OB : fetch_pending(limit = 20)\\nkhông tải cột ảnh (BLOB)
  S -> F : POST /api/violations/batch\\nX-Ingest-Token + X-Node-ID
  F -> F : so khớp token hằng thời gian\\n(MessageDigest.isEqual)
  F -> VS : chuyển tiếp khi token hợp lệ
  VS -> DB : existsByEventId cho từng hồ sơ
  VS -> DB : INSERT các hồ sơ mới (status = pending)
  VS --> S : 201 {accepted, duplicates, failed,\\nacceptedEventIds, duplicateEventIds}
  S -> OB : mark_sent(ids đã accepted)

  loop mỗi hồ sơ đã accepted
    S -> OB : fetch_media_pending(limit = 10)
    S -> M : POST /api/v1/violations/{eventId}/media\\n(multipart)
    M -> M : lưu object key\\nviolations/{eventId}/{uuid}.jpg
    M --> S : 201 {object_name, url}
    S -> OB : mark_media_sent(eventId)
  end
end

group Khi trung tâm không phản hồi / mất mạng
  S -> S : tăng attempts, lùi bước 5 → 10 → 20 → 40 → 60 giây
  note right of S
    Hồ sơ vẫn nằm trong outbox,
    pipeline tiếp tục chạy bình thường
    -> hệ thống chỉ TRỄ, không MẤT dữ liệu
  end note
end group

group Khởi động lại node
  S -> OB : đọc các hồ sơ pending còn lại\\n(volume outbox_data bền vững)
  S -> VS : gửi lại lô cũ
  VS -> DB : existsByEventId = true
  VS --> S : duplicates = N (không lỗi, không trùng hồ sơ)
end group
@enduml
"""

# ------------------------------------------------- 8. SEQUENCE: hiệu chuẩn từ xa
D["sequence-calibration-v2"] = """@startuml
skinparam backgroundColor #FFFFFF
skinparam shadowing false
autonumber "<b>0."
title Chuỗi hoạt động: hiệu chuẩn vạch dừng từ xa, hiệu lực tức thời

actor "Kỹ thuật viên\\n(Operator)" as Op
box "Web dashboard" #FEF3C7
  participant "Trang /nodes/[nodeId]" as Page
  participant "NodeCalibrationPanel\\n+ VideoPlayer" as Panel
  participant "apiClient\\n(lib/api.ts)" as Api
end box
box "Máy chủ trung tâm" #ECFDF5
  participant "JwtAuthFilter" as Jwt
  participant "EdgeNodeController" as Ctrl
  participant "EdgeProxyService" as Proxy
end box
box "Node biên" #EFF6FF
  participant "EdgeControlAPI" as Edge
  participant "Pipeline" as P
end box

Op -> Page : mở trang chi tiết node
Page -> Api : GET /api/v1/edge-nodes/{nodeId}
Api -> Jwt : Authorization: Bearer <JWT>
Jwt -> Ctrl : vai trò ADMIN / OPERATOR
Ctrl --> Page : thông tin node + trạng thái online
Page -> Panel : phát luồng HLS từ node
Panel -> Edge : GET /api/cameras/{id}/stream\\n(qua rewrite /edge-api)
Edge --> Panel : playlist .m3u8 + phân đoạn .ts

Op -> Panel : chọn chế độ vẽ vạch dừng
Op -> Panel : kéo hai điểm trên khung hình video
Panel -> Panel : quy đổi toạ độ hiển thị\\nvề pixel gốc của video
Panel -> Api : POST /api/v1/edge-nodes/{nodeId}/calibration/stop-line\\n{x1, y1, x2, y2, direction}
Api -> Jwt : Bearer <JWT>
Jwt -> Ctrl : kiểm tra vai trò
Ctrl -> Proxy : forward(nodeId, POST, /action/stop-line)
Proxy -> Edge : POST /action/stop-line\\nX-Edge-Token, HTTP/1.1
note right of Proxy
  Bắt buộc HTTP/1.1: nâng cấp h2c
  từng làm uvicorn rơi thân POST -> 422
end note
Edge -> Edge : kiểm token + giới hạn 30 yêu cầu/phút/IP
Edge -> P : set_active_tripwire(vạch mới)
P --> Edge : áp dụng NGAY khung hình kế tiếp\\n(khởi động lại tiến trình)
Edge --> Proxy : 200 {ok, calibration}
Proxy --> Ctrl : kết quả
Ctrl --> Panel : thông báo thành công (Toast)

Panel -> Api : GET /api/v1/edge-nodes/{nodeId}/calibration\\n(xác minh)
Api --> Panel : vạch dừng + vùng đèn hiện hành
Op -> Panel : vẽ khung hình trên video để kiểm tra trực quan
note over P
  Hồ sơ vi phạm chuyển từ
  DISABLED -> ENABLED kể từ khung hình này
end note
@enduml
"""

# ------------------------------------------------------- 9. KIẾN TRÚC HỆ THỐNG
D["architecture-v2"] = """@startuml
skinparam backgroundColor #FFFFFF
skinparam shadowing false
skinparam componentStyle rectangle
skinparam component {
  BackgroundColor #FFFFFF
  BorderColor #64748B
}
title Kiến trúc hệ thống RLVD — ba tầng

node "Điểm giám sát (tại nút giao)" #FEF2F2 {
  component "Camera IP\\n(RTSP) / tệp video" as Cam
  node "Node biên — máy GPU (Python 3.11)" #EFF6FF {
    component "Tầng thị giác\\nYOLO26m + ByteTrack\\n+ TrackMotion" as Vision
    component "Phân loại đèn\\nYOLO26n-cls + fusion HSV" as Light
    component "Biển số\\nPlateDetector + fast-plate-ocr\\n+ validator TT 24/2023" as Plate
    component "Xét vi phạm\\nRedLightStabilizer + Tripwire" as Logic
    database "Outbox\\nSQLite" as OB
    component "Control API\\nFastAPI :8080" as EdgeApi
    component "Camera stream\\nFFmpeg -> HLS" as HLS
  }
}

node "Máy chủ trung tâm (Docker)" #ECFDF5 {
  component "Central Server\\nSpring Boot 3.3 / Java 17" as Central {
    component "Controller\\n(Auth, Violation, EdgeNode,\\nMedia, AiQuery)" as Ctrl
    component "Service\\n(ingest idempotent, thống kê,\\nproxy, refresh token)" as Svc
    component "Security\\nJwtAuthFilter + IngestTokenFilter\\n+ IngestPaths" as Sec
    component "Repository\\nSpring Data JPA" as Repo
  }
  database "PostgreSQL 16\\n4 bảng, 10 index" as PG
  database "MinIO\\nảnh bằng chứng" as Minio
  component "Trợ lý ngôn ngữ\\nText-to-SQL" as AI
}

node "Máy trạm người dùng" #FFFBEB {
  component "Web dashboard\\nNext.js 16 / React 19" as Web {
    component "app/(dashboard)\\n7 trang" as Pages
    component "rewrite proxy\\n/api -> central\\n/edge-api -> edge" as Rewrite
  }
}

cloud "Gemini API\\n(bên ngoài)" as Gemini

Cam --> Vision : khung hình
Cam --> HLS : phát lại
Vision --> Light
Vision --> Logic
Light --> Logic
Plate --> Logic
Logic --> OB : ghi hồ sơ TRƯỚC khi gọi mạng
OB --> Ctrl : POST lô 20 hồ sơ / 5 giây\\n(X-Ingest-Token)
OB --> Ctrl : POST ảnh bằng chứng (multipart)
Ctrl --> Svc
Sec ..> Ctrl : lọc trước
Svc --> Repo
Repo --> PG
Svc --> Minio
Svc --> AI
AI --> Gemini : câu hỏi tiếng Việt -> SQL
Ctrl --> EdgeApi : gọi ngược (hiệu chuẩn, snapshot)\\nHTTP/1.1 + X-Edge-Token
Web --> Rewrite
Rewrite --> Ctrl : REST + JWT
Rewrite --> HLS : luồng .m3u8 / .ts
Pages ..> Rewrite
@enduml
"""

# ------------------------------------------------------------- 10. ERD (mới)
D["erd-v2"] = """@startuml
skinparam backgroundColor #FFFFFF
skinparam shadowing false
skinparam linetype ortho
hide circle
skinparam entity {
  BackgroundColor #EFF6FF
  BorderColor #3B82F6
}
title Mô hình thực thể - quan hệ (ERD) — CSDL rlvd_central

entity "users" as U {
  * id : BIGINT <<PK, tự tăng>>
  --
  * username : VARCHAR(64) <<UNIQUE>>
  * password_hash : VARCHAR(128) BCrypt
  full_name : VARCHAR(128)
  * role : VARCHAR(16) ADMIN|OPERATOR|OFFICER
  * enabled : BOOLEAN
  created_at, updated_at : TIMESTAMP
  ..
  index: idx_users_username (unique)
}

entity "refresh_tokens" as R {
  * id : BIGINT <<PK, tự tăng>>
  --
  * user_id : BIGINT <<FK logic -> users.id>>
  * token_hash : VARCHAR(64) <<UNIQUE>> SHA-256
  * expires_at : TIMESTAMP
  revoked_at : TIMESTAMP
  * created_at : TIMESTAMP
  user_agent : VARCHAR(255)
  ..
  index: idx_refresh_tokens_user,
  idx_refresh_tokens_hash (unique)
}

entity "edge_nodes" as N {
  * id : UUID <<PK>>
  --
  * node_id : VARCHAR <<UNIQUE>>
  * name : VARCHAR
  ip_address : VARCHAR
  * status : VARCHAR online|offline|degraded|maintenance
  last_ping : TIMESTAMP
  settings_json : TEXT
  created_at, updated_at : TIMESTAMP
  ..
  index: idx_edge_node_node_id (unique),
  idx_edge_node_status, idx_edge_node_last_ping
}

entity "violations" as V {
  * id : BIGINT <<PK, tự tăng>>
  --
  * event_id : VARCHAR <<UNIQUE>> rlv-{run}-{frame}-{track}
  * node_id : VARCHAR <<FK logic -> edge_nodes.node_id>>
  track_id, frame_index : INT
  timestamp_ms : DOUBLE
  crossing_point_x/y : DOUBLE
  previous_point_x/y : DOUBLE
  bbox_x1/y1/x2/y2 : DOUBLE
  light_state : VARCHAR red|yellow|green
  light_confidence : DOUBLE
  previous_side, current_side : INT
  plate_text : VARCHAR
  plate_confidence : DOUBLE
  * status : VARCHAR pending|approved|rejected
  media_url : VARCHAR(1024) (object key MinIO)
  metadata : TEXT (JSON)
  created_at, updated_at : TIMESTAMP
  ..
  index: idx_event_id (unique), idx_node_id,
  idx_status, idx_created_at
}

U ||--o{ R : "một người dùng\\ncó nhiều phiên đăng nhập"
N ||--o{ V : "một node gửi\\nnhiều hồ sơ vi phạm"
U ||--o{ V : "duyệt / từ chối\\n(theo vai trò)"

note bottom of V
  event_id UNIQUE là chốt chống trùng:
  node gửi lại lô sau khi mất mạng
  -> trung tâm đếm duplicates, không tạo bản sao
end note

note right of N
  Quan hệ node -> violations là quan hệ LOGIC
  (ghép theo chuỗi node_id, không có khoá ngoại
  vật lý) để node mới không chặn ghi hồ sơ
end note
@enduml
"""

# ------------------------------------------------------- 11. SƠ ĐỒ TRIỂN KHAI
D["deployment-v2"] = """@startuml
skinparam backgroundColor #FFFFFF
skinparam shadowing false
skinparam componentStyle rectangle
title Sơ đồ triển khai bằng Docker Compose (5 dịch vụ)

node "Máy trạm GPU — NVIDIA RTX 2060, CUDA 12.4, Linux" {
  frame "Mạng bridge nội bộ: rlvd-net (không lộ ra Internet)" {
    component "edge-pipeline\\nPython 3.11 + GPU\\ncổng host 8082 -> 8080" as Edge #EFF6FF
    component "central-server\\nJava 17 (build 2 giai đoạn Maven)\\ncổng host 8002 -> 8000" as Central #ECFDF5
    component "web-dashboard\\nNext.js standalone, user nextjs\\ncổng host 3000" as Web #FEF3C7
    database "postgres:16-alpine\\nhealthcheck pg_isready\\nvolume postgres_data" as PG #F1F5F9
    database "minio\\nhealthcheck mc ready\\nvolume minio_data" as Minio #F1F5F9
  }
  database "volume outbox_data\\n(SQLite của node biên)" as OBV
  database "volume hls_data\\n(phân đoạn video HLS)" as HLSV
}

actor "Người vận hành" as Op
component "Reverse proxy\\nNginx / Caddy — kết thúc TLS\\n(chỉ khi công bố ra Internet)" as Proxy #FEE2E2
cloud "Gemini API" as Gemini

Op --> Proxy : HTTPS 443
Proxy --> Web : HTTP 3000
Op ..> Web : hoặc HTTP 3000 trực tiếp (mạng nội bộ)

Edge --> PG
Edge --> Central : lô hồ sơ + ảnh
Central --> PG : JDBC (HikariCP, tối đa 10 kết nối)
Central --> Minio : SDK S3
Central --> Gemini : REST
Web --> Central : rewrite /api
Web --> Edge : rewrite /edge-api (HLS, hiệu chuẩn)
Central --> Edge : gọi ngược HTTP/1.1
Edge --> OBV
Edge --> HLSV

note bottom of Proxy
  Toàn bộ secret đọc từ .env qua ${VAR:-default}
  TZ=Asia/Ho_Chi_Minh đặt trên MỌI dịch vụ
  restart: unless-stopped cho cả 5 container
end note
@enduml
"""

# ------------------------------------------------------- 12. VÒNG ĐỜI HỒ SƠ
D["state-violation-v2"] = """@startuml
skinparam backgroundColor #FFFFFF
skinparam shadowing false
title Vòng đời hồ sơ vi phạm (trạng thái status trong CSDL)

[*] --> Pending : node biên phát hiện vi phạm\\nvà đẩy lô lên trung tâm
state "Pending\\n(chờ duyệt)" as Pending #FEF3C7
state "Approved\\n(đã phê duyệt)" as Approved #DCFCE7
state "Rejected\\n(đã từ chối)" as Rejected #FEE2E2
state "Đã xoá" as Deleted #E2E8F0

Pending --> Approved : Officer / Admin duyệt\\nPATCH /api/violations/{id}/status
Pending --> Rejected : Officer / Admin từ chối
Approved --> Rejected : Admin sửa lại kết luận
Rejected --> Approved : Admin sửa lại kết luận
Approved --> Deleted : chỉ Admin\\nDELETE /api/violations/{id}
Rejected --> Deleted : chỉ Admin
Pending --> Deleted : chỉ Admin

note right of Pending
  Hồ sơ mới luôn ở trạng thái pending:
  hệ thống KHÔNG tự kết luận vi phạm,
  con người là chốt cuối (human-in-the-loop)
end note

note right of Approved
  Approved phục vụ lập biên bản /
  thông báo phạt nguội
end note

Deleted --> [*]
@enduml
"""

# ------------------------------------------------------- 13. DATA FLOW
D["data-flow-v2"] = """@startuml
skinparam backgroundColor #FFFFFF
skinparam shadowing false
title Luồng dữ liệu hồ sơ vi phạm qua ba tầng

cloud "Camera /\\nluồng RTSP" as Cam
rectangle "Node biên" #EFF6FF {
  card "1. Khung hình\\n(ảnh BGR)" as F1
  card "2. Detection\\n(4 lớp + hộp giới hạn)" as F2
  card "3. Track\\n(mã theo dõi + quỹ đạo + vận tốc)" as F3
  card "4. Sự kiện vi phạm\\n(event_id, đèn, toạ độ)" as F4
  card "5. Hồ sơ + ảnh JPEG\\ntrong outbox SQLite" as F5
}
rectangle "Máy chủ trung tâm" #ECFDF5 {
  card "6. Lô JSON qua HTTP\\n(X-Ingest-Token)" as F6
  card "7. Bản ghi PostgreSQL\\n(status = pending)" as F7
  card "8. Ảnh trong MinIO\\n(object key)" as F8
}
rectangle "Web dashboard" #FEF3C7 {
  card "9. Danh sách phân trang\\n+ bộ lọc" as F9
  card "10. Ảnh bằng chứng\\n(stream qua proxy blob)" as F10
  card "11. Kết luận duyệt\\n(approved / rejected)" as F11
}
actor "Cán bộ nghiệp vụ" as Officer

Cam --> F1
F1 --> F2 : YOLO26m
F2 --> F3 : ByteTrack
F3 --> F4 : đèn đỏ ổn định + cắt vạch
F4 --> F5 : kèm OCR biển số
F5 --> F6 : thread gửi dữ liệu
F6 --> F7 : ingest idempotent
F6 --> F8 : upload multipart
F7 --> F9 : GET /api/violations/page
F8 --> F10 : GET .../media/blob
Officer --> F11
F11 --> F7 : PATCH status

note bottom of F5
  Điểm cắt quan trọng: dữ liệu được ghi BỀN VỮNG
  tại node trước khi đi qua mạng
end note
@enduml
"""

# ---------------------------------------------------- 14. BIỂU ĐỒ THÀNH PHẦN
D["component-v2"] = """@startuml
skinparam backgroundColor #FFFFFF
skinparam shadowing false
skinparam componentStyle rectangle
title Sơ đồ thành phần và giao diện giữa các tầng

package "Node biên (Python)" #EFF6FF {
  [core.pipeline] as c1
  [core.detector / byte_tracker / motion] as c2
  [core.traffic_light_yolo / traffic_light_cv] as c3
  [core.violation_logic / geometry / calibration] as c4
  [core.plate_detector / plate_associator / ocr_recognizer / vn_plate] as c5
  [outbox / violation_sender] as c6
  [api.server (FastAPI)] as c7
  [camera_stream (FFmpeg HLS)] as c8
  [central_client (đăng ký + heartbeat)] as c9
  [metrics / settings] as c10
}

package "Máy chủ trung tâm (Java)" #ECFDF5 {
  [controller.*] as j1
  [service.*] as j2
  [repository.*] as j3
  [entity.*] as j4
  [config.* (Security, Jwt, IngestPaths)] as j5
  [dto.*] as j6
}

package "Web dashboard (TypeScript)" #FEF3C7 {
  [app/(dashboard)/*] as w1
  [components/*] as w2
  [lib/api + lib/auth + lib/types] as w3
  [middleware.ts] as w4
  [next.config.ts (rewrites)] as w5
}

database "PostgreSQL" as DB
storage "MinIO" as MO
cloud "Gemini API" as GM

c1 --> c2
c1 --> c3
c1 --> c4
c1 --> c5
c1 --> c6
c1 --> c10
c7 --> c4 : cập nhật hiệu chuẩn live
c7 --> c8
c7 --> c10
c9 --> j1 : đăng ký / heartbeat
c6 --> j1 : lô hồ sơ + ảnh

j1 --> j2
j2 --> j3
j3 --> j4
j4 --> DB
j2 --> MO
j2 --> GM
j5 ..> j1 : lọc bảo vệ
j1 --> j6
j2 --> j5

w1 --> w2
w1 --> w3
w4 ..> w1 : chặn khi chưa đăng nhập
w3 --> w5
w5 --> j1 : /api -> central
w5 --> c7 : /edge-api -> edge
j2 --> c7 : gọi ngược (proxy)
@enduml
"""

# ---------------------------------------------------- 15. SECURITY LAYERS
D["security-layers-v2"] = """@startuml
skinparam backgroundColor #FFFFFF
skinparam shadowing false
title Các lớp bảo vệ của hệ thống

rectangle "Lớp 1 — Ranh giới mạng" #FEE2E2 {
  card "Mạng bridge nội bộ rlvd-net;\\nPostgreSQL / MinIO không mở cổng ra Internet;\\nreverse proxy kết thúc TLS, tường lửa chỉ mở 443" as L1
}
rectangle "Lớp 2 — Xác thực" #FFEDD5 {
  card "Người dùng: JWT HS256 (khoá bí mật >= 32 ký tự, từ chối khởi động nếu yếu), TTL 12 giờ\\nRefresh token: 32 byte ngẫu nhiên, chỉ lưu SHA-256, xoay vòng mỗi lần làm mới,\\nphát hiện dùng lại -> thu hồi TOÀN BỘ phiên của người dùng\\nNode biên: ingest token riêng (X-Ingest-Token), so khớp hằng thời gian\\nControl-plane của node: X-Edge-Token" as L2
}
rectangle "Lớp 3 — Phân quyền" #FEF9C3 {
  card "Ba vai trò ADMIN / OPERATOR / OFFICER khai báo tập trung trong SecurityConfig;\\nduyệt hồ sơ: ADMIN + OFFICER; hiệu chuẩn: ADMIN + OPERATOR;\\nxoá hồ sơ: chỉ ADMIN; đường dẫn ingest: IngestPaths là nguồn sự thật duy nhất" as L3
}
rectangle "Lớp 4 — Bảo vệ dữ liệu" #DCFCE7 {
  card "JPA tham số hoá mọi truy vấn (chống SQL injection);\\ntrợ lý ngôn ngữ chỉ cho phép SELECT + LIMIT 50,\\nchặn DROP/DELETE/INSERT/UPDATE/ALTER/TRUNCATE/GRANT/REVOKE;\\nmật khẩu băm BCrypt; ảnh bằng chứng đặt trong bucket riêng" as L4
}
rectangle "Lớp 5 — Chống lạm dụng" #DBEAFE {
  card "Giới hạn 30 yêu cầu/phút/IP cho POST /action/* trên node biên (cửa sổ trượt);\\nCORS: node biên dùng danh sách cho phép, trung tâm credentials=false;\\nsecurity headers: X-Content-Type-Options nosniff, X-Frame-Options DENY, no-store;\\ncontainer web chạy bằng user nextjs không phải root" as L5
}
rectangle "Lớp 6 — Kiểm soát thay đổi" #EDE9FE {
  card "Mọi secret đọc từ .env (đã gitignore), chỉ .env.example nằm trong git;\\nquét secret hard-code toàn kho mã nguồn;\\nCI 3 job: pytest + ruff, maven test-compile, next lint + build" as L6
}

L1 -[hidden]down- L2
L2 -[hidden]down- L3
L3 -[hidden]down- L4
L4 -[hidden]down- L5
L5 -[hidden]down- L6
@enduml
"""

# ---------------------------------------------------- 16. SEQUENCE: xác thực
D["sequence-auth-v2"] = """@startuml
skinparam backgroundColor #FFFFFF
skinparam shadowing false
autonumber "<b>0."
title Chuỗi hoạt động: đăng nhập, làm mới token và phát hiện dùng lại

actor "Người dùng" as U
box "Web dashboard" #FEF3C7
  participant "Trang /login" as L
  participant "authSession\\n(lib/auth.ts)" as S
  participant "apiClient\\n(lib/api.ts)" as A
end box
box "Máy chủ trung tâm" #ECFDF5
  participant "AuthController" as C
  participant "RefreshTokenService" as R
  participant "JwtService" as J
  database "PostgreSQL" as DB
end box

== Đăng nhập ==
U -> L : nhập tên đăng nhập + mật khẩu
L -> C : POST /api/auth/login
C -> DB : tìm user theo username
DB --> C : bản ghi + password_hash + role
C -> C : BCrypt.matches (so khớp)
alt Sai mật khẩu hoặc tài khoản bị khoá
  C --> L : 401 thông điệp chung\\n(không tiết lộ tài khoản có tồn tại)
else Hợp lệ
  C -> J : generateToken(username, role)
  J --> C : JWT HS256, TTL 12 giờ
  C -> R : issue(user)
  R -> R : sinh 32 byte ngẫu nhiên
  R -> DB : INSERT refresh_tokens\\n(token_hash = SHA-256, expires 7 ngày)
  C --> L : {token, refreshToken, role, expires_in}
  L -> S : lưu localStorage + cookie rlvd_token (cho SSR)
end

== Truy cập API ==
U -> A : mở trang /violations
A -> C : GET /api/violations/page\\nAuthorization: Bearer <JWT>
C --> A : 200 dữ liệu phân trang

== JWT hết hạn (401) ==
A -> C : GET /api/violations/page
C --> A : 401
A -> R : POST /api/auth/refresh {refreshToken}
R -> DB : tìm theo token_hash
alt Token còn hiệu lực, chưa thu hồi
  R -> DB : đánh dấu bản cũ revoked_at
  R -> DB : INSERT token mới (xoay vòng)
  R -> J : cấp JWT mới
  R --> A : {token, refreshToken mới}
  A -> C : thử lại yêu cầu gốc (đúng một lần)
  C --> A : 200
else Token đã bị thu hồi — dấu hiệu bị đánh cắp
  R -> DB : revokeAllByUser(user_id)\\nthu hồi TOÀN BỘ phiên
  R --> A : 401
  A -> S : clearSession()
  A --> U : chuyển về /login
end

== Đăng xuất ==
U -> A : bấm đăng xuất
A -> C : POST /api/auth/logout {refreshToken}
C -> DB : đánh dấu revoked_at
C --> A : 200
@enduml
"""

# ---------------------------------------------------- 17. SEQUENCE: trợ lý AI
D["sequence-ai-v2"] = """@startuml
skinparam backgroundColor #FFFFFF
skinparam shadowing false
autonumber "<b>0."
title Chuỗi hoạt động: trợ lý hỏi dữ liệu bằng tiếng Việt (Text-to-SQL)

actor "Cán bộ nghiệp vụ" as U
box "Web dashboard" #FEF3C7
  participant "AISidebar" as SB
  participant "apiClient" as A
end box
box "Máy chủ trung tâm" #ECFDF5
  participant "AiQueryController" as C
  participant "AiQueryService" as S
  database "PostgreSQL" as DB
end box
participant "Gemini API" as G

U -> SB : mở bảng trợ lý (nút nổi góc phải)
U -> SB : "Tháng này node nào phát hiện nhiều vi phạm nhất?"
SB -> A : POST /api/ai/query {question}
A -> C : kèm Authorization: Bearer <JWT>\\n(question <= 500 ký tự)
C -> S : process(question)
S -> S : dựng prompt: mô tả lược đồ 4 bảng\\n+ yêu cầu chỉ viết SELECT
S -> G : gọi REST qua java.net.http
G --> S : câu lệnh SQL + gợi ý loại biểu đồ

S -> S : KIỂM TRA AN TOÀN
note right of S
  chỉ chấp nhận SELECT,
  chặn DROP/DELETE/INSERT/UPDATE/
  ALTER/TRUNCATE/GRANT/REVOKE,
  tự thêm LIMIT 50
end note

alt SQL không an toàn
  S --> C : {error: "câu lệnh bị chặn"}
  C --> SB : hiển thị lỗi, KHÔNG thực thi
else SQL hợp lệ
  S -> DB : thực thi truy vấn chỉ đọc
  DB --> S : tập kết quả
  S --> C : {sql, columns, rows, count, chartType}
  C --> SB : 200
  SB -> SB : hiển thị câu SQL đã sinh\\n+ bảng kết quả / biểu đồ cột
end
SB --> U : câu trả lời kèm dữ liệu
@enduml
"""


def main() -> int:
    base.OUT = OUT  # ghi ảnh vào thư mục thesis/
    only = sys.argv[1:] or None
    names = [n for n in D if (only is None or n in only)]
    ok = 0
    for i, name in enumerate(names, 1):
        if base.render(name, D[name], i, len(names)):
            ok += 1
        time.sleep(1.2)
    print(f"\nHoàn thành: {ok}/{len(names)} -> {OUT}")
    return 0 if ok == len(names) else 1


if __name__ == "__main__":
    sys.exit(main())
