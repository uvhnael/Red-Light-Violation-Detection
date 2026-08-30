# -*- coding: utf-8 -*-
"""Sinh 17 diagram PlantUML tiếng Việt (có dấu) cho báo cáo RLVD v2.

Chiến lược font: PlantUML server render SVG (text còn nguyên, không bị phá dấu),
sau đó rasterize cục bộ bằng cairosvg + font Noto Sans (hỗ trợ đầy đủ tiếng Việt).
Ảnh PNG xuất tại docs/report-assets/vn/<name>.png
"""
import zlib
import urllib.request
import urllib.error
import sys
import time
import re
from pathlib import Path

OUT = Path("/home/uvhnael/projects/Red-Light-Violation-Detection/docs/report-assets/vn")
OUT.mkdir(parents=True, exist_ok=True)

D = {}

# ============================================================ 1. USE CASE (nâng cấp)
D["use-case"] = """@startuml
skinparam backgroundColor #FFFFFF
skinparam shadowing false
skinparam actorStyle awesome
left to right direction
skinparam packageStyle rectangle
skinparam usecase {
  BackgroundColor #EFF6FF
  BorderColor #3B82F6
}

actor "Cán bộ xử lý vi phạm\\n(Officer)" as Officer
actor "Kỹ thuật viên\\n(Operator)" as Operator
actor "Node biên\\n(Edge Node)" as Edge
actor "Máy chủ trung tâm\\n(Central Server)" as Central

rectangle "Hệ thống phát hiện vi phạm vượt đèn đỏ (RLVD)" {
  usecase "UC1. Đăng ký node,\\ngửi heartbeat" as UC1
  usecase "UC2. Hiệu chuẩn vạch dừng,\\nvùng đèn, hướng giám sát" as UC2
  usecase "UC3. Phát hiện xe vượt đèn đỏ\\n(detect + track + tripwire)" as UC3
  usecase "UC4. Đọc biển số xe (OCR)\\n+ chuẩn hoá biển VN" as UC4
  usecase "UC5. Gửi hồ sơ vi phạm\\n(outbox + batch idempotent)" as UC5
  usecase "UC6. Xem camera trực tiếp (HLS)\\n+ snapshot" as UC6
  usecase "UC7. Duyệt / từ chối\\nhồ sơ vi phạm" as UC7
  usecase "UC8. Tra cứu vi phạm\\nvà xem thống kê" as UC8
  usecase "UC9. Hỏi dữ liệu bằng AI\\n(Text-to-SQL tiếng Việt)" as UC9
  usecase "UC10. Quản lý node biên\\n(trạng thái, cấu hình)" as UC10
  usecase "UC11. Tải ảnh bằng chứng\\nlên MinIO" as UC11
}

Operator --> UC1
Operator --> UC2
Operator --> UC6
Operator --> UC10
Officer --> UC7
Officer --> UC8
Officer --> UC9

Edge ..> UC3 : <<thực hiện>>
Edge ..> UC4 : <<thực hiện>>
Edge ..> UC5 : <<thực hiện>>
Central ..> UC5 : <<nhận batch>>
Central ..> UC11 : <<nhận media>>
UC2 ..> UC3 : <<kích hoạt>>
UC3 ..> UC4 : <<include>>
UC3 ..> UC5 : <<include>>
UC5 ..> UC11 : <<include>>
UC5 ..> UC7 : <<phân phối>>
UC1 ..> UC2 : <<điều kiện>>
@enduml
"""

# ============================================================ 2. USE CASE chi tiết Officer
D["use-case-officer"] = """@startuml
skinparam backgroundColor #FFFFFF
skinparam shadowing false
skinparam actorStyle awesome
left to right direction

actor "Cán bộ xử lý vi phạm\\n(Officer)" as Officer

rectangle "Nghiệp vụ xử lý vi phạm" {
  usecase "Xem danh sách hồ sơ\\nchờ duyệt (pending)" as A
  usecase "Xem chi tiết hồ sơ:\\nảnh toàn cảnh, crop biển số,\\ntrạng thái đèn, toạ độ" as B
  usecase "Phê duyệt hồ sơ\\n(approved)" as C
  usecase "Từ chối hồ sơ\\n(rejected)" as D2
  usecase "Tra cứu theo biển số /\\nthời gian / node" as E
  usecase "Xem thống kê:\\nKPI, xu hướng giờ, theo node" as F
  usecase "Hỏi trợ lý AI\\nbằng tiếng Việt" as G
  usecase "Xuất dữ liệu phục vụ\\nlập biên bản" as H
}

Officer --> A
Officer --> E
Officer --> F
Officer --> G
A ..> B : <<include>>
B ..> C : <<extend>>
B ..> D2 : <<extend>>
C ..> H : <<mở rộng>>
D2 ..> H : <<mở rộng>>
@enduml
"""

# ============================================================ 3. USE CASE chi tiết Operator
D["use-case-operator"] = """@startuml
skinparam backgroundColor #FFFFFF
skinparam shadowing false
skinparam actorStyle awesome
left to right direction

actor "Kỹ thuật viên\\n(Operator)" as Operator

rectangle "Vận hành hệ thống" {
  usecase "Theo dõi danh sách node\\n(online/offline, heartbeat)" as A
  usecase "Chọn node để hiệu chuẩn" as B
  usecase "Vẽ vạch dừng\\n(kéo 2 điểm)" as C
  usecase "Vẽ mũi tên hướng giám sát\\n(đường hai chiều)" as E
  usecase "Khoanh vùng đèn tín hiệu\\n(2 điểm)" as F
  usecase "Xác nhận lưu — hiệu lực\\ntức thì, không restart" as G
  usecase "Xem camera live (HLS)\\nđể kiểm tra góc nhìn" as H
  usecase "Yêu cầu snapshot\\nframe hiện tại" as I
  usecase "Khởi động lại pipeline\\n(xa) khi cần" as J
}

Operator --> A
Operator --> H
Operator --> J
A ..> B : <<include>>
B ..> C : <<include>>
B ..> F : <<include>>
C ..> E : <<extend>>
C ..> G : <<include>>
F ..> G : <<include>>
H ..> I : <<include>>
@enduml
"""

# ============================================================ 4. ACTIVITY — phát hiện vi phạm (nâng cấp)
D["activity-detection"] = """@startuml
skinparam backgroundColor #FFFFFF
skinparam shadowing false
skinparam activityShape octagon
start
:Nguồn video\\n(tệp MP4 / RTSP camera);
repeat
  :Đọc frame từ nguồn\\n(hỗ trợ drop-frame realtime);
  :Phân loại trạng thái đèn tín hiệu\\n(YOLO26n-cls + fusion HSV + vị trí bóng đèn\\ntrên vùng đèn đã hiệu chuẩn);
  if (Đèn ĐỎ ổn định\\n≥ 3 frame liên tiếp?) then (Có)
    :Đánh dấu tín hiệu RED ổn định;
    :YOLO26m phát hiện phương tiện\\n(car / bike / van-bus / truck);
    :ByteTrack gán track ID ổn định;
    if (Vạch dừng đã được\\nhiệu chuẩn?) then (Đã có vạch)
      :Xác định phía của tâm hộp\\ngiới hạn so với vạch dừng;
      if (Tâm xe cắt qua vạch\\ntheo đúng hướng giám sát?) then (Vi phạm!)
        :Tạo hồ sơ vi phạm\\n(event_id, track_id, trạng thái đèn,\\ntoạ độ cắt vạch, bbox);
        :Chụp ảnh toàn cảnh\\n+ crop biển số;
        :OCR biển số (fast-plate-ocr)\\n+ kiểm tra cấu trúc biển VN\\n(fallback: biển tốt nhất đã nhớ);
        :Ghi hồ sơ vào outbox SQLite;
      else (Không vi phạm)
      endif
    else (Chưa có vạch)
      :Chỉ detect + track,\\nkhông xét vi phạm;
    endif
  else (Đèn khác / chưa ổn định)
    :Bỏ qua frame\\n(chưa đủ điều kiện xét vi phạm);
  endif
repeat while (Còn frame?) is (Có)
-> Hết;
:Dừng phát lại / chờ luồng mới;
stop
@enduml
"""

# ============================================================ 5. ACTIVITY — hiệu chuẩn (nâng cấp)
D["activity-calibration"] = """@startuml
skinparam backgroundColor #FFFFFF
skinparam shadowing false
start
:Node biên khởi động;
:Node tự đăng ký lên trung tâm\\n(kèm heartbeat định kỳ);
:Operator mở trang Nodes trên web;
:Chọn node → tải ảnh snapshot\\n(qua trung tâm proxy xuống edge);
:Vẽ vạch dừng — kéo chuột 2 điểm;
:Chọn hướng giám sát\\n(any / p→n / n→p);
:POST vạch: web → trung tâm → edge\\n(kèm X-Edge-Token);
:Pipeline cập nhật vạch đang hoạt động\\n(áp dụng ngay, không restart);
:Vẽ mũi tên hướng\\nnếu đường hai chiều;
:POST hướng kèm vạch\\n(hoặc chọn từ danh sách);
:Vẽ vùng đèn — kéo chuột 2 điểm\\nkhoanh vùng tín hiệu;
:POST vùng đèn: web → trung tâm → edge;
:Pipeline cập nhật vùng đèn\\n(hiệu lực tức thời);
:Hệ thống sẵn sàng xét vi phạm;
stop
@enduml
"""

# ============================================================ 6. ACTIVITY — duyệt hồ sơ (MỚI)
D["activity-review"] = """@startuml
skinparam backgroundColor #FFFFFF
skinparam shadowing false
start
:Cán bộ mở trang Review\\n(danh sách hồ sơ pending);
:Chọn hồ sơ cần kiểm tra;
:Hiển thị chi tiết: ảnh toàn cảnh,\\ncrop biển số, trạng thái đèn,\\nđộ tin cậy, toạ độ;
if (Ảnh bằng chứng\\ntải được?) then (Có)
  :Kiểm tra bằng chứng trực quan;
else (Không)
  :Căn cứ metadata\\n(đèn, toạ độ, biển số);
endif
if (Hồ sơ hợp lệ\\nđủ căn cứ?) then (Duyệt)
  :PATCH trạng thái = approved;
else (Từ chối)
  :PATCH trạng thái = rejected;
endif
:Cập nhật CSDL PostgreSQL;
:Thống kê KPI, badge số pending\\nlàm mới tự động;
if (Còn hồ sơ chờ?) then (Có)
  :Chọn hồ sơ tiếp theo;
  detach
else (Không)
  :Kết thúc phiên duyệt;
  stop
endif
@enduml
"""

# ============================================================ 7. SEQUENCE — vi phạm e2e (nâng cấp)
D["sequence-violation"] = """@startuml
skinparam backgroundColor #FFFFFF
skinparam shadowing false
autonumber "<b>0."
title Chuỗi hoạt động: phát hiện vi phạm và phân phối hồ sơ end-to-end

box "Node biên (Python)" #EFF6FF
  participant "Camera / video" as Cam
  participant "Pipeline\\n(YOLO + ByteTrack)" as Pipe
  participant "Outbox (SQLite)" as Outbox
  participant "ViolationSender" as Sender
end box
box "Máy chủ trung tâm (Spring Boot)" #ECFDF5
  participant "ViolationController" as Ctrl
  participant "ViolationService" as Svc
  participant "MinioStorageService" as Minio
  database "PostgreSQL" as DB
end box
box "Web (Next.js)" #FEF3C7
  participant "Dashboard" as Web
end box

Cam -> Pipe : frame mới
Pipe -> Pipe : đèn ĐỎ ổn định\\n+ xe cắt vạch đúng hướng
Pipe -> Outbox : ghi hồ sơ + 2 ảnh JPEG
Outbox -> Sender : fetch_pending (lô 20)
Sender -> Ctrl : POST /api/violations/batch
Ctrl -> Svc : lưu batch (dedup event_id)
Svc -> DB : INSERT / bỏ qua trùng
DB --> Svc : OK
Svc --> Ctrl : kết quả lô
Ctrl --> Sender : 200 + số bản ghi mới
Sender -> Ctrl : POST /{eventId}/media (multipart)
Ctrl -> Minio : upload violations/{eventId}/*.jpg
Minio --> Ctrl : presigned URL (1 giờ)
Ctrl -> DB : cập nhật media_url
Sender -> Outbox : đánh dấu sent, xoá ảnh đĩa
Web -> Ctrl : GET /api/violations/page
Ctrl --> Web : danh sách + media_url
Web -> Ctrl : PATCH /{id}/status (duyệt/từ chối)
Ctrl -> DB : cập nhật status
@enduml
"""

# ============================================================ 8. SEQUENCE — hiệu chuẩn (nâng cấp)
D["sequence-calibration"] = """@startuml
skinparam backgroundColor #FFFFFF
skinparam shadowing false
autonumber "<b>0."
title Chuỗi hoạt động: hiệu chuẩn vạch dừng / vùng đèn / hướng giám sát từ xa

actor "Operator" as Op
box "Web (Next.js)" #FEF3C7
  participant "CalibrationEditor" as CalEd
end box
box "Máy chủ trung tâm" #ECFDF5
  participant "EdgeNodeController" as Cen
  participant "EdgeProxyService" as Proxy
end box
box "Node biên" #EFF6FF
  participant "FastAPI control-plane" as EdgeAPI
  participant "Pipeline (runtime)" as Pipe
end box

Op -> CalEd : mở trang Node, chọn camera
CalEd -> Cen : GET .../calibration/snapshot
Cen -> Proxy : getBytesFromEdge()
Proxy -> EdgeAPI : GET /api/calibration/snapshot
EdgeAPI --> Proxy : frame JPEG
Proxy --> CalEd : ảnh snapshot
Op -> CalEd : kéo chuột vẽ vạch dừng
CalEd -> Cen : POST .../calibration/stop-line\\n{x1,y1,x2,y2,direction}
Cen -> Proxy : postToEdge(/action/stop-line)
Proxy -> EdgeAPI : POST /action/stop-line\\n(kèm X-Edge-Token)
EdgeAPI -> Pipe : set_active_tripwire()
Pipe --> EdgeAPI : OK — hiệu lực ngay
EdgeAPI --> Proxy : 200
Proxy --> CalEd : thành công
Op -> CalEd : vẽ mũi tên hướng giám sát
CalEd -> Cen : POST lại vạch + direction
Cen -> Proxy : postToEdge (như trên)
Proxy -> EdgeAPI : POST /action/stop-line
EdgeAPI -> Pipe : set_active_tripwire (kèm hướng)
Pipe --> EdgeAPI : OK
Op -> CalEd : khoanh vùng đèn
CalEd -> Cen : POST .../calibration/light-roi {x,y,w,h}
Cen -> Proxy : postToEdge(/action/light-roi)
Proxy -> EdgeAPI : POST /action/light-roi
EdgeAPI -> Pipe : set_active_light_roi()
Pipe --> EdgeAPI : OK — hiệu lực ngay
EdgeAPI --> Proxy : 200
Proxy --> CalEd : thành công
@enduml
"""

# ============================================================ 9. SEQUENCE — AI Text-to-SQL (MỚI)
D["sequence-ai"] = """@startuml
skinparam backgroundColor #FFFFFF
skinparam shadowing false
autonumber "<b>0."
title Chuỗi hoạt động: trợ lý AI Text-to-SQL hỏi dữ liệu tiếng Việt

actor "Cán bộ (Officer)" as Op
box "Web (Next.js)" #FEF3C7
  participant "AISidebar" as AI
  participant "route /api/ai-query\\n(proxy mỏng)" as Route
end box
box "Máy chủ trung tâm" #ECFDF5
  participant "AiQueryController" as Ctrl
  participant "AiQueryService" as Svc
  participant "Gemini API\\n(gemini-2.5-flash)" as Gem
  database "PostgreSQL" as DB
end box

Op -> AI : gõ câu hỏi tiếng Việt\\n("Hôm nay có bao nhiêu vi phạm?")
AI -> Route : POST /api/ai-query {question}
Route -> Ctrl : chuyển tiếp POST /api/ai/query
Ctrl -> Ctrl : kiểm tra độ dài ≤ 500 ký tự
Ctrl -> Svc : process(question)
Svc -> Gem : prompt: schema CSDL + câu hỏi\\n(yêu cầu chỉ sinh SELECT)
Gem --> Svc : chuỗi SQL SELECT
Svc -> Svc : kiểm tra an toàn\\n(chặn DROP/INSERT/UPDATE/ALTER…\\nchỉ cho SELECT/WITH + LIMIT 50)
alt SQL an toàn
  Svc -> DB : JdbcTemplate.queryForList(sql)
  DB --> Svc : rows + columns
  Svc -> Svc : determineChartType()\\n(table / bar)
else SQL nguy hiểm
  Svc --> Ctrl : lỗi — từ chối thực thi
end
Svc --> Ctrl : AiQueryResponse\\n{sql, columns, rows, chartType}
Ctrl --> Route : JSON
Route --> AI : JSON
AI --> Op : hiển thị SQL + kết quả\\n(bảng hoặc biểu đồ cột)
@enduml
"""

# ============================================================ 10. SEQUENCE — offline recovery (MỚI)
D["sequence-outbox"] = """@startuml
skinparam backgroundColor #FFFFFF
skinparam shadowing false
autonumber "<b>0."
title Chuỗi hoạt động: phân phối bền vững — mất mạng, khôi phục, chống trùng lặp

box "Node biên" #EFF6FF
  participant "Pipeline" as Pipe
  participant "Outbox (SQLite)" as Outbox
  participant "ViolationSender" as Sender
end box
box "Trung tâm" #ECFDF5
  participant "Central API" as Central
end box

== Chế độ bình thường ==
Pipe -> Outbox : ghi hồ sơ vi phạm (pending)
Sender -> Outbox : fetch_pending
Sender -> Central : POST batch + media
Central --> Sender : 200
Sender -> Outbox : mark sent + dọn ảnh

== Mất mạng / trung tâm ngừng ==
Pipe -> Outbox : vẫn ghi hồ sơ (pending)
Sender -> Central : POST batch
Central --x Sender : lỗi kết nối (retry)
note right of Sender : vòng lặp thử lại\\ntheo chu kỳ flush_interval
Sender -> Outbox : giữ nguyên pending

== Khôi phục kết nối ==
Sender -> Outbox : fetch_pending (toàn bộ)
Sender -> Central : POST batch
Central -> Central : dedup theo event_id\\n(bản trùng bị bỏ qua)
Central --> Sender : 200 + số bản ghi mới
Sender -> Outbox : mark sent

== Edge khởi động lại ==
note over Outbox : event_id chứa run_id (uuid)\\nnên lần chạy mới không\\nđụng id cũ dù frame lặp
@enduml
"""

# ============================================================ 11. CLASS (nâng cấp — thêm trung tâm + web)
D["class-diagram"] = """@startuml
skinparam backgroundColor #FFFFFF
skinparam shadowing false
skinparam classAttributeIconSize 0
skinparam maxMessageSize 250

package "Node biên (Python)" #EFF6FF {
  class RedLightViolationPipeline {
    -detector: YoloDetector
    -tracker: ByteTracker
    -lightClassifier: LightClassifier
    -violationDetector: ViolationDetector
    -outbox: ViolationOutbox
    +run()
  }
  class YoloDetector {
    -model: YOLO (yolo26m_vehicle)
    -confidence: float
    +detect(frame) : List[Detection]
  }
  class ByteTracker {
    +update_with_detections(dets) : Tracks
  }
  class YoloTrafficLightClassifier {
    +classify(roi_crop) : LightObservation
  }
  class ViolationDetector {
    -track_states : Dict[int, TrackCrossingState]
    +update(tracks, light_state) : List[Violation]
  }
  class TripwireConfig {
    x1, y1, x2, y2 : float
    direction : str
  }
  class PlateDetector
  class PlateAssociator {
    -plate_cache : Dict[int, PlateReading]
    +associate(track, boxes) : PlateObservation
  }
  class FastPlateOCR {
    +recognize_bbox(crop) : PlateObservation
  }
  class VnPlateValidator {
    +validate_and_format(text) : str | None
    +repair_plate_text(raw) : str | None
  }
  class ViolationOutbox {
    +append(violation, image)
    +fetch_pending() : List[dict]
    +mark_sent(event_ids)
  }
  class ViolationSender {
    -batch_size : int
    +run_forever()
    -push_batch()
  }
  class FastAPIServer {
    +POST /action/stop-line
    +POST /action/light-roi
    +GET /api/calibration/snapshot
    +GET /api/cameras
  }
}

package "Trung tâm (Java Spring Boot)" #ECFDF5 {
  class ViolationController {
    +POST /api/violations/batch
    +GET /api/violations/page
    +PATCH /{id}/status
    +GET /api/stats
  }
  class ViolationService {
    +saveBatch(items) : BatchResult
    +getPage(filters) : Page
    +getStats() : Stats
  }
  class EdgeNodeService {
    +register(node) : EdgeNode
    +heartbeat(nodeId)
  }
  class EdgeProxyService {
    -httpClient : HttpClient (HTTP/1.1)
    +postToEdge(nodeId, path, body)
    +getFromEdge(nodeId, path)
  }
  class AiQueryService {
    -geminiKey : String
    +process(question) : AiQueryResponse
    -validateSql(sql) : boolean
  }
  class MinioStorageService {
    +upload(eventId, file) : String
    +presign(key) : String
  }
  class Violation <<JPA Entity>> {
    eventId : String (unique)
    nodeId : String
    plateText : String
    lightState : String
    status : String
    mediaUrl : String
  }
  class EdgeNode <<JPA Entity>> {
    nodeId : String (unique)
    ipAddress : String
    status : String
    lastPing : LocalDateTime
  }
}

package "Web (Next.js)" #FEF3C7 {
  class CalibrationEditor {
    +sideOfLine(p, line) : int
    +directionFromArrow(a, b) : str
    +drawStopLine / drawLightRoi
  }
  class AISidebar {
    +sendQuestion(q)
    +renderTable / renderBarChart
  }
  class apiClient <<lib/api.ts>> {
    +getViolationsPage()
    +setStopLine / setLightRoi
  }
}

RedLightViolationPipeline *-- YoloDetector
RedLightViolationPipeline *-- ByteTracker
RedLightViolationPipeline *-- ViolationDetector
RedLightViolationPipeline o-- YoloTrafficLightClassifier
RedLightViolationPipeline o-- PlateDetector
RedLightViolationPipeline o-- PlateAssociator
RedLightViolationPipeline o-- ViolationOutbox
ViolationDetector --> TripwireConfig : đọc mỗi frame\\n(get_active_tripwire)
PlateAssociator --> FastPlateOCR : OCR từng crop
FastPlateOCR --> VnPlateValidator : kiểm tra cấu trúc biển
ViolationOutbox <.. ViolationSender : fetch_pending
ViolationSender --> ViolationController : HTTP batch + media
FastAPIServer ..> RedLightViolationPipeline : set_active_tripwire /\\nset_active_light_roi (live)
EdgeProxyService --> FastAPIServer : HTTP/1.1 + X-Edge-Token
ViolationController --> ViolationService
ViolationService --> Violation
ViolationService --> MinioStorageService
EdgeNodeService --> EdgeNode
AiQueryService ..> Violation : SELECT qua JdbcTemplate
CalibrationEditor --> apiClient
apiClient --> EdgeProxyService : qua /api rewrite
AISidebar --> apiClient
@enduml
"""

# ============================================================ 12. ARCHITECTURE (nâng cấp)
D["architecture"] = """@startuml
skinparam backgroundColor #FFFFFF
skinparam shadowing false
skinparam componentStyle rectangle
title Kiến trúc triển khai: Edge — Central — Web

node "Camera / RTSP / tệp video" as Cam

package "NODE BIÊN (Python — tại mỗi camera)" #EFF6FF {
  component "Nguồn frame\\n(file/RTSP, drop-frame realtime)" as VS
  component "Pipeline: YOLO26m detect\\n+ ByteTrack + đèn\\n(YOLO26n-cls + fusion HSV)" as PL
  component "Vạch dừng + hướng\\n(violation_logic)" as TW
  component "Biển số: detect + OCR\\n+ validator biển VN" as OCR
  component "Outbox bền vững (SQLite)" as OB
  component "ViolationSender (batch)" as SND
  component "FastAPI control-plane :8080" as API
  component "FFmpeg → HLS stream" as HLS
}

database "PostgreSQL 16\\n(metadata vi phạm, node)" as PG
database "MinIO (S3)\\n(ảnh bằng chứng)" as MINIO

package "MÁY CHỦ TRUNG TÂM (Spring Boot :8000)" #ECFDF5 {
  component "REST Controllers\\n(Violation / Media / EdgeNode / AI)" as CTRL
  component "Service layer\\n(Violation / EdgeNode / EdgeProxy / AI)" as SVC
  component "EdgeProxyService\\n(HTTP/1.1 + token)" as PRX
  component "AiQueryService\\n(Gemini Text-to-SQL)" as AIQ
}

package "WEB DASHBOARD (Next.js :3000)" #FEF3C7 {
  component "Trang: dashboard, violations,\\nreview, nodes, cameras" as PAGES
  component "CalibrationEditor\\n+ VideoPlayer (hls.js)" as CAL
  component "AISidebar (chat Text-to-SQL)" as AICHAT
}

Cam --> VS
VS --> PL
PL --> TW
TW --> OCR
TW --> OB
OB --> SND
SND --> CTRL : batch vi phạm\\n+ upload media
API <.. PRX : proxy hiệu chuẩn\\n(stop-line / light-roi)
HLS --> CAL : HLS live\\n(qua /edge-api rewrite)
PRX <.. CAL : POST vạch / vùng / hướng
PAGES --> CTRL : /api rewrite\\n(đọc dữ liệu)
AICHAT --> AIQ : /api/ai-query →\\n/api/ai/query
SVC --> PG : JPA
SVC --> MINIO : SDK
@enduml
"""

# ============================================================ 12b. ERD (nâng cấp tiếng Việt)
D["erd"] = """@startuml
skinparam backgroundColor #FFFFFF
skinparam shadowing false
skinparam linetype ortho
hide circle
skinparam entity {
  BackgroundColor #ECFDF5
  BorderColor #059669
}

entity "edge_nodes" as edge {
  * id : UUID <<PK>>
  --
  * node_id : VARCHAR(64) <<UNIQUE>>
  * name : VARCHAR
  ip_address : VARCHAR
  * status : VARCHAR(16) — online/offline/maintenance
  last_ping : TIMESTAMP
  settings_json : TEXT
  created_at / updated_at : TIMESTAMP
}
entity "violations" as vio {
  * id : BIGSERIAL <<PK>>
  --
  * event_id : VARCHAR <<UNIQUE>> — rlv-{run}-{frame}-{track}
  * node_id : VARCHAR(64) <<FK>>
  track_id / frame_index : INT
  timestamp_ms : DOUBLE
  crossing_point_x / crossing_point_y : DOUBLE
  previous_point_x / previous_point_y : DOUBLE
  bbox_x1 / bbox_y1 / bbox_x2 / bbox_y2 : DOUBLE
  light_state : VARCHAR(8) — red/yellow/green/unknown
  light_confidence : DOUBLE
  previous_side / current_side : INT
  plate_text / plate_confidence : VARCHAR / DOUBLE
  * status : VARCHAR(16) — pending/approved/rejected
  media_url : VARCHAR(1024)
  metadata : TEXT
  created_at / updated_at : TIMESTAMP
}
edge ||..o{ vio : "một node ghi nhận\\nnhiều vi phạm"
@enduml
"""

# ============================================================ 13. DEPLOYMENT (MỚI)
D["deployment"] = """@startuml
skinparam backgroundColor #FFFFFF
skinparam shadowing false
skinparam componentStyle rectangle
title Sơ đồ triển khai Docker Compose (5 service)

node "Máy trạm GPU (RTX 2060, CUDA 12.4)" {
  node "Docker Compose — mạng rlvd-net" {
    artifact "edge-pipeline\\n(python:3.11-slim + torch cu124, GPU)" as EDGE
    artifact "central-server\\n(temurin:17-jre-alpine)" as CENTRAL
    artifact "web-dashboard\\n(node:20-alpine, Next.js standalone)" as WEB
    artifact "postgres:16" as PG
    artifact "minio" as MINIO
  }
}

cloud "Telegram / trình duyệt người dùng" as USERS
node "Camera RTSP / tệp video" as CAM

CAM ..> EDGE : rtsp:// hoặc volume ./data
USERS --> WEB : HTTP :3000
WEB --> CENTRAL : rewrite /api → :8000
WEB --> EDGE : rewrite /edge-api → :8080\\n(HLS, snapshot)
EDGE --> CENTRAL : batch + media → :8000\\n(đăng ký, heartbeat)
CENTRAL --> PG : JDBC :5432
CENTRAL --> MINIO : S3 SDK :9000
EDGE ..> CENTRAL : GPU: nvidia runtime\\n(reserve device)
@enduml
"""

# ============================================================ 14. STATE — trạng thái hồ sơ (MỚI)
D["state-violation"] = """@startuml
skinparam backgroundColor #FFFFFF
skinparam shadowing false
title Trạng thái vòng đời hồ sơ vi phạm

[*] --> Pending : node biên phát hiện vi phạm,\\nđẩy batch lên trung tâm

state "Pending\\n(chờ duyệt)" as Pending
state "Approved\\n(đã phê duyệt)" as Approved
state "Rejected\\n(đã từ chối)" as Rejected

Pending : entry / gán status = 'pending'
Pending : hồ sơ đủ metadata + ảnh

Pending --> Approved : cán bộ duyệt\\n(PATCH /status)
Pending --> Rejected : cán bộ từ chối
Approved --> [*]
Rejected --> [*]

note right of Pending
  Hồ sơ vào CSDL với status mặc định
  'pending'; KPI và badge đếm theo
  status; duyệt là human-in-the-loop.
end note
@enduml
"""

# ============================================================ 15. STATE — trạng thái đèn (MỚI)
D["state-light"] = """@startuml
skinparam backgroundColor #FFFFFF
skinparam shadowing false
title Trạng thái đèn tín hiệu có ổn định (debounce + hysteresis)

[*] --> Unknown

state "Unknown\\n(không xác định)" as Unknown
state "Đỏ ổn định\\n(RED — được xét vi phạm)" as RedStable
state "Xanh ổn định\\n(GREEN)" as GreenStable
state "Vàng ổn định\\n(YELLOW)" as YellowStable

Unknown : chưa có vùng đèn\\nhoặc conf < 0.55

Unknown --> RedStable : quan sát RED liên tiếp\\n≥ 3 frame, conf ≥ 0.55
Unknown --> GreenStable : quan sát GREEN\\nliên tiếp ≥ 3 frame
Unknown --> YellowStable : quan sát YELLOW\\nliên tiếp ≥ 3 frame

RedStable --> GreenStable : chuyển trạng thái cần\\n≥ 7 frame liên tiếp
GreenStable --> RedStable : ≥ 7 frame liên tiếp
RedStable --> YellowStable : ≥ 7 frame
YellowStable --> GreenStable : ≥ 7 frame
GreenStable --> YellowStable : ≥ 7 frame
YellowStable --> RedStable : ≥ 7 frame
RedStable --> Unknown : mất tín hiệu kéo dài
GreenStable --> Unknown : mất tín hiệu kéo dài

note right of RedStable
  Chỉ trạng thái RED ổn định
  mới mở nhánh xét vi phạm
  (tripwire crossing).
end note
@enduml
"""

# ============================================================ 16. COMPONENT (MỚI)
D["component"] = """@startuml
skinparam backgroundColor #FFFFFF
skinparam shadowing false
skinparam componentStyle rectangle
title Sơ đồ thành phần tổng thể (component diagram)

package "Node biên" #EFF6FF {
  [Pipeline thị giác\\n(YOLO/ByteTrack/đèn/vạch)] as PIPE
  [Outbox SQLite] as OB
  [Control-plane FastAPI] as CPA
  [HLS streamer] as HLS
}

package "Trung tâm" #ECFDF5 {
  [Violation API] as VAPI
  [Media API] as MAPI
  [EdgeNode API] as NAPI
  [AI Query API] as AAPI
  [EdgeProxy] as PRX
}

package "Web" #FEF3C7 {
  [Trang nghiệp vụ\\n(dashboard/violations/review)] as PAGES
  [CalibrationEditor] as CE
  [AISidebar] as ASB
}

database "PostgreSQL" as PG
database "MinIO" as MIN

PIPE --> OB
OB --> VAPI : batch JSON
PIPE ..> MAPI : upload ảnh
VAPI --> PG
MAPI --> MIN
NAPI --> PG
NAPI --> PRX
PRX --> CPA
CPA --> PIPE : set_active_tripwire / light_roi
CE --> NAPI
PAGES --> VAPI
ASB --> AAPI
AAPI --> PG : SELECT (giới hạn)
CPA --> HLS
PIPE --> HLS : ffmpeg
@enduml
"""

# ============================================================ 17. DATA FLOW (MỚI)
D["data-flow"] = """@startuml
skinparam backgroundColor #FFFFFF
skinparam shadowing false
title Luồng dữ liệu hồ sơ vi phạm qua các tầng

cloud "Camera / RTSP" as Cam
cloud "Người duyệt\\n(xem ảnh bằng chứng)" as Viewer

package "Node biên" #EFF6FF {
  rectangle "frame video" as F
  rectangle "hồ sơ JSON + 2 ảnh JPEG" as V
  database "Outbox SQLite\\n(pending → sent)" as OB
}

package "Trung tâm" #ECFDF5 {
  database "PostgreSQL\\n(bản ghi violations)" as PG
  database "MinIO\\n(ảnh bằng chứng)" as MIN
}

Cam --> F
F --> V : pipeline\\ndetect/track/OCR
V --> OB
OB --> PG : POST batch JSON\\n(dedup event_id)
OB --> MIN : upload media\\n(multipart)
PG --> Viewer : GET page + media_url
MIN --> Viewer : stream blob / presigned
@enduml
"""


# ---------------------------------------------------------------- render machinery
def encode_plantuml(text: str) -> str:
    alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_"
    data = zlib.compress(text.encode("utf-8"), 9)[2:-4]
    out = []
    for i in range(0, len(data), 3):
        b = data[i:i + 3]
        b = b + b"\x00" * (3 - len(b))
        n = int.from_bytes(b, "big")
        chars = [alphabet[(n >> 18) & 63], alphabet[(n >> 12) & 63],
                 alphabet[(n >> 6) & 63], alphabet[n & 63]]
        out.extend(chars[:len(data[i:i + 3]) + 1])
    return "".join(out)


def fetch_svg(puml: str) -> str:
    url = "https://www.plantuml.com/plantuml/svg/" + encode_plantuml(puml)
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            r = urllib.request.urlopen(req, timeout=60)
            svg = r.read().decode("utf-8", errors="replace")
            if "Syntax Error" in svg:
                m = re.search(r"<text[^>]*>([^<]*[Ee]rror[^<]*)</text>", svg)
                raise RuntimeError("SYNTAX: " + (m.group(1) if m else svg[:200]))
            return svg
        except urllib.error.HTTPError as e:
            if e.code == 509:
                time.sleep(4)
                continue
            body = e.read().decode("utf-8", errors="replace")[:300]
            raise RuntimeError(f"HTTP {e.code}: {body}") from e
    raise RuntimeError("509 quá 4 lần")


def fix_svg_font(svg: str) -> str:
    """Đổi font sans-serif mặc định sang Noto Sans (đủ dấu tiếng Việt)
    và phóng kích thước 1.6x cho chữ nét rõ."""
    svg = svg.replace('font-family="sans-serif"',
                      'font-family="Noto Sans, DejaVu Sans, sans-serif"')
    # phóng 1.6x cả style lẫn width/height
    m = re.search(r'style="width:(\d+)px;height:(\d+)px', svg)
    if m:
        w, h = int(m.group(1)), int(m.group(2))
        svg = svg.replace(m.group(0),
                          f'style="width:{int(w*1.6)}px;height:{int(h*1.6)}px', 1)
    m2 = re.search(r'width="(\d+)px" height="(\d+)px"', svg)
    if m2:
        w, h = int(m2.group(1)), int(m2.group(2))
        svg = svg.replace(m2.group(0), f'width="{int(w*1.6)}px" height="{int(h*1.6)}px"', 1)
    return svg


def render(name: str, puml: str, i: int, total: int) -> bool:
    try:
        svg = fetch_svg(puml)
        svg = fix_svg_font(svg)
        import cairosvg
        png = cairosvg.svg2png(bytestring=svg.encode("utf-8"),
                               output_width=None, background_color="white")
        (OUT / f"{name}.png").write_bytes(png)
        print(f"[{i}/{total}] {name}.png OK ({len(png)//1024} KB)")
        return True
    except Exception as e:
        print(f"[{i}/{total}] {name} FAIL: {e}", file=sys.stderr)
        return False


if __name__ == "__main__":
    only = sys.argv[1:] or None
    names = [n for n in D if (only is None or n in only)]
    total = len(names)
    ok = 0
    for i, name in enumerate(names, 1):
        if render(name, D[name], i, total):
            ok += 1
        time.sleep(1.2)
    print(f"\nHoàn thành: {ok}/{total}")
    sys.exit(0 if ok == total else 1)
