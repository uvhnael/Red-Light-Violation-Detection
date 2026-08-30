"""Render các diagram PlantUML cho báo cáo RLVD qua plantuml.com.

Sinh PNG vào docs/report-assets/<name>.png. Trả về 0 nếu tất cả render thành công.
"""
import zlib
import urllib.request
import sys
import time
from pathlib import Path

OUT = Path("/home/uvhnael/projects/Red-Light-Violation-Detection/docs/report-assets")
OUT.mkdir(parents=True, exist_ok=True)


def encode_plantuml(text: str) -> str:
    """Mã hoá PlantUML text sang định dạng hex-ish của plantuml.com."""
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


def render(name: str, puml: str, idx: int, total: int) -> bool:
    url = "https://www.plantuml.com/plantuml/png/" + encode_plantuml(puml)
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            r = urllib.request.urlopen(req, timeout=60)
            content = r.read()
            if r.status == 200 and content[:8].startswith(b"\x89PNG"):
                (OUT / f"{name}.png").write_bytes(content)
                print(f"[{idx}/{total}] {name}.png OK ({len(content)} bytes)")
                return True
        except Exception as e:
            print(f"  {name} attempt {attempt+1} fail: {e}", file=sys.stderr)
            time.sleep(3)
    print(f"[{idx}/{total}] {name} FAILED")
    return False


DIAGRAMS: dict = {}

# ---------------------------------------------------------------- Use Case
DIAGRAMS["use-case"] = """@startuml
skinparam backgroundColor #FFFFFF
skinparam shadowing false
skinparam actorStyle awesome
left to right direction
skinparam packageStyle rectangle

actor "Van phong xu ly vi pham\\n(Can bo CSGT)" as Officer
actor "Ky thuat vien\\n(Operator)" as Operator
actor "Edge Node" as Edge
actor "Central Server" as Central

rectangle "He thong Phat hien Vuot Den Do (RLVD)" {
  usecase "UC1. Dang ky va ghi nhan camera" as UC1
  usecase "UC2. Kẻ vach dung, vung den,\\nhuong giam sat (Calibration)" as UC2
  usecase "UC3. Phat hien xe vuot den do\\n(detect + track + tripwire)" as UC3
  usecase "UC4. Doc bien so (OCR)" as UC4
  usecase "UC5. Gui ho so vi pham\\n(outbox + batch)" as UC5
  usecase "UC6. Xem camera live (HLS)" as UC6
  usecase "UC7. Duyet / tu choi ho so vi pham" as UC7
  usecase "UC8. Tra cuu vi pham, thong ke" as UC8
  usecase "UC9. Hoi du lieu bang AI\\n(Text-to-SQL)" as UC9
  usecase "UC10. Quan ly edge node\\n(heartbeat, status)" as UC10
}

Operator --> UC1
Operator --> UC2
Operator --> UC6
Operator --> UC10
Officer --> UC7
Officer --> UC8
Officer --> UC9

Edge ..> UC3 : <<thuc hien>>
Edge ..> UC4 : <<thuc hien>>
Edge ..> UC5 : <<thuc hien>>
Central ..> UC5 : <<nhan>>
UC2 ..> UC3 : <<kich hoat>>
UC3 ..> UC4 : <<include>>
UC3 ..> UC5 : <<include>>
UC5 ..> UC7 : <<phan phoi>>
@enduml
"""

# ---------------------------------------------------------------- Activity: violation detection
DIAGRAMS["activity-detection"] = """@startuml
skinparam backgroundColor #FFFFFF
skinparam shadowing false
start
:Camera/video stream;
repeat
  :Doc frame tu nguon (file/RTSP);
  :Phan loai trang thai den giao thong\\n(YOLO26n-cls + fusion HSV\\ntrong vung den da calibration);
  if (Den DO on dinh >= 3 frame?) then (Co)
    :Dan nhan trang thai den = RED;
    :YOLO26m phat hien phuong tien\\n(car/bike/van-bus/truck);
    :ByteTrack gan track ID;
    if (Operator da ke vach dung?) then (Da ke)
      :Do qua vi tri tam bbox (center)\\nso voi vach dung;
      if (Tam xe cat qua vach dung\\nva dung huong giam sat?) then (Vi pham!)
        :Tao ho so vi pham\\n(event_id, track_id, light_state);
        :Chup anh toan canh + crop bien so;
        :OCR bien so (fast-plate-ocr)\\n+ validate chuan bien VN;
        :Ghi vao SQLite outbox;
      else (Khong vi pham)
      endif
    else (Chua ke vach)
      :Khong xet vi pham\\n(van detect + track binh thuong);
    endif
  else (Den khac / khong on dinh)
    :Bo qua frame nay\\n(trang thai den chua on dinh);
  endif
repeat while (Con frame?) is (Co)
-> Khong;
:Ket thuc phien chay;
stop
@enduml
"""

# ---------------------------------------------------------------- Activity: calibration
DIAGRAMS["activity-calibration"] = """@startuml
skinparam backgroundColor #FFFFFF
skinparam shadowing false
start
:Edge node khoi dong, tu dang ky len Central;
:Operator mo trang Node tren web dashboard;
:Load anh snapshot tu edge\\n(qua Central proxy);
:Chon che do ve: vach dung;
:Keo chuot 2 diem tao duong thang vach dung;
:Chon huong giam sat (any /\\npositive_to_negative / negative_to_positive);
:POST stop-line qua Central -> EdgeProxy;
:Edge cap nhat active tripwire (live,\\nkhong can restart);
:Chon che do ve: vung den;
:Keo chuot 2 diem khoanh vung den giao thong;
:POST light-roi qua Central -> EdgeProxy;
:Edge cap nhat active light ROI (live);
:He thong san sang xet vi pham;
stop
@enduml
"""

# ---------------------------------------------------------------- Sequence: violation e2e
DIAGRAMS["sequence-violation"] = """@startuml
skinparam backgroundColor #FFFFFF
skinparam shadowing false
autonumber "<b>0."
title Chuoi hoat dong: Phat hien va gui ho so vi pham

box "Edge Node (Python)" #EFF6FF
  participant "Camera / Video" as Cam
  participant "Pipeline\\n(YOLO + ByteTrack)" as Pipe
  participant "Outbox (SQLite)" as Outbox
  participant "ViolationSender" as Sender
end box
box "Central Server (Spring Boot)" #ECFDF5
  participant "ViolationController" as Ctrl
  participant "ViolationService" as Svc
  participant "MinioStorageService" as Minio
  participant "PostgreSQL" as DB
end box
box "Web (Next.js)" #FEF3C7
  participant "Dashboard" as Web
end box

Cam -> Pipe : frame moi
Pipe -> Pipe : den RED on dinh + xe cat vach
Pipe -> Outbox : INSERT vi pham + anh JPEG
Outbox -> Sender : fetch_pending (batch)
Sender -> Ctrl : POST /api/violations/batch
Ctrl -> Svc : luu batch (dedup event_id)
Svc -> DB : INSERT / skip trung
DB --> Svc : OK
Svc --> Ctrl : ket qua batch
Ctrl --> Sender : 200 + so ban ghi moi
Sender -> Ctrl : POST /{eventId}/media (multipart)
Ctrl -> Minio : upload violations/{eventId}/xxx.jpg
Minio --> Ctrl : URL presigned (1h)
Ctrl -> DB : cap nhat media_url
Sender -> Outbox : danh dau sent + xoa anh
Web -> Ctrl : GET /api/violations/page
Ctrl --> Web : danh sach vi pham + media_url
Web -> Ctrl : PATCH /{id}/status (duyet/tu choi)
Ctrl -> DB : cap nhat status
@enduml
"""

# ---------------------------------------------------------------- Sequence: calibration
DIAGRAMS["sequence-calibration"] = """@startuml
skinparam backgroundColor #FFFFFF
skinparam shadowing false
autonumber "<b>0."
title Chuoi hoat dong: Calibration vach dung / vung den tu xa

participant "Operator" as Op
box "Web (Next.js)" #FEF3C7
  participant "CalibrationEditor" as CalEd
end box
box "Central Server" #ECFDF5
  participant "EdgeNodeController" as Cen
  participant "EdgeProxyService" as Proxy
end box
box "Edge Node" #EFF6FF
  participant "FastAPI Control-Plane" as EdgeAPI
  participant "Pipeline (runtime)" as Pipe
end box

Op -> CalEd : mo trang Node, chon camera
CalEd -> Cen : GET /api/v1/edge-nodes/{nodeId}/calibration/snapshot
Cen -> Proxy : getBytesFromEdge(snapshot)
Proxy -> EdgeAPI : GET /api/calibration/snapshot
EdgeAPI --> Proxy : JPEG frame
Proxy --> CalEd : anh snapshot (cache-busted)
Op -> CalEd : keo chuot ve vach dung + chon huong
CalEd -> Cen : POST .../calibration/stop-line\\n{x1,y1,x2,y2,direction}
Cen -> Proxy : postToEdge(/action/stop-line,\\nX-Edge-Token)
Proxy -> EdgeAPI : POST /action/stop-line
EdgeAPI -> Pipe : set_active_tripwire()
Pipe --> EdgeAPI : OK (ap dung ngay, khong restart)
EdgeAPI --> Proxy : 200
Proxy --> CalEd : thanh cong
Op -> CalEd : keo chuot khoanh vung den
CalEd -> Cen : POST .../calibration/light-roi {x,y,w,h}
Cen -> Proxy : postToEdge(/action/light-roi)
Proxy -> EdgeAPI : POST /action/light-roi
EdgeAPI -> Pipe : set_active_light_roi()
Pipe --> EdgeAPI : OK (ap dung ngay)
EdgeAPI --> Proxy : 200
Proxy --> CalEd : thanh cong
@enduml
"""

# ---------------------------------------------------------------- Class diagram
DIAGRAMS["class-diagram"] = """@startuml
skinparam backgroundColor #FFFFFF
skinparam shadowing false
skinparam classAttributeIconSize 0
skinparam maxMessageSize 250

class RedLightViolationPipeline {
  -settings: PipelineSettings
  -detector: YoloDetector
  -tracker: ByteTracker
  -lightClassifier: LightClassifier
  -violationDetector: ViolationDetector
  -outbox: ViolationOutbox | None
  +run()
  -process_frame(frame, idx)
}
class YoloDetector {
  -model: YOLO
  -confidence: float
  -img_size: int
  -device: str
  +detect(frame) -> list[Detection]
}
class ByteTracker {
  +update_with_detections(dets) -> Tracks
}
class YoloTrafficLightClassifier {
  -model: YOLO26n-cls
  +classify(roi_crop) -> LightObservation
}
class ViolationDetector {
  -tripwire: TripwireConfig | None
  -track_states: dict[int, _TrackCrossingState]
  +update(tracks, light_state) -> list[Violation]
}
class TripwireConfig {
  x1: float
  y1: float
  x2: float
  y2: float
  direction: str
}
class PlateDetector {
  +detect(frame) -> list[Detection]
}
class PlateAssociator {
  -plate_cache: dict[int, PlateReading]
  +associate(track, plate_boxes) -> PlateObservation
}
class FastPlateOCR {
  +recognize_bbox(crop) -> PlateObservation | None
}
class ViolationOutbox {
  -db_path: Path
  +append(violation, image)
  +fetch_pending() -> list[json]
  +fetch_media_pending() -> list[bytes]
  +mark_sent(event_ids)
}
class ViolationSender {
  -batch_size: int
  -flush_interval: float
  +run_forever()
  -push_batch()
}
class FastAPIServer {
  +GET /health
  +GET /api/cameras
  +GET /api/calibration
  +POST /action/stop-line
  +POST /action/light-roi
}

RedLightViolationPipeline *-- YoloDetector
RedLightViolationPipeline *-- ByteTracker
RedLightViolationPipeline *-- ViolationDetector
RedLightViolationPipeline o-- YoloTrafficLightClassifier
RedLightViolationPipeline o-- PlateDetector
RedLightViolationPipeline o-- PlateAssociator
RedLightViolationPipeline o-- ViolationOutbox : outbox_enabled
ViolationDetector --> TripwireConfig : doc moi frame\\n(get_active_tripwire)
PlateAssociator --> FastPlateOCR : OCR tung crop
ViolationOutbox <.. ViolationSender : fetch_pending
ViolationSender --> FastAPIServer : HTTP batch POST
RedLightViolationPipeline ..> FastAPIServer : control-plane\\n(set_active_tripwire)
@enduml
"""

# ---------------------------------------------------------------- System architecture
DIAGRAMS["architecture"] = """@startuml
skinparam backgroundColor #FFFFFF
skinparam shadowing false
skinparam componentStyle rectangle

title Kien truc tri khai: Edge - Central - Web

node "Camera / Video RTSP" as Cam

package "EDGE NODE (Python, tai moi camera)" #EFF6FF {
  component "Video Source\\n(file/RTSP + realtime drop-frame)" as VS
  component "Pipeline: YOLO26m detect\\n+ ByteTrack + den giao thong\\n(YOLO26n-cls + fusion HSV)" as PL
  component "Tripwire + Direction\\n(violation_logic)" as TW
  component "Plate detect + OCR\\n(fast-plate-ocr + validator VN)" as OCR
  component "Durable Outbox (SQLite)" as OB
  component "ViolationSender (batch)" as SND
  component "FastAPI Control-Plane :8080" as API
  component "FFmpeg HLS Stream" as HLS
}

database "PostgreSQL 16\\n(metadata vi pham, node)" as PG
database "MinIO (S3)\\n(anh bang chung)" as MINIO
package "CENTRAL SERVER (Spring Boot 3 :8000)" #ECFDF5 {
  component "REST Controller\\n(Violation/Media/EdgeNode/AI)" as CTRL
  component "Service Layer\\n(Violation/EdgeNode/EdgeProxy/AI)" as SVC
  component "EdgeProxyService\\n(HTTP_1_1 + X-Edge-Token)" as PRX
  component "AiQueryService\\n(Gemini Text-to-SQL)" as AIQ
}

package "WEB DASHBOARD (Next.js :3000)" #FEF3C7 {
  component "App Router pages\\n(dashboard/violations/review/nodes/cameras)" as PAGES
  component "CalibrationEditor + VideoPlayer (hls.js)" as CAL
  component "AISidebar (Text-to-SQL chat)" as AICHAT
}

Cam --> VS
VS --> PL
PL --> TW
TW --> OCR
TW --> OB
OB --> SND
SND --> CTRL : batch vi pham\\n+ media upload
API <.. PRX : proxy calibration
HLS --> CAL : HLS stream\\n(qua /edge-api rewrite)
PRX <.. CAL : POST vach/vung/huong
CAL --> PAGES
PAGES --> CTRL : /api/* rewrite
AICHAT --> AIQ : /api/ai-query ->\\n/api/ai/query
SVC --> PG
SVC --> MINIO
@enduml
"""

# ---------------------------------------------------------------- ERD
DIAGRAMS["erd"] = """@startuml
skinparam backgroundColor #FFFFFF
skinparam shadowing false
skinparam linetype ortho

entity "edge_nodes" as edge {
  * id : UUID <<PK>>
  --
  * node_id : VARCHAR(64) <<UNIQUE>>
  * name : VARCHAR
  ip_address : VARCHAR
  * status : VARCHAR(16)
  last_ping : TIMESTAMP
  settings_json : TEXT
  created_at / updated_at : TIMESTAMP
}
entity "violations" as vio {
  * id : BIGSERIAL <<PK>>
  --
  * event_id : VARCHAR <<UNIQUE>>
  * node_id : VARCHAR(64) <<FK>>
  track_id : INT
  frame_index : INT
  timestamp_ms : DOUBLE
  crossing_point_x/y : DOUBLE
  previous_point_x/y : DOUBLE
  bbox_x1/y1/x2/y2 : DOUBLE
  light_state : VARCHAR(8)
  light_confidence : DOUBLE
  previous_side / current_side : INT
  plate_text : VARCHAR
  plate_confidence : DOUBLE
  * status : VARCHAR(16)
  media_url : VARCHAR(1024)
  metadata : TEXT
  created_at / updated_at : TIMESTAMP
}
edge ||..o{ vio : "1 node ghi nhan\\nn vi pham"
@enduml
"""

if __name__ == "__main__":
    total = len(DIAGRAMS)
    ok = 0
    for i, (name, puml) in enumerate(DIAGRAMS.items(), 1):
        if render(name, puml, i, total):
            ok += 1
        time.sleep(1)
    print(f"\nDone: {ok}/{total}")
    sys.exit(0 if ok == total else 1)
