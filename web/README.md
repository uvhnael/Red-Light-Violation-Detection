# Web Dashboard — Red-Light Violation Detection

Giao diện quản trị (Next.js App Router + React 19 + Tailwind CSS 4) để giám sát hệ thống: xem thống kê, duyệt hồ sơ vi phạm, theo dõi edge node, xem camera live, kẻ vạch dừng / vùng đèn / hướng giám sát cho từng camera, và hỏi dữ liệu bằng tiếng Việt qua AI.

## Kiến trúc

```
Browser ──▶ Next.js (:3000)
              │  /api/*      ──rewrite──▶ Central Server (:8000)
              │  /edge-api/* ──rewrite──▶ Edge Node API (:8080)
              │  /api/ai-query (route handler) ──▶ Central /api/ai/query
              └─ Server Components đọc DB trực tiếp (pg) cho trang tổng quan
```

Web không gọi thẳng Central bằng URL tuyệt đối ở client — mọi request đi qua **rewrite** của Next.js (`next.config.ts`), tránh CORS và giấu địa chỉ backend.

## Tech stack

- **Next.js 16** (App Router, `output: 'standalone'`) + **React 19** + **TypeScript 5**
- **Tailwind CSS 4** (PostCSS)
- **recharts** — biểu đồ thống kê (BarChart, TrendChart)
- **hls.js** — phát live stream HLS từ edge node
- **lucide-react** — icon
- **@google/genai** + route handler `/api/ai-query` — AI chat hỏi dữ liệu
- **pg** — một số Server Component đọc PostgreSQL trực tiếp

## Trang

| Route | Mô tả |
|---|---|
| `/` | Dashboard tổng quan: thẻ thống kê, biểu đồ xu hướng vi phạm |
| `/violations` | Danh sách vi phạm (phân trang server-side, filter trạng thái) |
| `/violations/[id]` | Chi tiết 1 vi phạm: ảnh toàn cảnh, crop biển số, video bằng chứng |
| `/review` | Human-in-the-loop: duyệt / từ chối các case confidence thấp |
| `/nodes` | Danh sách edge node (online/offline) |
| `/nodes/[nodeId]` | Chi tiết node + **CalibrationEditor** |
| `/cameras` | Xem camera live (HLS) + snapshot |
| `/settings` | Cấu hình hệ thống |

## Component chính

- **CalibrationEditor** — kéo chuột kẻ vạch dừng, vẽ mũi tên chọn hướng giám sát (đường 2 chiều), khoanh vùng đèn trên ảnh snapshot; lưu xuống edge qua Central proxy.
- **AIChat / AISidebar** — hỏi dữ liệu vi phạm bằng tiếng Việt (Text-to-SQL).
- **VideoPlayer** — phát HLS bằng hls.js.
- **DataTable / BarChart / TrendChart / StatsCard / StatusBadge** — khối UI dùng lại.
- **CommandPalette** — điều hướng nhanh.

## Cấu trúc

```
web/
├── app/
│   ├── page.tsx                 # Dashboard
│   ├── violations/              # Danh sách + chi tiết
│   ├── review/                  # Duyệt vi phạm
│   ├── nodes/                   # Edge nodes + calibration
│   ├── cameras/                 # Live stream
│   ├── settings/
│   ├── api/ai-query/route.ts    # Proxy AI query
│   ├── layout.tsx / globals.css
├── components/                  # UI components
├── lib/
│   ├── api.ts                   # Client gọi /api + /edge-api
│   ├── ai.ts                    # AI helper
│   └── types.ts                 # TypeScript types
├── next.config.ts               # Rewrites đến central + edge
├── Dockerfile                   # standalone build
└── package.json
```

## Chạy

### Dev

```bash
cd web
npm install
npm run dev          # http://localhost:3000
```

Cần Central Server chạy ở `:8000` (mặc định của `CENTRAL_SERVER_URL`). Nếu central chạy cổng khác, export trước khi dev:

```bash
CENTRAL_SERVER_URL=http://localhost:8002 npm run dev
```

### Production (Docker — dùng chung stack)

```bash
# Từ gốc project
./start.sh            # full stack gồm web-dashboard
./start.sh minimal    # web + central + postgres + minio (không edge)
```

Web trong Docker expose `:3000`.

### Lint & type-check

```bash
cd web
npm run lint
npx tsc --noEmit
```

## Biến môi trường

| Variable | Default | Mô tả |
|---|---|---|
| `CENTRAL_SERVER_URL` | `http://localhost:8001` | Central để rewrite `/api/*` + AI query |
| `EDGE_SERVER_URL` | `http://localhost:8080` | Edge node để rewrite `/edge-api/*` |

> Trong `docker-compose.full.yml`, central được map ra host `:8002`, edge `:8082`; compose đặt `CENTRAL_SERVER_URL` trỏ về service nội bộ tương ứng.
