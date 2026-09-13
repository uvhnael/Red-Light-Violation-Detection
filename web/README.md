# Web Dashboard — Red-Light Violation Detection

Giao diện quản trị (Next.js App Router + React 19 + Tailwind CSS 4) để giám sát hệ thống: xem thống kê, duyệt hồ sơ vi phạm, theo dõi edge node, xem camera live, kẻ vạch dừng / vùng đèn / hướng giám sát cho từng camera, và hỏi dữ liệu bằng tiếng Việt qua AI.

## Kiến trúc

```
Browser ──▶ Next.js (:3000)
              │  /api/*      ──rewrite──▶ Central Server (:8002)
              │  /edge-api/* ──rewrite──▶ Edge Node API (:8080)
              │  /api/ai-query (route handler) ──▶ Central /api/ai/query
              └─ Server Components gọi Central qua HTTP (kèm JWT từ cookie)
```

Web không gọi thẳng Central bằng URL tuyệt đối ở client — mọi request đi qua **rewrite** của Next.js (`next.config.ts`), tránh CORS và giấu địa chỉ backend. Web layer **không** chứa database/AI library — mọi truy vấn dữ liệu sống ở Central Server.

## Tech stack

- **Next.js 16** (App Router, `output: 'standalone'`) + **React 19** + **TypeScript 5**
- **Tailwind CSS 4** (PostCSS) + **tw-animate-css** — design tokens + animation utility
- **recharts** — biểu đồ thống kê
- **hls.js** — phát live stream HLS từ edge node (dynamic import, chỉ tải ở trang camera/node)
- **lucide-react** — icon

## Trang

| Route | Mô tả |
|---|---|
| `/login` | **Đăng nhập** (JWT) — nhập username/password, lưu token vào localStorage + cookie `rlvd_token` |
| `/` | Dashboard tổng quan: thẻ thống kê, biểu đồ xu hướng vi phạm |
| `/violations` | Danh sách vi phạm (phân trang server-side, filter trạng thái) |
| `/violations/[id]` | Chi tiết 1 vi phạm: ảnh toàn cảnh, crop biển số, video bằng chứng |
| `/review` | Human-in-the-loop: duyệt / từ chối các case confidence thấp (OFFICER+) |
| `/nodes` | Danh sách edge node (online/offline) |
| `/nodes/[nodeId]` | Chi tiết node + **CalibrationEditor** (OPERATOR+) |
| `/cameras` | Xem camera live (HLS) + snapshot |
| `/settings` | Cấu hình hệ thống |

## Bảo mật & phân quyền (auth)

- **`proxy.ts`** (Next.js 16 — thay cho `middleware.ts` đã deprecated): mọi trang (trừ `/login`) chạy TRƯỚC khi render, kiểm tra cookie `rlvd_token` + **xác thực JWT** (`alg: HS256`, chữ ký bằng `JWT_SECRET`, `exp` còn hạn) — hỏng → 307 `/login?next=<đường-dẫn>` và xoá cookie chết. Đã đăng nhập mà vào `/login` → đẩy về `/`.
- **`lib/session-guard.ts`**: verify token dùng chung cho tầng server/proxy (không import React; thiếu `JWT_SECRET` thì chỉ kiểm tra `exp`).
- **`components/RequireSession.tsx`**: lớp guard thứ hai phía client cho nhóm route `(dashboard)` — bắt điều hướng SPA, phiên hết hạn khi tab đang mở, và sự kiện `rlvd:unauthorized` (401 sau khi refresh thất bại).
- **`lib/auth.ts`**: `login()` gọi `POST /api/auth/login`, lưu session (token, role, expiresAt) vào localStorage + đồng bộ cookie; `useSession()` cho component; `hasRole(session, minRole)` guard theo vai trò; `notifyUnauthorized()`/`clearSession()` khi phiên chết.
- **`lib/api.ts`**: mọi request API tự gắn `Authorization: Bearer <token>` từ session.
- **Header**: hiển thị tên + vai trò người dùng (ADMIN/OPERATOR/OFFICER), menu đăng xuất.
- Vai trò do central kiểm thử chặt ở tầng API — web chỉ guard UI; request vượt quyền bị central trả 403.

## Component chính

- **AppLayout** — layout chính: sidebar desktop (collapse được), **mobile drawer** (<lg), header với Command Palette + notification + menu user, AI floating FAB.
- **CalibrationEditor** — kéo chuột kẻ vạch dừng, vẽ mũi tên chọn hướng giám sát (đường 2 chiều), khoanh vùng đèn trên ảnh snapshot; lưu xuống edge qua Central proxy.
- **AISidebar** — hỏi dữ liệu vi phạm bằng tiếng Việt (Text-to-SQL), hiển thị SQL + bảng/biểu đồ kết quả.
- **VideoPlayer** — phát HLS bằng hls.js + overlay vạch/vùng đèn trên live stream.
- **Toast (ToastProvider/ConfirmDialog)** — thông báo + hộp thoại xác nhận dùng chung, thay `alert()/confirm()` native.
- **DataTable / BarChart / StatusBadge** — khối UI dùng lại.
- **CommandPalette** — điều hướng nhanh + tìm kiếm (Ctrl/Cmd+K).

## UX / Accessibility

- **Loading**: skeleton (không spinner trừ video), route-level `loading.tsx`.
- **Error**: route-level `error.tsx` + error state từng trang với nút thử lại; `not-found.tsx` trong và ngoài dashboard.
- **Mobile**: sidebar chuyển thành drawer overlay, table co giãn (cột ẩn theo breakpoint).
- **A11y**: skip-link "Nhảy tới nội dung chính", `aria-expanded/aria-label` cho các nút icon, click-outside + ESC đóng dropdown/dialog, focus input khi mở modal, `prefers-reduced-motion` được tôn trọng.
- **Ngôn ngữ**: UI tiếng Việt, định dạng ngày giờ `vi-VN`.
- **Theme**: Hệ thống hỗ trợ 5 tùy chọn theme độc lập: **VN Dark** (`vneid-dark` - mặc định), **VN Light** (`vneid-light`), **Dark** (`dark`), **Light** (`light`), và **System** (`system` - tự động theo hệ điều hành). Bộ theme VN mang nhận diện Đỏ mận & Vàng đồng đặc trưng CSGT Việt Nam; bộ theme Dark/Light mang phong cách Slate công nghệ cao. Hỗ trợ chuyển theme nhanh ngay trên Header và trong trang Cài đặt, kèm script chống flash theme (anti-FOUC) trước khi hydrate.

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
| `CENTRAL_SERVER_URL` | `http://localhost:8002` | Central để rewrite `/api/*` + AI query |
| `EDGE_SERVER_URL` | `http://localhost:8080` | Edge node để rewrite `/edge-api/*` |

> Trong `docker-compose.full.yml`, central được map ra host `:8002`, edge `:8082`; compose đặt `CENTRAL_SERVER_URL` trỏ về service nội bộ tương ứng.
