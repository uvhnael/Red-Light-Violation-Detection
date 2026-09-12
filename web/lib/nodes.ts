// Chuẩn hoá trạng thái edge node cho toàn bộ web dashboard.
//
// Vấn đề cũ: mỗi nơi hiển thị trạng thái node một kiểu — chỗ dùng
// `node.status` (chuỗi do node tự khai báo khi đăng ký, có thể là
// "maintenance"/"degraded" và KHÔNG tự cập nhật khi node mất kết nối),
// chỗ dùng `node.online` (central tính từ last_ping trong 2 phút cuối).
// Kết quả: một node vừa hiện online chỗ này, offline chỗ kia.
//
// Quy tắc thống nhất (áp dụng ở mọi page):
//   1. `online === false` (mất ping > 2 phút) → luôn "offline".
//   2. `online === true` → dùng status node khai báo (online/degraded/
//      maintenance) — degraded/maintenance vẫn đang kết nối nhưng báo
//      cáo vấn đề, không được gọi là "online" hẳn hoi.
// Nguồn chân lý về kết nối là `online` (tính từ last_ping), không phải
// `status`.

import { EdgeNodeResponse } from './types';

export type EffectiveNodeStatus = 'online' | 'offline' | 'degraded' | 'maintenance';

export function effectiveNodeStatus(node: Pick<EdgeNodeResponse, 'online' | 'status'>): EffectiveNodeStatus {
  if (!node.online) return 'offline';
  const status = node.status?.toLowerCase();
  if (status === 'degraded' || status === 'maintenance') return status;
  return 'online';
}

export function isNodeOnline(node: Pick<EdgeNodeResponse, 'online' | 'status'>): boolean {
  return effectiveNodeStatus(node) === 'online';
}
