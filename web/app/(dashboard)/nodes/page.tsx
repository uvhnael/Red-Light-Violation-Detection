import { cookies } from 'next/headers';
import { EdgeNodeResponse } from '@/lib/types';
import NodesClient from '@/components/NodesClient';

const API_BASE = process.env.CENTRAL_SERVER_URL || 'http://localhost:8002';
const COOKIE_NAME = 'rlvd_token';

/**
 * Server component: đọc JWT từ cookie do lib/auth.ts ghi khi đăng nhập
 * rồi gọi central kèm Authorization. Không có token → trả danh sách rỗng;
 * NodesClient phía client sẽ tự refetch bằng token trong localStorage.
 */
async function fetchNodes(): Promise<EdgeNodeResponse[]> {
  try {
    const token = (await cookies()).get(COOKIE_NAME)?.value;
    const headers: Record<string, string> = { 'Content-Type': 'application/json' };
    if (token) {
      headers.Authorization = `Bearer ${decodeURIComponent(token)}`;
    }
    const response = await fetch(`${API_BASE}/api/v1/edge-nodes`, {
      cache: 'no-store',
      headers,
    });
    if (!response.ok) {
      return [];
    }
    return response.json();
  } catch {
    return [];
  }
}

export default async function NodesPage() {
  const nodes = await fetchNodes();
  return <NodesClient initialNodes={nodes} />;
}
