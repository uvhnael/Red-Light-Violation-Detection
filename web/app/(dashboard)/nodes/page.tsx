import { EdgeNodeResponse } from '@/lib/types';
import NodesClient from '@/components/NodesClient';

const API_BASE = process.env.CENTRAL_SERVER_URL || 'http://localhost:8001';

async function fetchNodes(): Promise<EdgeNodeResponse[]> {
  try {
    const response = await fetch(`${API_BASE}/api/v1/edge-nodes`, { cache: 'no-store' });
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