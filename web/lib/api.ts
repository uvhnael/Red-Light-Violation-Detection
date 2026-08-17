import { ViolationResponse, Stats, HealthResponse, EdgeNodeResponse, EdgeNodeUpdateRequest, CameraInfo } from './types';

const API_BASE = '/api';
const EDGE_API_BASE = '/edge-api';

async function fetchAPI<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
  });
  if (!res.ok) {
    const error = await res.text().catch(() => 'Unknown error');
    throw new Error(`API Error ${res.status}: ${error}`);
  }
  return res.json();
}

// ----- Violations -----

export async function getViolations(params?: {
  status?: string;
  nodeId?: string;
  plateText?: string;
}): Promise<ViolationResponse[]> {
  const searchParams = new URLSearchParams();
  if (params?.status) searchParams.set('status', params.status);
  if (params?.nodeId) searchParams.set('nodeId', params.nodeId);
  if (params?.plateText) searchParams.set('plateText', params.plateText);
  const query = searchParams.toString();
  return fetchAPI<ViolationResponse[]>(`/violations${query ? `?${query}` : ''}`);
}

export async function getViolation(id: number): Promise<ViolationResponse> {
  return fetchAPI<ViolationResponse>(`/violations/${id}`);
}

export async function updateViolationStatus(
  id: number,
  status: string
): Promise<ViolationResponse> {
  return fetchAPI<ViolationResponse>(`/violations/${id}/status`, {
    method: 'PUT',
    body: JSON.stringify({ status }),
  });
}

export async function deleteViolation(id: number): Promise<void> {
  const res = await fetch(`${API_BASE}/violations/${id}`, { method: 'DELETE' });
  if (!res.ok) {
    const error = await res.text().catch(() => 'Unknown error');
    throw new Error(`API Error ${res.status}: ${error}`);
  }
}

// ----- Stats -----

export async function getStats(): Promise<Stats> {
  return fetchAPI<Stats>('/stats');
}

// ----- Health -----

export async function getHealth(): Promise<HealthResponse> {
  return fetchAPI<HealthResponse>('/health');
}

// ----- Edge Nodes -----

export async function getEdgeNodes(): Promise<EdgeNodeResponse[]> {
  return fetchAPI<EdgeNodeResponse[]>('/v1/edge-nodes');
}

export async function getEdgeNode(nodeId: string): Promise<EdgeNodeResponse> {
  return fetchAPI<EdgeNodeResponse>(`/v1/edge-nodes/${encodeURIComponent(nodeId)}`);
}

export async function updateEdgeNodeSettings(
  nodeId: string,
  payload: EdgeNodeUpdateRequest
): Promise<EdgeNodeResponse> {
  return fetchAPI<EdgeNodeResponse>(`/v1/edge-nodes/${encodeURIComponent(nodeId)}/settings`, {
    method: 'PUT',
    body: JSON.stringify({
      ...payload,
      settings: payload.settings ?? {},
    }),
  });
}

// ----- Cameras -----

export async function getCameras(): Promise<CameraInfo[]> {
  const res = await fetch(`${EDGE_API_BASE}/cameras`, { cache: 'no-store' });
  if (!res.ok) {
    throw new Error(`Failed to fetch cameras: ${res.status}`);
  }
  return res.json();
}
