import { ViolationResponse, ViolationPageResponse, ViolationCounts, Stats, HealthResponse, EdgeNodeResponse, EdgeNodeUpdateRequest, CameraInfo, CalibrationState } from './types';
import { getSession, refreshAccessToken, clearSession } from './auth';

const API_BASE = '/api';
const EDGE_API_BASE = '/edge-api';

async function authHeaders(): Promise<Record<string, string>> {
  // Chỉ chạy client-side; server components gọi fetch riêng.
  if (typeof window === 'undefined') return {};
  const session = getSession();
  if (!session) return {};
  return { Authorization: `Bearer ${session.token}` };
}

async function fetchAPI<T>(path: string, options?: RequestInit & { retry?: boolean }): Promise<T> {
  const { retry = true, ...init } = options ?? {};

  const attempt = async (): Promise<T> => {
    const headers = {
      'Content-Type': 'application/json',
      ...(await authHeaders()),
      ...init.headers,
    };
    let res: Response;
    try {
      res = await fetch(`${API_BASE}${path}`, { ...init, headers });
    } catch (err) {
      // Network error (CORS, offline, ECONNRESET…) — retry-able.
      throw new FetchError(0, err instanceof Error ? err.message : 'network error');
    }
    if (!res.ok) {
      const body = await res.text().catch(() => '');
      throw new FetchError(res.status, body || res.statusText);
    }
    return res.json() as Promise<T>;
  };

  try {
    return await attempt();
  } catch (err) {
    // 401 + còn refresh token → thử rotate 1 lần rồi retry request gốc.
    // Tránh loop vô hạn: chỉ retry 1 lần và chỉ khi init không tự là
    // /api/auth/refresh hay /api/auth/login (đã permit).
    if (
      err instanceof FetchError &&
      err.status === 401 &&
      retry &&
      !path.startsWith('/auth/')
    ) {
      const session = getSession();
      if (session?.refreshToken) {
        try {
          await refreshAccessToken(session);
          return await attempt();
        } catch {
          // Refresh thất bại → session đã clear, bubble 401.
          clearSession();
        }
      }
    }
    // Retry 5xx + network (như cũ).
    if (err instanceof FetchError && retry && err.isRetryable()) {
      await new Promise((r) => setTimeout(r, 500));
      return attempt();
    }
    throw err;
  }
}

// ----- Errors -----

export class FetchError extends Error {
  constructor(
    public readonly status: number,
    public readonly body: string,
  ) {
    super(`API Error ${status}: ${body}`);
    this.name = 'FetchError';
  }

  /** True cho 5xx và network (status=0). 4xx KHÔNG retry. */
  isRetryable(): boolean {
    return this.status === 0 || (this.status >= 500 && this.status < 600);
  }
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

/**
 * Paged violation list (lazy loading) — loads one page at a time instead of
 * the whole database. Prefer this over getViolations() for user-facing lists.
 */
export async function getViolationsPage(params?: {
  page?: number;
  size?: number;
  status?: string;
  nodeId?: string;
  plateText?: string;
}): Promise<ViolationPageResponse> {
  const searchParams = new URLSearchParams();
  searchParams.set('page', String(params?.page ?? 0));
  searchParams.set('size', String(params?.size ?? 20));
  if (params?.status) searchParams.set('status', params.status);
  if (params?.nodeId) searchParams.set('nodeId', params.nodeId);
  if (params?.plateText) searchParams.set('plateText', params.plateText);
  return fetchAPI<ViolationPageResponse>(`/violations/page?${searchParams.toString()}`);
}

/** Cheap status counts (COUNT queries server-side) for badges/headers. */
export async function getViolationCounts(): Promise<ViolationCounts> {
  return fetchAPI<ViolationCounts>('/violations/counts');
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
  await fetchAPI<void>(`/violations/${id}`, { method: 'DELETE' });
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

// ----- Calibration (manual stop line / light ROI drawn on the web UI) -----

/** Get the current calibration state (stop line + light ROI). */
export async function getCalibration(nodeId: string): Promise<CalibrationState> {
  return fetchAPI<CalibrationState>(
    `/v1/edge-nodes/${encodeURIComponent(nodeId)}/calibration`
  );
}

/** URL of the calibration frame JPEG (drawn under the overlay canvas). */
export function calibrationSnapshotUrl(nodeId: string): string {
  return `${API_BASE}/v1/edge-nodes/${encodeURIComponent(nodeId)}/calibration/snapshot?t=${Date.now()}`;
}

/** Manually set the stop line (drawn on the web UI). */
export async function setStopLine(
  nodeId: string,
  payload: { x1: number; y1: number; x2: number; y2: number; direction?: string }
): Promise<{ message: string }> {
  return fetchAPI<{ message: string }>(
    `/v1/edge-nodes/${encodeURIComponent(nodeId)}/calibration/stop-line`,
    { method: 'POST', body: JSON.stringify({ direction: 'any', ...payload }) }
  );
}

/** Manually set the traffic-light ROI box (drawn on the web UI). */
export async function setLightRoi(
  nodeId: string,
  payload: { x: number; y: number; w: number; h: number }
): Promise<{ message: string }> {
  return fetchAPI<{ message: string }>(
    `/v1/edge-nodes/${encodeURIComponent(nodeId)}/calibration/light-roi`,
    { method: 'POST', body: JSON.stringify(payload) }
  );
}
