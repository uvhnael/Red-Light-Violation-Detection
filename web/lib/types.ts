// TypeScript types matching the Central Server API responses

export interface PointDto {
  x: number;
  y: number;
}

export interface PlateDto {
  text: string;
  confidence: number;
}

export interface ViolationResponse {
  id: number;
  event_id: string;
  node_id: string;
  track_id: number;
  frame_index: number;
  timestamp_ms: number;
  crossing_point: PointDto | null;
  previous_point: PointDto | null;
  bbox_xyxy: number[] | null;
  light_state: string;
  light_confidence: number;
  previous_side: number;
  current_side: number;
  plate_text: string | null;
  plate_confidence: number | null;
  status: string;
  media_url: string | null;
  metadata: string | null;
  created_at: string;
  updated_at: string;
}

export interface ViolationPageResponse {
  content: ViolationResponse[];
  page: number;
  size: number;
  total_elements: number;
  total_pages: number;
  first: boolean;
  last: boolean;
}

export interface ViolationCounts {
  total: number;
  pending: number;
  approved: number;
  rejected: number;
}

export interface TrendPoint {
  hour: string;
  red: number;
  yellow: number;
}

export interface Stats {
  total: number;
  today_total: number;
  pending: number;
  approved: number;
  rejected: number;
  approval_rate: number;
  active_nodes: number;
  offline_nodes: number;
  violations_per_node: Record<string, number>;
  violations_per_node_today: Record<string, number>;
  violations_per_light_state: Record<string, number>;
  hourly_trend: TrendPoint[];
  recent_pending: ViolationResponse[];
}

export interface EdgeNodeResponse {
  id: string;
  node_id: string;
  name: string;
  ip_address: string | null;
  status: string;
  last_ping: string | null;
  settings: Record<string, unknown>;
  online: boolean;
  created_at: string;
  updated_at: string;
}

export interface EdgeNodeUpdateRequest {
  name?: string;
  ip_address?: string;
  status?: string;
  settings?: Record<string, unknown>;
}

export interface HealthResponse {
  status: string;
  service: string;
  timestamp: string;
}

export interface CameraInfo {
  id: string;
  name: string;
  status: string;
  resolution: string;
  location: string;
  stream_url: string;
  snapshot_url: string;
}

// ----- Calibration (re-detect traffic light + stop line) -----

export interface CalibrationPoint {
  x: number;
  y: number;
}

export interface CalibrationStopLine {
  start: CalibrationPoint;
  end: CalibrationPoint;
  direction: string;
}

export interface CalibrationRoi {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface CalibrationState {
  stop_line: CalibrationStopLine | null;
  light_roi: CalibrationRoi | null;
}
