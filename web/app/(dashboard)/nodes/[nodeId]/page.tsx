'use client';

import { useCallback, useEffect, useState } from 'react';
import { useParams } from 'next/navigation';
import Link from 'next/link';
import dynamic from 'next/dynamic';
import { EdgeNodeResponse, CalibrationState, CalibrationPoint } from '@/lib/types';
import { getEdgeNode, getCalibration, setStopLine, setLightRoi } from '@/lib/api';
import type { OverlayPoint } from '@/components/VideoPlayer';

// hls.js nặng (~500KB) — chỉ tải khi vào trang node có camera
const VideoPlayer = dynamic(() => import('@/components/VideoPlayer'), {
  ssr: false,
  loading: () => (
    <div className="w-full aspect-video bg-surface-3 rounded-xl flex items-center justify-center">
      <div className="w-10 h-10 border-4 border-primary-500/30 border-t-primary-500 rounded-full animate-spin" />
    </div>
  ),
});

type DrawMode = 'none' | 'line' | 'box' | 'arrow';
type Direction = 'any' | 'positive_to_negative' | 'negative_to_positive';

const DIRECTION_LABELS: Record<Direction, string> = {
  any: 'Cả 2 hướng',
  positive_to_negative: 'Một chiều: dương → âm',
  negative_to_positive: 'Một chiều: âm → dương',
};

/**
 * Dấu của cross(start, end, point) — port side_of_line phía edge.
 * >0 = phía dương của vạch start→end, <0 = phía âm.
 */
function sideOfLine(p: CalibrationPoint, start: CalibrationPoint, end: CalibrationPoint): number {
  const cross = (end.x - start.x) * (p.y - start.y) - (end.y - start.y) * (p.x - start.x);
  if (cross > 0) return 1;
  if (cross < 0) return -1;
  return 0;
}

/**
 * Suy hướng giám sát từ mũi tên user vẽ (port run_pipeline._direction_from_arrow).
 * tail (điểm đầu) nằm ở phía xe xuất phát; tail trên vạch thì lùi lại theo mũi tên.
 */
function directionFromArrow(
  lineStart: CalibrationPoint,
  lineEnd: CalibrationPoint,
  tail: CalibrationPoint,
  head: CalibrationPoint
): Direction {
  let side = sideOfLine(tail, lineStart, lineEnd);
  if (side === 0) {
    const dx = head.x - tail.x;
    const dy = head.y - tail.y;
    const len = Math.hypot(dx, dy) || 1;
    const backed = { x: tail.x - (dx / len) * 5, y: tail.y - (dy / len) * 5 };
    side = sideOfLine(backed, lineStart, lineEnd);
  }
  if (side > 0) return 'positive_to_negative';
  if (side < 0) return 'negative_to_positive';
  return 'any';
}

export default function NodeDetailPage() {
  const params = useParams();
  const nodeId = params.nodeId as string;

  const [node, setNode] = useState<EdgeNodeResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Live calibration overlay (stop line + light ROI) polled from the node
  const [calibration, setCalibration] = useState<CalibrationState | null>(null);
  const [drawMode, setDrawMode] = useState<DrawMode>('none');
  const [drawBusy, setDrawBusy] = useState(false);
  const [drawMessage, setDrawMessage] = useState<{ kind: 'ok' | 'err'; text: string } | null>(null);

  useEffect(() => {
    async function load() {
      try {
        const data = await getEdgeNode(nodeId);
        setNode(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Không thể tải thông tin node');
      } finally {
        setLoading(false);
      }
    }
    load();
  }, [nodeId]);

  // Poll calibration state so the overlay follows edits made anywhere
  const refreshCalibration = useCallback(async () => {
    try {
      const state = await getCalibration(nodeId);
      setCalibration(state);
    } catch {
      // node may not have calibration yet — overlay simply stays empty
    }
  }, [nodeId]);

  useEffect(() => {
    let active = true;
    const fetchState = async () => {
      try {
        const state = await getCalibration(nodeId);
        if (active) setCalibration(state);
      } catch {
        // ignore
      }
    };
    void fetchState();
    const timer = setInterval(fetchState, 3000);
    return () => {
      active = false;
      clearInterval(timer);
    };
  }, [nodeId]);

  const stopLine = calibration?.stop_line ?? null;
  const direction = (stopLine?.direction ?? 'any') as Direction;

  // Drawing on the live video: save the shape to the node, then refresh overlay
  const handleDraw = async (start: OverlayPoint, end: OverlayPoint) => {
    setDrawBusy(true);
    setDrawMessage(null);
    try {
      if (drawMode === 'line') {
        await setStopLine(nodeId, { x1: start.x, y1: start.y, x2: end.x, y2: end.y });
        setDrawMessage({
          kind: 'ok',
          text: 'Đã lưu stop line mới (hướng: cả 2). Vẽ mũi tên hoặc chọn hướng để giới hạn 1 chiều.',
        });
      } else if (drawMode === 'box') {
        await setLightRoi(nodeId, {
          x: Math.min(start.x, end.x),
          y: Math.min(start.y, end.y),
          w: Math.abs(end.x - start.x),
          h: Math.abs(end.y - start.y),
        });
        setDrawMessage({ kind: 'ok', text: 'Đã lưu vùng đèn tín hiệu mới.' });
      } else if (drawMode === 'arrow') {
        if (!stopLine) {
          setDrawMessage({ kind: 'err', text: 'Chưa có stop line — vẽ stop line trước.' });
        } else {
          const dir = directionFromArrow(stopLine.start, stopLine.end, start, end);
          if (dir === 'any') {
            setDrawMessage({
              kind: 'err',
              text: 'Mũi tên nằm trên vạch — vẽ rõ về một phía để xác định hướng xe chạy.',
            });
          } else {
            await setStopLine(nodeId, {
              x1: stopLine.start.x,
              y1: stopLine.start.y,
              x2: stopLine.end.x,
              y2: stopLine.end.y,
              direction: dir,
            });
            setDrawMessage({
              kind: 'ok',
              text: `Đã lưu hướng giám sát: ${DIRECTION_LABELS[dir]}. Xe đi ngược chiều sẽ bị bỏ qua.`,
            });
          }
        }
      }
      setDrawMode('none');
      await refreshCalibration();
    } catch (err) {
      setDrawMessage({
        kind: 'err',
        text: err instanceof Error ? err.message : 'Không lưu được cấu hình.',
      });
    } finally {
      setDrawBusy(false);
    }
  };

  // Đổi hướng qua dropdown — re-POST stop line hiện tại với direction mới
  const handleDirectionChange = async (dir: Direction) => {
    if (!stopLine) return;
    setDrawBusy(true);
    setDrawMessage(null);
    try {
      await setStopLine(nodeId, {
        x1: stopLine.start.x,
        y1: stopLine.start.y,
        x2: stopLine.end.x,
        y2: stopLine.end.y,
        direction: dir,
      });
      setDrawMessage({
        kind: 'ok',
        text:
          dir === 'any'
            ? 'Đã đặt hướng: cả 2 hướng.'
            : `Đã lưu hướng giám sát: ${DIRECTION_LABELS[dir]}.`,
      });
      await refreshCalibration();
    } catch (err) {
      setDrawMessage({
        kind: 'err',
        text: err instanceof Error ? err.message : 'Không lưu được hướng.',
      });
    } finally {
      setDrawBusy(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-[60vh]">
        <div className="flex flex-col items-center gap-4">
          <div className="w-12 h-12 border-4 border-primary-500/30 border-t-primary-500 rounded-full animate-spin" />
          <p className="text-text-secondary text-sm">Đang tải thông tin node...</p>
        </div>
      </div>
    );
  }

  if (error || !node) {
    return (
      <div className="flex items-center justify-center h-[60vh]">
        <div className="glass-card p-8 text-center max-w-md">
          <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-red-500/10 flex items-center justify-center">
            <svg className="w-8 h-8 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m9-.75a9 9 0 11-18 0 9 9 0 0118 0zm-9 3.75h.008v.008H12v-.008z" />
            </svg>
          </div>
          <h2 className="text-lg font-semibold text-text-primary mb-2">Lỗi</h2>
          <p className="text-sm text-text-secondary mb-4">{error || 'Node không tồn tại'}</p>
          <Link href="/nodes" className="btn-primary text-sm">
            Quay lại
          </Link>
        </div>
      </div>
    );
  }

  // Stream/snapshot URL đi qua rewrite proxy của Next.js (/edge-api → edge),
  // không dùng IP:port trực tiếp — IP Docker nội bộ (172.x) và port container
  // không tới được từ browser.
  const cameraId = `${nodeId}-cam-1`;
  const streamUrl = `/edge-api/cameras/${cameraId}/stream`;
  const snapshotUrl = `/edge-api/cameras/${cameraId}/snapshot`;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-start justify-between gap-4">
        <div>
          <Link
            href="/nodes"
            className="inline-flex items-center gap-1.5 text-sm text-text-muted hover:text-text-secondary transition-colors mb-3"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M10.5 19.5L3 12m0 0l7.5-7.5M3 12h18" />
            </svg>
            Danh sách node
          </Link>
          <h1 className="text-2xl font-bold gradient-text">{node.name}</h1>
          <p className="text-text-muted text-sm font-mono mt-1">{node.node_id}</p>
        </div>
      </div>

      {/* Node Info Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="glass-card p-4">
          <p className="text-xs text-text-muted mb-1">IP Address</p>
          <p className="text-sm font-medium text-text-primary font-mono">{node.ip_address || '—'}</p>
        </div>
        <div className="glass-card p-4">
          <p className="text-xs text-text-muted mb-1">API Port</p>
          <p className="text-sm font-medium text-text-primary font-mono">
            {(node.settings?.api_port as number) || 8080}
          </p>
        </div>
        <div className="glass-card p-4">
          <p className="text-xs text-text-muted mb-1">Trạng thái</p>
          <p className="text-sm font-medium text-text-primary capitalize">{node.status}</p>
        </div>
        <div className="glass-card p-4">
          <p className="text-xs text-text-muted mb-1">Last Ping</p>
          <p className="text-sm font-medium text-text-primary">
            {node.last_ping ? new Date(node.last_ping).toLocaleString('vi-VN') : '—'}
          </p>
        </div>
      </div>

      {/* Camera + hiệu chuẩn: MỘT bề mặt duy nhất — vẽ trực tiếp lên live stream */}
      <div className="glass-card overflow-hidden">
        <div className="px-6 py-4 border-b border-border flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-primary-500/15 flex items-center justify-center">
              <svg className="w-5 h-5 text-primary-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                <path strokeLinecap="round" d="M15.75 10.5l4.72-4.72a.75.75 0 011.28.53v11.38a.75.75 0 01-1.28.53l-4.72-4.72M4.5 18.75h9a2.25 2.25 0 002.25-2.25v-9a2.25 2.25 0 00-2.25-2.25h-9A2.25 2.25 0 002.25 7.5v9a2.25 2.25 0 002.25 2.25z" />
              </svg>
            </div>
            <div>
              <h2 className="text-sm font-semibold text-text-primary">Camera {nodeId} — Hiệu chuẩn trực tiếp</h2>
              <p className="text-xs text-text-muted">Vẽ vạch dừng, vùng đèn và hướng xe chạy ngay trên stream trực tiếp</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <div className="glow-dot bg-red-500 text-red-500" />
            <span className="text-xs font-medium text-red-400">LIVE</span>
          </div>
        </div>

        {/* Toolbar vẽ — tất cả trong một */}
        <div className="px-6 py-3 border-b border-border flex flex-wrap items-center gap-2">
          <button
            onClick={() => setDrawMode(drawMode === 'line' ? 'none' : 'line')}
            disabled={drawBusy}
            className={`text-xs px-3 py-1.5 rounded-lg border transition-colors ${
              drawMode === 'line'
                ? 'border-red-500 text-red-400 bg-red-500/10'
                : 'border-border text-text-secondary hover:border-red-500/50'
            }`}
          >
            Vẽ stop line
          </button>
          <button
            onClick={() => setDrawMode(drawMode === 'box' ? 'none' : 'box')}
            disabled={drawBusy}
            className={`text-xs px-3 py-1.5 rounded-lg border transition-colors ${
              drawMode === 'box'
                ? 'border-yellow-500 text-yellow-400 bg-yellow-500/10'
                : 'border-border text-text-secondary hover:border-yellow-500/50'
            }`}
          >
            Vẽ vùng đèn
          </button>
          <button
            onClick={() => setDrawMode(drawMode === 'arrow' ? 'none' : 'arrow')}
            disabled={drawBusy || !stopLine}
            title={stopLine ? 'Vẽ mũi tên: đầu ở phía xe xuất phát, chỉ hướng xe chạy qua vạch' : 'Cần vẽ stop line trước'}
            className={`text-xs px-3 py-1.5 rounded-lg border transition-colors ${
              drawMode === 'arrow'
                ? 'border-green-500 text-green-400 bg-green-500/10'
                : 'border-border text-text-secondary hover:border-green-500/50 disabled:opacity-40'
            }`}
          >
            Vẽ hướng xe chạy
          </button>

          {stopLine && (
            <label className="flex items-center gap-1.5 text-xs text-text-muted">
              Hướng giám sát:
              <select
                value={direction}
                disabled={drawBusy}
                onChange={(e) => handleDirectionChange(e.target.value as Direction)}
                className="text-xs px-2 py-1.5 rounded-lg border border-border bg-surface-3 text-text-secondary"
              >
                {(Object.keys(DIRECTION_LABELS) as Direction[]).map((d) => (
                  <option key={d} value={d}>
                    {DIRECTION_LABELS[d]}
                  </option>
                ))}
              </select>
            </label>
          )}

          {drawMode !== 'none' && (
            <span className="text-xs text-cyan-400">
              {drawMode === 'line'
                ? 'Kéo chuột trên video để vẽ stop line, thả để lưu.'
                : drawMode === 'arrow'
                  ? 'Kéo vẽ MŨI TÊN: điểm đầu phía xe xuất phát, điểm cuối theo hướng xe chạy. Xe ngược chiều sẽ bị bỏ qua.'
                  : 'Kéo chuột trên video để vẽ vùng đèn, thả để lưu.'}
            </span>
          )}
          {drawBusy && <span className="text-xs text-text-muted">Đang lưu…</span>}
          {drawMessage && (
            <span className={`text-xs ${drawMessage.kind === 'ok' ? 'text-green-400' : 'text-red-400'}`}>
              {drawMessage.text}
            </span>
          )}
        </div>

        <div className="p-4">
          <VideoPlayer
            src={streamUrl}
            className="w-full aspect-video"
            overlay={{
              stopLine: calibration?.stop_line
                ? { start: calibration.stop_line.start, end: calibration.stop_line.end }
                : null,
              lightRoi: calibration?.light_roi ?? null,
            }}
            direction={direction}
            drawMode={drawMode === 'none' ? null : drawMode}
            onDraw={handleDraw}
          />
        </div>

        <div className="px-6 py-3 border-t border-border flex items-center justify-between text-xs text-text-muted">
          <span>Stream: {streamUrl}</span>
          <a
            href={snapshotUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="text-primary-400 hover:text-primary-300 transition-colors"
          >
            Chụp ảnh snapshot →
          </a>
        </div>
      </div>

      {/* Node Settings */}
      <div className="glass-card p-6">
        <h3 className="text-sm font-semibold text-text-secondary uppercase tracking-wider mb-4">
          Cấu hình Node
        </h3>
        <pre className="bg-surface-3/50 rounded-xl p-4 text-xs text-text-secondary font-mono overflow-x-auto">
          {JSON.stringify(node.settings, null, 2)}
        </pre>
      </div>
    </div>
  );
}
