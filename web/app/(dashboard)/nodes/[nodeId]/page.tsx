'use client';

import { useCallback, useEffect, useState } from 'react';
import { useParams } from 'next/navigation';
import Link from 'next/link';
import dynamic from 'next/dynamic';
import CalibrationEditor from '@/components/CalibrationEditor';
import { EdgeNodeResponse, CalibrationState } from '@/lib/types';
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

type DrawMode = 'none' | 'line' | 'box';

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

  // Drawing on the live video: save the shape to the node, then refresh overlay
  const handleDraw = async (start: OverlayPoint, end: OverlayPoint) => {
    setDrawBusy(true);
    setDrawMessage(null);
    try {
      if (drawMode === 'line') {
        await setStopLine(nodeId, { x1: start.x, y1: start.y, x2: end.x, y2: end.y });
        setDrawMessage({ kind: 'ok', text: 'Đã lưu stop line mới.' });
      } else if (drawMode === 'box') {
        await setLightRoi(nodeId, {
          x: Math.min(start.x, end.x),
          y: Math.min(start.y, end.y),
          w: Math.abs(end.x - start.x),
          h: Math.abs(end.y - start.y),
        });
        setDrawMessage({ kind: 'ok', text: 'Đã lưu vùng đèn tín hiệu mới.' });
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

  // Build stream URL from edge node settings.
  // The edge node serves its video input as camera "{node_id}-cam-1".
  const apiPort = (node.settings?.api_port as number) || 8080;
  const nodeIp = node.ip_address || 'localhost';
  // Use localhost if IP looks like an internal Docker hostname
  const streamHost = (nodeIp.includes('.') || nodeIp === 'localhost') ? nodeIp : 'localhost';
  const cameraId = `${nodeId}-cam-1`;
  const streamUrl = `http://${streamHost}:${apiPort}/api/cameras/${cameraId}/stream`;
  const snapshotUrl = `http://${streamHost}:${apiPort}/api/cameras/${cameraId}/snapshot`;

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
            Quay lại Edge Nodes
          </Link>
          <h1 className="text-2xl font-bold gradient-text">{node.name}</h1>
          <p className="text-text-secondary text-sm mt-1 font-mono">{node.node_id}</p>
        </div>
        <span
          className={`inline-flex items-center rounded-full px-3 py-1 text-xs font-medium ${
            node.online ? 'status-confirmed' : 'status-rejected'
          }`}
        >
          {node.online ? 'Online' : 'Offline'}
        </span>
      </div>

      {/* Node Info Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="glass-card p-4">
          <p className="text-xs text-text-muted mb-1">IP Address</p>
          <p className="text-sm font-medium text-text-primary font-mono">{node.ip_address || '—'}</p>
        </div>
        <div className="glass-card p-4">
          <p className="text-xs text-text-muted mb-1">API Port</p>
          <p className="text-sm font-medium text-text-primary font-mono">{apiPort}</p>
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

      {/* Camera Stream */}
      <div className="glass-card overflow-hidden">
        <div className="px-6 py-4 border-b border-border flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-primary-500/15 flex items-center justify-center">
              <svg className="w-5 h-5 text-primary-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                <path strokeLinecap="round" d="M15.75 10.5l4.72-4.72a.75.75 0 011.28.53v11.38a.75.75 0 01-1.28.53l-4.72-4.72M4.5 18.75h9a2.25 2.25 0 002.25-2.25v-9a2.25 2.25 0 00-2.25-2.25h-9A2.25 2.25 0 002.25 7.5v9a2.25 2.25 0 002.25 2.25z" />
              </svg>
            </div>
            <div>
              <h2 className="text-sm font-semibold text-text-primary">Camera {nodeId}</h2>
              <p className="text-xs text-text-muted">HLS Stream — phát trực tiếp từ edge node</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <div className="glow-dot bg-red-500 text-red-500" />
            <span className="text-xs font-medium text-red-400">LIVE</span>
          </div>
        </div>

        {/* Draw-on-video toolbar */}
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
            Vẽ stop line trên video
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
            Vẽ vùng đèn trên video
          </button>
          {drawMode !== 'none' && (
            <span className="text-xs text-cyan-400">
              {drawMode === 'line'
                ? 'Kéo chuột trên video để vẽ stop line, thả để lưu.'
                : 'Kéo chuột trên video để vẽ vùng đèn, thả để lưu.'}
            </span>
          )}
          {drawBusy && <span className="text-xs text-text-muted">Đang lưu…</span>}
          {drawMessage && (
            <span
              className={`text-xs ${
                drawMessage.kind === 'ok' ? 'text-green-400' : 'text-red-400'
              }`}
            >
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

      {/* Calibration: re-detect traffic light + stop line */}
      <div className="glass-card overflow-hidden">
        <div className="px-6 py-4 border-b border-border flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-yellow-500/15 flex items-center justify-center">
              <svg className="w-5 h-5 text-yellow-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M10.5 6h9.75M10.5 6a1.5 1.5 0 11-3 0m3 0a1.5 1.5 0 10-3 0M3.75 6H7.5m3 12h9.75m-9.75 0a1.5 1.5 0 01-3 0m3 0a1.5 1.5 0 00-3 0m-3.75 0H7.5m9-6h3.75m-3.75 0a1.5 1.5 0 01-3 0m3 0a1.5 1.5 0 00-3 0m-9.75 0h9.75" />
              </svg>
            </div>
            <div>
              <h2 className="text-sm font-semibold text-text-primary">Hiệu chuẩn: Đèn tín hiệu & Stop line</h2>
              <p className="text-xs text-text-muted">Detect lại tự động hoặc vẽ thủ công trên ảnh</p>
            </div>
          </div>
        </div>
        <div className="p-4">
          <CalibrationEditor nodeId={nodeId} />
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
