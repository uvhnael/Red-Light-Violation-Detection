'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import {
  calibrationSnapshotUrl,
  getCalibration,
  setLightRoi,
  setStopLine,
} from '@/lib/api';
import type { CalibrationRoi, CalibrationStopLine } from '@/lib/types';

type DrawMode = 'none' | 'line' | 'box';

interface Props {
  nodeId: string;
}

/**
 * Calibration editor: shows the calibration frame from the edge node with
 * the configured stop line + traffic-light ROI overlaid. The user drags on
 * the image to draw the stop line or the light box, then saves.
 */
export default function CalibrationEditor({ nodeId }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imgRef = useRef<HTMLImageElement>(null);

  const [imageUrl, setImageUrl] = useState(() => calibrationSnapshotUrl(nodeId));
  const [imageLoaded, setImageLoaded] = useState(false);
  const [imgSize, setImgSize] = useState({ w: 0, h: 0 }); // natural pixels

  const [stopLine, setStopLineState] = useState<CalibrationStopLine | null>(null);
  const [lightRoi, setLightRoiState] = useState<CalibrationRoi | null>(null);

  const [mode, setMode] = useState<DrawMode>('none');
  const [drawing, setDrawing] = useState(false);
  const [draftStart, setDraftStart] = useState<{ x: number; y: number } | null>(null);
  const [draftEnd, setDraftEnd] = useState<{ x: number; y: number } | null>(null);

  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<{ kind: 'ok' | 'err'; text: string } | null>(null);

  // ---- load current calibration state from the edge node ----
  useEffect(() => {
    let cancelled = false;
    getCalibration(nodeId)
      .then((state) => {
        if (!cancelled && state) {
          setStopLineState(state.stop_line);
          setLightRoiState(state.light_roi);
        }
      })
      .catch(() => {
        // edge node may not have a calibration yet — not fatal
      });

    return () => {
      cancelled = true;
    };
  }, [nodeId]);

  // ---- drawing helpers ----
  const toImageCoords = useCallback(
    (clientX: number, clientY: number) => {
      const canvas = canvasRef.current;
      if (!canvas || imgSize.w === 0) return null;
      const rect = canvas.getBoundingClientRect();
      const scaleX = imgSize.w / rect.width;
      const scaleY = imgSize.h / rect.height;
      const x = Math.round((clientX - rect.left) * scaleX);
      const y = Math.round((clientY - rect.top) * scaleY);
      return {
        x: Math.max(0, Math.min(imgSize.w, x)),
        y: Math.max(0, Math.min(imgSize.h, y)),
      };
    },
    [imgSize]
  );

  const redraw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (imgSize.w === 0) return;

    // scale: canvas is natural-image sized; CSS shrinks it, so use natural px
    const sx = canvas.width / imgSize.w;
    const sy = canvas.height / imgSize.h;

    // stop line — red
    if (stopLine) {
      ctx.strokeStyle = '#ef4444';
      ctx.lineWidth = Math.max(3, 4 * sx);
      ctx.setLineDash([]);
      ctx.beginPath();
      ctx.moveTo(stopLine.start.x * sx, stopLine.start.y * sy);
      ctx.lineTo(stopLine.end.x * sx, stopLine.end.y * sy);
      ctx.stroke();
      // label
      ctx.fillStyle = '#ef4444';
      ctx.font = `${Math.max(14, 18 * sx)}px sans-serif`;
      ctx.fillText('STOP LINE', stopLine.start.x * sx + 8, stopLine.start.y * sy - 8);
    }

    // light ROI — yellow box
    if (lightRoi) {
      ctx.strokeStyle = '#facc15';
      ctx.lineWidth = Math.max(3, 4 * sx);
      ctx.setLineDash([]);
      ctx.strokeRect(lightRoi.x * sx, lightRoi.y * sy, lightRoi.w * sx, lightRoi.h * sy);
      ctx.fillStyle = '#facc15';
      ctx.font = `${Math.max(14, 18 * sx)}px sans-serif`;
      ctx.fillText('TRAFFIC LIGHT', lightRoi.x * sx + 8, lightRoi.y * sy - 8);
    }

    // draft while dragging
    if (drawing && draftStart && draftEnd) {
      ctx.setLineDash([10 * sx, 6 * sx]);
      if (mode === 'line') {
        ctx.strokeStyle = '#22d3ee';
        ctx.lineWidth = Math.max(2, 3 * sx);
        ctx.beginPath();
        ctx.moveTo(draftStart.x * sx, draftStart.y * sy);
        ctx.lineTo(draftEnd.x * sx, draftEnd.y * sy);
        ctx.stroke();
      } else if (mode === 'box') {
        ctx.strokeStyle = '#22d3ee';
        ctx.lineWidth = Math.max(2, 3 * sx);
        const x = Math.min(draftStart.x, draftEnd.x) * sx;
        const y = Math.min(draftStart.y, draftEnd.y) * sy;
        const w = Math.abs(draftEnd.x - draftStart.x) * sx;
        const h = Math.abs(draftEnd.y - draftStart.y) * sy;
        ctx.strokeRect(x, y, w, h);
      }
      ctx.setLineDash([]);
    }
  }, [stopLine, lightRoi, drawing, draftStart, draftEnd, mode, imgSize]);

  useEffect(() => {
    redraw();
  }, [redraw]);

  // ---- mouse handlers ----
  const onMouseDown = (e: React.MouseEvent) => {
    if (mode === 'none') return;
    const pt = toImageCoords(e.clientX, e.clientY);
    if (!pt) return;
    setDrawing(true);
    setDraftStart(pt);
    setDraftEnd(pt);
  };

  const onMouseMove = (e: React.MouseEvent) => {
    if (!drawing || mode === 'none') return;
    const pt = toImageCoords(e.clientX, e.clientY);
    if (pt) setDraftEnd(pt);
  };

  const onMouseUp = async () => {
    if (!drawing || !draftStart || !draftEnd) {
      setDrawing(false);
      return;
    }
    setDrawing(false);

    const start = draftStart;
    const end = draftEnd;
    setDraftStart(null);
    setDraftEnd(null);

    // ignore tiny accidental drags
    if (Math.abs(end.x - start.x) < 5 && Math.abs(end.y - start.y) < 5) return;

    setBusy(true);
    setMessage(null);
    try {
      if (mode === 'line') {
        await setStopLine(nodeId, { x1: start.x, y1: start.y, x2: end.x, y2: end.y });
        setStopLineState({ start, end, direction: 'any' });
        setMessage({ kind: 'ok', text: 'Đã lưu stop line mới.' });
      } else if (mode === 'box') {
        const roi = {
          x: Math.min(start.x, end.x),
          y: Math.min(start.y, end.y),
          w: Math.abs(end.x - start.x),
          h: Math.abs(end.y - start.y),
        };
        await setLightRoi(nodeId, roi);
        setLightRoiState(roi);
        setMessage({ kind: 'ok', text: 'Đã lưu vùng đèn tín hiệu mới.' });
      }
      setMode('none');
    } catch (err) {
      setMessage({
        kind: 'err',
        text: err instanceof Error ? err.message : 'Không lưu được cấu hình.',
      });
    } finally {
      setBusy(false);
    }
  };

  // ---- actions ----
  const handleRefreshImage = () => {
    setImageLoaded(false);
    setImageUrl(calibrationSnapshotUrl(nodeId));
  };

  const onImgLoad = () => {
    const img = imgRef.current;
    if (!img) return;
    setImgSize({ w: img.naturalWidth, h: img.naturalHeight });
    setImageLoaded(true);
  };

  const cursorClass =
    mode === 'none' ? 'cursor-default' : 'cursor-crosshair';

  return (
    <div className="space-y-4">
      {/* Toolbar */}
      <div className="flex flex-wrap items-center gap-2">
        <button
          onClick={() => setMode(mode === 'line' ? 'none' : 'line')}
          disabled={busy}
          className={`text-sm px-3 py-1.5 rounded-lg border transition-colors ${
            mode === 'line'
              ? 'border-red-500 text-red-400 bg-red-500/10'
              : 'border-border text-text-secondary hover:border-red-500/50'
          }`}
        >
          Vẽ lại stop line
        </button>
        <button
          onClick={() => setMode(mode === 'box' ? 'none' : 'box')}
          disabled={busy}
          className={`text-sm px-3 py-1.5 rounded-lg border transition-colors ${
            mode === 'box'
              ? 'border-yellow-500 text-yellow-400 bg-yellow-500/10'
              : 'border-border text-text-secondary hover:border-yellow-500/50'
          }`}
        >
          Vẽ lại vùng đèn
        </button>
        <button
          onClick={handleRefreshImage}
          disabled={busy}
          className="text-sm px-3 py-1.5 rounded-lg border border-border text-text-secondary hover:border-primary-500/50 transition-colors"
        >
          Làm mới ảnh
        </button>
      </div>

      {mode !== 'none' && (
        <p className="text-xs text-cyan-400">
          {mode === 'line'
            ? 'Kéo chuột trên ảnh để vẽ stop line mới, thả chuột để lưu.'
            : 'Kéo chuột trên ảnh để vẽ vùng đèn tín hiệu mới, thả chuột để lưu.'}
        </p>
      )}

      {message && (
        <div
          className={`text-sm rounded-lg px-4 py-2 ${
            message.kind === 'ok'
              ? 'bg-green-500/10 text-green-400 border border-green-500/30'
              : 'bg-red-500/10 text-red-400 border border-red-500/30'
          }`}
        >
          {message.text}
        </div>
      )}

      {/* Image + overlay canvas */}
      <div
        ref={containerRef}
        className="relative w-full rounded-xl overflow-hidden bg-surface-3/50 select-none"
      >
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          ref={imgRef}
          src={imageUrl}
          alt="Calibration frame"
          onLoad={onImgLoad}
          onError={() =>
            setMessage({ kind: 'err', text: 'Không tải được ảnh calibration từ edge node.' })
          }
          className="w-full h-auto block"
          draggable={false}
        />
        <canvas
          ref={canvasRef}
          width={imgSize.w || 1}
          height={imgSize.h || 1}
          className={`absolute inset-0 w-full h-full ${cursorClass}`}
          onMouseDown={onMouseDown}
          onMouseMove={onMouseMove}
          onMouseUp={onMouseUp}
          onMouseLeave={() => drawing && onMouseUp()}
        />
        {!imageLoaded && (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="w-8 h-8 border-4 border-primary-500/30 border-t-primary-500 rounded-full animate-spin" />
          </div>
        )}
      </div>

      {/* Legend */}
      <div className="flex flex-wrap items-center gap-4 text-xs text-text-muted">
        <span className="inline-flex items-center gap-1.5">
          <span className="w-4 h-0.5 bg-red-500 inline-block" /> Stop line
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="w-3 h-3 border-2 border-yellow-400 inline-block" /> Vùng đèn tín hiệu
        </span>
      </div>
    </div>
  );
}
