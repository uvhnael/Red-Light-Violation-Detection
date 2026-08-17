'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import Hls from 'hls.js';

// ----- Overlay types (calibration coordinates = native video pixels) -----

export interface OverlayPoint {
  x: number;
  y: number;
}

export interface OverlayStopLine {
  start: OverlayPoint;
  end: OverlayPoint;
}

export interface OverlayRoi {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface VideoOverlayData {
  stopLine?: OverlayStopLine | null;
  lightRoi?: OverlayRoi | null;
}

interface VideoPlayerProps {
  src: string;
  className?: string;
  /** Calibration boxes drawn over the live video (native video pixel coords). */
  overlay?: VideoOverlayData | null;
  /** When set, the user can drag on the video to draw a line/box. */
  drawMode?: 'line' | 'box' | null;
  /** Called with start/end points (native video pixel coords) after a drag. */
  onDraw?: (start: OverlayPoint, end: OverlayPoint) => void;
}

export default function VideoPlayer({
  src,
  className = '',
  overlay = null,
  drawMode = null,
  onDraw,
}: VideoPlayerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const hlsRef = useRef<Hls | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [mounted, setMounted] = useState(false);

  // Intrinsic video size (native pixels) and container size (CSS pixels)
  const [videoSize, setVideoSize] = useState({ w: 0, h: 0 });
  const [containerSize, setContainerSize] = useState({ w: 0, h: 0 });

  // Draft shape while dragging
  const [draftStart, setDraftStart] = useState<OverlayPoint | null>(null);
  const [draftEnd, setDraftEnd] = useState<OverlayPoint | null>(null);

  useEffect(() => {
    setMounted(true);
  }, []);

  // ----- HLS setup (unchanged behaviour) -----
  useEffect(() => {
    if (!mounted) return;
    const video = videoRef.current;
    if (!video || !src) return;

    let destroyed = false;

    if (Hls.isSupported()) {
      const hls = new Hls({
        enableWorker: false,
        lowLatencyMode: false,
        maxBufferLength: 60,
        maxMaxBufferLength: 120,
        liveSyncDurationCount: 5,
        liveMaxLatencyDurationCount: 12,
        manifestLoadingTimeOut: 10000,
        manifestLoadingMaxRetry: 10,
        levelLoadingTimeOut: 10000,
        levelLoadingMaxRetry: 10,
        fragLoadingTimeOut: 20000,
        fragLoadingMaxRetry: 20,
        startFragPrefetch: true,
      });

      hlsRef.current = hls;

      hls.on(Hls.Events.MANIFEST_PARSED, () => {
        if (!destroyed) {
          setLoading(false);
          setError(null);
          video.play().catch((e) => console.warn('Autoplay blocked:', e));
        }
      });

      hls.on(Hls.Events.ERROR, (_: unknown, data: { fatal: boolean; type: string; details: string }) => {
        console.warn('[HLS Error]', data.type, data.details);
        if (data.fatal) {
          setError('Stream không khả dụng. Đang thử lại...');
          setLoading(false);
          if (data.type === Hls.ErrorTypes.NETWORK_ERROR) {
            setTimeout(() => {
              if (!destroyed) {
                hls.startLoad();
                setLoading(true);
                setError(null);
              }
            }, 3000);
          }
        }
      });

      hls.loadSource(src);
      hls.attachMedia(video);
    } else if (video.canPlayType('application/vnd.apple.mpegurl')) {
      video.src = src;
      video.addEventListener('loadedmetadata', () => {
        if (!destroyed) {
          setLoading(false);
          video.play().catch(() => {});
        }
      });
      video.addEventListener('error', () => {
        if (!destroyed) {
          setError('Không thể phát stream camera.');
          setLoading(false);
        }
      });
    } else {
      setError('Trình duyệt không hỗ trợ HLS streaming.');
      setLoading(false);
    }

    return () => {
      destroyed = true;
      if (hlsRef.current) {
        hlsRef.current.destroy();
        hlsRef.current = null;
      }
    };
  }, [src, mounted]);

  // ----- Track intrinsic video size -----
  const syncVideoSize = useCallback(() => {
    const video = videoRef.current;
    if (!video) return;
    if (video.videoWidth > 0 && video.videoHeight > 0) {
      setVideoSize((prev) =>
        prev.w === video.videoWidth && prev.h === video.videoHeight
          ? prev
          : { w: video.videoWidth, h: video.videoHeight }
      );
    }
  }, []);

  // ----- Track container size (responsive canvas) -----
  useEffect(() => {
    if (!mounted) return;
    const container = containerRef.current;
    if (!container) return;

    const ro = new ResizeObserver(() => {
      setContainerSize({ w: container.clientWidth, h: container.clientHeight });
    });
    ro.observe(container);
    setContainerSize({ w: container.clientWidth, h: container.clientHeight });
    return () => ro.disconnect();
  }, [mounted]);

  // ----- Geometry: video is object-contain inside the container -----
  // Returns the fitted rect (CSS px) where the video pixels actually render.
  const getFitRect = useCallback(() => {
    if (videoSize.w === 0 || videoSize.h === 0 || containerSize.w === 0 || containerSize.h === 0) {
      return null;
    }
    const scale = Math.min(containerSize.w / videoSize.w, containerSize.h / videoSize.h);
    const dw = videoSize.w * scale;
    const dh = videoSize.h * scale;
    return {
      x: (containerSize.w - dw) / 2,
      y: (containerSize.h - dh) / 2,
      w: dw,
      h: dh,
      scale,
    };
  }, [videoSize, containerSize]);

  // ----- Overlay drawing -----
  const redraw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    // Keep the backing store in sync with the container size
    if (canvas.width !== containerSize.w || canvas.height !== containerSize.h) {
      canvas.width = containerSize.w;
      canvas.height = containerSize.h;
    }

    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const rect = getFitRect();
    if (!rect) return;

    const { x: ox, y: oy, scale } = rect;
    const toScreen = (p: OverlayPoint) => ({ x: ox + p.x * scale, y: oy + p.y * scale });

    // Stop line — red
    if (overlay?.stopLine) {
      const a = toScreen(overlay.stopLine.start);
      const b = toScreen(overlay.stopLine.end);
      ctx.strokeStyle = '#ef4444';
      ctx.lineWidth = Math.max(2, 3 * scale);
      ctx.setLineDash([]);
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.stroke();
      ctx.fillStyle = '#ef4444';
      ctx.font = `bold ${Math.max(11, 14 * scale)}px sans-serif`;
      ctx.fillText('STOP LINE', a.x + 6, a.y - 6);
    }

    // Traffic-light ROI — yellow box
    if (overlay?.lightRoi) {
      const r = overlay.lightRoi;
      ctx.strokeStyle = '#facc15';
      ctx.lineWidth = Math.max(2, 3 * scale);
      ctx.setLineDash([]);
      ctx.strokeRect(ox + r.x * scale, oy + r.y * scale, r.w * scale, r.h * scale);
      ctx.fillStyle = '#facc15';
      ctx.font = `bold ${Math.max(11, 14 * scale)}px sans-serif`;
      ctx.fillText('TRAFFIC LIGHT', ox + r.x * scale + 6, oy + r.y * scale - 6);
    }

    // Draft while dragging — cyan dashed
    if (drawMode && draftStart && draftEnd) {
      const a = toScreen(draftStart);
      const b = toScreen(draftEnd);
      ctx.setLineDash([8, 5]);
      ctx.strokeStyle = '#22d3ee';
      ctx.lineWidth = Math.max(2, 2.5 * scale);
      if (drawMode === 'line') {
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.stroke();
      } else {
        const x = Math.min(a.x, b.x);
        const y = Math.min(a.y, b.y);
        ctx.strokeRect(x, y, Math.abs(b.x - a.x), Math.abs(b.y - a.y));
      }
      ctx.setLineDash([]);
    }
  }, [overlay, drawMode, draftStart, draftEnd, containerSize, getFitRect]);

  useEffect(() => {
    redraw();
  }, [redraw]);

  // ----- Mouse → native video pixel coords -----
  const toImageCoords = useCallback(
    (clientX: number, clientY: number): OverlayPoint | null => {
      const canvas = canvasRef.current;
      const rect = getFitRect();
      if (!canvas || !rect) return null;
      const bounds = canvas.getBoundingClientRect();
      const cx = clientX - bounds.left;
      const cy = clientY - bounds.top;
      const x = Math.round((cx - rect.x) / rect.scale);
      const y = Math.round((cy - rect.y) / rect.scale);
      return {
        x: Math.max(0, Math.min(videoSize.w, x)),
        y: Math.max(0, Math.min(videoSize.h, y)),
      };
    },
    [getFitRect, videoSize]
  );

  const onMouseDown = (e: React.MouseEvent) => {
    if (!drawMode) return;
    const pt = toImageCoords(e.clientX, e.clientY);
    if (!pt) return;
    setDraftStart(pt);
    setDraftEnd(pt);
  };

  const onMouseMove = (e: React.MouseEvent) => {
    if (!drawMode || !draftStart) return;
    const pt = toImageCoords(e.clientX, e.clientY);
    if (pt) setDraftEnd(pt);
  };

  const onMouseUp = () => {
    if (!drawMode || !draftStart || !draftEnd) {
      setDraftStart(null);
      setDraftEnd(null);
      return;
    }
    const start = draftStart;
    const end = draftEnd;
    setDraftStart(null);
    setDraftEnd(null);
    // Ignore tiny accidental drags
    if (Math.abs(end.x - start.x) < 5 && Math.abs(end.y - start.y) < 5) return;
    onDraw?.(start, end);
  };

  const handleRetry = () => {
    setError(null);
    setLoading(true);
    if (hlsRef.current) {
      hlsRef.current.destroy();
      hlsRef.current = null;
    }
    window.location.reload();
  };

  return (
    <div
      ref={containerRef}
      className={`relative overflow-hidden rounded-2xl bg-black select-none ${className}`}
    >
      {loading && (
        <div className="absolute inset-0 z-10 flex flex-col items-center justify-center bg-black/80 backdrop-blur-sm">
          <div className="w-12 h-12 border-4 border-primary-500/30 border-t-primary-500 rounded-full animate-spin" />
          <p className="text-text-secondary text-sm mt-4">Đang kết nối stream...</p>
        </div>
      )}

      {error && (
        <div className="absolute inset-0 z-10 flex flex-col items-center justify-center bg-black/90 backdrop-blur-sm">
          <div className="w-16 h-16 mb-4 rounded-full bg-red-500/10 flex items-center justify-center">
            <svg className="w-8 h-8 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m9-.75a9 9 0 11-18 0 9 9 0 0118 0zm-9 3.75h.008v.008H12v-.008z" />
            </svg>
          </div>
          <p className="text-red-400 text-sm mb-4 text-center px-4">{error}</p>
          <button
            onClick={handleRetry}
            className="px-4 py-2 rounded-xl bg-primary-600/20 border border-primary-500/30 text-primary-400 text-sm font-medium hover:bg-primary-600/30 transition-colors"
          >
            Thử lại
          </button>
        </div>
      )}

      <video
        ref={videoRef}
        className="w-full h-full object-contain"
        autoPlay
        muted
        playsInline
        loop
        onLoadedMetadata={syncVideoSize}
        onResize={syncVideoSize}
      />

      {/* Calibration overlay canvas */}
      <canvas
        ref={canvasRef}
        className={`absolute inset-0 w-full h-full ${
          drawMode ? 'cursor-crosshair' : 'pointer-events-none'
        }`}
        onMouseDown={onMouseDown}
        onMouseMove={onMouseMove}
        onMouseUp={onMouseUp}
        onMouseLeave={() => draftStart && onMouseUp()}
      />
    </div>
  );
}
