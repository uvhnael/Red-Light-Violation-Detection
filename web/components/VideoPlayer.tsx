'use client';

import { useEffect, useRef, useState } from 'react';
import Hls from 'hls.js';

interface VideoPlayerProps {
  src: string;
  className?: string;
}

export default function VideoPlayer({ src, className = '' }: VideoPlayerProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const hlsRef = useRef<Hls | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

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
    <div className={`relative overflow-hidden rounded-2xl bg-black ${className}`}>
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
      />
    </div>
  );
}