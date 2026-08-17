'use client';

import { useEffect, useState } from 'react';
import { CameraInfo } from '@/lib/types';
import { getCameras } from '@/lib/api';
import VideoPlayer from '@/components/VideoPlayer';

export default function CamerasPage() {
  const [cameras, setCameras] = useState<CameraInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedCameraId, setSelectedCameraId] = useState<string | null>(null);

  useEffect(() => {
    getCameras()
      .then((data) => {
        setCameras(data);
        if (data.length > 0) {
          setSelectedCameraId(data[0].id);
        }
      })
      .catch(() => setError('Không thể kết nối tới camera'))
      .finally(() => setLoading(false));
  }, []);

  const selectedCamera = cameras.find((c) => c.id === selectedCameraId) ?? null;
  const hlsUrl = selectedCamera ? `/edge-api/cameras/${selectedCamera.id}/stream` : null;
  const snapshotUrl = selectedCamera ? `/edge-api/cameras/${selectedCamera.id}/snapshot` : null;

  if (loading) {
    return (
      <div className="flex items-center justify-center h-[60vh]">
        <div className="w-12 h-12 border-4 border-primary-500/30 border-t-primary-500 rounded-full animate-spin" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-[60vh]">
        <div className="glass-card p-8 text-center">
          <p className="text-red-400">{error}</p>
          <button onClick={() => window.location.reload()} className="btn-primary mt-4 text-sm">
            Thử lại
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold gradient-text">Camera</h1>
        <p className="text-text-secondary text-sm mt-1">Xem trực tiếp luồng video từ các edge node.</p>
      </div>

      {cameras.length === 0 ? (
        <div className="glass-card p-16 text-center text-text-muted">
          Chưa có camera nào. Hãy khởi động edge node với chế độ fake-camera.
        </div>
      ) : (
        <div className="grid grid-cols-1 xl:grid-cols-4 gap-6">
          <div className="xl:col-span-1 space-y-3">
            {cameras.map((cam) => (
              <button
                key={cam.id}
                onClick={() => setSelectedCameraId(cam.id)}
                className={`w-full text-left glass-card px-4 py-3 transition-colors ${
                  selectedCameraId === cam.id ? 'border-primary-500/50' : ''
                }`}
              >
                <div className="flex items-center justify-between mb-1">
                  <span className="font-medium text-sm text-text-primary">{cam.name}</span>
                  <span className={`text-xs px-2 py-0.5 rounded-full ${
                    cam.status === 'active' ? 'status-confirmed' : 'status-pending'
                  }`}>
                    {cam.status}
                  </span>
                </div>
                <p className="text-xs text-text-muted">{cam.resolution} — {cam.location}</p>
              </button>
            ))}
          </div>

          <div className="xl:col-span-3 space-y-4">
            {selectedCamera && (
              <>
                <div className="glass-card overflow-hidden">
                  {hlsUrl ? (
                    <VideoPlayer
                      key={hlsUrl}
                      src={hlsUrl}
                      className="w-full aspect-video"
                    />
                  ) : (
                    <div className="w-full aspect-video bg-black/60 flex items-center justify-center">
                      <p className="text-text-muted text-sm">Không có stream URL</p>
                    </div>
                  )}
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div className="glass-card p-4">
                    <p className="text-xs text-text-muted mb-1">Thông tin camera</p>
                    <p className="text-sm text-text-primary font-medium">{selectedCamera.name}</p>
                    <p className="text-xs text-text-muted mt-1">ID: {selectedCamera.id}</p>
                    <p className="text-xs text-text-muted">Độ phân giải: {selectedCamera.resolution}</p>
                    <p className="text-xs text-text-muted">Vị trí: {selectedCamera.location}</p>
                  </div>
                  <div className="glass-card p-4">
                    <p className="text-xs text-text-muted mb-1">Ảnh chụp nhanh</p>
                    {snapshotUrl && (
                      // eslint-disable-next-line @next/next/no-img-element
                      <img
                        src={snapshotUrl}
                        alt="Camera snapshot"
                        className="w-full rounded-xl border border-border object-cover aspect-video"
                      />
                    )}
                  </div>
                </div>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
