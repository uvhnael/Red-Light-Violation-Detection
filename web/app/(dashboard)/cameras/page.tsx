"use client";

import { useEffect, useState } from "react";
import { CameraInfo } from "@/lib/types";
import { getCameras } from "@/lib/api";
import dynamic from "next/dynamic";
import {
  Camera,
  RefreshCw,
  Video,
  MapPin,
  AlertTriangle,
} from "lucide-react";

// hls.js nặng (~500KB) — chỉ tải khi vào trang camera
const VideoPlayer = dynamic(() => import("@/components/VideoPlayer"), {
  ssr: false,
  loading: () => (
    <div className="w-full aspect-video bg-surface-3 rounded-xl flex items-center justify-center">
      <div className="w-10 h-10 border-4 border-indigo-500/30 border-t-indigo-500 rounded-full animate-spin" />
    </div>
  ),
});

export default function CamerasPage() {
  const [cameras, setCameras] = useState<CameraInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedCameraId, setSelectedCameraId] = useState<string | null>(null);

  const loadCameras = () => {
    setLoading(true);
    setError(null);
    getCameras()
      .then((data) => {
        setCameras(data);
        if (data.length > 0 && !selectedCameraId) {
          setSelectedCameraId(data[0].id);
        }
      })
      .catch(() => setError("Unable to connect to edge camera streams"))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    let active = true;
    getCameras()
      .then((data) => {
        if (active) {
          setCameras(data);
          if (data.length > 0) {
            setSelectedCameraId((prev) => prev || data[0].id);
          }
          setLoading(false);
        }
      })
      .catch(() => {
        if (active) {
          setError("Unable to connect to edge camera streams");
          setLoading(false);
        }
      });
    return () => {
      active = false;
    };
  }, []);

  const selectedCamera = cameras.find((c) => c.id === selectedCameraId) ?? cameras[0] ?? null;
  const hlsUrl = selectedCamera ? `/edge-api/cameras/${selectedCamera.id}/stream` : null;
  const snapshotUrl = selectedCamera ? `/edge-api/cameras/${selectedCamera.id}/snapshot` : null;

  if (loading && cameras.length === 0) {
    return (
      <div className="flex items-center justify-center h-[60vh]">
        <div className="flex flex-col items-center gap-3">
          <div className="w-12 h-12 border-4 border-indigo-500/30 border-t-indigo-500 rounded-full animate-spin" />
          <p className="text-xs text-text-muted">Connecting to live camera feeds...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <Video className="w-6 h-6 text-indigo-500" />
            <h1 className="text-2xl font-bold text-text-primary tracking-tight">
              Live Camera Streams
            </h1>
          </div>
          <p className="text-xs text-text-muted mt-1">
            Real-time HLS video feeds streaming from edge nodes.
          </p>
        </div>

        <button
          onClick={loadCameras}
          className="btn-ghost text-xs flex items-center gap-2 border border-border"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${loading ? "animate-spin text-indigo-500" : ""}`} />
          Refresh Feeds
        </button>
      </div>

      {error && (
        <div className="glass-card p-6 border-rose-500/20 text-center space-y-3 max-w-md mx-auto">
          <AlertTriangle className="w-8 h-8 text-rose-500 mx-auto" />
          <p className="text-xs text-rose-500 font-medium">{error}</p>
          <button onClick={loadCameras} className="btn-primary text-xs">
            Retry Connection
          </button>
        </div>
      )}

      {cameras.length === 0 && !error ? (
        <div className="glass-card p-16 text-center space-y-3 text-text-muted">
          <Camera className="w-10 h-10 text-text-muted mx-auto" />
          <p className="text-sm font-semibold text-text-primary">No cameras registered</p>
          <p className="text-xs text-text-muted max-w-sm mx-auto">
            Start an edge node server with an active camera input to stream video.
          </p>
        </div>
      ) : (
        <div className="grid grid-cols-1 xl:grid-cols-4 gap-6">
          {/* Left Camera List */}
          <div className="xl:col-span-1 space-y-3">
            <p className="text-[10px] font-bold text-text-muted uppercase tracking-wider px-1">
              Select Camera Stream ({cameras.length})
            </p>
            {cameras.map((cam) => {
              const isSelected = selectedCameraId === cam.id;
              return (
                <button
                  key={cam.id}
                  onClick={() => setSelectedCameraId(cam.id)}
                  className={`w-full text-left glass-card-hover p-4 border transition-all cursor-pointer ${
                    isSelected
                      ? "border-indigo-500/50 bg-indigo-500/10 shadow-lg shadow-indigo-500/10"
                      : "border-border hover:border-indigo-500/30"
                  }`}
                >
                  <div className="flex items-center justify-between mb-1.5">
                    <span className="font-bold text-sm text-text-primary flex items-center gap-2">
                      <Camera className="w-4 h-4 text-indigo-500" />
                      {cam.name}
                    </span>
                    <span
                      className={`text-[10px] px-2 py-0.5 rounded-full font-semibold ${
                        cam.status === "active" ? "status-confirmed" : "status-pending"
                      }`}
                    >
                      {cam.status}
                    </span>
                  </div>
                  <div className="text-[11px] text-text-muted flex items-center justify-between mt-2">
                    <span className="flex items-center gap-1 font-mono">
                      <MapPin className="w-3 h-3 text-text-muted" />
                      {cam.location || "Intersection #1"}
                    </span>
                    <span className="font-mono text-text-muted">{cam.resolution || "1080p"}</span>
                  </div>
                </button>
              );
            })}
          </div>

          {/* Right Main Video Player & Details */}
          <div className="xl:col-span-3 space-y-6">
            {selectedCamera && (
              <>
                <div className="glass-card overflow-hidden border-indigo-500/20">
                  <div className="px-6 py-4 border-b border-border flex items-center justify-between bg-surface-3/40">
                    <div className="flex items-center gap-3">
                      <div className="w-3 h-3 rounded-full bg-rose-500 animate-pulse shadow-[0_0_8px_#f43f5e]" />
                      <div>
                        <h2 className="text-sm font-bold text-text-primary">
                          {selectedCamera.name} — Live Feed
                        </h2>
                        <p className="text-[11px] font-mono text-text-muted">
                          ID: {selectedCamera.id}
                        </p>
                      </div>
                    </div>

                    <span className="px-2.5 py-1 rounded-full text-[10px] font-bold bg-rose-500/15 text-rose-500 border border-rose-500/30 flex items-center gap-1.5">
                      <span className="w-1.5 h-1.5 rounded-full bg-rose-500 animate-pulse" />
                      HLS LIVE
                    </span>
                  </div>

                  <div className="p-4 bg-surface-3">
                    {hlsUrl ? (
                      <VideoPlayer
                        key={hlsUrl}
                        src={hlsUrl}
                        className="w-full aspect-video rounded-xl shadow-2xl"
                      />
                    ) : (
                      <div className="w-full aspect-video bg-surface-3 flex items-center justify-center text-text-muted text-xs">
                        No stream URL specified
                      </div>
                    )}
                  </div>
                </div>

                {/* Info & Snapshot Cards */}
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-xs">
                  <div className="glass-card p-5 space-y-2">
                    <p className="text-[10px] font-bold text-text-muted uppercase tracking-wider">
                      Stream Metadata
                    </p>
                    <div className="space-y-1.5 text-text-secondary font-mono text-[11px]">
                      <p>
                        <span className="text-text-muted font-sans">Name:</span> {selectedCamera.name}
                      </p>
                      <p>
                        <span className="text-text-muted font-sans">ID:</span> {selectedCamera.id}
                      </p>
                      <p>
                        <span className="text-text-muted font-sans">Resolution:</span> {selectedCamera.resolution}
                      </p>
                      <p>
                        <span className="text-text-muted font-sans">Location:</span> {selectedCamera.location}
                      </p>
                    </div>
                  </div>

                  <div className="glass-card p-5 space-y-2">
                    <p className="text-[10px] font-bold text-text-muted uppercase tracking-wider">
                      Instant Frame Snapshot
                    </p>
                    {snapshotUrl ? (
                      // eslint-disable-next-line @next/next/no-img-element
                      <img
                        src={snapshotUrl}
                        alt="Camera snapshot"
                        className="w-full rounded-xl border border-border object-cover aspect-video shadow-md"
                      />
                    ) : (
                      <div className="w-full aspect-video bg-surface-3 rounded-xl flex items-center justify-center text-text-muted text-xs">
                        No snapshot available
                      </div>
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
