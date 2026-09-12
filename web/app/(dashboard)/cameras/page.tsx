"use client";

import { useEffect, useState } from "react";
import { CameraInfo } from "@/lib/types";
import { getCameras } from "@/lib/api";
import dynamic from "next/dynamic";
import {
  Camera,
  RefreshCw,
  Video,
  AlertTriangle,
  LayoutGrid,
  Grid2x2,
  Grid3x3,
  LayoutDashboard,
} from "lucide-react";
import { motion, AnimatePresence } from "motion/react";

// hls.js nặng (~500KB) — chỉ tải khi vào trang camera
const VideoPlayer = dynamic(() => import("@/components/VideoPlayer"), {
  ssr: false,
  loading: () => (
    <div className="w-full aspect-video bg-surface-3 rounded-xl flex items-center justify-center">
      <div className="w-10 h-10 border-4 border-indigo-500/30 border-t-indigo-500 rounded-full animate-spin" />
    </div>
  ),
});

/** Bố cục lưới: 1 = toàn màn hình, 4 = 2x2, 9 = 3x3, 16 = 4x4. */
type GridLayout = 1 | 4 | 9 | 16;

const LAYOUT_OPTIONS: { value: GridLayout; label: string; icon: typeof LayoutGrid; gridClass: string }[] = [
  { value: 1, label: "1 cam", icon: LayoutDashboard, gridClass: "grid-cols-1" },
  { value: 4, label: "4 cam", icon: Grid2x2, gridClass: "grid-cols-1 sm:grid-cols-2" },
  { value: 9, label: "9 cam", icon: Grid3x3, gridClass: "grid-cols-2 sm:grid-cols-3" },
  { value: 16, label: "16 cam", icon: LayoutGrid, gridClass: "grid-cols-2 sm:grid-cols-3 xl:grid-cols-4" },
];

export default function CamerasPage() {
  const [cameras, setCameras] = useState<CameraInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedCameraId, setSelectedCameraId] = useState<string | null>(null);
  const [layout, setLayout] = useState<GridLayout>(1);

  const loadCameras = () => {
    setLoading(true);
    setError(null);
    getCameras()
      .then((data) => {
        setCameras(data);
        if (data.length > 0) {
          setSelectedCameraId((prev) => (prev && data.some((c) => c.id === prev) ? prev : data[0].id));
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

  // Camera hiển thị trong lưới: layout 1 → chỉ cam đang chọn; N → N cam đầu
  // (hoặc tất cả nếu ít hơn N), đảm bảo cam đang chọn luôn có mặt.
  const gridCameras =
    layout === 1
      ? cameras.filter((c) => c.id === selectedCameraId)
      : (() => {
          const picked = cameras.slice(0, layout);
          if (selectedCameraId && !picked.some((c) => c.id === selectedCameraId)) {
            const selected = cameras.find((c) => c.id === selectedCameraId);
            if (selected) picked[0] = selected;
          }
          return picked;
        })();

  const activeLayout = LAYOUT_OPTIONS.find((o) => o.value === layout) ?? LAYOUT_OPTIONS[0];

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
              Luồng Camera trực tiếp
            </h1>
          </div>
          <p className="text-xs text-text-muted mt-1">
            Luồng truyền hình trực tiếp chuẩn HLS thời gian thực từ các Edge Node.
          </p>
        </div>

        <button
          onClick={loadCameras}
          className="btn-secondary btn-sm flex items-center gap-2"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${loading ? "animate-spin text-indigo-500" : ""}`} />
          Làm mới
        </button>
      </div>

      {error && (
        <div className="glass-card p-6 border-rose-500/20 text-center space-y-3 max-w-md mx-auto">
          <AlertTriangle className="w-8 h-8 text-rose-500 mx-auto" />
          <p className="text-xs text-rose-500 font-medium">{error}</p>
          <button onClick={loadCameras} className="btn-primary text-xs">
            Thử kết nối lại
          </button>
        </div>
      )}

      {cameras.length === 0 && !error ? (
        <div className="glass-card p-16 text-center space-y-3 text-text-muted">
          <Camera className="w-10 h-10 text-text-muted mx-auto" />
          <p className="text-sm font-semibold text-text-primary">Chưa có camera nào được đăng ký</p>
          <p className="text-xs text-text-muted max-w-sm mx-auto">
            Hãy khởi động Edge Node với nguồn camera hợp lệ để bắt đầu truyền phát video.
          </p>
        </div>
      ) : (
        <div className="grid grid-cols-1 xl:grid-cols-4 gap-6">
          {/* Left: Layout picker + Camera list (tối giản: tên + trạng thái) */}
          <div className="xl:col-span-1 space-y-4">
            {/* Layout picker */}
            <div className="glass-card p-3">
              <p className="text-[10px] font-bold text-text-muted uppercase tracking-wider px-1 mb-2">
                Cách xem camera
              </p>
              <div className="grid grid-cols-4 gap-1.5">
                {LAYOUT_OPTIONS.map((opt) => {
                  const Icon = opt.icon;
                  const isActive = layout === opt.value;
                  return (
                    <button
                      key={opt.value}
                      onClick={() => setLayout(opt.value)}
                      disabled={opt.value > 1 && cameras.length === 0}
                      title={opt.value === 1 ? "Xem 1 camera toàn màn hình" : `Xem lưới ${opt.value} camera`}
                        className={`flex flex-col items-center justify-center gap-1 py-2.5 rounded-xl border text-[10px] font-semibold transition-all cursor-pointer ${
                        isActive
                          ? "border-indigo-500/60 bg-indigo-500/10 text-indigo-500 shadow-sm"
                          : "border-border text-text-muted hover:border-indigo-500/30 hover:text-text-secondary"
                      }`}
                    >
                      <Icon className="w-4 h-4" />
                      {opt.value}
                    </button>
                  );
                })}
              </div>
              <p className="text-[10px] text-text-muted mt-2 px-1">
                {layout === 1
                  ? "Đang xem 1 camera toàn màn hình"
                  : `Lưới ${activeLayout.label} — ${Math.min(cameras.length, layout)} / ${cameras.length} camera`}
              </p>
            </div>

            {/* Camera list */}
            <div className="space-y-2">
              <p className="text-[10px] font-bold text-text-muted uppercase tracking-wider px-1">
                Danh sách Camera ({cameras.length})
              </p>
              {cameras.map((cam, i) => {
                const isSelected = selectedCameraId === cam.id;
                return (
                  <motion.button
                    key={cam.id}
                    initial={{ opacity: 0, x: -14 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: Math.min(i * 0.07, 0.3), duration: 0.35, ease: "easeOut" }}
                    onClick={() => setSelectedCameraId(cam.id)}
                    className={`w-full text-left glass-card-hover p-3.5 border transition-all cursor-pointer ${
                      isSelected
                        ? "border-indigo-500/50 bg-indigo-500/10 shadow-lg shadow-indigo-500/10"
                        : "border-border hover:border-indigo-500/30"
                    }`}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <span className="font-semibold text-sm text-text-primary flex items-center gap-2 min-w-0">
                        <Camera className="w-4 h-4 text-indigo-500 shrink-0" />
                        <span className="truncate">{cam.name}</span>
                      </span>
                      <span
                        className={`text-[10px] px-2 py-0.5 rounded-full font-semibold shrink-0 ${
                          cam.status === "active" ? "status-confirmed" : "status-pending"
                        }`}
                      >
                        {cam.status === "active" ? "Trực tuyến" : cam.status}
                      </span>
                    </div>
                  </motion.button>
                );
              })}
            </div>
          </div>

          {/* Right: Camera grid theo layout đã chọn */}
          <div className="xl:col-span-3">
            <AnimatePresence mode="wait">
              <motion.div
                key={`${layout}-${gridCameras.map((c) => c.id).join(",")}`}
                initial={{ opacity: 0, scale: 0.985 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.25, ease: "easeOut" }}
                className={`grid gap-3 ${activeLayout.gridClass}`}
              >
                {gridCameras.map((cam) => (
                  <div key={cam.id} className="glass-card overflow-hidden">
                    <div className="px-3 py-2 border-b border-border flex items-center justify-between bg-surface-3/40">
                      <span className="text-xs font-bold text-text-primary truncate flex items-center gap-1.5">
                        <span className="w-1.5 h-1.5 rounded-full bg-rose-500 animate-pulse shrink-0" />
                        {cam.name}
                      </span>
                      <span className="text-[10px] text-text-muted font-mono shrink-0">{cam.id}</span>
                    </div>
                    <div className="p-2 bg-surface-3">
                      <VideoPlayer
                        key={`${cam.id}-${layout}`}
                        src={`/edge-api/cameras/${cam.id}/stream`}
                        className="w-full aspect-video rounded-lg"
                      />
                    </div>
                  </div>
                ))}
              </motion.div>
            </AnimatePresence>
          </div>
        </div>
      )}
    </div>
  );
}
