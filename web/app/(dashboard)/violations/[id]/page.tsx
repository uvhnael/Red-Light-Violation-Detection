"use client";

import { useEffect, useState } from "react";
import Image from "next/image";
import { useParams, useRouter } from "next/navigation";
import Link from "next/link";
import StatusBadge from "@/components/StatusBadge";
import { useToast, ConfirmDialog } from "@/components/Toast";
import { ViolationResponse } from "@/lib/types";
import { getViolation, updateViolationStatus, deleteViolation } from "@/lib/api";
import {
  ArrowLeft,
  CheckCircle2,
  XCircle,
  Trash2,
  AlertTriangle,
} from "lucide-react";
import { motion } from "motion/react";

export default function ViolationDetailPage() {
  const params = useParams();
  const router = useRouter();
  const id = Number(params.id);

  const [violation, setViolation] = useState<ViolationResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [actionLoading, setActionLoading] = useState(false);
  const [deleteOpen, setDeleteOpen] = useState(false);
  const [deleteBusy, setDeleteBusy] = useState(false);
  const { show } = useToast();

  useEffect(() => {
    if (!id) return;
    getViolation(id)
      .then(setViolation)
      .catch(() => setViolation(null))
      .finally(() => setLoading(false));
  }, [id]);

  const handleStatusUpdate = async (status: string) => {
    if (!violation) return;
    setActionLoading(true);
    try {
      const updated = await updateViolationStatus(violation.id, status);
      setViolation(updated);
      show("Đã cập nhật trạng thái hồ sơ.", "success");
    } catch {
      show("Không cập nhật được trạng thái. Vui lòng thử lại.", "danger");
    } finally {
      setActionLoading(false);
    }
  };

  const handleDeleteConfirm = async () => {
    if (!violation) return;
    setDeleteBusy(true);
    try {
      await deleteViolation(violation.id);
      show("Đã xóa hồ sơ vĩnh viễn.", "success");
      router.push("/violations");
    } catch {
      show("Không xóa được hồ sơ. Vui lòng thử lại.", "danger");
      setDeleteBusy(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-[60vh]">
        <div className="w-12 h-12 border-4 border-indigo-500/30 border-t-indigo-500 rounded-full animate-spin" />
      </div>
    );
  }

  if (!violation) {
    return (
      <div className="flex items-center justify-center h-[60vh]">
        <div className="glass-card p-10 text-center space-y-4 max-w-md">
          <AlertTriangle className="w-12 h-12 text-amber-500 mx-auto" />
          <h2 className="text-lg font-bold text-text-primary">Không tìm thấy hồ sơ</h2>
          <p className="text-xs text-text-muted">Hồ sơ vi phạm #{id} không tồn tại trong cơ sở dữ liệu trung tâm.</p>
          <Link href="/violations" className="btn-primary text-xs inline-flex">
            ← Quay lại danh sách vi phạm
          </Link>
        </div>
      </div>
    );
  }

  const mediaUrl = violation.media_url
    ? `/api/v1/violations/${violation.event_id}/media/blob`
    : null;
  // Central quyết định content-type theo object MinIO; hiện tại edge chỉ
  // upload JPEG (bằng chứng frame), nên hiển thị ảnh là đường chính.
  const mediaIsVideo = Boolean(mediaUrl && /\.mp4(\?|$)/i.test(violation.media_url ?? ""));

  return (
    <motion.div
      className="space-y-6 max-w-5xl"
      initial={{ opacity: 0, y: 14 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
    >
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <Link
            href="/violations"
            className="p-2.5 rounded-xl bg-surface-3 border border-border text-text-muted hover:text-text-primary transition-colors"
          >
            <ArrowLeft className="w-4 h-4" />
          </Link>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-xl font-bold text-text-primary">Hồ sơ vi phạm #{violation.id}</h1>
              <StatusBadge status={violation.status} size="md" />
            </div>
            <p className="text-xs text-text-muted font-mono mt-0.5">{violation.event_id}</p>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Info & Evidence (2 Cols) */}
        <div className="lg:col-span-2 space-y-6">
          {/* ANPR Plate & Signal Overview */}
          <div className="glass-card p-6 space-y-6">
            <h2 className="text-xs font-bold text-text-muted uppercase tracking-wider">
              Tổng quan nhận diện tự động
            </h2>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              {/* Plate Card */}
              <div className="p-4 rounded-2xl bg-surface-3/60 border border-border space-y-2">
                <p className="text-[10px] font-bold text-text-muted uppercase tracking-wider">
                  Biển số xe nhận diện (ANPR)
                </p>
                <div className="flex items-center justify-between">
                  <span className="plate-badge text-xl">
                    {violation.plate_text || "KHÔNG ĐỌC ĐƯỢC"}
                  </span>
                  <div className="text-right">
                    <p className="text-xs font-bold text-indigo-500">
                      {violation.plate_confidence
                        ? `${(violation.plate_confidence * 100).toFixed(1)}%`
                        : "—"}
                    </p>
                    <p className="text-[10px] text-text-muted">Độ tin cậy OCR</p>
                  </div>
                </div>
              </div>

              {/* Light State Card */}
              <div className="p-4 rounded-2xl bg-surface-3/60 border border-border space-y-2">
                <p className="text-[10px] font-bold text-text-muted uppercase tracking-wider">
                  Trạng thái tín hiệu đèn
                </p>
                <div className="flex items-center justify-between">
                  <span
                    className={`inline-flex items-center gap-2 text-sm font-extrabold px-3 py-1.5 rounded-full ${
                      violation.light_state === "red"
                        ? "bg-rose-500/15 text-rose-500 border border-rose-500/30"
                        : "bg-amber-500/15 text-amber-500 border border-amber-500/30"
                    }`}
                  >
                    <span
                      className={`w-2.5 h-2.5 rounded-full ${
                        violation.light_state === "red"
                          ? "bg-rose-500 animate-pulse shadow-[0_0_8px_#f43f5e]"
                          : "bg-amber-400"
                      }`}
                    />
                    {violation.light_state === "red" ? "ĐÈN ĐỎ" : violation.light_state?.toUpperCase()}
                  </span>
                  <div className="text-right">
                    <p className="text-xs font-bold text-text-primary">
                      {violation.light_confidence
                        ? `${(violation.light_confidence * 100).toFixed(1)}%`
                        : "—"}
                    </p>
                    <p className="text-[10px] text-text-muted">Độ tin cậy đèn</p>
                  </div>
                </div>
              </div>
            </div>

            {/* Coordinates Grid */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
              <div className="p-3 rounded-xl bg-surface-3/60 border border-border">
                <p className="text-[10px] text-text-muted">Node ID</p>
                <p className="font-mono font-semibold text-text-primary mt-0.5">{violation.node_id}</p>
              </div>
              <div className="p-3 rounded-xl bg-surface-3/60 border border-border">
                <p className="text-[10px] text-text-muted">Mã theo dõi (Track)</p>
                <p className="font-mono font-semibold text-text-primary mt-0.5">#{violation.track_id}</p>
              </div>
              <div className="p-3 rounded-xl bg-surface-3/60 border border-border">
                <p className="text-[10px] text-text-muted">Tọa độ cắt vạch</p>
                <p className="font-mono font-semibold text-text-primary mt-0.5">
                  ({violation.crossing_point?.x?.toFixed(1) || 0}, {violation.crossing_point?.y?.toFixed(1) || 0})
                </p>
              </div>
              <div className="p-3 rounded-xl bg-surface-3/60 border border-border">
                <p className="text-[10px] text-text-muted">Chuyển hướng vạch</p>
                <p className="font-mono font-semibold text-text-primary mt-0.5">
                  Phía {violation.previous_side} → {violation.current_side}
                </p>
              </div>
            </div>
          </div>

          {/* Evidence Media Preview */}
          {mediaUrl && (
            <div className="glass-card p-6 space-y-4">
              <h2 className="text-xs font-bold text-text-muted uppercase tracking-wider">
                Bằng chứng ghi hình vi phạm
              </h2>
              <div className="relative aspect-video w-full rounded-2xl overflow-hidden border border-border bg-surface-3 shadow-2xl">
                {mediaIsVideo ? (
                  <video src={mediaUrl} controls autoPlay loop muted playsInline className="w-full h-full object-contain" />
                ) : (
                  <Image
                    src={mediaUrl}
                    alt={`Bằng chứng vi phạm ${violation.event_id}`}
                    fill
                    sizes="(max-width: 1024px) 100vw, 60vw"
                    className="object-contain"
                    // Ảnh bằng chứng là nội dung chính (LCP) — tải ngay, không lazy
                    priority
                    unoptimized
                  />
                )}
              </div>
            </div>
          )}
        </div>

        {/* Action Sidebar (1 Col) */}
        <div className="space-y-6">
          {/* Operator Decision Actions */}
          <div className="glass-card p-6 space-y-4">
            <h2 className="text-xs font-bold text-text-muted uppercase tracking-wider">
              Quyết định xử lý
            </h2>
            <p className="text-xs text-text-muted leading-relaxed">
              Kiểm tra hình ảnh bằng chứng và biển số xe trước khi phê duyệt lập biên bản xử phạt.
            </p>

            <div className="space-y-3">
              <button
                onClick={() => handleStatusUpdate("approved")}
                disabled={actionLoading || violation.status === "approved"}
                className="btn-success w-full py-2.5 text-xs font-bold disabled:opacity-40"
              >
                <CheckCircle2 className="w-4 h-4" />
                Duyệt &amp; Lập biên bản
              </button>

              <button
                onClick={() => handleStatusUpdate("rejected")}
                disabled={actionLoading || violation.status === "rejected"}
                className="btn-danger w-full py-2.5 text-xs font-bold disabled:opacity-40"
              >
                <XCircle className="w-4 h-4" />
                Từ chối (Báo động giả)
              </button>

              <button
                onClick={() => handleStatusUpdate("pending")}
                disabled={actionLoading || violation.status === "pending"}
                className="btn-secondary w-full py-2 text-xs disabled:opacity-40"
              >
                Đặt lại thành Chờ duyệt
              </button>
            </div>
          </div>

          {/* Record Timestamps */}
          <div className="glass-card p-6 space-y-3 text-xs">
            <h2 className="text-xs font-bold text-text-muted uppercase tracking-wider">
              Nhật ký kiểm toán
            </h2>
            <div className="space-y-2">
              <div>
                <p className="text-text-muted text-[10px]">Thời điểm ghi nhận</p>
                <p className="text-text-primary font-medium mt-0.5">
                  {violation.created_at ? new Date(violation.created_at).toLocaleString("vi-VN") : "—"}
                </p>
              </div>
              <div>
                <p className="text-text-muted text-[10px]">Cập nhật lần cuối</p>
                <p className="text-text-primary font-medium mt-0.5">
                  {violation.updated_at ? new Date(violation.updated_at).toLocaleString("vi-VN") : "—"}
                </p>
              </div>
            </div>
          </div>

          {/* Danger Zone */}
          <div className="glass-card p-6 border-rose-500/20 space-y-3">
            <h2 className="text-xs font-bold text-rose-500 uppercase tracking-wider">
              Danger Zone
            </h2>
            <button
              onClick={() => setDeleteOpen(true)}
              className="text-xs text-rose-500 hover:text-rose-600 font-semibold flex items-center gap-2 transition-colors cursor-pointer"
            >
              <Trash2 className="w-4 h-4" />
              Xóa hồ sơ vĩnh viễn
            </button>
          </div>
        </div>
      </div>

      <ConfirmDialog
        open={deleteOpen}
        title="Xóa hồ sơ vĩnh viễn?"
        message={`Hồ sơ #${violation.id} (${violation.plate_text || "không đọc được biển số"}) sẽ bị xóa khỏi cơ sở dữ liệu và không thể khôi phục.`}
        confirmLabel="Xóa vĩnh viễn"
        busy={deleteBusy}
        onConfirm={handleDeleteConfirm}
        onCancel={() => setDeleteOpen(false)}
      />
    </motion.div>
  );
}
