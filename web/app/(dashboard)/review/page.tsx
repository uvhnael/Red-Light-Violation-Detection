"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import Image from "next/image";
import StatusBadge from "@/components/StatusBadge";
import { ViolationResponse } from "@/lib/types";
import { getViolationsPage, updateViolationStatus } from "@/lib/api";
import {
  ShieldCheck,
  CheckCircle2,
  XCircle,
  ChevronLeft,
  ChevronRight,
  ExternalLink,
  Clock,
  Sparkles,
  AlertTriangle,
  FileCheck,
} from "lucide-react";
import { motion, AnimatePresence } from "motion/react";

const BATCH = 50;

export default function ReviewPage() {
  // Buffer pending items loaded lazily in batches — never the whole queue.
  const [violations, setViolations] = useState<ViolationResponse[]>([]);
  const [totalPending, setTotalPending] = useState(0);
  const [loading, setLoading] = useState(true);
  const [fetchingMore, setFetchingMore] = useState(false);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [reviewedCount, setReviewedCount] = useState(0);
  const [actionLoading, setActionLoading] = useState(false);
  const [toastMessage, setToastMessage] = useState<{
    text: string;
    type: "success" | "danger";
  } | null>(null);
  const bufferRef = useRef<ViolationResponse[]>([]);
  const fetchInFlight = useRef(false);

  const current = useMemo(
    () => violations[currentIndex] ?? null,
    [violations, currentIndex]
  );

  useEffect(() => {
    let active = true;
    getViolationsPage({ status: "pending", page: 0, size: BATCH })
      .then((data) => {
        if (!active) return;
        bufferRef.current = data.content;
        setViolations(data.content);
        setTotalPending(data.total_elements);
      })
      .catch(() => {
        if (!active) return;
        bufferRef.current = [];
        setViolations([]);
        setTotalPending(0);
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, []);

  // Nạp thêm batch pending tiếp theo vào buffer. Luôn fetch page 0 vì item
  // đã review tự rớt khỏi danh sách pending phía server; nếu page 0 toàn
  // duplicate (người dùng chỉ lướt xem mà chưa review) thì fallback sang
  // page offset dựa trên độ dài buffer.
  const loadMore = useCallback(async () => {
    if (fetchInFlight.current) return;
    fetchInFlight.current = true;
    setFetchingMore(true);
    try {
      const seen = new Set(bufferRef.current.map((v) => v.id));
      let data = await getViolationsPage({ status: "pending", page: 0, size: BATCH });
      setTotalPending(data.total_elements);
      let fresh = data.content.filter((v) => !seen.has(v.id));
      if (fresh.length === 0 && !data.last) {
        const offsetPage = Math.floor(bufferRef.current.length / BATCH);
        data = await getViolationsPage({
          status: "pending",
          page: offsetPage,
          size: BATCH,
        });
        setTotalPending(data.total_elements);
        fresh = data.content.filter((v) => !seen.has(v.id));
      }
      if (fresh.length > 0) {
        bufferRef.current = [...bufferRef.current, ...fresh];
        setViolations(bufferRef.current);
      }
    } catch {
      // Lỗi mạng — giữ buffer hiện tại, thử lại ở lần trigger sau
    } finally {
      fetchInFlight.current = false;
      setFetchingMore(false);
    }
  }, []);

  // Prefetch khi duyệt gần hết buffer (hoặc buffer vừa cạn) mà vẫn còn
  // pending trên server
  useEffect(() => {
    if (
      !loading &&
      bufferRef.current.length < totalPending &&
      (violations.length === 0 || currentIndex >= violations.length - 5)
    ) {
      void loadMore();
    }
  }, [currentIndex, violations.length, totalPending, loading, loadMore]);

  const handleAction = useCallback(
    async (status: "approved" | "rejected") => {
      if (!current || actionLoading) return;
      setActionLoading(true);
      try {
        await updateViolationStatus(current.id, status);
        const actionText =
          status === "approved"
            ? `Đã duyệt hồ sơ #${current.id} (${current.plate_text || "Không rõ biển số"})`
            : `Đã từ chối hồ sơ #${current.id}`;
        setToastMessage({
          text: actionText,
          type: status === "approved" ? "success" : "danger",
        });

        bufferRef.current = bufferRef.current.filter((_, i) => i !== currentIndex);
        setViolations(bufferRef.current);
        setTotalPending((t) => Math.max(0, t - 1));
        setReviewedCount((c) => c + 1);
        if (currentIndex >= bufferRef.current.length && bufferRef.current.length > 0) {
          setCurrentIndex(bufferRef.current.length - 1);
        }
      } catch {
        setToastMessage({ text: "Lỗi cập nhật trạng thái hồ sơ", type: "danger" });
      } finally {
        setActionLoading(false);
        setTimeout(() => setToastMessage(null), 3000);
      }
    },
    [current, currentIndex, actionLoading]
  );

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (violations.length === 0 || actionLoading) return;
      if (
        document.activeElement?.tagName === "INPUT" ||
        document.activeElement?.tagName === "TEXTAREA"
      ) {
        return;
      }

      if (event.key === "ArrowLeft" || event.key.toLowerCase() === "a") {
        setCurrentIndex((v) => Math.max(0, v - 1));
      }
      if (event.key === "ArrowRight" || event.key.toLowerCase() === "d") {
        setCurrentIndex((v) => Math.min(violations.length - 1, v + 1));
      }
      if (event.key === "Enter" || event.key.toLowerCase() === "v") {
        event.preventDefault();
        void handleAction("approved");
      }
      if (event.key === "Backspace" || event.key.toLowerCase() === "r") {
        event.preventDefault();
        void handleAction("rejected");
      }
    };

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [violations.length, actionLoading, handleAction]);

  if (loading) {
    return (
      <div className="max-w-5xl mx-auto space-y-6">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div className="space-y-2">
            <div className="skeleton h-6 w-64" />
            <div className="skeleton h-3 w-96 max-w-full" />
          </div>
          <div className="skeleton h-11 w-44 rounded-xl" />
        </div>
        <div className="skeleton h-2 w-full rounded-full" />
        <div className="glass-card overflow-hidden grid grid-cols-1 lg:grid-cols-12">
          <div className="lg:col-span-5 p-6 space-y-5 bg-surface-3/30">
            <div className="flex items-center justify-between">
              <div className="space-y-2">
                <div className="skeleton h-3 w-24" />
                <div className="skeleton h-7 w-20" />
              </div>
              <div className="skeleton h-6 w-20 rounded-full" />
            </div>
            <div className="skeleton h-24 w-full rounded-xl" />
            <div className="grid grid-cols-2 gap-3">
              <div className="skeleton h-20 rounded-xl" />
              <div className="skeleton h-20 rounded-xl" />
            </div>
            <div className="skeleton h-24 w-full rounded-xl" />
          </div>
          <div className="lg:col-span-7 p-6 space-y-6">
            <div className="skeleton aspect-video w-full rounded-2xl" />
            <div className="grid grid-cols-2 gap-3">
              <div className="skeleton h-12 rounded-xl" />
              <div className="skeleton h-12 rounded-xl" />
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6 max-w-5xl mx-auto">
      {/* Toast Feedback */}
      {toastMessage && (
        <div
          className={`toast toast-${toastMessage.type} fixed top-20 right-6 z-50 px-4 py-3 rounded-xl border shadow-xl flex items-center gap-3 animate-in fade-in slide-in-from-top-4 duration-200`}
        >
          {toastMessage.type === "success" ? (
            <CheckCircle2 className="w-5 h-5 text-success shrink-0" />
          ) : (
            <XCircle className="w-5 h-5 text-danger shrink-0" />
          )}
          <span className="text-xs font-semibold">{toastMessage.text}</span>
        </div>
      )}

      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <ShieldCheck className="w-6 h-6 text-indigo-500" />
            <h1 className="text-2xl font-bold text-text-primary tracking-tight">
              Trạm kiểm duyệt hồ sơ vi phạm
            </h1>
          </div>
          <p className="text-xs text-text-muted mt-1">
            Kiểm tra và xác thực các phát hiện vi phạm tự động trước khi ban hành biên bản xử phạt.
          </p>
        </div>
        <div className="glass-card px-4 py-2 flex items-center gap-3 border-amber-500/20">
          <span className="w-2.5 h-2.5 rounded-full bg-amber-400 animate-pulse shadow-[0_0_8px_#fbbf24]" />
          <span className="text-sm font-extrabold text-text-primary">
            {totalPending.toLocaleString("vi-VN")}
          </span>
          <span className="text-xs text-text-muted">hồ sơ chờ duyệt</span>
        </div>
      </div>

      {violations.length === 0 && totalPending > 0 ? (
        <div className="flex items-center justify-center h-[40vh]">
          <div className="flex flex-col items-center gap-3">
            <div className="w-10 h-10 border-4 border-indigo-500/30 border-t-indigo-500 rounded-full animate-spin" />
            <p className="text-text-muted text-xs font-medium">Đang tải lô tiếp theo…</p>
          </div>
        </div>
      ) : violations.length === 0 ? (
        <div className="glass-card p-16 text-center space-y-4">
          <div className="w-20 h-20 mx-auto rounded-2xl bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center">
            <FileCheck className="w-10 h-10 text-emerald-500" />
          </div>
          <div>
            <h2 className="text-xl font-extrabold text-text-primary">
              Hàng chờ duyệt đã hoàn thành!
            </h2>
            <p className="text-xs text-text-muted mt-1 max-w-md mx-auto">
              Tất cả các bản ghi vi phạm tự động đã được kiểm tra và xử lý xong.
            </p>
          </div>
          <Link href="/" className="btn-primary text-xs inline-flex">
            Quay lại Bảng điều khiển
          </Link>
        </div>
      ) : current ? (
        <div className="space-y-6">
          {/* Queue Progress Bar */}
          <div className="space-y-1.5">
            <div className="flex items-center justify-between text-xs text-text-muted">
              <span className="font-semibold text-text-secondary">
                Đang duyệt hồ sơ {reviewedCount + currentIndex + 1} /{" "}
                {(totalPending + reviewedCount).toLocaleString("vi-VN")}
                {fetchingMore && (
                  <span className="ml-2 text-indigo-500 animate-pulse">đang tải thêm…</span>
                )}
              </span>
              <span className="text-indigo-500 font-mono font-bold">
                {totalPending + reviewedCount > 0
                  ? Math.round((reviewedCount / (totalPending + reviewedCount)) * 100)
                  : 0}
                % Hoàn thành
              </span>
            </div>
            <div className="w-full h-2 bg-surface-3 rounded-full overflow-hidden border border-border">
              <div
                className="h-full bg-gradient-to-r from-indigo-500 via-violet-500 to-indigo-400 rounded-full transition-all duration-300"
                style={{
                  width: `${
                    totalPending + reviewedCount > 0
                      ? (reviewedCount / (totalPending + reviewedCount)) * 100
                      : 0
                  }%`,
                }}
              />
            </div>
          </div>

          {/* Main Inspection Station Card */}
          <AnimatePresence mode="wait">
            <motion.div
              key={current.id}
              initial={{ opacity: 0, y: 24, scale: 0.985 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: -18, scale: 0.985 }}
              transition={{ duration: 0.28, ease: [0.22, 1, 0.36, 1] }}
              className="glass-card overflow-hidden"
            >
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-0">
              {/* Left Details Column (5 cols) */}
              <div className="lg:col-span-5 p-6 border-b lg:border-b-0 lg:border-r border-border space-y-5 bg-surface-3/30">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-[10px] uppercase font-bold tracking-wider text-text-muted">
                      Hồ sơ vi phạm
                    </p>
                    <h2 className="text-xl font-mono font-bold text-text-primary mt-0.5">
                      #{current.id}
                    </h2>
                  </div>
                  <StatusBadge status={current.status} size="md" />
                </div>

                {/* License Plate Display */}
                <div className="p-4 rounded-xl bg-surface-3 border border-border space-y-2">
                  <p className="text-[10px] font-bold text-text-muted uppercase tracking-wider">
                    Nhận diện biển số xe (ANPR)
                  </p>
                  <div className="flex items-center justify-between">
                    <span className="plate-badge text-lg">
                      {current.plate_text || "KHÔNG RÕ"}
                    </span>
                    <div className="text-right">
                      <p className="text-xs font-bold text-indigo-500">
                        {current.plate_confidence
                          ? `${(current.plate_confidence * 100).toFixed(1)}%`
                          : "Kiểm tra thủ công"}
                      </p>
                      <p className="text-[10px] text-text-muted">Độ tin cậy OCR</p>
                    </div>
                  </div>
                </div>

                {/* Signal State Display */}
                <div className="grid grid-cols-2 gap-3">
                  <div className="p-3.5 rounded-xl bg-surface-3/80 border border-border">
                    <p className="text-[10px] font-bold text-text-muted uppercase tracking-wider mb-1">
                      Trạng thái tín hiệu đèn
                    </p>
                    <p
                      className={`text-sm font-extrabold flex items-center gap-1.5 ${
                        current.light_state === "red"
                          ? "text-rose-500"
                          : "text-amber-500"
                      }`}
                    >
                      <span
                        className={`w-2.5 h-2.5 rounded-full ${
                          current.light_state === "red"
                            ? "bg-rose-500 animate-pulse shadow-[0_0_8px_#f43f5e]"
                            : "bg-amber-400"
                        }`}
                      />
                      {current.light_state === "red" ? "ĐÈN ĐỎ" : current.light_state?.toUpperCase()}
                    </p>
                    <p className="text-[10px] text-text-muted mt-1">
                      Độ tin cậy đèn:{" "}
                      {current.light_confidence
                        ? `${(current.light_confidence * 100).toFixed(0)}%`
                        : "—"}
                    </p>
                  </div>

                  <div className="p-3.5 rounded-xl bg-surface-3/80 border border-border">
                    <p className="text-[10px] font-bold text-text-muted uppercase tracking-wider mb-1">
                      Edge Node ID
                    </p>
                    <p className="text-xs font-mono font-bold text-text-primary">
                      {current.node_id}
                    </p>
                    <p className="text-[10px] text-text-muted mt-1 flex items-center gap-1">
                      <Clock className="w-3 h-3 text-text-muted" />
                      {current.created_at
                        ? new Date(current.created_at).toLocaleTimeString("vi-VN", {
                            hour: "2-digit",
                            minute: "2-digit",
                          })
                        : "—"}
                    </p>
                  </div>
                </div>

                {/* Coordinates & Technical Metadata */}
                <div className="space-y-2 text-xs">
                  <p className="text-[10px] font-bold text-text-muted uppercase tracking-wider">
                    Tọa độ nhận diện
                  </p>
                  <div className="grid grid-cols-2 gap-2 font-mono text-[11px]">
                    <div className="p-2.5 rounded-lg bg-surface-3/60 border border-border">
                      <p className="text-[10px] text-text-muted font-sans">
                        Điểm cắt vạch
                      </p>
                      <p className="text-text-secondary">
                        ({current.crossing_point?.x?.toFixed(1) || 0},{" "}
                        {current.crossing_point?.y?.toFixed(1) || 0})
                      </p>
                    </div>
                    <div className="p-2.5 rounded-lg bg-surface-3/60 border border-border">
                      <p className="text-[10px] text-text-muted font-sans">
                        Chuyển hướng vạch
                      </p>
                      <p className="text-text-secondary">
                        Phía {current.previous_side} → {current.current_side}
                      </p>
                    </div>
                  </div>
                </div>

                {/* Keyboard Shortcuts Help */}
                <div className="p-3 rounded-xl bg-indigo-500/10 border border-indigo-500/20 text-[11px] text-indigo-500 space-y-1">
                  <p className="font-semibold flex items-center gap-1.5 text-xs text-indigo-500">
                    <Sparkles className="w-3.5 h-3.5" /> Phím tắt thao tác
                  </p>
                  <div className="grid grid-cols-2 gap-1 text-[10px] text-indigo-500/90 font-medium">
                    <span>
                      <kbd className="bg-indigo-500/20 px-1 py-0.5 rounded text-indigo-600 dark:text-indigo-300">
                        Enter
                      </kbd>{" "}
                      /{" "}
                      <kbd className="bg-indigo-500/20 px-1 py-0.5 rounded text-indigo-600 dark:text-indigo-300">
                        V
                      </kbd>{" "}
                      Duyệt
                    </span>
                    <span>
                      <kbd className="bg-indigo-500/20 px-1 py-0.5 rounded text-indigo-600 dark:text-indigo-300">
                        Backspace
                      </kbd>{" "}
                      /{" "}
                      <kbd className="bg-indigo-500/20 px-1 py-0.5 rounded text-indigo-600 dark:text-indigo-300">
                        R
                      </kbd>{" "}
                      Từ chối
                    </span>
                    <span>
                      <kbd className="bg-indigo-500/20 px-1 py-0.5 rounded text-indigo-600 dark:text-indigo-300">
                        ←
                      </kbd>{" "}
                      /{" "}
                      <kbd className="bg-indigo-500/20 px-1 py-0.5 rounded text-indigo-600 dark:text-indigo-300">
                        A
                      </kbd>{" "}
                      Trước
                    </span>
                    <span>
                      <kbd className="bg-indigo-500/20 px-1 py-0.5 rounded text-indigo-600 dark:text-indigo-300">
                        →
                      </kbd>{" "}
                      /{" "}
                      <kbd className="bg-indigo-500/20 px-1 py-0.5 rounded text-indigo-600 dark:text-indigo-300">
                        D
                      </kbd>{" "}
                      Sau
                    </span>
                  </div>
                </div>
              </div>

              {/* Right Media & Decision Column (7 cols) */}
              <div className="lg:col-span-7 p-6 flex flex-col justify-between space-y-6">
                {/* Media Preview Box */}
                <div className="space-y-2">
                  <div className="flex items-center justify-between text-xs text-text-muted">
                    <span className="font-bold text-text-primary uppercase tracking-wider text-[11px]">
                      Bằng chứng ghi hình vi phạm
                    </span>
                    <Link
                      href={`/violations/${current.id}`}
                      className="text-indigo-500 hover:text-indigo-600 font-medium flex items-center gap-1"
                    >
                      Xem chi tiết <ExternalLink className="w-3 h-3" />
                    </Link>
                  </div>

                  <div className="relative aspect-video w-full rounded-2xl overflow-hidden border border-border bg-surface-3 flex items-center justify-center shadow-xl">
                    {current.media_url ? (
                      current.media_url.match(/\.(mp4|webm|ogg)(\?|$)/i) ? (
                        <video
                          src={current.media_url}
                          controls
                          autoPlay
                          loop
                          muted
                          playsInline
                          className="w-full h-full object-contain"
                        />
                      ) : (
                        <Image
                          src={current.media_url}
                          alt={`Bằng chứng vi phạm ${current.event_id}`}
                          fill
                          sizes="(max-width: 1024px) 100vw, 60vw"
                          className="object-contain"
                          // Ảnh duyệt là nội dung chính (LCP) — tải ngay, không lazy
                          priority
                          unoptimized
                        />
                      )
                    ) : (
                      <div className="text-center py-12 text-text-muted space-y-2">
                        <AlertTriangle className="w-8 h-8 mx-auto text-text-muted" />
                        <p className="text-xs">Không có hình ảnh bằng chứng</p>
                      </div>
                    )}
                  </div>
                </div>

                {/* Primary Decision Action Buttons */}
                <div className="space-y-3">
                  <div className="grid grid-cols-2 gap-3">
                    <button
                      onClick={() => handleAction("approved")}
                      disabled={actionLoading}
                      className="btn-success py-3.5 text-sm font-bold flex items-center justify-center gap-2 rounded-xl disabled:opacity-50"
                    >
                      <CheckCircle2 className="w-5 h-5" />
                      Duyệt &amp; Lập biên bản
                    </button>
                    <button
                      onClick={() => handleAction("rejected")}
                      disabled={actionLoading}
                      className="btn-danger py-3.5 text-sm font-bold flex items-center justify-center gap-2 rounded-xl disabled:opacity-50"
                    >
                      <XCircle className="w-5 h-5" />
                      Từ chối (Báo động giả)
                    </button>
                  </div>
                </div>
              </div>
            </div>
            </motion.div>
          </AnimatePresence>

          {/* Bottom Pagination controls */}
          <div className="flex items-center justify-between">
            <button
              onClick={() => setCurrentIndex((v) => Math.max(0, v - 1))}
              disabled={currentIndex === 0}
              className="btn-secondary btn-sm flex items-center gap-1.5 disabled:opacity-30"
            >
              <ChevronLeft className="w-4 h-4" /> Hồ sơ trước
            </button>
            <span className="text-xs text-text-muted font-mono">
              Hồ sơ {currentIndex + 1} / {violations.length}
            </span>
            <button
              onClick={() =>
                setCurrentIndex((v) => Math.min(violations.length - 1, v + 1))
              }
              disabled={currentIndex >= violations.length - 1}
              className="btn-secondary btn-sm flex items-center gap-1.5 disabled:opacity-30"
            >
              Hồ sơ tiếp theo <ChevronRight className="w-4 h-4" />
            </button>
          </div>
        </div>
      ) : null}
    </div>
  );
}
