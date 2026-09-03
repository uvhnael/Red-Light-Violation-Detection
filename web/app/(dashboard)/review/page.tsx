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
            ? `Approved Ticket #${current.id} (${current.plate_text || "Unreadable"})`
            : `Rejected Ticket #${current.id}`;
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
        setToastMessage({ text: "Error updating violation status", type: "danger" });
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
      <div className="flex items-center justify-center h-[60vh]">
        <div className="flex flex-col items-center gap-3">
          <div className="w-12 h-12 border-4 border-indigo-500/30 border-t-indigo-500 rounded-full animate-spin" />
          <p className="text-text-muted text-xs font-medium">
            Loading pending review queue...
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6 max-w-5xl mx-auto">
      {/* Toast Feedback */}
      {toastMessage && (
        <div
          className={`fixed top-20 right-6 z-50 px-4 py-3 rounded-xl border shadow-xl flex items-center gap-3 animate-in fade-in slide-in-from-top-4 duration-200 ${
            toastMessage.type === "success"
              ? "bg-emerald-950/90 text-emerald-300 border-emerald-500/30 dark:bg-emerald-950/90"
              : "bg-rose-950/90 text-rose-300 border-rose-500/30 dark:bg-rose-950/90"
          }`}
        >
          {toastMessage.type === "success" ? (
            <CheckCircle2 className="w-5 h-5 text-emerald-400" />
          ) : (
            <XCircle className="w-5 h-5 text-rose-400" />
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
              Human-in-the-Loop Review Station
            </h1>
          </div>
          <p className="text-xs text-text-muted mt-1">
            Review low-confidence automated violation detections before fine generation.
          </p>
        </div>
        <div className="glass-card px-4 py-2 flex items-center gap-3 border-amber-500/20">
          <span className="w-2.5 h-2.5 rounded-full bg-amber-400 animate-pulse shadow-[0_0_8px_#fbbf24]" />
          <span className="text-sm font-extrabold text-text-primary">
            {totalPending.toLocaleString("vi-VN")}
          </span>
          <span className="text-xs text-text-muted">cases remaining</span>
        </div>
      </div>

      {violations.length === 0 && totalPending > 0 ? (
        <div className="flex items-center justify-center h-[40vh]">
          <div className="flex flex-col items-center gap-3">
            <div className="w-10 h-10 border-4 border-indigo-500/30 border-t-indigo-500 rounded-full animate-spin" />
            <p className="text-text-muted text-xs font-medium">Loading next batch…</p>
          </div>
        </div>
      ) : violations.length === 0 ? (
        <div className="glass-card p-16 text-center space-y-4">
          <div className="w-20 h-20 mx-auto rounded-2xl bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center">
            <FileCheck className="w-10 h-10 text-emerald-500" />
          </div>
          <div>
            <h2 className="text-xl font-extrabold text-text-primary">
              Review Queue Completed!
            </h2>
            <p className="text-xs text-text-muted mt-1 max-w-md mx-auto">
              All automated detection records have been verified and processed by the operator.
            </p>
          </div>
          <Link href="/" className="btn-primary text-xs inline-flex">
            Return to Dashboard
          </Link>
        </div>
      ) : current ? (
        <div className="space-y-6">
          {/* Queue Progress Bar */}
          <div className="space-y-1.5">
            <div className="flex items-center justify-between text-xs text-text-muted">
              <span className="font-semibold text-text-secondary">
                Reviewing Case {reviewedCount + currentIndex + 1} of{" "}
                {(totalPending + reviewedCount).toLocaleString("vi-VN")}
                {fetchingMore && (
                  <span className="ml-2 text-indigo-500 animate-pulse">loading more…</span>
                )}
              </span>
              <span className="text-indigo-500 font-mono font-bold">
                {totalPending + reviewedCount > 0
                  ? Math.round((reviewedCount / (totalPending + reviewedCount)) * 100)
                  : 0}
                % Completed
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
          <div className="glass-card overflow-hidden border-indigo-500/20">
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-0">
              {/* Left Details Column (5 cols) */}
              <div className="lg:col-span-5 p-6 border-b lg:border-b-0 lg:border-r border-border space-y-5 bg-surface-3/30">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-[10px] uppercase font-bold tracking-wider text-text-muted">
                      Violation Record
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
                    ANPR Plate Recognition
                  </p>
                  <div className="flex items-center justify-between">
                    <span className="plate-badge text-lg">
                      {current.plate_text || "NO-READ"}
                    </span>
                    <div className="text-right">
                      <p className="text-xs font-bold text-indigo-500">
                        {current.plate_confidence
                          ? `${(current.plate_confidence * 100).toFixed(1)}%`
                          : "Manual Check"}
                      </p>
                      <p className="text-[10px] text-text-muted">OCR Confidence</p>
                    </div>
                  </div>
                </div>

                {/* Signal State Display */}
                <div className="grid grid-cols-2 gap-3">
                  <div className="p-3.5 rounded-xl bg-surface-3/80 border border-border">
                    <p className="text-[10px] font-bold text-text-muted uppercase tracking-wider mb-1">
                      Signal Light State
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
                      {current.light_state?.toUpperCase()}
                    </p>
                    <p className="text-[10px] text-text-muted mt-1">
                      Signal Conf:{" "}
                      {current.light_confidence
                        ? `${(current.light_confidence * 100).toFixed(0)}%`
                        : "N/A"}
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
                    Detection Coordinates
                  </p>
                  <div className="grid grid-cols-2 gap-2 font-mono text-[11px]">
                    <div className="p-2.5 rounded-lg bg-surface-3/60 border border-border">
                      <p className="text-[10px] text-text-muted font-sans">
                        Crossing Point
                      </p>
                      <p className="text-text-secondary">
                        ({current.crossing_point?.x?.toFixed(1) || 0},{" "}
                        {current.crossing_point?.y?.toFixed(1) || 0})
                      </p>
                    </div>
                    <div className="p-2.5 rounded-lg bg-surface-3/60 border border-border">
                      <p className="text-[10px] text-text-muted font-sans">
                        Stopline Side
                      </p>
                      <p className="text-text-secondary">
                        Side {current.previous_side} → {current.current_side}
                      </p>
                    </div>
                  </div>
                </div>

                {/* Keyboard Shortcuts Help */}
                <div className="p-3 rounded-xl bg-indigo-500/10 border border-indigo-500/20 text-[11px] text-indigo-500 space-y-1">
                  <p className="font-semibold flex items-center gap-1.5 text-xs text-indigo-500">
                    <Sparkles className="w-3.5 h-3.5" /> Keyboard Shortcuts
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
                      Approve
                    </span>
                    <span>
                      <kbd className="bg-indigo-500/20 px-1 py-0.5 rounded text-indigo-600 dark:text-indigo-300">
                        Backspace
                      </kbd>{" "}
                      /{" "}
                      <kbd className="bg-indigo-500/20 px-1 py-0.5 rounded text-indigo-600 dark:text-indigo-300">
                        R
                      </kbd>{" "}
                      Reject
                    </span>
                    <span>
                      <kbd className="bg-indigo-500/20 px-1 py-0.5 rounded text-indigo-600 dark:text-indigo-300">
                        ←
                      </kbd>{" "}
                      /{" "}
                      <kbd className="bg-indigo-500/20 px-1 py-0.5 rounded text-indigo-600 dark:text-indigo-300">
                        A
                      </kbd>{" "}
                      Previous
                    </span>
                    <span>
                      <kbd className="bg-indigo-500/20 px-1 py-0.5 rounded text-indigo-600 dark:text-indigo-300">
                        →
                      </kbd>{" "}
                      /{" "}
                      <kbd className="bg-indigo-500/20 px-1 py-0.5 rounded text-indigo-600 dark:text-indigo-300">
                        D
                      </kbd>{" "}
                      Next
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
                      Violation Evidence Media
                    </span>
                    <Link
                      href={`/violations/${current.id}`}
                      className="text-indigo-500 hover:text-indigo-600 font-medium flex items-center gap-1"
                    >
                      Full Details <ExternalLink className="w-3 h-3" />
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
                          alt={current.event_id}
                          fill
                          className="object-contain"
                          unoptimized
                        />
                      )
                    ) : (
                      <div className="text-center py-12 text-text-muted space-y-2">
                        <AlertTriangle className="w-8 h-8 mx-auto text-text-muted" />
                        <p className="text-xs">No media preview available</p>
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
                      className="btn-success py-3.5 text-sm font-bold flex items-center justify-center gap-2 rounded-xl shadow-lg shadow-emerald-500/20 disabled:opacity-50"
                    >
                      <CheckCircle2 className="w-5 h-5" />
                      Approve &amp; Issue Fine
                    </button>
                    <button
                      onClick={() => handleAction("rejected")}
                      disabled={actionLoading}
                      className="btn-danger py-3.5 text-sm font-bold flex items-center justify-center gap-2 rounded-xl shadow-lg shadow-rose-500/20 disabled:opacity-50"
                    >
                      <XCircle className="w-5 h-5" />
                      Reject (False Positive)
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Bottom Pagination controls */}
          <div className="flex items-center justify-between">
            <button
              onClick={() => setCurrentIndex((v) => Math.max(0, v - 1))}
              disabled={currentIndex === 0}
              className="btn-ghost text-xs flex items-center gap-1.5 disabled:opacity-30 border border-border"
            >
              <ChevronLeft className="w-4 h-4" /> Previous Record
            </button>
            <span className="text-xs text-text-muted font-mono">
              Case {currentIndex + 1} / {violations.length}
            </span>
            <button
              onClick={() =>
                setCurrentIndex((v) => Math.min(violations.length - 1, v + 1))
              }
              disabled={currentIndex >= violations.length - 1}
              className="btn-ghost text-xs flex items-center gap-1.5 disabled:opacity-30 border border-border"
            >
              Next Record <ChevronRight className="w-4 h-4" />
            </button>
          </div>
        </div>
      ) : null}
    </div>
  );
}
