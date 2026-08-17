'use client';

import { useCallback, useEffect, useMemo, useState } from 'react';
import Link from 'next/link';
import Image from 'next/image';
import StatusBadge from '@/components/StatusBadge';
import { ViolationResponse } from '@/lib/types';
import { getViolations, updateViolationStatus } from '@/lib/api';

export default function ReviewPage() {
  const [violations, setViolations] = useState<ViolationResponse[]>([]);
  const [loading, setLoading] = useState(true);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [actionLoading, setActionLoading] = useState(false);
  const current = useMemo(() => violations[currentIndex] ?? null, [violations, currentIndex]);

  useEffect(() => {
    getViolations({ status: 'pending' })
      .then(setViolations)
      .catch(() => setViolations([]))
      .finally(() => setLoading(false));
  }, []);

  const handleAction = useCallback(async (status: string) => {
    if (!current) return;
    setActionLoading(true);
    try {
      await updateViolationStatus(current.id, status);
      const updated = violations.filter((_, i) => i !== currentIndex);
      setViolations(updated);
      if (currentIndex >= updated.length && updated.length > 0) {
        setCurrentIndex(updated.length - 1);
      }
    } catch {
      alert('Lỗi cập nhật');
    } finally {
      setActionLoading(false);
    }
  }, [current, currentIndex, violations]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (violations.length === 0 || actionLoading) {
        return;
      }
      if (event.key === 'ArrowLeft') {
        setCurrentIndex((value) => Math.max(0, value - 1));
      }
      if (event.key === 'ArrowRight') {
        setCurrentIndex((value) => Math.min(violations.length - 1, value + 1));
      }
      if (event.key === 'Enter' && current) {
        event.preventDefault();
        void handleAction('approved');
      }
    };

    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [violations.length, actionLoading, current, handleAction]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-[60vh]">
        <div className="w-12 h-12 border-4 border-primary-500/30 border-t-primary-500 rounded-full animate-spin" />
      </div>
    );
  }

  return (
    <div className="space-y-6 max-w-4xl mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold gradient-text">Duyệt hồ sơ vi phạm</h1>
          <p className="text-text-secondary text-sm mt-1">
            Human-in-the-loop — CSGT duyệt các case có confidence thấp
          </p>
        </div>
        <div className="glass-card px-4 py-2 flex items-center gap-2">
          <div className={`glow-dot ${violations.length > 0 ? 'bg-amber-400 text-amber-400' : 'bg-emerald-400 text-emerald-400'}`} />
          <span className="text-sm font-medium text-text-primary">{violations.length}</span>
          <span className="text-sm text-text-muted">chờ duyệt</span>
        </div>
      </div>

      {violations.length === 0 ? (
        <div className="glass-card p-16 text-center">
          <div className="w-20 h-20 mx-auto mb-6 rounded-full bg-emerald-500/10 flex items-center justify-center">
            <svg className="w-10 h-10 text-emerald-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75L11.25 15 15 9.75M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </div>
          <h2 className="text-xl font-semibold text-text-primary mb-2">Không có hồ sơ cần duyệt</h2>
          <p className="text-sm text-text-secondary">Tất cả hồ sơ vi phạm đã được xử lý.</p>
        </div>
      ) : current ? (
        <div className="space-y-6">
          {/* Progress */}
          <div className="flex items-center gap-4">
            <div className="flex-1 h-1.5 bg-surface-3 rounded-full overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-primary-500 to-blue-500 rounded-full transition-all duration-500"
                style={{ width: `${((currentIndex + 1) / (violations.length)) * 100}%` }}
              />
            </div>
            <span className="text-xs text-text-muted whitespace-nowrap">
              {currentIndex + 1} / {violations.length}
            </span>
          </div>

          {/* Review Card */}
          <div className="glass-card overflow-hidden">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-0">
              <div className="p-6 border-b lg:border-b-0 lg:border-r border-border space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <h2 className="text-lg font-bold text-text-primary">Vi phạm #{current.id}</h2>
                    <p className="text-xs text-text-muted font-mono mt-1">{current.event_id}</p>
                  </div>
                  <StatusBadge status={current.status} size="md" />
                </div>

                <div className="grid grid-cols-2 gap-3">
                  <div className="p-4 bg-surface-3/50 rounded-xl">
                    <p className="text-xs text-text-muted mb-1">Biển số</p>
                    <p className="text-lg font-bold text-primary-400">{current.plate_text || '—'}</p>
                    <p className="text-xs text-text-muted mt-1">{current.plate_confidence ? `${(current.plate_confidence * 100).toFixed(0)}% conf.` : 'OCR chưa có'}</p>
                  </div>
                  <div className="p-4 bg-surface-3/50 rounded-xl">
                    <p className="text-xs text-text-muted mb-1">Đèn</p>
                    <p className={`text-lg font-bold ${current.light_state === 'red' ? 'text-red-400' : 'text-amber-400'}`}>{current.light_state?.toUpperCase()}</p>
                    <p className="text-xs text-text-muted mt-1">{current.light_confidence ? `${(current.light_confidence * 100).toFixed(0)}% conf.` : ''}</p>
                  </div>
                  <div className="p-4 bg-surface-3/50 rounded-xl">
                    <p className="text-xs text-text-muted mb-1">Node</p>
                    <p className="text-sm font-medium text-text-primary">{current.node_id}</p>
                  </div>
                  <div className="p-4 bg-surface-3/50 rounded-xl">
                    <p className="text-xs text-text-muted mb-1">Thời gian</p>
                    <p className="text-sm font-medium text-text-primary">{current.created_at ? new Date(current.created_at).toLocaleString('vi-VN') : '—'}</p>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-3">
                  <div className="p-3 bg-surface-3/30 rounded-xl">
                    <p className="text-xs text-text-muted">Crossing Point</p>
                    <p className="text-sm text-text-primary font-mono">({current.crossing_point?.x?.toFixed(1)}, {current.crossing_point?.y?.toFixed(1)})</p>
                  </div>
                  <div className="p-3 bg-surface-3/30 rounded-xl">
                    <p className="text-xs text-text-muted">Side Transition</p>
                    <p className="text-sm text-text-primary font-mono">{current.previous_side} → {current.current_side}</p>
                  </div>
                </div>

                <div className="flex items-center gap-3 text-xs text-text-muted">
                  <span>← / → để chuyển hồ sơ</span>
                  <span>Enter để duyệt</span>
                </div>
              </div>

              <div className="p-6 space-y-4">
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  <button
                    onClick={() => handleAction('approved')}
                    disabled={actionLoading}
                    className="btn-success py-3 flex items-center justify-center gap-2 text-base disabled:opacity-50"
                  >
                    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M4.5 12.75l6 6 9-13.5" />
                    </svg>
                    Approve &amp; Generate Ticket
                  </button>
                  <button
                    onClick={() => handleAction('rejected')}
                    disabled={actionLoading}
                    className="btn-danger py-3 flex items-center justify-center gap-2 text-base disabled:opacity-50"
                  >
                    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                    </svg>
                    Reject (False Positive)
                  </button>
                </div>
                <Link href={`/violations/${current.id}`} className="btn-ghost w-full py-3 flex items-center justify-center gap-2">
                  Mở hồ sơ chi tiết →
                </Link>

                <div className="p-4 bg-surface-3/40 rounded-xl">
                  <p className="text-xs text-text-muted uppercase tracking-wider mb-3">Media preview</p>
                  {current.media_url ? (
                    current.media_url.match(/\.(mp4|webm|ogg)(\?|$)/i) ? (
                      <video src={current.media_url} className="w-full rounded-xl border border-border bg-black/40" controls autoPlay loop muted playsInline />
                    ) : (
                      <div className="relative aspect-video w-full overflow-hidden rounded-xl border border-border bg-surface-3">
                        <Image src={current.media_url} alt={current.event_id} fill className="object-cover" />
                      </div>
                    )
                  ) : (
                    <div className="flex h-64 items-center justify-center rounded-xl border border-dashed border-border text-sm text-text-muted">
                      Chưa có media từ server
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>

          {/* Navigation */}
          <div className="flex items-center justify-between">
            <button
              onClick={() => setCurrentIndex(Math.max(0, currentIndex - 1))}
              disabled={currentIndex === 0}
              className="btn-ghost text-sm disabled:opacity-30"
            >
              ← Trước
            </button>
            <button
              onClick={() => setCurrentIndex(Math.min(violations.length - 1, currentIndex + 1))}
              disabled={currentIndex >= violations.length - 1}
              className="btn-ghost text-sm disabled:opacity-30"
            >
              Tiếp →
            </button>
          </div>
        </div>
      ) : null}
    </div>
  );
}
