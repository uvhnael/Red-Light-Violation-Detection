'use client';

import { useEffect, useState } from 'react';
import Image from 'next/image';
import { useParams, useRouter } from 'next/navigation';
import Link from 'next/link';
import StatusBadge from '@/components/StatusBadge';
import { ViolationResponse } from '@/lib/types';
import { getViolation, updateViolationStatus, deleteViolation } from '@/lib/api';

export default function ViolationDetailPage() {
  const params = useParams();
  const router = useRouter();
  const id = Number(params.id);

  const [violation, setViolation] = useState<ViolationResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [actionLoading, setActionLoading] = useState(false);

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
    } catch {
      alert('Lỗi cập nhật trạng thái');
    } finally {
      setActionLoading(false);
    }
  };

  const handleDelete = async () => {
    if (!violation || !confirm('Bạn có chắc muốn xóa hồ sơ vi phạm này?')) return;
    try {
      await deleteViolation(violation.id);
      router.push('/violations');
    } catch {
      alert('Lỗi khi xóa');
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-[60vh]">
        <div className="w-12 h-12 border-4 border-primary-500/30 border-t-primary-500 rounded-full animate-spin" />
      </div>
    );
  }

  if (!violation) {
    return (
      <div className="flex items-center justify-center h-[60vh]">
        <div className="glass-card p-8 text-center">
          <h2 className="text-lg font-semibold text-text-primary mb-2">Không tìm thấy</h2>
          <p className="text-sm text-text-secondary mb-4">Hồ sơ vi phạm #{id} không tồn tại.</p>
          <Link href="/violations" className="btn-primary text-sm">← Quay lại</Link>
        </div>
      </div>
    );
  }

  const infoRows = [
    { label: 'Event ID', value: violation.event_id, mono: true },
    { label: 'Node ID', value: violation.node_id },
    { label: 'Track ID', value: violation.track_id },
    { label: 'Frame Index', value: violation.frame_index },
    { label: 'Timestamp (ms)', value: violation.timestamp_ms?.toFixed(1) },
    { label: 'Biển số', value: violation.plate_text || '—', highlight: true },
    { label: 'Biển số (conf.)', value: violation.plate_confidence ? `${(violation.plate_confidence * 100).toFixed(1)}%` : '—' },
    { label: 'Trạng thái đèn', value: violation.light_state, color: violation.light_state === 'red' ? 'text-red-400' : 'text-amber-400' },
    { label: 'Đèn (conf.)', value: violation.light_confidence ? `${(violation.light_confidence * 100).toFixed(1)}%` : '—' },
  ];

  const mediaUrl = violation.media_url
      ? `/api/v1/violations/${violation.event_id}/media/blob`
      : null;
    const mediaIsVideo = Boolean(mediaUrl && violation.media_url?.endsWith('.mp4'));

  return (
    <div className="space-y-6 max-w-5xl">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <Link href="/violations" className="p-2 rounded-xl hover:bg-surface-3 transition-colors">
            <svg className="w-5 h-5 text-text-secondary" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M10.5 19.5L3 12m0 0l7.5-7.5M3 12h18" />
            </svg>
          </Link>
          <div>
            <h1 className="text-xl font-bold text-text-primary">Vi phạm #{violation.id}</h1>
            <p className="text-sm text-text-muted font-mono">{violation.event_id}</p>
          </div>
        </div>
        <StatusBadge status={violation.status} size="md" />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Info Panel */}
        <div className="lg:col-span-2 glass-card p-6">
          <h2 className="text-sm font-semibold text-text-secondary uppercase tracking-wider mb-4">Thông tin vi phạm</h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {infoRows.map((row) => (
              <div key={row.label} className="p-3 bg-surface-3/50 rounded-xl">
                <p className="text-xs text-text-muted mb-1">{row.label}</p>
                <p className={`text-sm font-medium ${
                  row.color || (row.highlight ? 'text-primary-400 text-lg font-bold' : 'text-text-primary')
                } ${row.mono ? 'font-mono text-xs' : ''}`}>
                  {row.value}
                </p>
              </div>
            ))}
          </div>

          {/* Coordinates */}
          <h3 className="text-sm font-semibold text-text-secondary uppercase tracking-wider mt-6 mb-3">Tọa độ</h3>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div className="p-3 bg-surface-3/50 rounded-xl">
              <p className="text-xs text-text-muted mb-1">Crossing Point</p>
              <p className="text-sm text-text-primary font-mono">
                ({violation.crossing_point?.x?.toFixed(1)}, {violation.crossing_point?.y?.toFixed(1)})
              </p>
            </div>
            <div className="p-3 bg-surface-3/50 rounded-xl">
              <p className="text-xs text-text-muted mb-1">Previous Point</p>
              <p className="text-sm text-text-primary font-mono">
                ({violation.previous_point?.x?.toFixed(1)}, {violation.previous_point?.y?.toFixed(1)})
              </p>
            </div>
            <div className="p-3 bg-surface-3/50 rounded-xl">
              <p className="text-xs text-text-muted mb-1">Bounding Box (xyxy)</p>
              <p className="text-sm text-text-primary font-mono">
                [{violation.bbox_xyxy?.map(v => v?.toFixed(0)).join(', ')}]
              </p>
            </div>
            <div className="p-3 bg-surface-3/50 rounded-xl">
              <p className="text-xs text-text-muted mb-1">Side Transition</p>
              <p className="text-sm text-text-primary font-mono">
                {violation.previous_side} → {violation.current_side}
              </p>
            </div>
          </div>

          {/* Metadata */}
          {violation.metadata && (
            <>
              <h3 className="text-sm font-semibold text-text-secondary uppercase tracking-wider mt-6 mb-3">Metadata</h3>
              <pre className="p-4 bg-surface-3/50 rounded-xl text-xs text-text-secondary font-mono overflow-x-auto">
                {(() => {
                  try { return JSON.stringify(JSON.parse(violation.metadata), null, 2); }
                  catch { return violation.metadata; }
                })()}
              </pre>
            </>
          )}

          {mediaUrl && (
            <>
              <h3 className="text-sm font-semibold text-text-secondary uppercase tracking-wider mt-6 mb-3">Media</h3>
              <div className="p-4 bg-surface-3/50 rounded-xl">
                {mediaIsVideo ? (
                  <video src={mediaUrl} controls autoPlay loop muted playsInline className="w-full rounded-xl border border-border bg-black/40" />
                ) : (
                  <div className="relative aspect-video w-full overflow-hidden rounded-xl border border-border bg-surface-3">
                    <Image src={mediaUrl} alt={violation.event_id} fill className="object-cover" unoptimized />
                  </div>
                )}
              </div>
            </>
          )}
        </div>

        {/* Actions Panel */}
        <div className="space-y-6">
          {/* Review Actions */}
          <div className="glass-card p-6">
            <h2 className="text-sm font-semibold text-text-secondary uppercase tracking-wider mb-4">Duyệt hồ sơ</h2>
            <p className="text-xs text-text-muted mb-4">
              Xác nhận hoặc từ chối vi phạm này. Dành cho CSGT duyệt lại các case có confidence thấp.
            </p>
            <div className="space-y-3">
              <button
                onClick={() => handleStatusUpdate('approved')}
                disabled={actionLoading || violation.status === 'approved'}
                className="btn-success w-full flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M4.5 12.75l6 6 9-13.5" />
                </svg>
                Approve &amp; Generate Ticket
              </button>
              <button
                onClick={() => handleStatusUpdate('rejected')}
                disabled={actionLoading || violation.status === 'rejected'}
                className="btn-danger w-full flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                </svg>
                Reject
              </button>
              <button
                onClick={() => handleStatusUpdate('pending')}
                disabled={actionLoading || violation.status === 'pending'}
                className="btn-ghost w-full text-sm disabled:opacity-50 disabled:cursor-not-allowed"
              >
                ↩ Đặt lại Chờ duyệt
              </button>
            </div>
          </div>

          {/* Timestamps */}
          <div className="glass-card p-6">
            <h2 className="text-sm font-semibold text-text-secondary uppercase tracking-wider mb-4">Thời gian</h2>
            <div className="space-y-3">
              <div>
                <p className="text-xs text-text-muted">Tạo lúc</p>
                <p className="text-sm text-text-primary">
                  {violation.created_at ? new Date(violation.created_at).toLocaleString('vi-VN') : '—'}
                </p>
              </div>
              <div>
                <p className="text-xs text-text-muted">Cập nhật</p>
                <p className="text-sm text-text-primary">
                  {violation.updated_at ? new Date(violation.updated_at).toLocaleString('vi-VN') : '—'}
                </p>
              </div>
            </div>
          </div>

          {/* Danger zone */}
          <div className="glass-card p-6 border-red-500/20">
            <h2 className="text-sm font-semibold text-red-400 uppercase tracking-wider mb-3">Vùng nguy hiểm</h2>
            <button onClick={handleDelete} className="text-sm text-red-400 hover:text-red-300 transition-colors">
              Xóa hồ sơ vi phạm
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
