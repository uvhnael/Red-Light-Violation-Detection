'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import StatusBadge from '@/components/StatusBadge';
import { ViolationResponse } from '@/lib/types';
import { getViolations } from '@/lib/api';

export default function ViolationsPage() {
  const [violations, setViolations] = useState<ViolationResponse[]>([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState({ status: '', nodeId: '', plateText: '' });

  useEffect(() => {
    const timer = window.setTimeout(() => {
      void (async () => {
        setLoading(true);
        try {
          const params: Record<string, string> = {};
          if (filter.status) params.status = filter.status;
          if (filter.nodeId) params.nodeId = filter.nodeId;
          if (filter.plateText) params.plateText = filter.plateText;
          const data = await getViolations(params);
          setViolations(data);
        } catch {
          setViolations([]);
        } finally {
          setLoading(false);
        }
      })();
    }, 0);

    return () => window.clearTimeout(timer);
  }, [filter.status, filter.nodeId, filter.plateText]);

  const handleFilter = (e: React.FormEvent) => {
    e.preventDefault();
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold gradient-text">Danh sách vi phạm</h1>
          <p className="text-text-secondary text-sm mt-1">
            {violations.length} hồ sơ vi phạm
          </p>
        </div>
        <button onClick={() => setFilter({ ...filter })} className="btn-ghost flex items-center gap-2 text-sm">
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.993 0l3.181 3.183a8.25 8.25 0 0013.803-3.7M4.031 9.865a8.25 8.25 0 0113.803-3.7l3.181 3.182" />
          </svg>
          Làm mới
        </button>
      </div>

      {/* Filters */}
      <form onSubmit={handleFilter} className="glass-card p-4 flex flex-wrap gap-3 items-end">
        <div className="flex-1 min-w-[180px]">
          <label className="block text-xs text-text-muted mb-1.5 font-medium">Trạng thái</label>
          <select
            value={filter.status}
            onChange={(e) => setFilter({ ...filter, status: e.target.value })}
            className="input-field w-full"
          >
            <option value="">Tất cả</option>
            <option value="pending">Chờ duyệt</option>
            <option value="confirmed">Đã xác nhận</option>
            <option value="rejected">Từ chối</option>
          </select>
        </div>
        <div className="flex-1 min-w-[180px]">
          <label className="block text-xs text-text-muted mb-1.5 font-medium">Node ID</label>
          <input
            type="text"
            value={filter.nodeId}
            onChange={(e) => setFilter({ ...filter, nodeId: e.target.value })}
            className="input-field w-full"
            placeholder="edge-node-01"
          />
        </div>
        <div className="flex-1 min-w-[180px]">
          <label className="block text-xs text-text-muted mb-1.5 font-medium">Biển số xe</label>
          <input
            type="text"
            value={filter.plateText}
            onChange={(e) => setFilter({ ...filter, plateText: e.target.value })}
            className="input-field w-full"
            placeholder="29A-12345"
          />
        </div>
        <button type="submit" className="btn-primary text-sm flex items-center gap-2">
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M21 21l-5.197-5.197m0 0A7.5 7.5 0 105.196 5.196a7.5 7.5 0 0010.607 10.607z" />
          </svg>
          Tìm kiếm
        </button>
      </form>

      {/* Table */}
      <div className="glass-card overflow-hidden">
        {loading ? (
          <div className="flex items-center justify-center py-20">
            <div className="w-10 h-10 border-4 border-primary-500/30 border-t-primary-500 rounded-full animate-spin" />
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-border">
                  <th className="px-5 py-3 text-left text-xs font-semibold text-text-muted uppercase">ID</th>
                  <th className="px-5 py-3 text-left text-xs font-semibold text-text-muted uppercase">Event ID</th>
                  <th className="px-5 py-3 text-left text-xs font-semibold text-text-muted uppercase">Node</th>
                  <th className="px-5 py-3 text-left text-xs font-semibold text-text-muted uppercase">Biển số</th>
                  <th className="px-5 py-3 text-left text-xs font-semibold text-text-muted uppercase">Đèn</th>
                  <th className="px-5 py-3 text-left text-xs font-semibold text-text-muted uppercase">Conf.</th>
                  <th className="px-5 py-3 text-left text-xs font-semibold text-text-muted uppercase">Trạng thái</th>
                  <th className="px-5 py-3 text-left text-xs font-semibold text-text-muted uppercase">Thời gian</th>
                  <th className="px-5 py-3 text-left text-xs font-semibold text-text-muted uppercase"></th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border/50">
                {violations.length > 0 ? (
                  violations.map((v) => (
                    <tr key={v.id} className="hover:bg-surface-3/50 transition-colors">
                      <td className="px-5 py-3 text-sm text-text-muted">#{v.id}</td>
                      <td className="px-5 py-3 text-sm text-text-primary font-mono">
                        {v.event_id.length > 18 ? v.event_id.slice(0, 18) + '...' : v.event_id}
                      </td>
                      <td className="px-5 py-3 text-sm text-text-secondary">{v.node_id}</td>
                      <td className="px-5 py-3 text-sm text-text-primary font-semibold">
                        {v.plate_text || <span className="text-text-muted font-normal">—</span>}
                      </td>
                      <td className="px-5 py-3">
                        <span className={`inline-flex items-center gap-1.5 text-xs font-medium ${
                          v.light_state === 'red' ? 'text-red-400' : 'text-amber-400'
                        }`}>
                          <span className={`w-1.5 h-1.5 rounded-full ${
                            v.light_state === 'red' ? 'bg-red-400' : 'bg-amber-400'
                          }`} />
                          {v.light_state}
                        </span>
                      </td>
                      <td className="px-5 py-3 text-sm text-text-secondary">
                        {v.light_confidence ? `${(v.light_confidence * 100).toFixed(0)}%` : '—'}
                      </td>
                      <td className="px-5 py-3"><StatusBadge status={v.status} /></td>
                      <td className="px-5 py-3 text-xs text-text-muted">
                        {v.created_at ? new Date(v.created_at).toLocaleString('vi-VN') : '—'}
                      </td>
                      <td className="px-5 py-3">
                        <Link href={`/violations/${v.id}`} className="text-primary-400 hover:text-primary-300 text-sm font-medium transition-colors">
                          Chi tiết
                        </Link>
                      </td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan={9} className="px-6 py-16 text-center text-sm text-text-muted">
                      Không tìm thấy vi phạm nào.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
