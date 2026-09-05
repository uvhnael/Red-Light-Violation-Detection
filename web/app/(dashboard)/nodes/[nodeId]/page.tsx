'use client';

import { useEffect, useState } from 'react';
import { useParams } from 'next/navigation';
import Link from 'next/link';
import { EdgeNodeResponse } from '@/lib/types';
import { getEdgeNode } from '@/lib/api';
import NodeCalibrationPanel from '@/components/NodeCalibrationPanel';

export default function NodeDetailPage() {
  const params = useParams();
  const nodeId = params.nodeId as string;

  const [node, setNode] = useState<EdgeNodeResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    setLoading(true);
    setError(null);
    (async () => {
      try {
        const data = await getEdgeNode(nodeId);
        if (active) setNode(data);
      } catch (err) {
        if (active) {
          setError(err instanceof Error ? err.message : 'Không thể tải thông tin node');
        }
      } finally {
        if (active) setLoading(false);
      }
    })();
    return () => {
      active = false;
    };
  }, [nodeId]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-[60vh]">
        <div className="flex flex-col items-center gap-4">
          <div className="w-12 h-12 border-4 border-primary-500/30 border-t-primary-500 rounded-full animate-spin" />
          <p className="text-text-secondary text-sm">Đang tải thông tin node...</p>
        </div>
      </div>
    );
  }

  if (error || !node) {
    return (
      <div className="flex items-center justify-center h-[60vh]">
        <div className="glass-card p-8 text-center max-w-md">
          <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-red-500/10 flex items-center justify-center">
            <svg className="w-8 h-8 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m9-.75a9 9 0 11-18 0 9 9 0 0118 0zm-9 3.75h.008v.008H12v-.008z" />
            </svg>
          </div>
          <h2 className="text-lg font-semibold text-text-primary mb-2">Lỗi</h2>
          <p className="text-sm text-text-secondary mb-4">{error || 'Node không tồn tại'}</p>
          <Link href="/nodes" className="btn-primary text-sm">
            Quay lại
          </Link>
        </div>
      </div>
    );
  }

  // Stream/snapshot URL đi qua rewrite proxy của Next.js (/edge-api → edge),
  // không dùng IP:port trực tiếp — IP Docker nội bộ (172.x) và port container
  // không tới được từ browser.
  const cameraId = `${nodeId}-cam-1`;
  const streamUrl = `/edge-api/cameras/${cameraId}/stream`;
  const snapshotUrl = `/edge-api/cameras/${cameraId}/snapshot`;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-start justify-between gap-4">
        <div>
          <Link
            href="/nodes"
            className="inline-flex items-center gap-1.5 text-sm text-text-muted hover:text-text-secondary transition-colors mb-3"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M10.5 19.5L3 12m0 0l7.5-7.5M3 12h18" />
            </svg>
            Danh sách node
          </Link>
          <h1 className="text-2xl font-bold gradient-text">{node.name}</h1>
          <p className="text-text-muted text-sm font-mono mt-1">{node.node_id}</p>
        </div>
      </div>

      {/* Node Info Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="glass-card p-4">
          <p className="text-xs text-text-muted mb-1">IP Address</p>
          <p className="text-sm font-medium text-text-primary font-mono">{node.ip_address || '—'}</p>
        </div>
        <div className="glass-card p-4">
          <p className="text-xs text-text-muted mb-1">API Port</p>
          <p className="text-sm font-medium text-text-primary font-mono">
            {(node.settings?.api_port as number) || 8080}
          </p>
        </div>
        <div className="glass-card p-4">
          <p className="text-xs text-text-muted mb-1">Trạng thái</p>
          <p className="text-sm font-medium text-text-primary capitalize">{node.status}</p>
        </div>
        <div className="glass-card p-4">
          <p className="text-xs text-text-muted mb-1">Last Ping</p>
          <p className="text-sm font-medium text-text-primary">
            {node.last_ping ? new Date(node.last_ping).toLocaleString('vi-VN') : '—'}
          </p>
        </div>
      </div>

      {/* Live camera + calibration (delegated to panel) */}
      <NodeCalibrationPanel
        nodeId={nodeId}
        nodeName={node.name}
        cameraId={cameraId}
        streamUrl={streamUrl}
        snapshotUrl={snapshotUrl}
      />

      {/* Node Settings */}
      <div className="glass-card p-6">
        <h3 className="text-sm font-semibold text-text-secondary uppercase tracking-wider mb-4">
          Cấu hình Node
        </h3>
        <pre className="bg-surface-3/50 rounded-xl p-4 text-xs text-text-secondary font-mono overflow-x-auto">
          {JSON.stringify(node.settings, null, 2)}
        </pre>
      </div>
    </div>
  );
}