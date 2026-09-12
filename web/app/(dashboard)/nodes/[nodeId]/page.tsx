'use client';

import { useEffect, useState } from 'react';
import { useParams } from 'next/navigation';
import Link from 'next/link';
import { EdgeNodeResponse } from '@/lib/types';
import { getEdgeNode } from '@/lib/api';
import NodeCalibrationPanel from '@/components/NodeCalibrationPanel';
import StatusBadge from '@/components/StatusBadge';
import { effectiveNodeStatus } from '@/lib/nodes';
import { ArrowLeft, AlertTriangle } from 'lucide-react';
import { motion } from 'motion/react';

export default function NodeDetailPage() {
  const params = useParams();
  const nodeId = params.nodeId as string;

  const [node, setNode] = useState<EdgeNodeResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    (async () => {
      setLoading(true);
      setError(null);
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
          <div className="w-12 h-12 border-4 border-indigo-500/30 border-t-indigo-500 rounded-full animate-spin" />
          <p className="text-text-muted text-xs font-medium">Đang tải thông tin node...</p>
        </div>
      </div>
    );
  }

  if (error || !node) {
    return (
      <div className="flex items-center justify-center h-[60vh]">
        <div className="glass-card p-8 text-center max-w-md space-y-3">
          <div className="w-14 h-14 mx-auto rounded-2xl bg-rose-500/10 flex items-center justify-center">
            <AlertTriangle className="w-7 h-7 text-rose-500" />
          </div>
          <h2 className="text-lg font-bold text-text-primary">Lỗi kết nối Node</h2>
          <p className="text-xs text-text-muted">{error || 'Node không tồn tại trong hệ thống'}</p>
          <Link href="/nodes" className="btn-primary text-xs inline-flex mt-2">
            ← Quay lại danh sách node
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

  return (
    <motion.div
      className="space-y-6"
      initial={{ opacity: 0, y: 14 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
    >
      {/* Header */}
      <div className="flex items-center gap-4">
        <Link
          href="/nodes"
          className="p-2.5 rounded-xl bg-surface-3 border border-border text-text-muted hover:text-text-primary transition-colors"
        >
          <ArrowLeft className="w-4 h-4" />
        </Link>
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-xl font-bold text-text-primary">{node.name}</h1>
            <StatusBadge status={effectiveNodeStatus(node)} size="md" />
          </div>
          <p className="text-xs text-text-muted font-mono mt-0.5">{node.node_id}</p>
        </div>
      </div>

      {/* Node Info Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="glass-card p-4">
          <p className="text-[10px] uppercase font-bold tracking-wider text-text-muted mb-1">Địa chỉ IP</p>
          <p className="text-sm font-semibold text-text-primary font-mono">{node.ip_address || '—'}</p>
        </div>
        <div className="glass-card p-4">
          <p className="text-[10px] uppercase font-bold tracking-wider text-text-muted mb-1">Cổng API Port</p>
          <p className="text-sm font-semibold text-text-primary font-mono">
            {(node.settings?.api_port as number) || 8080}
          </p>
        </div>
        <div className="glass-card p-4">
          <p className="text-[10px] uppercase font-bold tracking-wider text-text-muted mb-1">Ping gần nhất</p>
          <p className="text-xs font-medium text-text-primary mt-0.5">
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
    </motion.div>
  );
}