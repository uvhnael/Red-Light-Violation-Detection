'use client';

import { useEffect, useState } from 'react';
import { useParams } from 'next/navigation';
import Link from 'next/link';
import VideoPlayer from '@/components/VideoPlayer';
import { EdgeNodeResponse } from '@/lib/types';
import { getEdgeNode } from '@/lib/api';

export default function NodeDetailPage() {
  const params = useParams();
  const nodeId = params.nodeId as string;

  const [node, setNode] = useState<EdgeNodeResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function load() {
      try {
        const data = await getEdgeNode(nodeId);
        setNode(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Không thể tải thông tin node');
      } finally {
        setLoading(false);
      }
    }
    load();
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

  // Build stream URL from edge node settings
  const apiPort = (node.settings?.api_port as number) || 8080;
  const nodeIp = node.ip_address || 'localhost';
  // Use localhost if IP looks like an internal Docker hostname
  const streamHost = (nodeIp.includes('.') || nodeIp === 'localhost') ? nodeIp : 'localhost';
  const streamUrl = `http://${streamHost}:${apiPort}/api/cameras/fake-cam-1/stream`;
  const snapshotUrl = `http://${streamHost}:${apiPort}/api/cameras/fake-cam-1/snapshot`;

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
            Quay lại Edge Nodes
          </Link>
          <h1 className="text-2xl font-bold gradient-text">{node.name}</h1>
          <p className="text-text-secondary text-sm mt-1 font-mono">{node.node_id}</p>
        </div>
        <span
          className={`inline-flex items-center rounded-full px-3 py-1 text-xs font-medium ${
            node.online ? 'status-confirmed' : 'status-rejected'
          }`}
        >
          {node.online ? 'Online' : 'Offline'}
        </span>
      </div>

      {/* Node Info Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="glass-card p-4">
          <p className="text-xs text-text-muted mb-1">IP Address</p>
          <p className="text-sm font-medium text-text-primary font-mono">{node.ip_address || '—'}</p>
        </div>
        <div className="glass-card p-4">
          <p className="text-xs text-text-muted mb-1">API Port</p>
          <p className="text-sm font-medium text-text-primary font-mono">{apiPort}</p>
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

      {/* Camera Stream */}
      <div className="glass-card overflow-hidden">
        <div className="px-6 py-4 border-b border-border flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-primary-500/15 flex items-center justify-center">
              <svg className="w-5 h-5 text-primary-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                <path strokeLinecap="round" d="M15.75 10.5l4.72-4.72a.75.75 0 011.28.53v11.38a.75.75 0 01-1.28.53l-4.72-4.72M4.5 18.75h9a2.25 2.25 0 002.25-2.25v-9a2.25 2.25 0 00-2.25-2.25h-9A2.25 2.25 0 002.25 7.5v9a2.25 2.25 0 002.25 2.25z" />
              </svg>
            </div>
            <div>
              <h2 className="text-sm font-semibold text-text-primary">Fake Camera 1</h2>
              <p className="text-xs text-text-muted">HLS Stream — aziz1.MP4</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <div className="glow-dot bg-red-500 text-red-500" />
            <span className="text-xs font-medium text-red-400">LIVE</span>
          </div>
        </div>

        <div className="p-4">
          <VideoPlayer src={streamUrl} className="w-full aspect-video" />
        </div>

        <div className="px-6 py-3 border-t border-border flex items-center justify-between text-xs text-text-muted">
          <span>Stream: {streamUrl}</span>
          <a
            href={snapshotUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="text-primary-400 hover:text-primary-300 transition-colors"
          >
            Chụp ảnh snapshot →
          </a>
        </div>
      </div>

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
