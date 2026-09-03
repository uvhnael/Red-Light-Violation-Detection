'use client';

import { useMemo, useState } from 'react';
import Link from 'next/link';
import { EdgeNodeResponse } from '@/lib/types';
import { getEdgeNodes, updateEdgeNodeSettings } from '@/lib/api';
import { useToast } from '@/components/Toast';

type NodeFormState = {
  name: string;
  ipAddress: string;
  status: string;
  settingsJson: string;
};

const emptyForm: NodeFormState = {
  name: '',
  ipAddress: '',
  status: 'online',
  settingsJson: '{\n  "note": ""\n}',
};

function toFormState(node: EdgeNodeResponse): NodeFormState {
  return {
    name: node.name ?? '',
    ipAddress: node.ip_address ?? '',
    status: node.status ?? 'online',
    settingsJson: JSON.stringify(node.settings ?? {}, null, 2),
  };
}

interface NodesClientProps {
  initialNodes: EdgeNodeResponse[];
}

export default function NodesClient({ initialNodes }: NodesClientProps) {
  const [nodes, setNodes] = useState<EdgeNodeResponse[]>(initialNodes);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(initialNodes[0]?.node_id ?? null);
  const [saving, setSaving] = useState(false);
  const [form, setForm] = useState<NodeFormState>(initialNodes[0] ? toFormState(initialNodes[0]) : emptyForm);
  const { show } = useToast();

  const selectedNode = useMemo(
    () => nodes.find((node) => node.node_id === selectedNodeId) ?? null,
    [nodes, selectedNodeId]
  );

  const loadNodes = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await getEdgeNodes();
      setNodes(data);

      const nextSelected = data.find((node) => node.node_id === selectedNodeId) ?? data[0] ?? null;
      setSelectedNodeId(nextSelected?.node_id ?? null);
      setForm(nextSelected ? toFormState(nextSelected) : emptyForm);
    } catch (fetchError) {
      setError(fetchError instanceof Error ? fetchError.message : 'Không tải được danh sách node');
    } finally {
      setLoading(false);
    }
  };

  const handleSelectNode = (node: EdgeNodeResponse) => {
    setSelectedNodeId(node.node_id);
    setForm(toFormState(node));
  };

  const handleSave = async () => {
    if (!selectedNode) return;

    let parsedSettings: Record<string, unknown> = {};
    try {
      parsedSettings = form.settingsJson.trim() ? JSON.parse(form.settingsJson) : {};
    } catch {
      show('JSON settings không hợp lệ — kiểm tra lại cú pháp.', 'danger');
      return;
    }

    setSaving(true);
    try {
      const updated = await updateEdgeNodeSettings(selectedNode.node_id, {
        name: form.name,
        ip_address: form.ipAddress,
        status: form.status,
        settings: parsedSettings,
      });

      setNodes((current) => current.map((node) => (node.node_id === updated.node_id ? updated : node)));
      setSelectedNodeId(updated.node_id);
      setForm(toFormState(updated));
      show(`Đã lưu cấu hình node "${updated.name}".`, 'success');
    } catch (saveError) {
      show(saveError instanceof Error ? saveError.message : 'Lỗi cập nhật node', 'danger');
    } finally {
      setSaving(false);
    }
  };

  const onlineCount = nodes.filter((node) => node.online).length;

  return (
    <div className="space-y-6">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold gradient-text">Edge Nodes</h1>
          <p className="text-text-secondary text-sm mt-1">Danh sách node đã đăng ký và cấu hình đang chạy.</p>
        </div>
        <button onClick={loadNodes} className="btn-ghost text-sm">
          Làm mới
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
        <div className="glass-card p-5">
          <p className="text-sm text-text-secondary">Registered</p>
          <p className="text-3xl font-bold mt-2">{nodes.length}</p>
        </div>
        <div className="glass-card p-5">
          <p className="text-sm text-text-secondary">Online</p>
          <p className="text-3xl font-bold mt-2 text-emerald-400">{onlineCount}</p>
        </div>
        <div className="glass-card p-5">
          <p className="text-sm text-text-secondary">Offline</p>
          <p className="text-3xl font-bold mt-2 text-red-400">{Math.max(0, nodes.length - onlineCount)}</p>
        </div>
      </div>

      {loading ? (
        <div className="glass-card p-12 text-center text-text-muted">Đang tải node...</div>
      ) : error ? (
        <div className="glass-card p-12 text-center text-red-400">{error}</div>
      ) : (
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
          <div className="glass-card overflow-hidden">
            <div className="px-6 py-4 border-b border-border flex items-center justify-between">
              <h2 className="text-sm font-semibold text-text-secondary uppercase tracking-wider">Node registry</h2>
              <span className="text-xs text-text-muted">{nodes.length} nodes</span>
            </div>
            <div className="divide-y divide-border/50">
              {nodes.length > 0 ? nodes.map((node) => (
                <button
                  key={node.node_id}
                  onClick={() => handleSelectNode(node)}
                  className={`w-full text-left px-6 py-4 transition-colors ${selectedNodeId === node.node_id ? 'bg-primary-600/10' : 'hover:bg-surface-3/50'}`}
                >
                  <div className="flex items-center justify-between gap-3">
                    <div>
                      <p className="font-semibold text-text-primary">{node.name}</p>
                      <p className="text-xs text-text-muted font-mono mt-1">{node.node_id}</p>
                    </div>
                    <div className="text-right">
                      <span className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium ${node.online ? 'status-confirmed' : 'status-rejected'}`}>
                        {node.online ? 'Online' : 'Offline'}
                      </span>
                      <p className="text-xs text-text-muted mt-2">{node.ip_address || '—'}</p>
                    </div>
                  </div>
                  <p className="text-xs text-text-muted mt-2">
                    Last ping: {node.last_ping ? new Date(node.last_ping).toLocaleString('vi-VN') : '—'}
                  </p>
                  <Link
                    href={`/nodes/${node.node_id}`}
                    className="mt-2 inline-flex items-center gap-1.5 text-xs text-primary-400 hover:text-primary-300 font-medium transition-colors"
                    onClick={(e) => e.stopPropagation()}
                  >
                    <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                      <path strokeLinecap="round" d="M15.75 10.5l4.72-4.72a.75.75 0 011.28.53v11.38a.75.75 0 01-1.28.53l-4.72-4.72M4.5 18.75h9a2.25 2.25 0 002.25-2.25v-9a2.25 2.25 0 00-2.25-2.25h-9A2.25 2.25 0 002.25 7.5v9a2.25 2.25 0 002.25 2.25z" />
                    </svg>
                    Xem Camera
                  </Link>
                </button>
              )) : (
                <div className="px-6 py-12 text-center text-text-muted">Chưa có edge node nào đăng ký.</div>
              )}
            </div>
          </div>

          <div className="glass-card p-6 space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-sm font-semibold text-text-secondary uppercase tracking-wider">Node settings</h2>
                <p className="text-xs text-text-muted mt-1">Chỉnh cấu hình cho node đang chọn.</p>
              </div>
              {selectedNode && (
                <span className={`inline-flex items-center rounded-full px-3 py-1 text-xs font-medium ${selectedNode.online ? 'status-confirmed' : 'status-rejected'}`}>
                  {selectedNode.online ? 'Live' : 'Stale'}
                </span>
              )}
            </div>

            {selectedNode ? (
              <div className="space-y-4">
                <div>
                  <label className="block text-xs text-text-muted mb-1.5 font-medium">Tên node</label>
                  <input className="input-field w-full" value={form.name} onChange={(e) => setForm((current) => ({ ...current, name: e.target.value }))} />
                </div>
                <div>
                  <label className="block text-xs text-text-muted mb-1.5 font-medium">IP address</label>
                  <input className="input-field w-full" value={form.ipAddress} onChange={(e) => setForm((current) => ({ ...current, ipAddress: e.target.value }))} />
                </div>
                <div>
                  <label className="block text-xs text-text-muted mb-1.5 font-medium">Status</label>
                  <select className="input-field w-full" value={form.status} onChange={(e) => setForm((current) => ({ ...current, status: e.target.value }))}>
                    <option value="online">online</option>
                    <option value="offline">offline</option>
                    <option value="degraded">degraded</option>
                    <option value="maintenance">maintenance</option>
                  </select>
                </div>
                <div>
                  <label className="block text-xs text-text-muted mb-1.5 font-medium">Settings JSON</label>
                  <textarea
                    className="input-field w-full min-h-64 font-mono text-xs"
                    value={form.settingsJson}
                    onChange={(e) => setForm((current) => ({ ...current, settingsJson: e.target.value }))}
                  />
                </div>
                <button onClick={handleSave} disabled={saving} className="btn-primary w-full disabled:opacity-60">
                  {saving ? 'Đang lưu...' : 'Lưu settings'}
                </button>
              </div>
            ) : (
              <div className="text-text-muted text-sm">Chọn một node ở danh sách bên trái.</div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}