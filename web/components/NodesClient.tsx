'use client';

import { useMemo, useState } from 'react';
import Link from 'next/link';
import { EdgeNodeResponse } from '@/lib/types';
import { getEdgeNodes, updateEdgeNodeSettings } from '@/lib/api';
import { useToast } from '@/components/Toast';
import StatusBadge from '@/components/StatusBadge';
import { effectiveNodeStatus } from '@/lib/nodes';
import { Server, Activity, WifiOff, RefreshCw, Eye } from 'lucide-react';

type NodeFormState = {
  name: string;
  ipAddress: string;
  settingsJson: string;
};

const emptyForm: NodeFormState = {
  name: '',
  ipAddress: '',
  settingsJson: '{\n  "note": ""\n}',
};

function toFormState(node: EdgeNodeResponse): NodeFormState {
  return {
    name: node.name ?? '',
    ipAddress: node.ip_address ?? '',
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

  const onlineCount = nodes.filter((node) => effectiveNodeStatus(node) === 'online').length;

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <Server className="w-6 h-6 text-indigo-500" />
            <h1 className="text-2xl font-bold text-text-primary tracking-tight">
              Quản lý Edge Nodes
            </h1>
          </div>
          <p className="text-xs text-text-muted mt-1">
            Danh sách node trạm xử lý biên và cấu hình tham số trực tiếp.
          </p>
        </div>
        <button
          onClick={loadNodes}
          className="btn-secondary btn-sm flex items-center gap-2"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${loading ? "animate-spin text-indigo-500" : ""}`} />
          Làm mới
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="glass-card p-5 space-y-1">
          <div className="flex items-center justify-between">
            <p className="text-xs text-text-muted font-medium">Tổng số Edge Node</p>
            <div className="w-8 h-8 rounded-lg bg-indigo-500/10 flex items-center justify-center">
              <Server className="w-4 h-4 text-indigo-500" />
            </div>
          </div>
          <p className="text-3xl font-extrabold text-text-primary mt-2">{nodes.length}</p>
        </div>

        <div className="glass-card p-5 space-y-1">
          <div className="flex items-center justify-between">
            <p className="text-xs text-text-muted font-medium">Đang hoạt động (Online)</p>
            <div className="w-8 h-8 rounded-lg bg-emerald-500/10 flex items-center justify-center">
              <Activity className="w-4 h-4 text-emerald-500" />
            </div>
          </div>
          <p className="text-3xl font-extrabold text-emerald-500 mt-2">{onlineCount}</p>
        </div>

        <div className="glass-card p-5 space-y-1">
          <div className="flex items-center justify-between">
            <p className="text-xs text-text-muted font-medium">Mất kết nối (Offline)</p>
            <div className="w-8 h-8 rounded-lg bg-rose-500/10 flex items-center justify-center">
              <WifiOff className="w-4 h-4 text-rose-500" />
            </div>
          </div>
          <p className="text-3xl font-extrabold text-rose-500 mt-2">{Math.max(0, nodes.length - onlineCount)}</p>
        </div>
      </div>

      {loading && nodes.length === 0 ? (
        <div className="glass-card p-12 text-center text-text-muted">
          <div className="w-10 h-10 border-4 border-indigo-500/30 border-t-indigo-500 rounded-full animate-spin mx-auto mb-3" />
          <p className="text-xs">Đang tải danh sách Edge Node...</p>
        </div>
      ) : error ? (
        <div className="glass-card p-8 text-center text-rose-500 text-xs font-medium">{error}</div>
      ) : (
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
          <div className="glass-card overflow-hidden">
            <div className="px-6 py-4 border-b border-border flex items-center justify-between bg-surface-3/40">
              <h2 className="text-xs font-bold text-text-muted uppercase tracking-wider">Danh sách Edge Node</h2>
              <span className="text-xs font-mono text-text-muted">{nodes.length} nodes</span>
            </div>
            <div className="divide-y divide-border">
              {nodes.length > 0 ? nodes.map((node) => (
                <div
                  key={node.node_id}
                  className={`w-full text-left px-6 py-4 transition-colors cursor-pointer ${
                    selectedNodeId === node.node_id
                      ? 'bg-indigo-600/10 border-l-4 border-indigo-500'
                      : 'hover:bg-surface-3/50 border-l-4 border-transparent'
                  }`}
                  onClick={() => handleSelectNode(node)}
                >
                  <div className="flex items-center justify-between gap-3">
                    <div>
                      <p className="font-semibold text-text-primary text-sm">{node.name}</p>
                      <p className="text-xs text-text-muted font-mono mt-0.5">{node.node_id}</p>
                    </div>
                    <div className="text-right flex flex-col items-end gap-1">
                      <StatusBadge status={effectiveNodeStatus(node)} size="sm" />
                      <p className="text-xs font-mono text-text-muted">{node.ip_address || '—'}</p>
                    </div>
                  </div>
                  <div className="flex items-center justify-between mt-3 text-xs text-text-muted">
                    <span>
                      Ping gần nhất: {node.last_ping ? new Date(node.last_ping).toLocaleString('vi-VN') : '—'}
                    </span>
                    <Link
                      href={`/nodes/${node.node_id}`}
                      className="btn-secondary btn-sm inline-flex items-center gap-1.5"
                      onClick={(e) => e.stopPropagation()}
                    >
                      <Eye className="w-3.5 h-3.5" />
                      Chi tiết
                    </Link>
                  </div>
                </div>
              )) : (
                <div className="px-6 py-12 text-center text-text-muted text-xs">Chưa có edge node nào đăng ký.</div>
              )}
            </div>
          </div>

          <div className="glass-card p-6 space-y-4">
            <div className="flex items-center justify-between border-b border-border pb-4">
              <div>
                <h2 className="text-xs font-bold text-text-muted uppercase tracking-wider">Cấu hình Node</h2>
                <p className="text-xs text-text-muted mt-1">Chỉnh sửa thông số cho node đang chọn.</p>
              </div>
              {selectedNode && (
                <StatusBadge status={effectiveNodeStatus(selectedNode)} size="sm" />
              )}
            </div>

            {selectedNode ? (
              <div className="space-y-4">
                <div>
                  <label className="block text-xs text-text-muted mb-1.5 font-medium">Tên node</label>
                  <input
                    className="input-field w-full"
                    value={form.name}
                    onChange={(e) => setForm((current) => ({ ...current, name: e.target.value }))}
                    placeholder="VD: Edge Node Trạm 1"
                  />
                </div>
                <div>
                  <label className="block text-xs text-text-muted mb-1.5 font-medium">Địa chỉ IP</label>
                  <input
                    className="input-field w-full font-mono"
                    value={form.ipAddress}
                    onChange={(e) => setForm((current) => ({ ...current, ipAddress: e.target.value }))}
                    placeholder="VD: 192.168.1.100"
                  />
                </div>
                <div>
                  <label className="block text-xs text-text-muted mb-1.5 font-medium">Tham số JSON (Settings)</label>
                  <textarea
                    className="input-field w-full min-h-56 font-mono text-xs"
                    value={form.settingsJson}
                    onChange={(e) => setForm((current) => ({ ...current, settingsJson: e.target.value }))}
                  />
                </div>
                <button
                  onClick={handleSave}
                  disabled={saving}
                  className="btn-primary w-full py-2.5 disabled:opacity-60"
                >
                  {saving ? 'Đang lưu cấu hình...' : 'Lưu cấu hình'}
                </button>
              </div>
            ) : (
              <div className="text-text-muted text-xs py-8 text-center">Chọn một node ở danh sách bên trái để cấu hình.</div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}