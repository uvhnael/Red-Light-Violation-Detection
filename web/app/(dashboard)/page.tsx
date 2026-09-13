"use client";

import { useEffect, useState } from "react";
import {
  Server,
  Activity,
  WifiOff,
  RefreshCw,
  ArrowRight,
  ShieldAlert,
  CheckCircle2,
  Clock,
  Eye,
  Sparkles,
  MapPin,
} from "lucide-react";
import Link from "next/link";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  BarChart as ReBarChart,
  Bar,
  Cell,
} from "recharts";
import StatusBadge from "@/components/StatusBadge";
import { Stats, EdgeNodeResponse, ViolationResponse } from "@/lib/types";
import { getStats, getEdgeNodes } from "@/lib/api";
import { effectiveNodeStatus, isNodeOnline } from "@/lib/nodes";
import { FadeItem, StaggerList } from "@/components/motion";

function formatDate(iso: string) {
  if (!iso) return "—";
  const d = new Date(iso);
  return d.toLocaleDateString("vi-VN", {
    day: "2-digit",
    month: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export default function DashboardPage() {
  const [stats, setStats] = useState<Stats | null>(null);
  const [nodes, setNodes] = useState<EdgeNodeResponse[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<"trend" | "dist">("trend");

  const loadData = async () => {
    setLoading(true);
    setError(null);
    try {
      const [s, n] = await Promise.all([getStats(), getEdgeNodes()]);
      setStats(s);
      setNodes(n);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Cannot connect to server");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    let active = true;
    Promise.all([getStats(), getEdgeNodes()])
      .then(([s, n]) => {
        if (active) {
          setStats(s);
          setNodes(n);
          setLoading(false);
        }
      })
      .catch((err) => {
        if (active) {
          setError(err instanceof Error ? err.message : "Cannot connect to server");
          setLoading(false);
        }
      });

    return () => {
      active = false;
    };
  }, []);

  // Loading skeleton
  if (loading && !stats) {
    return (
      <div className="space-y-6">
        <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4">
          {Array.from({ length: 4 }).map((_, i) => (
            <div key={i} className="glass-card p-5 space-y-4 animate-pulse">
              <div className="skeleton h-4 w-24" />
              <div className="skeleton h-8 w-16" />
              <div className="skeleton h-3 w-32" />
            </div>
          ))}
        </div>
        <div className="glass-card p-6 space-y-4 animate-pulse">
          <div className="skeleton h-5 w-48" />
          <div className="skeleton h-80 w-full rounded-xl" />
        </div>
      </div>
    );
  }

  // Error state
  if (error && !stats) {
    return (
      <div className="flex items-center justify-center h-[60vh]">
        <div className="glass-card p-10 text-center max-w-md space-y-4">
          <div className="w-16 h-16 mx-auto rounded-2xl bg-rose-500/10 flex items-center justify-center">
            <WifiOff className="w-8 h-8 text-rose-500" />
          </div>
          <h2 className="text-lg font-semibold text-text-primary">Connection Error</h2>
          <p className="text-sm text-text-muted">{error}</p>
          <button onClick={loadData} className="btn-primary text-sm">
            <RefreshCw className="w-4 h-4" /> Retry Connection
          </button>
        </div>
      </div>
    );
  }

  const d = stats || {
    total: 0,
    today_total: 0,
    pending: 0,
    approved: 0,
    rejected: 0,
    approval_rate: 0,
    active_nodes: 0,
    offline_nodes: 0,
    violations_per_node: {},
    violations_per_node_today: {},
    violations_per_light_state: {},
    hourly_trend: [],
    recent_pending: [],
  };

  // Trend theo giờ: backend trả {hour, red, yellow} — tổng = red + yellow
  const chartData = (d.hourly_trend || []).map((pt) => ({
    hour: pt.hour ?? "00:00",
    red: pt.red ?? 0,
    yellow: pt.yellow ?? 0,
  }));
  const peak = chartData.reduce(
    (best, p) => (p.red + p.yellow > best.red + best.yellow ? p : best),
    chartData[0] ?? { hour: "—", red: 0, yellow: 0 }
  );
  const peakCount = peak.red + peak.yellow;

  const lightDistData = [
    { name: "Đèn đỏ", count: d.violations_per_light_state?.red || 0, color: "#f43f5e" },
    { name: "Đèn vàng", count: d.violations_per_light_state?.yellow || 0, color: "#f59e0b" },
    { name: "Khác / không rõ", count: d.violations_per_light_state?.green || 0, color: "#10b981" },
  ];

  // Top node theo vi phạm HÔM NAY (kèm tổng tích lũy để so góc độ)
  const nodeRankToday = Object.entries(d.violations_per_node_today || {})
    .map(([nodeId, count]) => ({
      nodeId,
      today: count,
      total: d.violations_per_node?.[nodeId] ?? count,
    }))
    .sort((a, b) => b.today - a.today)
    .slice(0, 5);
  const maxTodayNode = Math.max(1, ...nodeRankToday.map((n) => n.today));

  const recent: ViolationResponse[] = d.recent_pending || [];
  const onlineNodes = nodes.filter(isNodeOnline);
  const offlineNodeCount = nodes.length - onlineNodes.length;

  return (
    <StaggerList className="space-y-6">
      {/* ── HEADER BANNER ── */}
      <FadeItem className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 glass-card p-6 border-indigo-500/20">
        <div>
          <div className="flex items-center gap-2">
            <span className="px-2.5 py-0.5 rounded-full text-[10px] font-bold bg-indigo-500/15 text-indigo-500 border border-indigo-500/30 uppercase tracking-wider">
              Giám sát trực tiếp
            </span>
            <span className="text-xs text-text-muted capitalize">
              {new Date().toLocaleDateString("vi-VN", { weekday: "long", day: "2-digit", month: "2-digit", year: "numeric" })}
            </span>
          </div>
          <h1 className="text-2xl font-bold text-text-primary mt-1 tracking-tight">
            Tổng quan Giám sát Vi phạm
          </h1>
          <p className="text-xs text-text-muted mt-1">
            Hệ thống phát hiện vi phạm vượt đèn đỏ tự động bằng AI thời gian thực trên toàn mạng lưới Edge Nodes.
          </p>
        </div>
        <div className="flex items-center gap-2.5">
          <button
            onClick={loadData}
            className="btn-secondary btn-md"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? "animate-spin text-indigo-500" : ""}`} />
            Làm mới dữ liệu
          </button>
          <Link href="/review" className="btn-primary btn-md">
            <ShieldAlert className="w-4 h-4" />
            Hàng chờ duyệt ({d.pending})
          </Link>
        </div>
      </FadeItem>

      {/* ── KPI BENTO GRID ── */}
      <FadeItem className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4">
        {/* Violations Today */}
        <div className="glass-card-hover p-5 flex flex-col justify-between border-indigo-500/20">
          <div className="flex items-center justify-between">
            <p className="text-[11px] font-bold text-text-muted uppercase tracking-wider">
              Vi phạm hôm nay
            </p>
            <div className="w-9 h-9 rounded-xl bg-indigo-500/15 flex items-center justify-center text-indigo-500">
              <Activity className="w-5 h-5" />
            </div>
          </div>
          <div className="mt-4">
            <div className="flex items-baseline gap-2">
              <p className="text-3xl font-extrabold text-text-primary tracking-tight">
                {d.today_total.toLocaleString("vi-VN")}
              </p>
              <span className="text-xs text-text-muted">vụ vi phạm</span>
            </div>
            <p className="text-[11px] text-text-muted mt-1">
              Tổng số tích lũy: <span className="text-text-secondary font-semibold">{d.total.toLocaleString("vi-VN")}</span>
            </p>
          </div>
        </div>

        {/* Pending Review */}
        <div className="glass-card-hover p-5 flex flex-col justify-between border-amber-500/20">
          <div className="flex items-center justify-between">
            <p className="text-[11px] font-bold text-text-muted uppercase tracking-wider">
              Hồ sơ chờ duyệt
            </p>
            <div className="w-9 h-9 rounded-xl bg-amber-500/15 flex items-center justify-center text-amber-500">
              <Clock className="w-5 h-5" />
            </div>
          </div>
          <div className="mt-4">
            <div className="flex items-baseline gap-2">
              <p className="text-3xl font-extrabold text-amber-500 tracking-tight">
                {d.pending.toLocaleString("vi-VN")}
              </p>
              <span className="text-xs text-amber-500 font-medium">cần cán bộ xử lý</span>
            </div>
            <p className="text-[11px] text-text-muted mt-1">
              Đã duyệt: <span className="text-emerald-500 font-medium">{d.approved}</span> · Từ chối: <span className="text-rose-500 font-medium">{d.rejected}</span>
            </p>
          </div>
        </div>

        {/* Active Edge Nodes */}
        <div className="glass-card-hover p-5 flex flex-col justify-between border-emerald-500/20">
          <div className="flex items-center justify-between">
            <p className="text-[11px] font-bold text-text-muted uppercase tracking-wider">
              Edge Nodes trực tuyến
            </p>
            <div className="w-9 h-9 rounded-xl bg-emerald-500/15 flex items-center justify-center text-emerald-500">
              <Server className="w-5 h-5" />
            </div>
          </div>
          <div className="mt-4">
            <div className="flex items-baseline gap-2">
              <p className="text-3xl font-extrabold text-text-primary tracking-tight">
                {onlineNodes.length}
              </p>
              <span className="text-base text-text-muted">/ {nodes.length || 1} hoạt động</span>
            </div>
            <div className="flex items-center gap-2 mt-1">
              <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse shadow-[0_0_8px_#10b981]" />
              <span className="text-[11px] text-text-muted">{offlineNodeCount} node mất kết nối</span>
            </div>
          </div>
        </div>

        {/* Approval Rate */}
        <div className="glass-card-hover p-5 flex flex-col justify-between border-violet-500/20">
          <div className="flex items-center justify-between">
            <p className="text-[11px] font-bold text-text-muted uppercase tracking-wider">
              Tỷ lệ xử lý hợp lệ
            </p>
            <div className="w-9 h-9 rounded-xl bg-violet-500/15 flex items-center justify-center text-violet-500">
              <CheckCircle2 className="w-5 h-5" />
            </div>
          </div>
          <div className="mt-4">
            <p className="text-3xl font-extrabold text-text-primary tracking-tight">
              {d.approval_rate ? d.approval_rate.toFixed(1) : "0.0"}%
            </p>
            <div className="w-full bg-surface-3 rounded-full h-2 mt-2 overflow-hidden border border-border">
              <div
                className="bg-gradient-to-r from-violet-500 to-indigo-500 h-full rounded-full transition-all duration-700"
                style={{ width: `${Math.min(d.approval_rate || 0, 100)}%` }}
              />
            </div>
          </div>
        </div>
      </FadeItem>

      {/* ── CHARTS SECTION ── */}
      <FadeItem className="glass-card p-6 space-y-6">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <h3 className="text-base font-bold text-text-primary">
              Phân tích Thống kê Vi phạm
            </h3>
            <p className="text-xs text-text-muted mt-0.5">
              Phân bổ vi phạm theo khung giờ (phân tách đèn đỏ/vàng) và theo trạng thái đèn tín hiệu — phạm vi hôm nay.
            </p>
          </div>
          <div className="segmented-control">
            <button
              onClick={() => setActiveTab("trend")}
              className={`tab-btn ${activeTab === "trend" ? "tab-active" : ""}`}
            >
              Biểu đồ theo giờ
            </button>
            <button
              onClick={() => setActiveTab("dist")}
              className={`tab-btn ${activeTab === "dist" ? "tab-active" : ""}`}
            >
              Phân bổ đèn tín hiệu
            </button>
          </div>
        </div>

        {activeTab === "trend" && peakCount > 0 && (
          <div className="flex items-center gap-2 text-xs text-text-muted -mt-3 pb-1">
            <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-rose-500/10 border border-rose-500/20 text-rose-500 font-semibold">
              <span className="w-1.5 h-1.5 rounded-full bg-rose-500 animate-pulse" />
              Giờ cao điểm: {peak.hour}
            </span>
            <span>với {peakCount.toLocaleString("vi-VN")} vụ — gồm {peak.red.toLocaleString("vi-VN")} đèn đỏ, {peak.yellow.toLocaleString("vi-VN")} đèn vàng</span>
          </div>
        )}

        {activeTab === "trend" ? (
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={chartData} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
              <defs>
                <linearGradient id="redGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#f43f5e" stopOpacity={0.4} />
                  <stop offset="95%" stopColor="#f43f5e" stopOpacity={0} />
                </linearGradient>
                <linearGradient id="yellowGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#f59e0b" stopOpacity={0.35} />
                  <stop offset="95%" stopColor="#f59e0b" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(100,116,139,0.15)" vertical={false} />
              <XAxis
                dataKey="hour"
                axisLine={false}
                tickLine={false}
                tick={{ fill: "#64748b", fontSize: 11 }}
                dy={6}
                interval="preserveStartEnd"
                minTickGap={28}
              />
              <YAxis
                axisLine={false}
                tickLine={false}
                tick={{ fill: "#64748b", fontSize: 11 }}
                allowDecimals={false}
                width={48}
                tickFormatter={(v: number) => v.toLocaleString("vi-VN")}
              />
              <Tooltip
                formatter={(value: unknown, name: unknown) => [
                  Number(value).toLocaleString("vi-VN"),
                  name === "red" ? "Vượt đèn đỏ" : "Vượt đèn vàng",
                ] as [string, string]}
                labelFormatter={(l: unknown) => `Khung giờ ${String(l)}`}
                contentStyle={{
                  borderRadius: 12,
                  border: "1px solid rgba(100,116,139,0.3)",
                  background: "rgba(15,23,42,0.92)",
                  fontSize: 12,
                }}
                labelStyle={{ color: "#e2e8f0", fontWeight: 600 }}
              />
              <Area
                type="monotone"
                dataKey="yellow"
                name="yellow"
                stackId="1"
                stroke="#f59e0b"
                strokeWidth={2}
                fill="url(#yellowGrad)"
              />
              <Area
                type="monotone"
                dataKey="red"
                name="red"
                stackId="1"
                stroke="#f43f5e"
                strokeWidth={3}
                fill="url(#redGrad)"
                activeDot={{ r: 6, fill: "#f43f5e", stroke: "#fff", strokeWidth: 2 }}
              />
            </AreaChart>
          </ResponsiveContainer>
        ) : (
          <ResponsiveContainer width="100%" height={300}>
            <ReBarChart data={lightDistData} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(100,116,139,0.15)" vertical={false} />
              <XAxis dataKey="name" axisLine={false} tickLine={false} tick={{ fill: "#64748b", fontSize: 12 }} />
              <YAxis
                axisLine={false}
                tickLine={false}
                tick={{ fill: "#64748b", fontSize: 11 }}
                allowDecimals={false}
                width={48}
                tickFormatter={(v: number) => v.toLocaleString("vi-VN")}
              />
              <Tooltip
                formatter={(value: unknown) => [
                  Number(value).toLocaleString("vi-VN"),
                  "Số vi phạm",
                ] as [string, string]}
                contentStyle={{
                  borderRadius: 12,
                  border: "1px solid rgba(100,116,139,0.3)",
                  background: "rgba(15,23,42,0.92)",
                  fontSize: 12,
                }}
                labelStyle={{ color: "#e2e8f0", fontWeight: 600 }}
                cursor={{ fill: "rgba(100,116,139,0.08)" }}
              />
              <Bar dataKey="count" radius={[8, 8, 0, 0]}>
                {lightDistData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Bar>
            </ReBarChart>
          </ResponsiveContainer>
        )}
      </FadeItem>

      {/* ── LOWER GRID: PENDING FEED & NODE RANKING HÔM NAY & EDGE STATUS ── */}
      <FadeItem className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left: Pending Violations Feed (5 cols) */}
        <div className="lg:col-span-5 glass-card p-6 space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-base font-bold text-text-primary flex items-center gap-2">
                <Clock className="w-4.5 h-4.5 text-amber-500" />
                Hồ sơ chờ cán bộ xét duyệt
              </h3>
              <p className="text-xs text-text-muted mt-0.5">
                Mỗi hồ sơ phải được mở và xem xét trực tiếp trước khi duyệt.
              </p>
            </div>
            <Link
              href="/review"
              className="text-xs font-semibold flex items-center gap-1 transition-colors"
              style={{ color: "rgb(var(--accent-rgb))" }}
            >
              Trạm duyệt <ArrowRight className="w-3.5 h-3.5" />
            </Link>
          </div>

          <div className="space-y-3">
                      {recent.length > 0 ? (
                        recent.slice(0, 3).map((v) => (
                          <div
                            key={v.id}
                            className="p-4 rounded-xl bg-surface-3/40 border border-border hover:border-indigo-500/30 transition-all flex flex-col sm:flex-row sm:items-center justify-between gap-3"
                          >
                            <div className="space-y-2 min-w-0">
                              <div className="flex items-center gap-2 flex-wrap">
                                <span className="plate-badge text-xs">
                                  {v.plate_text || "KHÔNG ĐỌC ĐƯỢC"}
                                </span>
                                <StatusBadge status={v.status} />
                                <span className="text-[10px] text-text-muted font-mono">
                                  #{v.id}
                                </span>
                                <span
                                  className={`inline-flex items-center gap-1.5 text-[10px] font-bold px-2 py-0.5 rounded-full ${
                                    v.light_state === "red"
                                      ? "bg-rose-500/15 text-rose-500 border border-rose-500/30"
                                      : "bg-amber-500/15 text-amber-500 border border-amber-500/30"
                                  }`}
                                >
                                  <span
                                    className={`w-1.5 h-1.5 rounded-full ${
                                      v.light_state === "red"
                                        ? "bg-rose-500 animate-pulse shadow-[0_0_6px_#f43f5e]"
                                        : "bg-amber-400"
                                    }`}
                                  />
                                  {v.light_state === "red" ? "ĐÈN ĐỎ" : v.light_state?.toUpperCase()}
                                </span>
                              </div>
                              <div className="flex items-center gap-3 text-[10px] text-text-muted">
                                <span className="flex items-center gap-1">
                                  <MapPin className="w-3 h-3" /> Node: {v.node_id}
                                </span>
                                <span>•</span>
                                <span className="flex items-center gap-1">
                                  <Clock className="w-3 h-3" /> {formatDate(v.created_at)}
                                </span>
                              </div>
                            </div>

                            <div className="flex items-center gap-2 shrink-0">
                              <Link
                                href={`/violations/${v.id}`}
                                className="btn-secondary btn-sm inline-flex items-center gap-1.5"
                              >
                                <Eye className="w-3.5 h-3.5" />
                                Chi tiết
                              </Link>
                            </div>
                          </div>
                        ))
                      ) : (
                        <div className="text-center py-10 text-text-muted space-y-2">
                          <CheckCircle2 className="w-8 h-8 text-emerald-500 mx-auto" />
                          <p className="text-xs">Hiện tại không có hồ sơ nào chờ kiểm tra.</p>
                        </div>
                      )}
                    </div>
        </div>

        {/* Middle: Node vi phạm nhiều nhất HÔM NAY (4 cols) */}
        <div className="lg:col-span-4 glass-card p-6 space-y-4">
          <div>
            <h3 className="text-base font-bold text-text-primary flex items-center gap-2">
              <Activity className="w-4.5 h-4.5 text-rose-500" />
              Node vi phạm nhiều nhất hôm nay
            </h3>
            <p className="text-xs text-text-muted mt-0.5">
              Xếp hạng theo số vụ ghi nhận trong ngày.
            </p>
          </div>

          {nodeRankToday.length > 0 ? (
            <div className="space-y-3">
              {nodeRankToday.map((n, i) => (
                <Link
                  key={n.nodeId}
                  href={`/nodes/${n.nodeId}`}
                  className="block p-3.5 rounded-xl bg-surface-3/40 border border-border hover:border-indigo-500/30 transition-all group"
                >
                  <div className="flex items-center justify-between gap-3">
                    <div className="flex items-center gap-2.5 min-w-0">
                      <span
                        className={`w-6 h-6 rounded-lg flex items-center justify-center text-[11px] font-extrabold shrink-0 ${
                          i === 0
                            ? "bg-rose-500/20 text-rose-500 border border-rose-500/40"
                            : i === 1
                              ? "bg-amber-500/20 text-amber-500 border border-amber-500/40"
                              : "bg-surface-4/60 text-text-muted border border-border"
                        }`}
                      >
                        {i + 1}
                      </span>
                      <div className="min-w-0">
                        <p className="text-xs font-semibold text-text-primary group-hover:text-indigo-500 transition-colors truncate">
                          {n.nodeId}
                        </p>
                        <p className="text-[10px] text-text-muted">
                          Tích lũy: {n.total.toLocaleString("vi-VN")}
                        </p>
                      </div>
                    </div>
                    <p className="text-sm font-extrabold text-rose-500 shrink-0">
                      {n.today.toLocaleString("vi-VN")}
                      <span className="text-[10px] font-medium text-text-muted ml-1">vụ</span>
                    </p>
                  </div>
                  <div className="mt-2.5 h-1.5 rounded-full bg-surface-4/60 overflow-hidden">
                    <div
                      className={`h-full rounded-full transition-all duration-700 ${
                        i === 0
                          ? "bg-gradient-to-r from-rose-500 to-orange-500"
                          : "bg-gradient-to-r from-indigo-500 to-violet-500"
                      }`}
                      style={{ width: `${Math.max(6, (n.today / maxTodayNode) * 100)}%` }}
                    />
                  </div>
                </Link>
              ))}
            </div>
          ) : (
            <div className="text-center py-10 text-text-muted space-y-2">
              <Activity className="w-8 h-8 text-indigo-500/50 mx-auto" />
              <p className="text-xs">Hôm nay chưa ghi nhận vi phạm nào từ node nào.</p>
            </div>
          )}
        </div>

        {/* Right: Edge Node Quick Status (3 cols) */}
        <div className="lg:col-span-3 glass-card p-6 flex flex-col justify-between space-y-4">
          <div>
            <div className="flex items-center justify-between mb-4">
              <div>
                <h3 className="text-base font-bold text-text-primary flex items-center gap-2">
                  <Server className="w-4.5 h-4.5 text-indigo-500" />
                  Trạng thái Edge Nodes
                </h3>
                <p className="text-xs text-text-muted mt-0.5">
                  Phần cứng camera biên đang xử lý suy luận AI tại chỗ.
                </p>
              </div>
              <Link
                href="/nodes"
                className="text-xs font-semibold flex items-center gap-1"
                style={{ color: "rgb(var(--accent-rgb))" }}
              >
                Tất cả Nodes <ArrowRight className="w-3.5 h-3.5" />
              </Link>
            </div>

            <div className="space-y-2.5">
              {nodes.length > 0 ? (
                nodes.slice(0, 3).map((node) => (
                  <Link
                    key={node.node_id}
                    href={`/nodes/${node.node_id}`}
                    className="p-3.5 rounded-xl bg-surface-3/40 border border-border hover:border-indigo-500/30 transition-all flex items-center justify-between text-xs group"
                  >
                    <div className="space-y-0.5">
                      <p className="font-semibold text-text-primary group-hover:text-indigo-500 transition-colors">
                        {node.name}
                      </p>
                      <p className="text-[10px] text-text-muted font-mono">
                        {node.node_id} · {node.ip_address || "Mạng nội bộ"}
                      </p>
                    </div>

                    <div className="text-right space-y-1">
                      <StatusBadge status={effectiveNodeStatus(node)} />
                    </div>
                  </Link>
                ))
              ) : (
                <div className="p-6 text-center text-text-muted text-xs">
                  Chưa có edge node nào được kết nối.
                </div>
              )}
            </div>
          </div>

          <div className="mt-6 pt-4 border-t border-border">
            <p className="text-[10px] font-bold text-text-muted uppercase tracking-wider mb-2">
              Công cụ Hiệu chuẩn AI
            </p>
            <p className="text-xs text-text-muted leading-relaxed">
              Vẽ vạch dừng, vùng đèn tín hiệu và hướng xe chạy trực tiếp trên luồng camera của từng node.
            </p>
            <div className="flex flex-wrap gap-2 mt-3">
              {onlineNodes.slice(0, 2).map((node) => (
                <Link
                  key={node.node_id}
                  href={`/nodes/${node.node_id}`}
                  className="btn-secondary btn-sm inline-flex items-center gap-1.5"
                >
                  <Sparkles className="w-3.5 h-3.5" />
                  {node.name}
                </Link>
              ))}
              {nodes.length === 0 && (
                <span className="text-xs text-text-muted">Chưa có node nào đang hoạt động.</span>
              )}
            </div>
          </div>
        </div>
      </FadeItem>
    </StaggerList>
  );
}