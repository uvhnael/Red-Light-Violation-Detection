"use client";

import { useEffect, useState } from "react";
import {
  TrendingUp,
  Server,
  Activity,
  MapPin,
  Database,
  WifiOff,
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
} from "recharts";
import { Stats, ViolationResponse, EdgeNodeResponse } from "@/lib/types";
import { getStats, getEdgeNodes } from "@/lib/api";

function fmtTime(iso: string) {
  const d = new Date(iso);
  const now = Date.now();
  const mins = Math.floor((now - d.getTime()) / 60000);
  if (mins < 1) return "Vừa xong";
  if (mins < 60) return `${mins} phút trước`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs} giờ trước`;
  return d.toLocaleDateString("vi-VN", { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });
}

function statusClass(s: string) {
  switch (s) {
    case "approved":
      return "status-approved";
    case "rejected":
      return "status-rejected";
    default:
      return "status-pending";
  }
}

function CheckCircleIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75L11.25 15 15 9.75M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
    </svg>
  );
}

// ── Dashboard ──
export default function DashboardPage() {
  const [stats, setStats] = useState<Stats | null>(null);
  const [nodes, setNodes] = useState<EdgeNodeResponse[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function load() {
      try {
        const [s, n] = await Promise.all([getStats(), getEdgeNodes()]);
        setStats(s);
        setNodes(n);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Cannot connect to server");
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  // Loading skeleton
  if (loading) {
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
        <div className="grid grid-cols-1 xl:grid-cols-[1fr_340px] gap-5">
          <div className="glass-card p-6 space-y-4 animate-pulse">
            <div className="skeleton h-5 w-40" />
            <div className="skeleton h-64 w-full rounded-xl" />
          </div>
          <div className="glass-card p-6 space-y-4 animate-pulse">
            <div className="skeleton h-5 w-32" />
            {Array.from({ length: 4 }).map((_, i) => (
              <div key={i} className="skeleton h-14 w-full rounded-xl" />
            ))}
          </div>
        </div>
      </div>
    );
  }

  // Error state
  if (error) {
    return (
      <div className="flex items-center justify-center h-[60vh]">
        <div className="glass-card p-10 text-center max-w-md">
          <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-rose-500/10 flex items-center justify-center">
            <WifiOff className="w-8 h-8 text-rose-400" />
          </div>
          <h2 className="text-lg font-semibold text-zinc-100 mb-2">Connection Error</h2>
          <p className="text-sm text-zinc-400 mb-6">{error}</p>
          <button
            onClick={() => window.location.reload()}
            className="btn-primary text-sm"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  // Empty state — no data at all
  const isEmpty = !stats || stats.total === 0;

  if (isEmpty) {
    return (
      <div className="flex items-center justify-center h-[60vh]">
        <div className="glass-card p-8 text-center space-y-4 max-w-md">
          <div className="w-16 h-16 mx-auto rounded-2xl bg-indigo-500/10 flex items-center justify-center">
            <Database className="w-8 h-8 text-indigo-400" />
          </div>
          <div>
            <h2 className="text-lg font-semibold text-zinc-100">No Data Yet</h2>
            <p className="text-sm text-zinc-400 mt-1">
              Run the edge node with fake-camera mode to start sending violations.
            </p>
          </div>
          <Link href="/cameras" className="btn-primary text-sm inline-flex">
            Go to Cameras →
          </Link>
        </div>
      </div>
    );
  }

  // Real data
  const d = stats!;
  const chartData: Record<string, unknown>[] = d.hourly_trend.map((pt: { hour?: string; red?: number; yellow?: number; count?: number }) => ({
    hour: pt.hour ?? "00:00",
    red: pt.red ?? pt.count ?? 0,
    yellow: pt.yellow ?? Math.floor((pt.count ?? 0) * 0.3),
  }));

  const recent: ViolationResponse[] = d.recent_pending ?? [];
  const onlineNodes = nodes.filter((n) => n.status === "online" || n.online);
  const offlineNodeCount = nodes.length - onlineNodes.length;

  return (
    <div className="space-y-6">
      {/* ── KPI CARDS ── */}
      <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4">
        {/* Today's Total */}
        <div className="glass-card p-5 flex flex-col justify-between gap-3">
          <div className="flex items-center justify-between">
            <p className="text-xs text-zinc-500 font-semibold uppercase tracking-wider">
              Violations Today
            </p>
            <div className="w-9 h-9 rounded-xl bg-indigo-500/10 flex items-center justify-center">
              <Activity className="w-4.5 h-4.5 text-indigo-400" />
            </div>
          </div>
          <div>
            <p className="text-3xl font-bold text-zinc-100 tracking-tight">
              {d.today_total}
            </p>
            <div className="flex items-center gap-1.5 mt-1">
              <span className="text-xs text-zinc-500">Total: {d.total}</span>
            </div>
          </div>
        </div>

        {/* Active Nodes */}
        <div className="glass-card p-5 flex flex-col justify-between">
          <div className="flex items-center justify-between">
            <p className="text-xs text-zinc-500 font-semibold uppercase tracking-wider">
              Edge Nodes
            </p>
            <div className="w-9 h-9 rounded-xl bg-emerald-500/10 flex items-center justify-center">
              <Server className="w-4.5 h-4.5 text-emerald-400" />
            </div>
          </div>
          <div>
            <div className="flex items-baseline gap-2">
              <p className="text-3xl font-bold text-zinc-100 tracking-tight">
                {onlineNodes.length}
              </p>
              <span className="text-lg text-zinc-400">
                / {nodes.length}
              </span>
            </div>
            <div className="flex items-center gap-1.5 mt-1">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse shadow-[0_0_6px_#34d399]" />
              <span className="text-xs text-zinc-500">{offlineNodeCount} offline</span>
            </div>
          </div>
        </div>

        {/* Approval Rate */}
        <div className="glass-card p-5 flex flex-col justify-between">
          <div className="flex items-center justify-between">
            <p className="text-xs text-zinc-500 font-semibold uppercase tracking-wider">
              Approval Rate
            </p>
            <div className="w-9 h-9 rounded-xl bg-violet-500/10 flex items-center justify-center">
              <CheckCircleIcon className="w-4.5 h-4.5 text-violet-400" />
            </div>
          </div>
          <div>
            <p className="text-3xl font-bold text-zinc-100 tracking-tight">
              {d.approval_rate.toFixed(1)}%
            </p>
            <div className="flex items-end gap-[2px] mt-1 h-5">
              {[0.4, 0.75, 0.6, 0.9, 0.7, 0.85, 0.95, 0.8, 0.7, 0.88].map((v, i) => (
                <div key={i} className="flex-1 h-full rounded-sm bg-violet-500/10 overflow-hidden">
                  <div
                    className="w-full rounded-sm bg-gradient-to-t from-violet-500 to-indigo-400 mt-auto"
                    style={{ height: `${v * 100}%` }}
                  />
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Pending Review */}
        <div className="glass-card p-5 flex flex-col justify-between">
          <div className="flex items-center justify-between">
            <p className="text-xs text-zinc-500 font-semibold uppercase tracking-wider">
              Pending Review
            </p>
            <div className="w-9 h-9 rounded-xl bg-amber-500/10 flex items-center justify-center">
              <CheckCircleIcon className="w-4.5 h-4.5 text-amber-400" />
            </div>
          </div>
          <div>
            <p className="text-3xl font-bold text-zinc-100 tracking-tight">
              {d.pending}
            </p>
            <div className="flex items-center gap-1.5 mt-1">
              <span className="text-xs text-zinc-500">
                Approved: {d.approved} · Rejected: {d.rejected}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* ── VIOLATIONS OVER TIME (Area Chart) ── */}
      <div className="glass-card p-6">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h3 className="text-sm font-semibold text-zinc-100">
              Violations Over Time
            </h3>
            <p className="text-xs text-zinc-500 mt-0.5">Last 24 hours · hourly breakdown</p>
          </div>
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <span className="w-2.5 h-2.5 rounded-full bg-rose-500" />
              <span className="text-[12px] text-zinc-400">Red Light</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="w-2.5 h-2.5 rounded-full bg-sky-500" />
              <span className="text-[12px] text-zinc-400">Yellow Light</span>
            </div>
          </div>
        </div>

        <ResponsiveContainer width="100%" height={320}>
          <AreaChart data={chartData} margin={{ top: 5, right: 10, left: -10, bottom: 0 }}>
            <defs>
              <linearGradient id="redGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#f43f5e" stopOpacity={0.35} />
                <stop offset="95%" stopColor="#f43f5e" stopOpacity={0} />
              </linearGradient>
              <linearGradient id="blueGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#38bdf8" stopOpacity={0.35} />
                <stop offset="95%" stopColor="#38bdf8" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
            <XAxis
              dataKey="hour"
              axisLine={false}
              tickLine={false}
              tick={{ fill: "#71717a", fontSize: 11 }}
              dy={8}
            />
            <YAxis
              axisLine={false}
              tickLine={false}
              tick={{ fill: "#71717a", fontSize: 11 }}
              dx={-4}
              allowDecimals={false}
            />
            <Tooltip
              contentStyle={{
                background: "rgba(24,24,27,0.95)",
                border: "1px solid rgba(255,255,255,0.08)",
                borderRadius: "12px",
                boxShadow: "0 8px 32px rgba(0,0,0,0.5)",
              }}
              labelStyle={{ color: "#a1a1aa", fontSize: 12 }}
              itemStyle={{ color: "#f4f4f5", fontSize: 13 }}
            />
            <Area
              type="monotone"
              dataKey="red"
              stroke="#f43f5e"
              strokeWidth={2.5}
              fill="url(#redGrad)"
              dot={false}
              activeDot={{ r: 5, fill: "#f43f5e", strokeWidth: 2, stroke: "#fff" }}
            />
            <Area
              type="monotone"
              dataKey="yellow"
              stroke="#38bdf8"
              strokeWidth={2.5}
              fill="url(#blueGrad)"
              dot={false}
              activeDot={{ r: 5, fill: "#38bdf8", strokeWidth: 2, stroke: "#fff" }}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {/* ── RECENT VIOLATIONS + EDGE NODES ── */}
      <div className="grid grid-cols-1 xl:grid-cols-[1fr_340px] gap-5">
        {/* Recent Violations Table */}
        <div className="glass-card overflow-hidden">
          <div className="px-5 py-4 border-b border-white/5 flex items-center justify-between">
            <div>
              <h3 className="text-sm font-semibold text-zinc-100">Recent Violations</h3>
              <p className="text-[11px] text-zinc-500 mt-0.5">
                Latest pending review items
              </p>
            </div>
            <Link
              href="/violations"
              className="text-[12px] text-indigo-400 hover:text-indigo-300 font-medium transition-colors"
            >
              View all →
            </Link>
          </div>

          {recent.length === 0 ? (
            <div className="py-16 text-center">
              <Database className="w-8 h-8 text-zinc-600 mx-auto mb-3" />
              <p className="text-sm text-zinc-500">No pending violations</p>
              <p className="text-xs text-zinc-600 mt-1">Edge nodes have not sent any data yet</p>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-white/5">
                    <th className="px-5 py-3 text-left text-[11px] font-semibold text-zinc-500 uppercase tracking-wider">
                      Plate Number
                    </th>
                    <th className="px-5 py-3 text-left text-[11px] font-semibold text-zinc-500 uppercase tracking-wider">
                      Type
                    </th>
                    <th className="px-5 py-3 text-left text-[11px] font-semibold text-zinc-500 uppercase tracking-wider hidden md:table-cell">
                      Node
                    </th>
                    <th className="px-5 py-3 text-left text-[11px] font-semibold text-zinc-500 uppercase tracking-wider hidden sm:table-cell">
                      Time
                    </th>
                    <th className="px-5 py-3 text-left text-[11px] font-semibold text-zinc-500 uppercase tracking-wider hidden lg:table-cell">
                      Confidence
                    </th>
                    <th className="px-5 py-3 text-left text-[11px] font-semibold text-zinc-500 uppercase tracking-wider">
                      Status
                    </th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/5">
                  {recent.map((v) => (
                    <tr
                      key={v.id}
                      className="hover:bg-white/[0.03] transition-colors cursor-pointer group"
                    >
                      <td className="px-5 py-3.5">
                        <Link href={`/violations/${v.id}`}>
                          <p className="text-sm font-mono font-semibold text-zinc-100 hover:text-indigo-400 transition-colors">
                            {v.plate_text ?? v.event_id.slice(0, 14) + "..."}
                          </p>
                        </Link>
                      </td>
                      <td className="px-5 py-3.5">
                        <span
                          className={`inline-flex items-center gap-1.5 text-[12px] font-medium px-2.5 py-1 rounded-full ${
                            v.light_state === "red"
                              ? "bg-rose-500/10 text-rose-400 border border-rose-500/20"
                              : v.light_state === "yellow"
                              ? "bg-amber-500/10 text-amber-400 border border-amber-500/20"
                              : "bg-sky-500/10 text-sky-400 border border-sky-500/20"
                          }`}
                        >
                          <span
                            className={`w-1.5 h-1.5 rounded-full ${
                              v.light_state === "red"
                                ? "bg-rose-500"
                                : v.light_state === "yellow"
                                ? "bg-amber-400"
                                : "bg-sky-500"
                            }`}
                          />
                          {v.light_state === "red"
                            ? "Red Light"
                            : v.light_state === "yellow"
                            ? "Yellow"
                            : v.light_state}
                        </span>
                      </td>
                      <td className="px-5 py-3.5 hidden md:table-cell">
                        <div className="flex items-center gap-1.5">
                          <MapPin className="w-3.5 h-3.5 text-zinc-600" />
                          <span className="text-[13px] text-zinc-400">{v.node_id}</span>
                        </div>
                      </td>
                      <td className="px-5 py-3.5 hidden sm:table-cell">
                        <span className="text-[13px] text-zinc-500 font-medium">
                          {v.created_at ? fmtTime(v.created_at) : "—"}
                        </span>
                      </td>
                      <td className="px-5 py-3.5 hidden lg:table-cell">
                        <div className="flex items-center gap-2">
                          <div className="flex-1 h-1.5 rounded-full bg-zinc-700 overflow-hidden max-w-[80px]">
                            <div
                              className={`h-full rounded-full transition-all duration-500 ${
                                (v.light_confidence ?? 0) > 0.95
                                  ? "bg-emerald-500"
                                  : (v.light_confidence ?? 0) > 0.85
                                  ? "bg-amber-500"
                                  : "bg-rose-500"
                              }`}
                              style={{ width: `${Math.min((v.light_confidence ?? 0.9) * 100, 100)}%` }}
                            />
                          </div>
                          <span className="text-[12px] font-medium text-zinc-400">
                            {v.light_confidence != null ? (v.light_confidence * 100).toFixed(1) + "%" : "—"}
                          </span>
                        </div>
                      </td>
                      <td className="px-5 py-3.5">
                        <span className={statusClass(v.status)}>{v.status}</span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* Edge Node Status (live from API) */}
        <div className="glass-card p-5 flex flex-col">
          <h3 className="text-sm font-semibold text-zinc-100 mb-4">
            Edge Nodes
          </h3>
          <div className="space-y-3 flex-1">
            {nodes.length === 0 ? (
              <div className="text-center py-6">
                <Server className="w-8 h-8 text-zinc-600 mx-auto mb-2" />
                <p className="text-xs text-zinc-500">No nodes registered</p>
              </div>
            ) : (
              nodes.slice(0, 6).map((node) => (
                <Link
                  key={node.node_id}
                  href={`/nodes/${node.node_id}`}
                  className="flex items-center gap-3 px-3 py-2.5 rounded-xl bg-zinc-900/40 border border-white/5 hover:border-indigo-500/15 transition-colors group cursor-pointer"
                >
                  <div className="w-8 h-8 rounded-lg bg-indigo-500/10 flex items-center justify-center shrink-0">
                    <Server className="w-4 h-4 text-indigo-400" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-[13px] font-medium text-zinc-100 truncate">
                      {node.name || node.node_id}
                    </p>
                    <p className="text-[11px] font-mono text-zinc-500 truncate">
                      {node.ip_address || "—"}
                    </p>
                  </div>
                  <div className="flex items-center gap-1.5">
                    <span
                      className={`w-2 h-2 rounded-full ${
                        node.status === "online" || node.online
                          ? "bg-emerald-500 shadow-[0_0_6px_#34d399] animate-pulse"
                          : "bg-zinc-600"
                      }`}
                    />
                    <span
                      className={`text-[11px] font-medium ${
                        node.status === "online" || node.online
                          ? "text-emerald-400"
                          : "text-zinc-500"
                      }`}
                    >
                      {node.online ? "online" : node.status || "offline"}
                    </span>
                  </div>
                </Link>
              ))
            )}
          </div>

          <Link
            href="/nodes"
            className="mt-4 text-center text-[12px] text-indigo-400 hover:text-indigo-300 transition-colors"
          >
            View all nodes →
          </Link>
        </div>
      </div>
    </div>
  );
}