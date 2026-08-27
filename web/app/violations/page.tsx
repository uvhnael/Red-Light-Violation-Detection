"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import StatusBadge from "@/components/StatusBadge";
import { ViolationResponse } from "@/lib/types";
import { getViolations } from "@/lib/api";
import {
  AlertTriangle,
  Search,
  RefreshCw,
  Filter,
  ExternalLink,
  MapPin,
  Clock,
  FileText,
} from "lucide-react";

export default function ViolationsPage() {
  const [violations, setViolations] = useState<ViolationResponse[]>([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState({ status: "", nodeId: "", plateText: "" });

  const loadData = async () => {
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
  };

  useEffect(() => {
    let active = true;
    const timer = setTimeout(() => {
      const params: Record<string, string> = {};
      if (filter.status) params.status = filter.status;
      if (filter.nodeId) params.nodeId = filter.nodeId;
      if (filter.plateText) params.plateText = filter.plateText;
      setLoading(true);
      getViolations(params)
        .then((data) => {
          if (active) setViolations(data);
        })
        .catch(() => {
          if (active) setViolations([]);
        })
        .finally(() => {
          if (active) setLoading(false);
        });
    }, 200);

    return () => {
      active = false;
      clearTimeout(timer);
    };
  }, [filter.status, filter.nodeId, filter.plateText]);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <AlertTriangle className="w-6 h-6 text-indigo-500" />
            <h1 className="text-2xl font-bold text-text-primary tracking-tight">
              Violation Database
            </h1>
          </div>
          <p className="text-xs text-text-muted mt-1">
            Total of {violations.length} recorded traffic violation events.
          </p>
        </div>

        <button
          onClick={loadData}
          className="btn-ghost text-xs flex items-center gap-2 border border-border"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${loading ? "animate-spin text-indigo-500" : ""}`} />
          Refresh List
        </button>
      </div>

      {/* Filter Tabs & Search Bar */}
      <div className="glass-card p-4 space-y-4 border-indigo-500/10">
        {/* Quick Status Tabs */}
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-border pb-3">
          <div className="flex items-center gap-1.5 bg-surface-3 p-1 rounded-xl border border-border">
            {[
              { label: "All Statuses", value: "" },
              { label: "Pending", value: "pending" },
              { label: "Approved", value: "approved" },
              { label: "Rejected", value: "rejected" },
            ].map((tab) => (
              <button
                key={tab.value}
                onClick={() => setFilter({ ...filter, status: tab.value })}
                className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all cursor-pointer ${
                  filter.status === tab.value
                    ? "bg-indigo-600 text-white shadow-sm"
                    : "text-text-muted hover:text-text-primary"
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>

          <span className="text-xs text-text-muted font-mono">
            Showing {violations.length} entries
          </span>
        </div>

        {/* Inputs */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
          {/* Plate search */}
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-text-muted" />
            <input
              type="text"
              value={filter.plateText}
              onChange={(e) => setFilter({ ...filter, plateText: e.target.value })}
              placeholder="Search by license plate (e.g. 29A-12345)..."
              className="w-full bg-surface-3 border border-border rounded-xl pl-9 pr-4 py-2 text-xs text-text-primary placeholder:text-text-muted focus:outline-none focus:border-indigo-500/40"
            />
          </div>

          {/* Node ID */}
          <div className="relative">
            <Filter className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-text-muted" />
            <input
              type="text"
              value={filter.nodeId}
              onChange={(e) => setFilter({ ...filter, nodeId: e.target.value })}
              placeholder="Filter by Node ID (e.g. edge-node-01)..."
              className="w-full bg-surface-3 border border-border rounded-xl pl-9 pr-4 py-2 text-xs text-text-primary placeholder:text-text-muted focus:outline-none focus:border-indigo-500/40"
            />
          </div>

          {/* Clear filters */}
          {(filter.status || filter.nodeId || filter.plateText) && (
            <button
              onClick={() => setFilter({ status: "", nodeId: "", plateText: "" })}
              className="btn-ghost text-xs text-rose-500 hover:text-rose-600"
            >
              Reset Filters
            </button>
          )}
        </div>
      </div>

      {/* Main Table */}
      <div className="glass-card overflow-hidden">
        {loading ? (
          <div className="flex items-center justify-center py-20">
            <div className="w-10 h-10 border-4 border-indigo-500/30 border-t-indigo-500 rounded-full animate-spin" />
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left">
              <thead>
                <tr className="border-b border-border bg-surface-3/60 text-[10px] font-bold text-text-muted uppercase tracking-wider">
                  <th className="px-5 py-3">ID / Event</th>
                  <th className="px-5 py-3">License Plate</th>
                  <th className="px-5 py-3">Signal State</th>
                  <th className="px-5 py-3 hidden md:table-cell">Node</th>
                  <th className="px-5 py-3 hidden lg:table-cell">Confidence</th>
                  <th className="px-5 py-3">Status</th>
                  <th className="px-5 py-3 hidden sm:table-cell">Timestamp</th>
                  <th className="px-5 py-3 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border text-xs">
                {violations.length > 0 ? (
                  violations.map((v) => (
                    <tr
                      key={v.id}
                      className="hover:bg-surface-3/50 transition-colors cursor-pointer group"
                    >
                      <td className="px-5 py-3.5">
                        <Link href={`/violations/${v.id}`}>
                          <p className="font-mono font-bold text-text-primary group-hover:text-indigo-500 transition-colors">
                            #{v.id}
                          </p>
                          <p className="text-[10px] text-text-muted font-mono truncate max-w-[120px]">
                            {v.event_id}
                          </p>
                        </Link>
                      </td>

                      <td className="px-5 py-3.5">
                        <span className="plate-badge text-xs">
                          {v.plate_text || "UNREADABLE"}
                        </span>
                      </td>

                      <td className="px-5 py-3.5">
                        <span
                          className={`inline-flex items-center gap-1.5 text-[11px] font-bold px-2.5 py-1 rounded-full ${
                            v.light_state === "red"
                              ? "bg-rose-500/15 text-rose-500 border border-rose-500/30"
                              : "bg-amber-500/15 text-amber-500 border border-amber-500/30"
                          }`}
                        >
                          <span
                            className={`w-2 h-2 rounded-full ${
                              v.light_state === "red"
                                ? "bg-rose-500 animate-pulse shadow-[0_0_6px_#f43f5e]"
                                : "bg-amber-400"
                            }`}
                          />
                          {v.light_state?.toUpperCase()}
                        </span>
                      </td>

                      <td className="px-5 py-3.5 hidden md:table-cell">
                        <div className="flex items-center gap-1.5 text-text-secondary font-mono text-[11px]">
                          <MapPin className="w-3.5 h-3.5 text-text-muted" />
                          {v.node_id}
                        </div>
                      </td>

                      <td className="px-5 py-3.5 hidden lg:table-cell">
                        <div className="flex items-center gap-2 max-w-[120px]">
                          <div className="flex-1 h-1.5 bg-surface-3 rounded-full overflow-hidden border border-border">
                            <div
                              className="h-full bg-gradient-to-r from-indigo-500 to-violet-500 rounded-full"
                              style={{
                                width: `${Math.min((v.light_confidence || 0.9) * 100, 100)}%`,
                              }}
                            />
                          </div>
                          <span className="text-[10px] text-text-muted font-mono">
                            {v.light_confidence
                              ? `${(v.light_confidence * 100).toFixed(0)}%`
                              : "—"}
                          </span>
                        </div>
                      </td>

                      <td className="px-5 py-3.5">
                        <StatusBadge status={v.status} />
                      </td>

                      <td className="px-5 py-3.5 hidden sm:table-cell">
                        <span className="text-text-muted text-[11px] flex items-center gap-1">
                          <Clock className="w-3 h-3 text-text-muted" />
                          {v.created_at
                            ? new Date(v.created_at).toLocaleString("en-US", {
                                month: "short",
                                day: "numeric",
                                hour: "2-digit",
                                minute: "2-digit",
                              })
                            : "—"}
                        </span>
                      </td>

                      <td className="px-5 py-3.5 text-right">
                        <Link
                          href={`/violations/${v.id}`}
                          className="inline-flex items-center gap-1 text-xs text-indigo-500 hover:text-indigo-600 font-semibold transition-colors"
                        >
                          Details <ExternalLink className="w-3.5 h-3.5" />
                        </Link>
                      </td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan={8} className="px-6 py-16 text-center text-text-muted space-y-2">
                      <FileText className="w-8 h-8 mx-auto text-text-muted" />
                      <p className="text-sm font-medium">No violation records found</p>
                      <p className="text-xs text-text-muted">
                        Try adjusting your search criteria or status filter above.
                      </p>
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
