"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import StatusBadge from "@/components/StatusBadge";
import { useToast } from "@/components/Toast";
import { ViolationPageResponse } from "@/lib/types";
import { getViolationsPage, updateViolationStatus } from "@/lib/api";
import {
  AlertTriangle,
  Search,
  RefreshCw,
  Filter,
  ExternalLink,
  MapPin,
  Clock,
  CheckCircle2,
  XCircle,
  FileText,
  ChevronLeft,
  ChevronRight,
} from "lucide-react";
import { motion } from "motion/react";

const PAGE_SIZE = 20;

export default function ViolationsPage() {
  const [pageData, setPageData] = useState<ViolationPageResponse | null>(null);
  const [page, setPage] = useState(0);
  const [loading, setLoading] = useState(true);
  const [actionId, setActionId] = useState<number | null>(null);
  const [filter, setFilter] = useState({ status: "", nodeId: "", plateText: "" });
  const { show } = useToast();

  const violations = pageData?.content ?? [];
  const totalElements = pageData?.total_elements ?? 0;
  const totalPages = pageData?.total_pages ?? 0;

  useEffect(() => {
    let active = true;
    const timer = setTimeout(() => {
      const params: Record<string, string> = {};
      if (filter.status) params.status = filter.status;
      if (filter.nodeId) params.nodeId = filter.nodeId;
      if (filter.plateText) params.plateText = filter.plateText;
      setLoading(true);
      getViolationsPage({ ...params, page, size: PAGE_SIZE })
        .then((data) => {
          if (active) setPageData(data);
        })
        .catch(() => {
          if (active) setPageData(null);
        })
        .finally(() => {
          if (active) setLoading(false);
        });
    }, 200);

    return () => {
      active = false;
      clearTimeout(timer);
    };
  }, [filter.status, filter.nodeId, filter.plateText, page]);

  // Đổi filter thì quay về trang đầu
  const applyFilter = (next: typeof filter) => {
    setFilter(next);
    setPage(0);
  };

  const reloadPage = async () => {
    try {
      const params: Record<string, string> = {};
      if (filter.status) params.status = filter.status;
      if (filter.nodeId) params.nodeId = filter.nodeId;
      if (filter.plateText) params.plateText = filter.plateText;
      const data = await getViolationsPage({ ...params, page, size: PAGE_SIZE });
      setPageData(data);
    } catch {
      /* giữ data hiện tại */
    }
  };

  const handleQuickStatus = async (id: number, status: "approved" | "rejected") => {
    setActionId(id);
    try {
      await updateViolationStatus(id, status);
      show(
        status === "approved"
          ? `Đã duyệt hồ sơ #${id}`
          : `Đã từ chối hồ sơ #${id}`,
        status === "approved" ? "success" : "info"
      );
      await reloadPage();
    } catch {
      show("Không cập nhật được trạng thái. Vui lòng thử lại.", "danger");
    } finally {
      setActionId(null);
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <AlertTriangle className="w-6 h-6 text-indigo-500" />
            <h1 className="text-2xl font-bold text-text-primary tracking-tight">
              Cơ sở dữ liệu vi phạm
            </h1>
          </div>
          <p className="text-xs text-text-muted mt-1">
            Tổng số {totalElements.toLocaleString("vi-VN")} bản ghi sự kiện vi phạm vượt đèn đỏ.
          </p>
        </div>

        <button
          onClick={() => {
            setLoading(true);
            getViolationsPage({
              ...(filter.status ? { status: filter.status } : {}),
              ...(filter.nodeId ? { nodeId: filter.nodeId } : {}),
              ...(filter.plateText ? { plateText: filter.plateText } : {}),
              page,
              size: PAGE_SIZE,
            })
              .then(setPageData)
              .catch(() => setPageData(null))
              .finally(() => setLoading(false));
          }}
          className="btn-secondary btn-sm flex items-center gap-2"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${loading ? "animate-spin text-indigo-500" : ""}`} />
          Làm mới
        </button>
      </div>

      {/* Filter Tabs & Search Bar */}
      <div className="glass-card p-4 space-y-4">
        {/* Quick Status Tabs */}
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-border pb-3">
          <div className="segmented-control">
            {[
              { label: "Tất cả", value: "" },
              { label: "Chờ duyệt", value: "pending" },
              { label: "Đã duyệt", value: "approved" },
              { label: "Từ chối", value: "rejected" },
            ].map((tab) => (
              <button
                key={tab.value}
                onClick={() => applyFilter({ ...filter, status: tab.value })}
                className={`tab-btn ${filter.status === tab.value ? "tab-active" : ""}`}
              >
                {tab.label}
              </button>
            ))}
          </div>

          <span className="text-xs text-text-muted font-mono">
            Hiển thị {violations.length} / {totalElements.toLocaleString("vi-VN")} bản ghi
          </span>
        </div>

        {/* Inputs */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
          {/* plate search */}
          <div className="relative">
            <label htmlFor="filter-plate" className="sr-only">Tìm theo biển số</label>
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-text-muted pointer-events-none" />
            <input
              id="filter-plate"
              type="search"
              value={filter.plateText}
              onChange={(e) => applyFilter({ ...filter, plateText: e.target.value })}
              placeholder="Tìm theo biển số (VD: 29-H12345)..."
              className="input-field w-full pl-9 pr-4"
            />
          </div>

          {/* Node ID */}
          <div className="relative">
            <label htmlFor="filter-node" className="sr-only">Lọc theo Node ID</label>
            <Filter className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-text-muted pointer-events-none" />
            <input
              id="filter-node"
              type="search"
              value={filter.nodeId}
              onChange={(e) => applyFilter({ ...filter, nodeId: e.target.value })}
              placeholder="Lọc theo Node ID (VD: edge-node-01)..."
              className="input-field w-full pl-9 pr-4"
            />
          </div>

          {/* Clear filters */}
          {(filter.status || filter.nodeId || filter.plateText) && (
            <div className="flex items-center">
              <button
                onClick={() => applyFilter({ status: "", nodeId: "", plateText: "" })}
                className="btn-ghost btn-sm text-rose-500 hover:text-rose-600 hover:bg-rose-500/10"
              >
                Đặt lại bộ lọc
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Main Table */}
      <div className="glass-card overflow-hidden">
        {loading && !pageData ? (
          <div className="overflow-x-auto">
            <table className="w-full text-left">
              <thead>
                <tr className="border-b border-border bg-surface-3/60 text-[10px] font-bold text-text-muted uppercase tracking-wider">
                  <th className="px-5 py-3">Mã hồ sơ / Event ID</th>
                  <th className="px-5 py-3">Biển số ANPR</th>
                  <th className="px-5 py-3">Trạng thái đèn</th>
                  <th className="px-5 py-3 hidden md:table-cell">Edge Node</th>
                  <th className="px-5 py-3 hidden lg:table-cell">Độ tin cậy</th>
                  <th className="px-5 py-3">Trạng thái</th>
                  <th className="px-5 py-3 hidden sm:table-cell">Thời gian</th>
                  <th className="px-5 py-3 text-right">Hành động</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border text-xs">
                {Array.from({ length: 8 }).map((_, i) => (
                  <motion.tr
                    key={`sk-${i}`}
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: i * 0.05, duration: 0.3 }}
                  >
                    <td className="px-5 py-4"><div className="skeleton h-3 w-20" /></td>
                    <td className="px-5 py-4"><div className="skeleton h-5 w-24 rounded-md" /></td>
                    <td className="px-5 py-4"><div className="skeleton h-4 w-16 rounded-full" /></td>
                    <td className="px-5 py-4 hidden md:table-cell"><div className="skeleton h-3 w-24" /></td>
                    <td className="px-5 py-4 hidden lg:table-cell"><div className="skeleton h-1.5 w-full rounded-full" /></td>
                    <td className="px-5 py-4"><div className="skeleton h-4 w-16 rounded-full" /></td>
                    <td className="px-5 py-4 hidden sm:table-cell"><div className="skeleton h-3 w-24" /></td>
                    <td className="px-5 py-4 text-right"><div className="skeleton h-6 w-20 ml-auto" /></td>
                  </motion.tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className={`overflow-x-auto transition-opacity ${loading ? "opacity-50 pointer-events-none" : ""}`}>
            <table className="w-full text-left">
              <thead>
                <tr className="border-b border-border bg-surface-3/60 text-[10px] font-bold text-text-muted uppercase tracking-wider">
                  <th className="px-5 py-3">Mã hồ sơ / Event ID</th>
                  <th className="px-5 py-3">Biển số ANPR</th>
                  <th className="px-5 py-3">Trạng thái đèn</th>
                  <th className="px-5 py-3 hidden md:table-cell">Edge Node</th>
                  <th className="px-5 py-3 hidden lg:table-cell">Độ tin cậy</th>
                  <th className="px-5 py-3">Trạng thái</th>
                  <th className="px-5 py-3 hidden sm:table-cell">Thời gian</th>
                  <th className="px-5 py-3 text-right">Hành động</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border text-xs">
                {violations.length > 0 ? (
                  violations.map((v, i) => (
                    <motion.tr
                      key={v.id}
                      initial={{ opacity: 0, y: 8 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: Math.min(i * 0.03, 0.3), duration: 0.3, ease: "easeOut" }}
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
                          {v.plate_text || "KHÔNG ĐỌC ĐƯỢC"}
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
                          {v.light_state === "red" ? "ĐÈN ĐỎ" : v.light_state?.toUpperCase()}
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
                            ? new Date(v.created_at).toLocaleString("vi-VN", {
                                day: "2-digit",
                                month: "2-digit",
                                hour: "2-digit",
                                minute: "2-digit",
                              })
                            : "—"}
                        </span>
                      </td>

                      <td className="px-5 py-3.5 text-right">
                        <div className="inline-flex items-center gap-1.5 justify-end">
                          {v.status === "pending" && (
                            <>
                              <button
                                onClick={() => handleQuickStatus(v.id, "approved")}
                                disabled={actionId === v.id}
                                className="btn-success btn-sm py-1 px-2 disabled:opacity-40"
                                title="Duyệt hồ sơ"
                              >
                                <CheckCircle2 className="w-3.5 h-3.5" />
                              </button>
                              <button
                                onClick={() => handleQuickStatus(v.id, "rejected")}
                                disabled={actionId === v.id}
                                className="btn-danger btn-sm py-1 px-2 disabled:opacity-40"
                                title="Từ chối hồ sơ"
                              >
                                <XCircle className="w-3.5 h-3.5" />
                              </button>
                            </>
                          )}
                          <Link
                            href={`/violations/${v.id}`}
                            className="inline-flex items-center gap-1 text-xs text-indigo-500 hover:text-indigo-600 font-semibold transition-colors"
                          >
                            Xem chi tiết <ExternalLink className="w-3.5 h-3.5" />
                          </Link>
                        </div>
                      </td>
                    </motion.tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan={8} className="px-6 py-16 text-center text-text-muted space-y-2">
                      <FileText className="w-8 h-8 mx-auto text-text-muted" />
                      <p className="text-sm font-medium">Không tìm thấy dữ liệu vi phạm</p>
                      <p className="text-xs text-text-muted">
                        Hãy thử điều chỉnh điều kiện tìm kiếm hoặc bộ lọc trạng thái phía trên.
                      </p>
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        )}

        {/* Pagination footer */}
        {totalPages > 1 && (
          <div className="flex items-center justify-between border-t border-border px-5 py-3">
            <button
              onClick={() => setPage((p) => Math.max(0, p - 1))}
              disabled={page === 0 || loading}
              className="btn-secondary btn-sm flex items-center gap-1.5 disabled:opacity-30"
            >
              <ChevronLeft className="w-4 h-4" /> Trang trước
            </button>
            <span className="text-xs text-text-muted font-mono">
              Trang {page + 1} / {totalPages}
            </span>
            <button
              onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
              disabled={page >= totalPages - 1 || loading}
              className="btn-secondary btn-sm flex items-center gap-1.5 disabled:opacity-30"
            >
              Trang sau <ChevronRight className="w-4 h-4" />
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
