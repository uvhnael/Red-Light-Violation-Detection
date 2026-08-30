"use client";

import { useEffect, useState } from "react";
import Image from "next/image";
import { useParams, useRouter } from "next/navigation";
import Link from "next/link";
import StatusBadge from "@/components/StatusBadge";
import { ViolationResponse } from "@/lib/types";
import { getViolation, updateViolationStatus, deleteViolation } from "@/lib/api";
import {
  ArrowLeft,
  CheckCircle2,
  XCircle,
  Trash2,
  AlertTriangle,
} from "lucide-react";

export default function ViolationDetailPage() {
  const params = useParams();
  const router = useRouter();
  const id = Number(params.id);

  const [violation, setViolation] = useState<ViolationResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [actionLoading, setActionLoading] = useState(false);

  useEffect(() => {
    if (!id) return;
    getViolation(id)
      .then(setViolation)
      .catch(() => setViolation(null))
      .finally(() => setLoading(false));
  }, [id]);

  const handleStatusUpdate = async (status: string) => {
    if (!violation) return;
    setActionLoading(true);
    try {
      const updated = await updateViolationStatus(violation.id, status);
      setViolation(updated);
    } catch {
      alert("Error updating violation status");
    } finally {
      setActionLoading(false);
    }
  };

  const handleDelete = async () => {
    if (!violation || !confirm("Are you sure you want to delete this violation record?")) return;
    try {
      await deleteViolation(violation.id);
      router.push("/violations");
    } catch {
      alert("Error deleting record");
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-[60vh]">
        <div className="w-12 h-12 border-4 border-indigo-500/30 border-t-indigo-500 rounded-full animate-spin" />
      </div>
    );
  }

  if (!violation) {
    return (
      <div className="flex items-center justify-center h-[60vh]">
        <div className="glass-card p-10 text-center space-y-4 max-w-md">
          <AlertTriangle className="w-12 h-12 text-amber-500 mx-auto" />
          <h2 className="text-lg font-bold text-text-primary">Record Not Found</h2>
          <p className="text-xs text-text-muted">Violation record #{id} does not exist in central database.</p>
          <Link href="/violations" className="btn-primary text-xs inline-flex">
            ← Back to Database
          </Link>
        </div>
      </div>
    );
  }

  const mediaUrl = violation.media_url
    ? `/api/v1/violations/${violation.event_id}/media/blob`
    : null;
  const mediaIsVideo = Boolean(mediaUrl && violation.media_url?.endsWith(".mp4"));

  return (
    <div className="space-y-6 max-w-5xl">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <Link
            href="/violations"
            className="p-2.5 rounded-xl bg-surface-3 border border-border text-text-muted hover:text-text-primary transition-colors"
          >
            <ArrowLeft className="w-4 h-4" />
          </Link>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-xl font-bold text-text-primary">Violation Event #{violation.id}</h1>
              <StatusBadge status={violation.status} size="md" />
            </div>
            <p className="text-xs text-text-muted font-mono mt-0.5">{violation.event_id}</p>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Info & Evidence (2 Cols) */}
        <div className="lg:col-span-2 space-y-6">
          {/* ANPR Plate & Signal Overview */}
          <div className="glass-card p-6 border-indigo-500/20 space-y-6">
            <h2 className="text-xs font-bold text-text-muted uppercase tracking-wider">
              Automated Detection Overview
            </h2>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              {/* Plate Card */}
              <div className="p-4 rounded-2xl bg-surface-3/60 border border-border space-y-2">
                <p className="text-[10px] font-bold text-text-muted uppercase tracking-wider">
                  Recognized License Plate
                </p>
                <div className="flex items-center justify-between">
                  <span className="plate-badge text-xl">
                    {violation.plate_text || "NO-PLATE"}
                  </span>
                  <div className="text-right">
                    <p className="text-xs font-bold text-indigo-500">
                      {violation.plate_confidence
                        ? `${(violation.plate_confidence * 100).toFixed(1)}%`
                        : "N/A"}
                    </p>
                    <p className="text-[10px] text-text-muted">OCR Confidence</p>
                  </div>
                </div>
              </div>

              {/* Light State Card */}
              <div className="p-4 rounded-2xl bg-surface-3/60 border border-border space-y-2">
                <p className="text-[10px] font-bold text-text-muted uppercase tracking-wider">
                  Traffic Light Signal State
                </p>
                <div className="flex items-center justify-between">
                  <span
                    className={`inline-flex items-center gap-2 text-sm font-extrabold px-3 py-1.5 rounded-full ${
                      violation.light_state === "red"
                        ? "bg-rose-500/15 text-rose-500 border border-rose-500/30"
                        : "bg-amber-500/15 text-amber-500 border border-amber-500/30"
                    }`}
                  >
                    <span
                      className={`w-2.5 h-2.5 rounded-full ${
                        violation.light_state === "red"
                          ? "bg-rose-500 animate-pulse shadow-[0_0_8px_#f43f5e]"
                          : "bg-amber-400"
                      }`}
                    />
                    {violation.light_state?.toUpperCase()}
                  </span>
                  <div className="text-right">
                    <p className="text-xs font-bold text-text-primary">
                      {violation.light_confidence
                        ? `${(violation.light_confidence * 100).toFixed(1)}%`
                        : "N/A"}
                    </p>
                    <p className="text-[10px] text-text-muted">Detector Conf.</p>
                  </div>
                </div>
              </div>
            </div>

            {/* Coordinates Grid */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
              <div className="p-3 rounded-xl bg-surface-3/60 border border-border">
                <p className="text-[10px] text-text-muted">Node ID</p>
                <p className="font-mono font-semibold text-text-primary mt-0.5">{violation.node_id}</p>
              </div>
              <div className="p-3 rounded-xl bg-surface-3/60 border border-border">
                <p className="text-[10px] text-text-muted">Track ID</p>
                <p className="font-mono font-semibold text-text-primary mt-0.5">#{violation.track_id}</p>
              </div>
              <div className="p-3 rounded-xl bg-surface-3/60 border border-border">
                <p className="text-[10px] text-text-muted">Crossing (X, Y)</p>
                <p className="font-mono font-semibold text-text-primary mt-0.5">
                  ({violation.crossing_point?.x?.toFixed(1) || 0}, {violation.crossing_point?.y?.toFixed(1) || 0})
                </p>
              </div>
              <div className="p-3 rounded-xl bg-surface-3/60 border border-border">
                <p className="text-[10px] text-text-muted">Side Transition</p>
                <p className="font-mono font-semibold text-text-primary mt-0.5">
                  Side {violation.previous_side} → {violation.current_side}
                </p>
              </div>
            </div>
          </div>

          {/* Evidence Media Preview */}
          {mediaUrl && (
            <div className="glass-card p-6 space-y-4">
              <h2 className="text-xs font-bold text-text-muted uppercase tracking-wider">
                Evidence Recording Media
              </h2>
              <div className="relative aspect-video w-full rounded-2xl overflow-hidden border border-border bg-surface-3 shadow-2xl">
                {mediaIsVideo ? (
                  <video src={mediaUrl} controls autoPlay loop muted playsInline className="w-full h-full object-contain" />
                ) : (
                  <Image src={mediaUrl} alt={violation.event_id} fill className="object-contain" unoptimized />
                )}
              </div>
            </div>
          )}
        </div>

        {/* Action Sidebar (1 Col) */}
        <div className="space-y-6">
          {/* Operator Decision Actions */}
          <div className="glass-card p-6 space-y-4 border-indigo-500/20">
            <h2 className="text-xs font-bold text-text-muted uppercase tracking-wider">
              Enforcement Decision
            </h2>
            <p className="text-xs text-text-muted leading-relaxed">
              Verify video evidence and vehicle license plate before approving fine ticket creation.
            </p>

            <div className="space-y-3">
              <button
                onClick={() => handleStatusUpdate("approved")}
                disabled={actionLoading || violation.status === "approved"}
                className="btn-success w-full py-3 text-xs font-bold disabled:opacity-40"
              >
                <CheckCircle2 className="w-4 h-4" />
                Approve &amp; Fine Ticket
              </button>

              <button
                onClick={() => handleStatusUpdate("rejected")}
                disabled={actionLoading || violation.status === "rejected"}
                className="btn-danger w-full py-3 text-xs font-bold disabled:opacity-40"
              >
                <XCircle className="w-4 h-4" />
                Reject False Positive
              </button>

              <button
                onClick={() => handleStatusUpdate("pending")}
                disabled={actionLoading || violation.status === "pending"}
                className="btn-ghost w-full py-2 text-xs border border-border disabled:opacity-40"
              >
                Reset to Pending
              </button>
            </div>
          </div>

          {/* Record Timestamps */}
          <div className="glass-card p-6 space-y-3 text-xs">
            <h2 className="text-xs font-bold text-text-muted uppercase tracking-wider">
              Audit Logs
            </h2>
            <div className="space-y-2">
              <div>
                <p className="text-text-muted text-[10px]">Created Timestamp</p>
                <p className="text-text-primary font-medium mt-0.5">
                  {violation.created_at ? new Date(violation.created_at).toLocaleString() : "—"}
                </p>
              </div>
              <div>
                <p className="text-text-muted text-[10px]">Last Updated</p>
                <p className="text-text-primary font-medium mt-0.5">
                  {violation.updated_at ? new Date(violation.updated_at).toLocaleString() : "—"}
                </p>
              </div>
            </div>
          </div>

          {/* Danger Zone */}
          <div className="glass-card p-6 border-rose-500/20 space-y-3">
            <h2 className="text-xs font-bold text-rose-500 uppercase tracking-wider">
              Danger Zone
            </h2>
            <button
              onClick={handleDelete}
              className="text-xs text-rose-500 hover:text-rose-600 font-semibold flex items-center gap-2 transition-colors cursor-pointer"
            >
              <Trash2 className="w-4 h-4" />
              Delete Record Permanently
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
