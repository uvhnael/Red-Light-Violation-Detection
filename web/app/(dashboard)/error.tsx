"use client";

/**
 * Route-level error boundary cho nhóm trang dashboard.
 * Lỗi render/fetch trầm trọng sẽ dừng ở đây thay vì màn hình trắng.
 */
import { AlertTriangle, RefreshCw } from "lucide-react";
import { useEffect } from "react";

export default function DashboardError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error("[DashboardError]", error);
  }, [error]);

  return (
    <div className="flex items-center justify-center h-[60vh]">
      <div className="glass-card p-10 text-center max-w-md space-y-4">
        <div className="w-16 h-16 mx-auto rounded-2xl bg-rose-500/10 flex items-center justify-center">
          <AlertTriangle className="w-8 h-8 text-rose-500" />
        </div>
        <h2 className="text-lg font-semibold text-text-primary">
          Không thể tải trang
        </h2>
        <p className="text-sm text-text-muted">
          Đã có lỗi xảy ra khi hiển thị trang này. Vui lòng thử lại.
        </p>
        <button onClick={reset} className="btn-primary text-sm">
          <RefreshCw className="w-4 h-4" /> Thử lại
        </button>
      </div>
    </div>
  );
}
