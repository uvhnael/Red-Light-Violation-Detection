import Link from "next/link";
import { FileQuestion, Home } from "lucide-react";

/** 404 toàn cục. */
export default function NotFound() {
  return (
    <div className="min-h-screen bg-surface-0 flex items-center justify-center px-4">
      <div className="glass-card p-10 text-center max-w-md space-y-4">
        <div className="w-16 h-16 mx-auto rounded-2xl bg-indigo-500/10 flex items-center justify-center">
          <FileQuestion className="w-8 h-8 text-indigo-500" />
        </div>
        <h1 className="text-xl font-bold text-text-primary">404 — Không tìm thấy trang</h1>
        <p className="text-sm text-text-muted">
          Đường dẫn không tồn tại hoặc đã bị di chuyển.
        </p>
        <Link href="/" className="btn-primary text-sm inline-flex">
          <Home className="w-4 h-4" /> Về bảng điều khiển
        </Link>
      </div>
    </div>
  );
}
