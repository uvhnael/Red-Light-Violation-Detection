import Link from "next/link";
import { FileQuestion, Home } from "lucide-react";

/** Trang 404 trong dashboard (giữ layout sidebar). */
export default function DashboardNotFound() {
  return (
    <div className="flex items-center justify-center h-[60vh]">
      <div className="glass-card p-10 text-center max-w-md space-y-4">
        <div className="w-16 h-16 mx-auto rounded-2xl bg-indigo-500/10 flex items-center justify-center">
          <FileQuestion className="w-8 h-8 text-indigo-500" />
        </div>
        <h2 className="text-lg font-semibold text-text-primary">
          Không tìm thấy trang
        </h2>
        <p className="text-sm text-text-muted">
          Trang bạn truy cập không tồn tại hoặc đã bị di chuyển.
        </p>
        <Link href="/" className="btn-primary text-sm inline-flex">
          <Home className="w-4 h-4" /> Về bảng điều khiển
        </Link>
      </div>
    </div>
  );
}
