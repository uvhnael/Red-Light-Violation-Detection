import { AppLayout } from "@/components/AppLayout";
import { RequireSession } from "@/components/RequireSession";

/**
 * Layout cho các trang đã đăng nhập — guard phiên trước, rồi mới tới
 * sidebar + AI sidebar + header.
 *
 * Lớp 1 là proxy.ts (chặn ở server, trước khi HTML render). Lớp 2 ở đây
 * bắt điều hướng client-side, phiên hết hạn giữa chừng và 401 sau refresh.
 */
export default function DashboardLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <RequireSession>
      <AppLayout>{children}</AppLayout>
    </RequireSession>
  );
}
