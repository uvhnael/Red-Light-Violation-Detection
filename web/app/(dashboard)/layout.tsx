import { AppLayout } from "@/components/AppLayout";

/** Layout cho các trang đã đăng nhập — sidebar + AI sidebar + header. */
export default function DashboardLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return <AppLayout>{children}</AppLayout>;
}
