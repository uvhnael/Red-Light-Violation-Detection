import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { AppLayout } from "@/components/AppLayout";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

export const metadata: Metadata = {
  title: "TrafficAI — Distributed Traffic Violation Detection",
  description:
    "Smart City control center for AI-powered red light violation detection. Monitor edge nodes, review violations, and manage enforcement in real-time.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={inter.variable}>
      <body>
        <AppLayout>{children}</AppLayout>
      </body>
    </html>
  );
}