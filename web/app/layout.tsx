import type { Metadata, Viewport } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { ThemeProvider } from "@/components/ThemeProvider";

const inter = Inter({ subsets: ["latin"], variable: "--font-inter", display: "swap" });

export const metadata: Metadata = {
  title: {
    default: "TrafficAI — Giám sát vi phạm đèn đỏ",
    template: "%s | TrafficAI",
  },
  description:
    "Trung tâm điều khiển Smart City: phát hiện vi phạm đèn đỏ bằng AI, phân tán edge-to-central.",
  applicationName: "TrafficAI",
  robots: { index: false, follow: false },
  formatDetection: { telephone: false },
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  themeColor: [
    { media: "(prefers-color-scheme: dark)", color: "#16120e" },
    { media: "(prefers-color-scheme: light)", color: "#f6f2e9" },
  ],
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="vi" className={`${inter.variable} vneid-dark`} suppressHydrationWarning>
      <head>
        {/* Chặn flash theme sai trước khi ThemeProvider kịp chạy */}
        <script
          dangerouslySetInnerHTML={{
            __html: `try{var t=localStorage.getItem('app-theme')||'vneid-dark';if(t==='vn-dark')t='vneid-dark';if(t==='vn-light')t='vneid-light';var r=document.documentElement;r.classList.remove('dark','light','vneid-dark','vneid-light');var d=window.matchMedia('(prefers-color-scheme: dark)').matches;var c=t;if(t==='system'){c=d?'dark':'light'}else if(t==='vneid-system'){c=d?'vneid-dark':'vneid-light'}r.classList.add(c);r.setAttribute('data-theme',c)}catch(e){}`,
          }}
        />
      </head>
      <body>
        <a
          href="#main-content"
          className="sr-only focus:not-sr-only focus:fixed focus:top-3 focus:left-3 focus:z-[100] focus:px-4 focus:py-2 focus:rounded-lg focus:text-white focus:text-sm font-semibold"
          style={{ background: "rgb(var(--accent-rgb))" }}
        >
          Nhảy tới nội dung chính
        </a>
        <ThemeProvider>{children}</ThemeProvider>
      </body>
    </html>
  );
}
