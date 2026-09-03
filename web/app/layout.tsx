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
    { media: "(prefers-color-scheme: dark)", color: "#09090b" },
    { media: "(prefers-color-scheme: light)", color: "#f1f5f9" },
  ],
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="vi" className={`${inter.variable} dark`} suppressHydrationWarning>
      <head>
        {/* Chặn flash theme sai trước khi ThemeProvider kịp chạy */}
        <script
          dangerouslySetInnerHTML={{
            __html: `try{var t=localStorage.getItem('app-theme')||'dark';if(t==='light'){document.documentElement.classList.add('light');document.documentElement.classList.remove('dark');document.documentElement.setAttribute('data-theme','light')}}catch(e){}`,
          }}
        />
      </head>
      <body>
        <a
          href="#main-content"
          className="sr-only focus:not-sr-only focus:fixed focus:top-3 focus:left-3 focus:z-[100] focus:px-4 focus:py-2 focus:rounded-lg focus:bg-indigo-600 focus:text-white focus:text-sm font-semibold"
        >
          Nhảy tới nội dung chính
        </a>
        <ThemeProvider>{children}</ThemeProvider>
      </body>
    </html>
  );
}
