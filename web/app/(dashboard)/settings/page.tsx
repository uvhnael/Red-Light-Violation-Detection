"use client";

import { useState } from "react";
import { useTheme, type Theme } from "@/components/ThemeProvider";
import {
  Settings as SettingsIcon,
  Moon,
  Sun,
  Laptop,
  Check,
  Sliders,
  Cpu,
  Sparkles,
  Save,
  CheckCircle2,
} from "lucide-react";

interface ThemeOption {
  id: Theme;
  title: string;
  subtitle: string;
  badge: string;
  badgeClass: string;
  desc: string;
  icon: typeof Moon;
  accentColor: string;
  swatches: {
    bg: string;
    card: string;
    accent: string;
    accentSecondary: string;
  };
}

const THEME_OPTIONS: ThemeOption[] = [
  {
    id: "vneid-dark",
    title: "VN Dark",
    subtitle: "Việt Nam — Tối",
    badge: "🇻🇳 Khuyên dùng",
    badgeClass: "bg-rose-500/15 text-rose-500 border-rose-500/30",
    desc: "Nền đen nâu ấm sâu, màu nhấn Đỏ mận & Vàng đồng đặc trưng CSGT Việt Nam",
    icon: Moon,
    accentColor: "#c4282f",
    swatches: {
      bg: "#14110e",
      card: "#1c1713",
      accent: "#c4282f",
      accentSecondary: "#c9a227",
    },
  },
  {
    id: "vneid-light",
    title: "VN Light",
    subtitle: "Việt Nam — Sáng",
    badge: "🇻🇳 Cơ quan Sáng",
    badgeClass: "bg-amber-500/15 text-amber-600 dark:text-amber-400 border-amber-500/30",
    desc: "Nền kem ấm sang trọng, thẻ trắng tinh khôi, tương phản cao cho ca trực ban ngày",
    icon: Sun,
    accentColor: "#b7121b",
    swatches: {
      bg: "#f7f4ed",
      card: "#ffffff",
      accent: "#b7121b",
      accentSecondary: "#b45309",
    },
  },
  {
    id: "dark",
    title: "Dark",
    subtitle: "Chế độ Tối",
    badge: "🌙 Slate Dark",
    badgeClass: "bg-indigo-500/15 text-indigo-500 border-indigo-500/30",
    desc: "Giao diện Slate đen tuyền hiện đại, màu nhấn Indigo dịu mắt cho trung tâm chỉ huy",
    icon: Moon,
    accentColor: "#6366f1",
    swatches: {
      bg: "#090b10",
      card: "#0f131d",
      accent: "#6366f1",
      accentSecondary: "#8b5cf6",
    },
  },
  {
    id: "light",
    title: "Light",
    subtitle: "Chế độ Sáng",
    badge: "☀️ Slate Light",
    badgeClass: "bg-sky-500/15 text-sky-600 dark:text-sky-400 border-sky-500/30",
    desc: "Giao diện Slate trắng sáng tiêu chuẩn, các đường viền rõ nét, tương phản cao",
    icon: Sun,
    accentColor: "#4f46e5",
    swatches: {
      bg: "#f8fafc",
      card: "#ffffff",
      accent: "#4f46e5",
      accentSecondary: "#818cf8",
    },
  },
  {
    id: "system",
    title: "System",
    subtitle: "Hệ thống",
    badge: "💻 Tự động OS",
    badgeClass: "bg-purple-500/15 text-purple-600 dark:text-purple-400 border-purple-500/30",
    desc: "Tự động chuyển đổi giữa Sáng và Tối đồng bộ theo cấu hình hệ điều hành thiết bị",
    icon: Laptop,
    accentColor: "#8b5cf6",
    swatches: {
      bg: "linear-gradient(135deg, #090b10 50%, #f8fafc 50%)",
      card: "linear-gradient(135deg, #0f131d 50%, #ffffff 50%)",
      accent: "#8b5cf6",
      accentSecondary: "#6366f1",
    },
  },
  {
    id: "vneid-system",
    title: "VN System",
    subtitle: "VN — Theo OS",
    badge: "🇻🇳 Tự động OS",
    badgeClass: "bg-rose-500/15 text-rose-600 dark:text-rose-400 border-rose-500/30",
    desc: "Theo OS nhưng giữ bảng màu Đỏ mận + Vàng đồng của cơ quan nhà nước Việt Nam",
    icon: Laptop,
    accentColor: "#b7121b",
    swatches: {
      bg: "linear-gradient(135deg, #14110e 50%, #f5efe1 50%)",
      card: "linear-gradient(135deg, #1c1713 50%, #ffffff 50%)",
      accent: "#b7121b",
      accentSecondary: "#b45309",
    },
  },
];

export default function SettingsPage() {
  const { theme, setTheme, resolvedTheme } = useTheme();
  const [ocrConfidence, setOcrConfidence] = useState(85);
  const [refreshInterval, setRefreshInterval] = useState(5);
  const [soundAlerts, setSoundAlerts] = useState(true);
  const [savedToast, setSavedToast] = useState(false);

  const handleSaveSettings = () => {
    setSavedToast(true);
    setTimeout(() => setSavedToast(false), 3000);
  };

  const getActiveLabel = () => {
    switch (theme) {
      case "vneid-dark":
        return "🇻🇳 VN Dark (Việt Nam — Tối ấm)";
      case "vneid-light":
        return "🇻🇳 VN Light (Việt Nam — Sáng kem)";
      case "dark":
        return "🌙 Dark Mode (Slate Tối)";
      case "light":
        return "☀️ Light Mode (Slate Sáng)";
      case "system":
        return `💻 Hệ thống OS (${resolvedTheme === "dark" ? "🌙 Đang áp dụng Dark Mode" : "☀️ Đang áp dụng Light Mode"})`;
      case "vneid-system":
        return `🇻🇳 VN System (${resolvedTheme === "dark" ? "Đang áp dụng VN Dark" : "Đang áp dụng VN Light"})`;
      default:
        return theme;
    }
  };

  return (
    <div className="space-y-6 max-w-5xl mx-auto">
      {/* Save Notification Toast */}
      {savedToast && (
        <div className="toast toast-success fixed top-20 right-6 z-50 px-4 py-3 rounded-xl border shadow-xl flex items-center gap-3 animate-in fade-in slide-in-from-top-4 duration-200">
          <CheckCircle2 className="w-5 h-5 text-success shrink-0" />
          <span className="text-xs font-semibold">Đã lưu cài đặt cấu hình thành công!</span>
        </div>
      )}

      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <div className="flex items-center gap-2">
            <SettingsIcon className="w-6 h-6" style={{ color: "rgb(var(--accent-rgb))" }} />
            <h1 className="text-2xl font-bold text-text-primary tracking-tight">
              Cài đặt Hệ thống &amp; Giao diện
            </h1>
          </div>
          <p className="text-xs text-text-muted mt-1">
            Cấu hình chế độ hiển thị Sáng/Tối, tham số nhận diện ANPR và tùy chỉnh máy chủ.
          </p>
        </div>

        <button
          onClick={handleSaveSettings}
          className="btn-primary text-xs flex items-center gap-2 shadow-lg"
          style={{ boxShadow: "0 8px 24px -8px rgba(var(--accent-soft-rgb), 0.5)" }}
        >
          <Save className="w-4 h-4" />
          Lưu cài đặt
        </button>
      </div>

      {/* ── SECTION 1: THEME & APPEARANCE ── */}
      <div className="glass-card p-6 space-y-6">
        <div>
          <h2 className="text-sm font-bold text-text-primary flex items-center gap-2">
            <Sun className="w-4 h-4" style={{ color: "rgb(var(--accent-rgb))" }} />
            Chế độ Giao diện (Theme &amp; Colors)
          </h2>
          <p className="text-xs text-text-muted mt-0.5">
            Chọn chủ đề màu sắc mong muốn cho bảng điều khiển Trung tâm TrafficAI. Nhấp vào để áp dụng ngay.
          </p>
        </div>

        {/* 6 Theme Cards Grid: VN Dark, VN Light, Dark, Light, System, VN System */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6 gap-3.5">
          {THEME_OPTIONS.map((opt) => {
            const Icon = opt.icon;
            const isSelected =
              theme === opt.id ||
              (theme === "vneid-system" && opt.id === "vneid-system");

            return (
              <button
                key={opt.id}
                type="button"
                onClick={() => setTheme(opt.id)}
                className={`p-3.5 rounded-2xl border text-left transition-all duration-200 flex flex-col justify-between space-y-3 group cursor-pointer relative ${
                  isSelected
                    ? "border-[rgb(var(--accent-rgb))] bg-[rgba(var(--accent-rgb),0.10)] shadow-lg shadow-[rgba(var(--accent-rgb),0.15)] ring-2 ring-[rgba(var(--accent-rgb),0.35)]"
                    : "bg-surface-2/50 border-border hover:border-border-light hover:bg-surface-3/50"
                }`}
              >
                {/* Top: Icon + Badge + Check */}
                <div className="flex items-start justify-between gap-2">
                  <div
                    className={`w-9 h-9 rounded-xl flex items-center justify-center transition-colors shrink-0 ${
                      isSelected
                        ? "text-white shadow-md"
                        : "bg-surface-3 text-text-muted group-hover:text-text-primary"
                    }`}
                    style={isSelected ? { background: "rgb(var(--accent-rgb))" } : undefined}
                  >
                    <Icon className="w-4 h-4" />
                  </div>

                  <div className="flex items-center gap-1.5 shrink-0">
                    <span
                      className={`text-[9px] font-semibold px-2 py-0.5 rounded-full border ${opt.badgeClass}`}
                    >
                      {opt.badge}
                    </span>
                    {isSelected && (
                      <span
                        className="w-5 h-5 rounded-full text-white flex items-center justify-center shadow-sm"
                        style={{ background: "rgb(var(--accent-rgb))" }}
                      >
                        <Check className="w-3 h-3 stroke-[3]" />
                      </span>
                    )}
                  </div>
                </div>

                {/* Middle: Title & Description */}
                <div className="space-y-1">
                  <div className="flex items-baseline gap-1.5">
                    <h3 className="text-sm font-bold text-text-primary leading-tight">
                      {opt.title}
                    </h3>
                    <span className="text-[10px] text-text-muted">
                      ({opt.subtitle})
                    </span>
                  </div>
                  <p className="text-[11px] text-text-muted leading-relaxed line-clamp-2">
                    {opt.desc}
                  </p>
                </div>

                {/* Bottom: Palette Swatches Preview */}
                <div className="pt-2 border-t border-border/40 flex items-center justify-between">
                  <span className="text-[10px] text-text-muted font-medium">Bảng màu:</span>
                  <div className="flex items-center gap-1.5 p-1 rounded-lg bg-surface-3/50 border border-border/50">
                    <span
                      className="w-3.5 h-3.5 rounded-full border border-black/10 shadow-sm shrink-0"
                      style={{ background: opt.swatches.bg }}
                      title="Nền trang"
                    />
                    <span
                      className="w-3.5 h-3.5 rounded-full border border-black/10 shadow-sm shrink-0"
                      style={{ background: opt.swatches.card }}
                      title="Thẻ nội dung"
                    />
                    <span
                      className="w-3.5 h-3.5 rounded-full border border-black/10 shadow-sm shrink-0"
                      style={{ background: opt.swatches.accent }}
                      title="Màu nhấn chính"
                    />
                    <span
                      className="w-3.5 h-3.5 rounded-full border border-black/10 shadow-sm shrink-0"
                      style={{ background: opt.swatches.accentSecondary }}
                      title="Màu nhấn phụ"
                    />
                  </div>
                </div>
              </button>
            );
          })}
        </div>

        {/* Active Theme Info Banner */}
        <div className="p-3.5 rounded-xl bg-surface-3/40 border border-border flex flex-col sm:flex-row sm:items-center justify-between gap-2 text-xs">
          <div className="flex items-center gap-2">
            <span className="text-text-muted">Chủ đề đang áp dụng:</span>
            <span className="font-bold" style={{ color: "rgb(var(--accent-rgb))" }}>
              {getActiveLabel()}
            </span>
          </div>
          <div className="flex items-center gap-3">
            <span className="text-[11px] text-text-muted font-mono">
              data-theme=&quot;{theme === "system" ? resolvedTheme : theme}&quot;
            </span>
            <span className="text-[11px] text-emerald-500 flex items-center gap-1 font-medium">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
              Đã đồng bộ
            </span>
          </div>
        </div>
      </div>

      {/* ── SECTION 2: DETECTION & OCR SETTINGS ── */}
      <div className="glass-card p-6 space-y-6">
        <div>
          <h2 className="text-sm font-bold text-text-primary flex items-center gap-2">
            <Sliders className="w-4 h-4" style={{ color: "rgb(var(--accent-rgb))" }} />
            Tham số Nhận diện &amp; Cảnh báo ANPR
          </h2>
          <p className="text-xs text-text-muted mt-0.5">
            Cấu hình ngưỡng độ tin cậy OCR biển số xe và chu kỳ cập nhật dữ liệu tự động.
          </p>
        </div>

        <div className="space-y-5 text-xs">
          {/* Slider for ANPR confidence threshold */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <label className="font-semibold text-text-primary">
                Ngưỡng Confidence duyệt tự động (ANPR Minimum Confidence)
              </label>
              <span className="font-mono font-bold text-sm" style={{ color: "rgb(var(--accent-rgb))" }}>
                {ocrConfidence}%
              </span>
            </div>
            <input
              type="range"
              min="50"
              max="95"
              step="5"
              value={ocrConfidence}
              onChange={(e) => setOcrConfidence(Number(e.target.value))}
              className="w-full h-2 bg-surface-3 rounded-lg appearance-none cursor-pointer"
              style={{ accentColor: "rgb(var(--accent-rgb))" }}
            />
            <p className="text-[11px] text-text-muted">
              Các vi phạm có độ tin cậy OCR thấp hơn {ocrConfidence}% sẽ tự động được chuyển sang hàng chờ <strong>Review Queue</strong> để CSGT kiểm tra thủ công.
            </p>
          </div>

          {/* Refresh interval */}
          <div className="space-y-2 pt-3 border-t border-border">
            <div className="flex items-center justify-between">
              <label className="font-semibold text-text-primary">
                Chu kỳ cập nhật danh sách camera &amp; vi phạm (Auto-refresh)
              </label>
              <select
                value={refreshInterval}
                onChange={(e) => setRefreshInterval(Number(e.target.value))}
                className="input-field text-xs py-1.5 px-3"
              >
                <option value={3}>3 giây</option>
                <option value={5}>5 giây (Mặc định)</option>
                <option value={10}>10 giây</option>
                <option value={0}>Tắt cập nhật tự động</option>
              </select>
            </div>
          </div>

          {/* Sound alert toggle */}
          <div className="flex items-center justify-between pt-3 border-t border-border">
            <div>
              <p className="font-semibold text-text-primary">
                Âm thanh Cảnh báo Vi phạm Mới (Audio Alerts)
              </p>
              <p className="text-[11px] text-text-muted mt-0.5">
                Phát âm thanh thông báo khi có vi phạm vượt đèn đỏ mới được phát hiện từ Edge Node.
              </p>
            </div>
            <button
              onClick={() => setSoundAlerts(!soundAlerts)}
              className={`w-12 h-6 rounded-full transition-colors relative cursor-pointer ${
                soundAlerts ? "" : "bg-surface-3"
              }`}
              style={soundAlerts ? { background: "rgb(var(--accent-rgb))" } : undefined}
            >
              <span
                className={`w-4 h-4 rounded-full bg-white absolute top-1 transition-transform ${
                  soundAlerts ? "translate-x-7" : "translate-x-1"
                }`}
              />
            </button>
          </div>
        </div>
      </div>

      {/* ── SECTION 3: SYSTEM INFO & AI ENGINE ── */}
      <div className="glass-card p-6 space-y-4">
        <div>
          <h2 className="text-sm font-bold text-text-primary flex items-center gap-2">
            <Cpu className="w-4 h-4" style={{ color: "rgb(var(--accent-rgb))" }} />
            Thông tin Máy chủ Central Server &amp; Trợ lý AI
          </h2>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-xs">
          <div className="p-4 rounded-xl bg-surface-2/50 border border-border space-y-1">
            <p className="text-[10px] text-text-muted font-bold uppercase tracking-wider">
              Central Server URL
            </p>
            <p className="font-mono font-semibold text-text-primary">
              http://localhost:8002
            </p>
            <p className="text-[10px] text-emerald-400 mt-1 flex items-center gap-1">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
              Connected &amp; Healthy
            </p>
          </div>

          <div className="p-4 rounded-xl bg-surface-2/50 border border-border space-y-1">
            <p className="text-[10px] text-text-muted font-bold uppercase tracking-wider">
              AI Query Engine (Text-to-SQL)
            </p>
            <p className="font-mono font-semibold text-text-primary">
              Google Gemini 2.0 / 1.5 Pro
            </p>
            <p className="text-[10px] mt-1 flex items-center gap-1" style={{ color: "rgb(var(--accent-rgb))" }}>
              <Sparkles className="w-3 h-3" />
              Natural Language SQL Active
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}