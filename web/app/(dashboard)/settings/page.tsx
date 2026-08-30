"use client";

import { useState } from "react";
import { useTheme, Theme } from "@/components/ThemeProvider";
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

export default function SettingsPage() {
  const { theme, setTheme, resolvedTheme } = useTheme();
  const [ocrConfidence, setOcrConfidence] = useState(85);
  const [refreshInterval, setRefreshInterval] = useState(5);
  const [soundAlerts, setSoundAlerts] = useState(true);
  const [accentColor, setAccentColor] = useState("indigo");
  const [savedToast, setSavedToast] = useState(false);

  const handleSaveSettings = () => {
    setSavedToast(true);
    setTimeout(() => setSavedToast(false), 3000);
  };

  const THEME_OPTIONS: { id: Theme; label: string; desc: string; icon: typeof Sun }[] = [
    {
      id: "dark",
      label: "Dark Mode (Tối)",
      desc: "Giao diện tối màu chuyên dụng cho phòng điều hành & ban đêm",
      icon: Moon,
    },
    {
      id: "light",
      label: "Light Mode (Sáng)",
      desc: "Giao diện sáng rực rỡ, độ tương phản cao dễ quan sát ban ngày",
      icon: Sun,
    },
    {
      id: "system",
      label: "Hệ thống (System)",
      desc: "Tự động thay đổi theo cấu hình giao diện của hệ điều hành OS",
      icon: Laptop,
    },
  ];

  const ACCENT_COLORS = [
    { id: "indigo", name: "Indigo Violet", colorClass: "bg-indigo-500" },
    { id: "emerald", name: "Emerald Mint", colorClass: "bg-emerald-500" },
    { id: "rose", name: "Rose Crimson", colorClass: "bg-rose-500" },
    { id: "amber", name: "Amber Gold", colorClass: "bg-amber-500" },
  ];

  return (
    <div className="space-y-6 max-w-4xl mx-auto">
      {/* Save Notification Toast */}
      {savedToast && (
        <div className="fixed top-20 right-6 z-50 px-4 py-3 rounded-xl bg-emerald-950/90 text-emerald-300 border border-emerald-500/30 shadow-xl flex items-center gap-3 animate-in fade-in slide-in-from-top-4 duration-200">
          <CheckCircle2 className="w-5 h-5 text-emerald-400" />
          <span className="text-xs font-semibold">Đã lưu cài đặt cấu hình thành công!</span>
        </div>
      )}

      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <div className="flex items-center gap-2">
            <SettingsIcon className="w-6 h-6 text-indigo-400" />
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
          className="btn-primary text-xs flex items-center gap-2 shadow-lg shadow-indigo-500/20"
        >
          <Save className="w-4 h-4" />
          Lưu cài đặt
        </button>
      </div>

      {/* ── SECTION 1: THEME & APPEARANCE ── */}
      <div className="glass-card p-6 space-y-6">
        <div>
          <h2 className="text-sm font-bold text-text-primary flex items-center gap-2">
            <Sun className="w-4 h-4 text-indigo-400" />
            Chế độ Giao diện (Light / Dark Theme)
          </h2>
          <p className="text-xs text-text-muted mt-0.5">
            Chọn chủ đề màu sắc mong muốn cho bảng điều khiển Trung tâm TrafficAI.
          </p>
        </div>

        {/* Theme cards selector */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          {THEME_OPTIONS.map((opt) => {
            const Icon = opt.icon;
            const isSelected = theme === opt.id;
            return (
              <button
                key={opt.id}
                onClick={() => setTheme(opt.id)}
                className={`p-4 rounded-2xl border text-left transition-all duration-200 flex flex-col justify-between space-y-4 group cursor-pointer ${
                  isSelected
                    ? "bg-indigo-600/15 border-indigo-500/50 shadow-lg shadow-indigo-500/10 ring-1 ring-indigo-500/30"
                    : "bg-surface-2/40 border-border hover:border-border-light hover:bg-surface-3/50"
                }`}
              >
                <div className="flex items-center justify-between">
                  <div
                    className={`w-10 h-10 rounded-xl flex items-center justify-center transition-colors ${
                      isSelected
                        ? "bg-indigo-600 text-white shadow-md shadow-indigo-500/30"
                        : "bg-surface-3 text-text-muted group-hover:text-text-primary"
                    }`}
                  >
                    <Icon className="w-5 h-5" />
                  </div>
                  {isSelected && (
                    <span className="w-6 h-6 rounded-full bg-indigo-500 text-white flex items-center justify-center">
                      <Check className="w-3.5 h-3.5 stroke-[3]" />
                    </span>
                  )}
                </div>

                <div>
                  <h3 className="text-sm font-bold text-text-primary">{opt.label}</h3>
                  <p className="text-[11px] text-text-muted mt-1 leading-relaxed">
                    {opt.desc}
                  </p>
                </div>
              </button>
            );
          })}
        </div>

        {/* Active Theme Info Banner */}
        <div className="p-3.5 rounded-xl bg-surface-3/40 border border-border flex items-center justify-between text-xs">
          <div className="flex items-center gap-2">
            <span className="text-text-muted">Chủ đề đang áp dụng:</span>
            <span className="font-bold text-primary-400 capitalize">
              {resolvedTheme === "dark" ? "🌙 Dark Mode" : "☀️ Light Mode"}
            </span>
          </div>
          <span className="text-[11px] text-text-muted font-mono">
            data-theme=&quot;{resolvedTheme}&quot;
          </span>
        </div>

        {/* Accent Color Selection */}
        <div className="pt-4 border-t border-border space-y-3">
          <label className="text-xs font-bold text-text-secondary uppercase tracking-wider block">
            Màu điểm nhấn Chủ đạo (Accent Color)
          </label>
          <div className="flex flex-wrap gap-3">
            {ACCENT_COLORS.map((c) => (
              <button
                key={c.id}
                onClick={() => setAccentColor(c.id)}
                className={`flex items-center gap-2.5 px-3.5 py-2 rounded-xl text-xs font-medium border transition-all ${
                  accentColor === c.id
                    ? "border-primary-500 bg-primary-500/10 text-text-primary"
                    : "border-border text-text-muted hover:border-text-secondary"
                }`}
              >
                <span className={`w-3.5 h-3.5 rounded-full ${c.colorClass}`} />
                {c.name}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* ── SECTION 2: DETECTION & OCR SETTINGS ── */}
      <div className="glass-card p-6 space-y-6">
        <div>
          <h2 className="text-sm font-bold text-text-primary flex items-center gap-2">
            <Sliders className="w-4 h-4 text-indigo-400" />
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
              <span className="font-mono font-bold text-indigo-400 text-sm">
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
              className="w-full h-2 bg-surface-3 rounded-lg appearance-none cursor-pointer accent-indigo-500"
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
                soundAlerts ? "bg-indigo-600" : "bg-surface-3"
              }`}
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
            <Cpu className="w-4 h-4 text-indigo-400" />
            Thông tin Máy chủ Central Server &amp; Trợ lý AI
          </h2>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-xs">
          <div className="p-4 rounded-xl bg-surface-2/50 border border-border space-y-1">
            <p className="text-[10px] text-text-muted font-bold uppercase tracking-wider">
              Central Server URL
            </p>
            <p className="font-mono font-semibold text-text-primary">
              http://localhost:8001
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
            <p className="text-[10px] text-indigo-400 mt-1 flex items-center gap-1">
              <Sparkles className="w-3 h-3" />
              Natural Language SQL Active
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
