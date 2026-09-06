"use client";

/**
 * Toast + Confirm toàn app — thay cho alert()/confirm() native.
 *
 * - ToastProvider bọc ở dashboard layout; các trang gọi useToast().show().
 * - ConfirmDialog: hộp thoại xác nhận nền overlay, focus trap đơn giản
 *   (focus vào nút hủy khi mở), Enter=Xác nhận, Escape=Hủy.
 */
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { AlertTriangle, CheckCircle2, Info, XCircle } from "lucide-react";
import { motion, AnimatePresence } from "motion/react";

type ToastKind = "success" | "danger" | "info";

interface ToastItem {
  id: number;
  text: string;
  kind: ToastKind;
}

interface ToastContextValue {
  show: (text: string, kind?: ToastKind) => void;
}

const ToastContext = createContext<ToastContextValue | undefined>(undefined);

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<ToastItem[]>([]);
  const idRef = useRef(0);

  const show = useCallback((text: string, kind: ToastKind = "info") => {
    const id = ++idRef.current;
    setToasts((prev) => [...prev.slice(-3), { id, text, kind }]);
    setTimeout(() => {
      setToasts((prev) => prev.filter((t) => t.id !== id));
    }, 4000);
  }, []);

  return (
    <ToastContext.Provider value={{ show }}>
      {children}
      {/* Vùng thông báo: aria-live để screen reader đọc được */}
      <div
        aria-live="polite"
        role="status"
        className="fixed top-20 right-4 sm:right-6 z-[70] flex flex-col gap-2 items-end pointer-events-none"
      >
        <AnimatePresence>
          {toasts.map((t) => (
            <motion.div
              key={t.id}
              layout
              initial={{ opacity: 0, y: -24, scale: 0.96 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: -12, scale: 0.97 }}
              transition={{ duration: 0.22, ease: [0.22, 1, 0.36, 1] }}
              className={`toast toast-${t.kind} pointer-events-auto flex items-center gap-3 px-4 py-3 rounded-xl border shadow-xl max-w-[calc(100vw-2rem)]`}
            >
              {t.kind === "success" && (
                <CheckCircle2 className="w-5 h-5 text-success shrink-0" />
              )}
              {t.kind === "danger" && (
                <XCircle className="w-5 h-5 text-danger shrink-0" />
              )}
              {t.kind === "info" && (
                <Info className="w-5 h-5 shrink-0" style={{ color: "rgb(var(--accent-rgb))" }} />
              )}
              <span className="text-xs font-semibold">{t.text}</span>
              <button
                onClick={() => setToasts((prev) => prev.filter((x) => x.id !== t.id))}
                className="ml-1 p-0.5 rounded-md text-text-muted hover:text-text-primary transition-colors"
                aria-label="Đóng thông báo"
              >
                <XCircle className="w-3.5 h-3.5" />
              </button>
            </motion.div>
          ))}
        </AnimatePresence>
      </div>
    </ToastContext.Provider>
  );
}

export function useToast(): ToastContextValue {
  const ctx = useContext(ToastContext);
  if (!ctx) throw new Error("useToast phải dùng bên trong ToastProvider");
  return ctx;
}

/* ──────────────────────────────────────────────────────────── */

interface ConfirmDialogProps {
  open: boolean;
  title: string;
  message: string;
  confirmLabel?: string;
  cancelLabel?: string;
  danger?: boolean;
  busy?: boolean;
  onConfirm: () => void;
  onCancel: () => void;
}

export function ConfirmDialog({
  open,
  title,
  message,
  confirmLabel = "Xác nhận",
  cancelLabel = "Hủy",
  danger = true,
  busy = false,
  onConfirm,
  onCancel,
}: ConfirmDialogProps) {
  const cancelRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (!open) return;
    cancelRef.current?.focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onCancel();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onCancel]);

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-[80] flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm"
      onClick={onCancel}
      role="presentation"
    >
      <div
        role="alertdialog"
        aria-modal="true"
        aria-labelledby="confirm-title"
        aria-describedby="confirm-desc"
        className="w-full max-w-sm bg-surface border border-border rounded-2xl shadow-2xl p-6 space-y-4 animate-in fade-in zoom-in-95 duration-150"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-start gap-3">
          <div
            className={`w-10 h-10 rounded-xl flex items-center justify-center shrink-0 ${
              danger ? "bg-rose-500/10" : "bg-indigo-500/10"
            }`}
          >
            <AlertTriangle
              className={`w-5 h-5 ${danger ? "text-rose-500" : "text-indigo-500"}`}
            />
          </div>
          <div className="space-y-1">
            <h2
              id="confirm-title"
              className="text-sm font-bold text-text-primary"
            >
              {title}
            </h2>
            <p
              id="confirm-desc"
              className="text-xs text-text-muted leading-relaxed"
            >
              {message}
            </p>
          </div>
        </div>
        <div className="flex justify-end gap-2 pt-2">
          <button
            ref={cancelRef}
            onClick={onCancel}
            disabled={busy}
            className="btn-ghost text-xs border border-border px-4 disabled:opacity-40"
          >
            {cancelLabel}
          </button>
          <button
            onClick={onConfirm}
            disabled={busy}
            className={`${danger ? "btn-danger" : "btn-primary"} text-xs px-4 disabled:opacity-40`}
          >
            {busy ? "Đang xử lý..." : confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
}
