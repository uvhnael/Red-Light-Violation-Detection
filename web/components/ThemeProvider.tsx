"use client";

import React, { createContext, useContext, useEffect, useState } from "react";

export type Theme =
  | "dark"
  | "light"
  | "system"
  | "vneid-dark"
  | "vneid-light"
  | "vneid-system";

/** Theme ánh xạ sang chế độ sáng/tối dùng cho `resolvedTheme` (đồ thị, ảnh). */
export type ResolvedTheme = "dark" | "light";

/** True nếu theme thuộc họ VNeID (đỏ mận + vàng đồng). */
function isVneid(t: Theme): boolean {
  return t === "vneid-dark" || t === "vneid-light" || t === "vneid-system";
}

interface ThemeContextType {
  theme: Theme;
  setTheme: (theme: Theme) => void;
  resolvedTheme: ResolvedTheme;
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

const STORAGE_THEME = "app-theme";

/** Theme mặc định cho user mới. */
const DEFAULT_THEME: Theme = "vneid-dark";

function isTheme(v: unknown): v is Theme {
  return (
    v === "dark" ||
    v === "light" ||
    v === "system" ||
    v === "vneid-dark" ||
    v === "vneid-light" ||
    v === "vneid-system"
  );
}

function normalizeTheme(raw: unknown): Theme {
  if (typeof raw !== "string") return DEFAULT_THEME;
  if (raw === "vn-dark") return "vneid-dark";
  if (raw === "vn-light") return "vneid-light";
  if (isTheme(raw)) return raw;
  return DEFAULT_THEME;
}

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const [theme, setThemeState] = useState<Theme>(() => {
    if (typeof window === "undefined") return DEFAULT_THEME;
    const raw = localStorage.getItem(STORAGE_THEME);
    return normalizeTheme(raw);
  });

  const [resolvedTheme, setResolvedTheme] = useState<ResolvedTheme>("dark");

  useEffect(() => {
    const root = document.documentElement;

    const applyTheme = (t: Theme) => {
      let active: ResolvedTheme;
      let targetClass: "dark" | "light" | "vneid-dark" | "vneid-light";

      if (t === "system") {
        active = window.matchMedia("(prefers-color-scheme: dark)").matches
          ? "dark"
          : "light";
        targetClass = active;
      } else if (t === "vneid-system") {
        active = window.matchMedia("(prefers-color-scheme: dark)").matches
          ? "dark"
          : "light";
        targetClass = active === "dark" ? "vneid-dark" : "vneid-light";
      } else if (t === "vneid-light") {
        active = "light";
        targetClass = "vneid-light";
      } else if (t === "vneid-dark") {
        active = "dark";
        targetClass = "vneid-dark";
      } else if (t === "light") {
        active = "light";
        targetClass = "light";
      } else {
        active = "dark";
        targetClass = "dark";
      }

      setResolvedTheme(active);

      // Đồng bộ data-theme và class
      root.setAttribute("data-theme", targetClass);
      root.classList.remove("dark", "light", "vneid-dark", "vneid-light");
      root.classList.add(targetClass);
    };

    applyTheme(theme);
    localStorage.setItem(STORAGE_THEME, theme);

    if (theme === "system" || theme === "vneid-system") {
      const mediaQuery = window.matchMedia("(prefers-color-scheme: dark)");
      const handleChange = () => {
        applyTheme(theme);
      };
      mediaQuery.addEventListener("change", handleChange);
      return () => mediaQuery.removeEventListener("change", handleChange);
    }
  }, [theme]);

  const value = React.useMemo(
    () => ({ theme, setTheme: setThemeState, resolvedTheme }),
    [theme, resolvedTheme]
  );

  return (
    <ThemeContext.Provider value={value}>
      {children}
    </ThemeContext.Provider>
  );
}

export function useTheme() {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error("useTheme must be used within a ThemeProvider");
  }
  return context;
}

/** Helper: cho biết 1 theme có dùng palette VNeID (đỏ mận + vàng đồng). */
export { isVneid };