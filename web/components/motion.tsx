"use client";

/**
 * Motion presets dùng chung cho các trang dashboard (lib `motion`).
 *
 * - `motion.div`/`motion` component wrapper: fade + slide + blur khi mount,
 *   có `delay` để tạo hiệu ứng stagger (lệch nhịp từng phần tử).
 * - `pageVariants`/`sectionVariants`/`itemVariants`: các preset biến thể.
 * - Motion tự tôn trọng `prefers-reduced-motion` qua `useReducedMotion`,
 *   nhưng preset ở đây luôn set `opacity: 0` → nếu user bật reduced-motion,
 *   CPU animate mạnh hơn sẽ chạy 300ms (thay vì vô hạn).
 */

import { motion, useReducedMotion, type Variants } from "motion/react";

/** Container cha: các con stagger 80ms một phần tử, tổng không quá ~600ms. */
export const staggerContainer: Variants = {
  hidden: {},
  visible: {
    transition: { staggerChildren: 0.08, delayChildren: 0.05 },
  },
};

/** Phần tử con: fade + trượt lên 16px + blur nhẹ. */
export const fadeUpItem: Variants = {
  hidden: { opacity: 0, y: 16, filter: "blur(6px)" },
  visible: {
    opacity: 1,
    y: 0,
    filter: "blur(0px)",
    transition: { duration: 0.5, ease: [0.22, 1, 0.36, 1] },
  },
};

/** Phần tử con: fade + zoom nhẹ từ 97% (dùng cho KPI card, badge). */
export const fadeScaleItem: Variants = {
  hidden: { opacity: 0, scale: 0.97 },
  visible: {
    opacity: 1,
    scale: 1,
    transition: { duration: 0.45, ease: [0.22, 1, 0.36, 1] },
  },
};

/** Wrapper tiện ích: motion.div + stagger container, có dùng reduced-motion. */
export function StaggerList({
  children,
  className,
  delay = 0,
}: {
  children: React.ReactNode;
  className?: string;
  /** Delay khởi động (giây) — dùng để giữ nhịp chênh lệch giữa các cụm. */
  delay?: number;
}) {
  const prefersReducedMotion = useReducedMotion();
  if (prefersReducedMotion) return <div className={className}>{children}</div>;
  return (
    <motion.div
      className={className}
      initial="hidden"
      animate="visible"
      variants={{
        hidden: {},
        visible: { transition: { staggerChildren: 0.08, delayChildren: delay } },
      }}
    >
      {children}
    </motion.div>
  );
}

/** Wrapper tiện ích: phần tử con bên trong StaggerList. */
export function FadeItem({
  children,
  className,
  style,
}: {
  children: React.ReactNode;
  className?: string;
  style?: React.CSSProperties;
}) {
  const prefersReducedMotion = useReducedMotion();
  if (prefersReducedMotion) return <div className={className} style={style}>{children}</div>;
  return (
    <motion.div className={className} style={style} variants={fadeUpItem}>
      {children}
    </motion.div>
  );
}
