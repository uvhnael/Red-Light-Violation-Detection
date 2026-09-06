"use client";

/** Skeleton khi chuyển route trong dashboard — stagger từng cụm cho có nhịp. */
import { motion, type Transition } from "motion/react";

const fade = (delay: number): { initial: { opacity: number; y: number }; animate: { opacity: number; y: number }; transition: Transition } => ({
  initial: { opacity: 0, y: 10 },
  animate: { opacity: 1, y: 0 },
  transition: { delay, duration: 0.3, ease: [0.22, 1, 0.36, 1] },
});

export default function DashboardLoading() {
  return (
    <div className="space-y-6">
      <motion.div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4" {...fade(0)}>
        {Array.from({ length: 4 }).map((_, i) => (
          <div key={i} className="glass-card p-5 space-y-4">
            <div className="flex justify-between items-center">
              <div className="skeleton h-3 w-24" />
              <div className="skeleton h-9 w-9 rounded-xl" />
            </div>
            <div className="space-y-2">
              <div className="skeleton h-8 w-16" />
              <div className="skeleton h-3 w-32" />
            </div>
          </div>
        ))}
      </motion.div>
      <motion.div className="glass-card p-6 space-y-4" {...fade(0.12)}>
        <div className="skeleton h-5 w-48" />
        <div className="skeleton h-72 w-full rounded-xl" />
      </motion.div>
      <motion.div className="grid grid-cols-1 lg:grid-cols-12 gap-6" {...fade(0.24)}>
        <div className="lg:col-span-7 glass-card p-6 space-y-4">
          <div className="skeleton h-5 w-56" />
          <div className="skeleton h-16 w-full rounded-xl" />
          <div className="skeleton h-16 w-full rounded-xl" />
          <div className="skeleton h-16 w-full rounded-xl" />
        </div>
        <div className="lg:col-span-5 glass-card p-6 space-y-4">
          <div className="skeleton h-5 w-40" />
          <div className="skeleton h-14 w-full rounded-xl" />
          <div className="skeleton h-14 w-full rounded-xl" />
          <div className="skeleton h-14 w-full rounded-xl" />
        </div>
      </motion.div>
    </div>
  );
}
