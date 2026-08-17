'use client';

import { TrendPoint } from '@/lib/types';

interface TrendChartProps {
  points: TrendPoint[];
}

export default function TrendChart({ points }: TrendChartProps) {
  const data = points.length > 0 ? points : Array.from({ length: 24 }, (_, hour) => ({ hour: String(hour).padStart(2, '0'), count: 0 }));
  const maxValue = Math.max(...data.map((item) => item.count), 1);

  return (
    <div className="glass-card p-6">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-sm font-semibold text-text-secondary uppercase tracking-wider">Xu hướng vi phạm theo giờ</h3>
          <p className="text-xs text-text-muted mt-1">Dữ liệu trong 24 giờ gần nhất</p>
        </div>
        <div className="text-xs text-text-muted">
          {maxValue > 0 ? `Đỉnh: ${maxValue}` : 'Chưa có dữ liệu'}
        </div>
      </div>

      <div className="grid grid-cols-12 gap-2 items-end h-64">
        {data.map((point) => {
          const height = `${Math.max(6, (point.count / maxValue) * 100)}%`;
          return (
            <div key={point.hour} className="flex h-full flex-col items-center justify-end gap-2">
              <div className="w-full flex-1 flex items-end">
                <div
                  className="w-full rounded-t-lg bg-gradient-to-t from-primary-600 via-primary-500 to-cyan-400 transition-all duration-500"
                  style={{ height }}
                  title={`${point.hour}:00 - ${point.count}`}
                />
              </div>
              <div className="text-[10px] text-text-muted">{point.hour}</div>
            </div>
          );
        })}
      </div>
    </div>
  );
}