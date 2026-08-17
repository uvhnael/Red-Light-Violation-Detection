'use client';

interface BarChartProps {
  columns: string[];
  rows: Record<string, unknown>[];
}

export function BarChart({ columns, rows }: BarChartProps) {
  if (!columns?.length || !rows?.length) {
    return <p className="text-text-muted text-sm text-center py-4">Không có dữ liệu.</p>;
  }

  const labelCol = columns.find((c) => rows.every((r) => typeof r[c] === 'string')) || columns[0];
  const valueCol = columns.find((c) => rows.every((r) => typeof r[c] === 'number')) || columns[1] || columns[0];
  const maxValue = Math.max(...rows.map((r) => Number(r[valueCol]) || 0));

  const barColors = [
    'bg-gradient-to-r from-primary-500 to-blue-500',
    'bg-gradient-to-r from-emerald-500 to-green-500',
    'bg-gradient-to-r from-amber-400 to-orange-500',
    'bg-gradient-to-r from-red-500 to-pink-500',
    'bg-gradient-to-r from-violet-500 to-indigo-500',
  ];

  return (
    <div className="space-y-2 p-4">
      <p className="text-xs text-text-muted mb-3">
        Biểu đồ: {labelCol.replace(/_/g, ' ')} theo {valueCol.replace(/_/g, ' ')}
      </p>
      {rows.map((row, i) => {
        const val = Number(row[valueCol]) || 0;
        const pct = maxValue > 0 ? (val / maxValue) * 100 : 0;
        return (
          <div key={i} className="flex items-center gap-2">
            <span className="text-text-secondary text-xs w-36 truncate text-right">
              {String(row[labelCol] || '').replace(/_/g, ' ')}
            </span>
            <div className="flex-1 h-6 bg-surface-3 rounded-full overflow-hidden relative">
              <div
                className={`h-full rounded-full transition-all duration-700 ease-out ${barColors[i % barColors.length]}`}
                style={{ width: `${Math.max(pct, 2)}%` }}
              />
              <span className="absolute inset-0 flex items-center px-2 text-xs font-semibold text-white/90">
                {val.toLocaleString('vi-VN')}
              </span>
            </div>
          </div>
        );
      })}
    </div>
  );
}