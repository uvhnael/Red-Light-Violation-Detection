'use client';

interface DataTableProps {
  columns: string[];
  rows: Record<string, unknown>[];
  maxHeight?: number;
}

function formatCell(value: unknown): string {
  if (value === null || value === undefined) return '—';
  if (typeof value === 'number') {
    if (Number.isInteger(value)) return value.toLocaleString('vi-VN');
    if (value < 1) return value.toFixed(4);
    return value.toFixed(2);
  }
  if (value instanceof Date) return value.toLocaleString('vi-VN');
  if (typeof value === 'string' && value.match(/^\d{4}-\d{2}-\d{2}T[\d:.]+Z?$/)) {
    return new Date(value).toLocaleString('vi-VN');
  }
  const str = String(value);
  return str.length > 100 ? str.slice(0, 100) + '...' : str;
}

export function DataTable({ columns, rows, maxHeight = 400 }: DataTableProps) {
  if (!columns?.length || !rows?.length) {
    return (
      <div className="text-center py-6 text-text-muted text-sm">
        Không có dữ liệu.
      </div>
    );
  }

  return (
    <div
      className="overflow-auto border border-border rounded-lg bg-surface-2/50"
      style={{ maxHeight: maxHeight }}
    >
      <table className="w-full text-xs">
        <thead className="sticky top-0 z-10">
          <tr className="bg-surface-3 border-b border-border">
            {columns.map((col) => (
              <th
                key={col}
                className="px-3 py-2 text-left font-semibold text-text-secondary uppercase tracking-wider whitespace-nowrap border-r border-border/50 last:border-r-0"
              >
                {col.replace(/_/g, ' ')}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-border/30">
          {rows.map((row, i) => (
            <tr
              key={i}
              className={i % 2 === 0 ? 'bg-surface-2/30' : 'bg-transparent hover:bg-surface-3/50 transition-colors'}
            >
              {columns.map((col) => (
                <td
                  key={col}
                  className="px-3 py-2 text-text-primary whitespace-nowrap border-r border-border/20 last:border-r-0"
                >
                  {formatCell(row[col])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}