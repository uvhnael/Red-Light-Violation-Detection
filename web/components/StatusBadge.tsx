'use client';

interface StatusBadgeProps {
  status: string;
  size?: 'sm' | 'md';
}

const statusConfig: Record<string, { label: string; className: string; dotColor: string }> = {
  pending: { label: 'Chờ duyệt', className: 'status-pending', dotColor: 'bg-amber-500' },
  approved: { label: 'Đã duyệt', className: 'status-confirmed', dotColor: 'bg-emerald-500' },
  confirmed: { label: 'Đã duyệt', className: 'status-confirmed', dotColor: 'bg-emerald-500' },
  rejected: { label: 'Từ chối', className: 'status-rejected', dotColor: 'bg-rose-500' },
  online: { label: 'Trực tuyến', className: 'status-confirmed', dotColor: 'bg-emerald-500' },
  offline: { label: 'Ngoại tuyến', className: 'status-rejected', dotColor: 'bg-rose-500' },
  degraded: { label: 'Suy giảm', className: 'status-pending', dotColor: 'bg-amber-500' },
  maintenance: { label: 'Bảo trì', className: 'status-pending', dotColor: 'bg-amber-500' },
};

export default function StatusBadge({ status, size = 'sm' }: StatusBadgeProps) {
  const config = statusConfig[status.toLowerCase()] || {
    label: status,
    className: 'bg-surface-3 text-text-secondary border border-border',
    dotColor: 'bg-text-muted',
  };
  const sizeClass = size === 'sm' ? 'text-[11px] px-2.5 py-0.5 gap-1.5' : 'text-xs px-3 py-1 gap-2';

  return (
    <span className={`inline-flex items-center rounded-full font-semibold ${config.className} ${sizeClass} select-none`}>
      <span className={`w-1.5 h-1.5 rounded-full ${config.dotColor} shrink-0`} />
      {config.label}
    </span>
  );
}
