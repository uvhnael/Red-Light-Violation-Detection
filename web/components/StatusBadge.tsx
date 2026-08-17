'use client';

interface StatusBadgeProps {
  status: string;
  size?: 'sm' | 'md';
}

const statusConfig: Record<string, { label: string; className: string }> = {
  pending: { label: 'Chờ duyệt', className: 'status-pending' },
  approved: { label: 'Đã duyệt', className: 'status-confirmed' },
  confirmed: { label: 'Đã duyệt', className: 'status-confirmed' },
  rejected: { label: 'Từ chối', className: 'status-rejected' },
};

export default function StatusBadge({ status, size = 'sm' }: StatusBadgeProps) {
  const config = statusConfig[status] || { label: status, className: 'bg-surface-3 text-text-secondary border border-border' };
  const sizeClass = size === 'sm' ? 'text-xs px-2.5 py-0.5' : 'text-sm px-3 py-1';

  return (
    <span className={`inline-flex items-center rounded-full font-medium ${config.className} ${sizeClass}`}>
      {config.label}
    </span>
  );
}
