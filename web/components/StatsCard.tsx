'use client';

interface StatsCardProps {
  title: string;
  value: number | string;
  icon: React.ReactNode;
  trend?: string;
  color: 'primary' | 'success' | 'warning' | 'danger';
}

const colorMap = {
  primary: {
    bg: 'from-primary-600/20 to-primary-700/10',
    border: 'border-primary-500/20',
    icon: 'bg-primary-500/15 text-primary-400',
    glow: 'shadow-primary-500/10',
  },
  success: {
    bg: 'from-emerald-600/20 to-emerald-700/10',
    border: 'border-emerald-500/20',
    icon: 'bg-emerald-500/15 text-emerald-400',
    glow: 'shadow-emerald-500/10',
  },
  warning: {
    bg: 'from-amber-600/20 to-amber-700/10',
    border: 'border-amber-500/20',
    icon: 'bg-amber-500/15 text-amber-400',
    glow: 'shadow-amber-500/10',
  },
  danger: {
    bg: 'from-red-600/20 to-red-700/10',
    border: 'border-red-500/20',
    icon: 'bg-red-500/15 text-red-400',
    glow: 'shadow-red-500/10',
  },
};

export default function StatsCard({ title, value, icon, trend, color }: StatsCardProps) {
  const c = colorMap[color];

  return (
    <div className={`glass-card-hover p-5 bg-gradient-to-br ${c.bg} ${c.border} shadow-lg ${c.glow}`}>
      <div className="flex items-start justify-between">
        <div>
          <p className="text-sm text-text-secondary font-medium">{title}</p>
          <p className="text-3xl font-bold mt-2 text-text-primary">{value}</p>
          {trend && (
            <p className="text-xs mt-1 text-text-muted">{trend}</p>
          )}
        </div>
        <div className={`p-3 rounded-xl ${c.icon}`}>
          {icon}
        </div>
      </div>
    </div>
  );
}
