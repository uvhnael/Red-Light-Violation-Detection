/** Skeleton khi chuyển route trong dashboard — không spinner, không màn hình trắng. */
export default function DashboardLoading() {
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4">
        {Array.from({ length: 4 }).map((_, i) => (
          <div key={i} className="glass-card p-5 space-y-4">
            <div className="skeleton h-4 w-24" />
            <div className="skeleton h-8 w-16" />
            <div className="skeleton h-3 w-32" />
          </div>
        ))}
      </div>
      <div className="glass-card p-6 space-y-4">
        <div className="skeleton h-5 w-48" />
        <div className="skeleton h-72 w-full rounded-xl" />
      </div>
    </div>
  );
}
