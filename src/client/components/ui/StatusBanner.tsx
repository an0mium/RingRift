import React from 'react';
import clsx from 'clsx';

export type StatusBannerVariant = 'error' | 'warning' | 'info' | 'success';

export interface StatusBannerProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: StatusBannerVariant;
  title?: string;
  icon?: React.ReactNode;
  actions?: React.ReactNode;
  role?: 'alert' | 'status';
  children?: React.ReactNode;
}

const VARIANT_CLASSES: Record<StatusBannerVariant, string> = {
  error: 'bg-red-900/30 border-red-500/60 text-red-100',
  warning: 'bg-amber-900/30 border-amber-500/60 text-amber-100',
  info: 'bg-slate-900/60 border-slate-600 text-slate-100',
  success: 'bg-emerald-900/30 border-emerald-500/60 text-emerald-100',
};

export function StatusBanner({
  variant = 'info',
  title,
  icon,
  actions,
  role,
  className,
  children,
  ...props
}: StatusBannerProps) {
  const resolvedRole = role ?? (variant === 'error' ? 'alert' : 'status');

  return (
    <div
      role={resolvedRole}
      aria-live={resolvedRole === 'alert' ? 'assertive' : 'polite'}
      className={clsx(
        'rounded-xl border px-4 py-3 text-sm flex flex-wrap items-center justify-between gap-3',
        VARIANT_CLASSES[variant],
        className
      )}
      {...props}
    >
      <div className="flex items-start gap-3 min-w-0">
        {icon ? (
          <div className="mt-0.5 flex-shrink-0" aria-hidden="true">
            {icon}
          </div>
        ) : null}
        <div className="min-w-0">
          {title ? <div className="font-semibold">{title}</div> : null}
          {children ? (
            <div className={clsx(title ? 'mt-1 text-slate-100/90' : undefined)}>{children}</div>
          ) : null}
        </div>
      </div>
      {actions ? <div className="flex items-center gap-2">{actions}</div> : null}
    </div>
  );
}
