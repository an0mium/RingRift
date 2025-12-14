import React from 'react';
import clsx from 'clsx';

export type InlineAlertVariant = 'error' | 'warning' | 'info' | 'success';

export interface InlineAlertProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: InlineAlertVariant;
  title?: string;
  role?: 'alert' | 'status';
  children: React.ReactNode;
}

const VARIANT_CLASSES: Record<InlineAlertVariant, string> = {
  error: 'bg-red-900/40 border-red-700 text-red-300',
  warning: 'bg-amber-900/40 border-amber-500/60 text-amber-200',
  info: 'bg-slate-900/60 border-slate-600 text-slate-200',
  success: 'bg-emerald-900/30 border-emerald-500/60 text-emerald-200',
};

export function InlineAlert({
  variant = 'error',
  title,
  role,
  className,
  children,
  ...props
}: InlineAlertProps) {
  const resolvedRole = role ?? (variant === 'error' ? 'alert' : 'status');

  return (
    <div
      role={resolvedRole}
      aria-live={resolvedRole === 'alert' ? 'assertive' : 'polite'}
      className={clsx('rounded-lg border px-3 py-2 text-sm', VARIANT_CLASSES[variant], className)}
      {...props}
    >
      {title ? <div className="font-semibold text-slate-50">{title}</div> : null}
      <div className={clsx(title ? 'mt-1' : undefined)}>{children}</div>
    </div>
  );
}
