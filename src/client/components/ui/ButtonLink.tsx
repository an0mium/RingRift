import React from 'react';
import clsx from 'clsx';
import { Link, type LinkProps } from 'react-router-dom';

export type ButtonLinkVariant = 'primary' | 'secondary' | 'ghost' | 'outline' | 'danger';
export type ButtonLinkSize = 'sm' | 'md' | 'lg';

export interface ButtonLinkProps extends Omit<LinkProps, 'className'> {
  variant?: ButtonLinkVariant;
  size?: ButtonLinkSize;
  fullWidth?: boolean;
  className?: string;
}

const baseClasses =
  'inline-flex items-center justify-center rounded-md text-sm font-semibold transition-colors focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-slate-900';

const variantClasses: Record<ButtonLinkVariant, string> = {
  primary: 'bg-emerald-600 text-white hover:bg-emerald-500 focus:ring-emerald-500',
  secondary: 'bg-slate-700 text-slate-100 hover:bg-slate-600 focus:ring-slate-500',
  ghost: 'bg-transparent text-slate-200 hover:bg-slate-800/60 focus:ring-slate-500',
  outline:
    'border border-slate-600 bg-transparent text-slate-100 hover:bg-slate-800/40 focus:ring-slate-500',
  danger: 'bg-red-600 text-white hover:bg-red-500 focus:ring-red-500',
};

const sizeClasses: Record<ButtonLinkSize, string> = {
  sm: 'px-3 py-1.5 text-xs',
  md: 'px-4 py-2 text-sm',
  lg: 'px-5 py-2.5 text-base',
};

export const ButtonLink = React.forwardRef<HTMLAnchorElement, ButtonLinkProps>(
  ({ variant = 'primary', size = 'md', fullWidth, className, ...props }, ref) => {
    return (
      <Link
        ref={ref}
        className={clsx(
          baseClasses,
          variantClasses[variant],
          sizeClasses[size],
          fullWidth && 'w-full',
          className
        )}
        {...props}
      />
    );
  }
);

ButtonLink.displayName = 'ButtonLink';
