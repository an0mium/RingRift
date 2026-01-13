import React from 'react';

interface LoadingSpinnerProps {
  size?: 'sm' | 'md' | 'lg';
  /** Color variant for the spinner */
  variant?: 'primary' | 'secondary' | 'muted';
  className?: string;
  /** Optional text to display below the spinner */
  text?: string;
}

const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({
  size = 'md',
  variant = 'primary',
  className = '',
  text,
}) => {
  const sizeClasses = {
    sm: 'w-4 h-4',
    md: 'w-8 h-8',
    lg: 'w-12 h-12',
  };

  const variantClasses = {
    primary: 'spinner-primary',
    secondary: 'spinner-secondary',
    muted: 'spinner-muted',
  };

  if (text) {
    return (
      <div className={`flex flex-col items-center gap-2 ${className}`}>
        <div
          className={`spinner ${variantClasses[variant]} ${sizeClasses[size]}`}
          role="status"
          aria-label={text}
        />
        <span className="text-sm text-slate-400">{text}</span>
      </div>
    );
  }

  return (
    <div
      className={`spinner ${variantClasses[variant]} ${sizeClasses[size]} ${className}`}
      role="status"
      aria-label="Loading"
    />
  );
};

export default LoadingSpinner;
