import { cn } from '@/lib/utils'
import { ButtonHTMLAttributes, forwardRef } from 'react'

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'outline' | 'ghost'
  size?: 'sm' | 'md'
}

const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = 'primary', size = 'md', ...props }, ref) => (
    <button
      ref={ref}
      className={cn(
        'inline-flex items-center justify-center gap-1.5 rounded-lg font-medium transition-colors',
        'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-500',
        'disabled:pointer-events-none disabled:opacity-40',
        variant === 'primary' && 'bg-violet-600 text-white hover:bg-violet-700',
        variant === 'outline' &&
          'border border-zinc-700 text-zinc-300 hover:bg-zinc-800 hover:text-white',
        variant === 'ghost' && 'text-zinc-400 hover:bg-zinc-800 hover:text-white',
        size === 'sm' && 'px-3 py-1.5 text-xs',
        size === 'md' && 'px-4 py-2.5 text-sm',
        className,
      )}
      {...props}
    />
  ),
)
Button.displayName = 'Button'

export { Button }
