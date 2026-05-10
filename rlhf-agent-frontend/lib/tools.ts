import type { ToolName } from '@/types'

export const TOOLS: ToolName[] = [
  'faq_search',
  'order_status',
  'refund_policy',
  'raise_ticket',
]

export const TOOL_LABELS: Record<ToolName, string> = {
  faq_search: 'FAQ Search',
  order_status: 'Order Status',
  refund_policy: 'Refund Policy',
  raise_ticket: 'Raise Ticket',
}

// Tailwind classes for each tool badge (bg + text + border)
export const TOOL_BADGE_CLASS: Record<ToolName, string> = {
  faq_search: 'bg-blue-500/10 text-blue-400 border-blue-500/25',
  order_status: 'bg-green-500/10 text-green-400 border-green-500/25',
  refund_policy: 'bg-yellow-500/10 text-yellow-400 border-yellow-500/25',
  raise_ticket: 'bg-red-500/10 text-red-400 border-red-500/25',
}
