'use client'

import { useState } from 'react'
import { TOOLS, TOOL_LABELS } from '@/lib/tools'
import type { ToolName } from '@/types'

interface Props {
  defaultTool?: ToolName
  onSubmit: (correctTool: ToolName) => Promise<void>
  onCancel: () => void
  isSubmitting: boolean
}

export function ToolCorrection({ defaultTool = 'faq_search', onSubmit, onCancel, isSubmitting }: Props) {
  const [selected, setSelected] = useState<ToolName>(defaultTool)

  return (
    <div className="flex flex-wrap items-center gap-2 pt-1">
      <span className="text-xs text-zinc-500">Select correct tool:</span>

      <select
        value={selected}
        onChange={(e) => setSelected(e.target.value as ToolName)}
        disabled={isSubmitting}
        className="bg-zinc-900 border border-zinc-700 rounded-lg px-3 py-1.5 text-xs text-white
                   focus:outline-none focus:border-violet-500 cursor-pointer
                   disabled:opacity-50 transition-colors"
      >
        {TOOLS.map((tool) => (
          <option key={tool} value={tool}>
            {TOOL_LABELS[tool]}
          </option>
        ))}
      </select>

      <button
        onClick={() => onSubmit(selected)}
        disabled={isSubmitting}
        className="text-xs px-3 py-1.5 rounded-lg bg-violet-600 text-white
                   hover:bg-violet-700 disabled:opacity-50 transition-colors"
      >
        {isSubmitting ? 'Saving…' : 'Confirm'}
      </button>

      <button
        onClick={onCancel}
        disabled={isSubmitting}
        className="text-xs px-3 py-1.5 rounded-lg border border-zinc-700 text-zinc-400
                   hover:bg-zinc-800 hover:text-white disabled:opacity-50 transition-colors"
      >
        Cancel
      </button>
    </div>
  )
}
