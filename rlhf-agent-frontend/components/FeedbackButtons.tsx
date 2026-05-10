'use client'

import { useState } from 'react'
import { ThumbsUp, ThumbsDown, CheckCircle2 } from 'lucide-react'
import { ToolCorrection } from './ToolCorrection'
import type { Message, Rating, ToolName } from '@/types'

interface Props {
  message: Message
  onFeedback: (message: Message, rating: Rating, correctTool: ToolName) => Promise<void>
}

type LocalView = 'idle' | 'correcting'

export function FeedbackButtons({ message, onFeedback }: Props) {
  const [view, setView] = useState<LocalView>('idle')
  const [busy, setBusy] = useState(false)

  // Parent flips message.feedbackState to 'submitted' after API call succeeds.
  if (message.feedbackState === 'submitted') {
    return (
      <div className="flex items-center gap-1.5 text-xs text-zinc-500 pt-1">
        <CheckCircle2 className="h-3.5 w-3.5 text-green-500 flex-shrink-0" />
        Thanks for your feedback!
      </div>
    )
  }

  async function handleThumbsUp() {
    if (!message.queryResult) return
    setBusy(true)
    await onFeedback(message, 'thumbs_up', message.queryResult.selected_tool)
    setBusy(false)
  }

  async function handleConfirmCorrection(correctTool: ToolName) {
    setBusy(true)
    await onFeedback(message, 'thumbs_down', correctTool)
    setBusy(false)
  }

  if (view === 'correcting') {
    return (
      <ToolCorrection
        defaultTool={message.queryResult?.selected_tool}
        onSubmit={handleConfirmCorrection}
        onCancel={() => setView('idle')}
        isSubmitting={busy}
      />
    )
  }

  return (
    <div className="flex items-center gap-2 pt-1">
      <span className="text-xs text-zinc-600">Was this the right tool?</span>

      <button
        onClick={handleThumbsUp}
        disabled={busy}
        className="flex items-center gap-1 text-xs px-2.5 py-1 rounded-md
                   border border-zinc-700 text-zinc-400
                   hover:border-green-600 hover:text-green-400 hover:bg-green-950/40
                   disabled:opacity-40 transition-colors"
      >
        <ThumbsUp className="h-3 w-3" />
        Correct
      </button>

      <button
        onClick={() => setView('correcting')}
        disabled={busy}
        className="flex items-center gap-1 text-xs px-2.5 py-1 rounded-md
                   border border-zinc-700 text-zinc-400
                   hover:border-red-600 hover:text-red-400 hover:bg-red-950/40
                   disabled:opacity-40 transition-colors"
      >
        <ThumbsDown className="h-3 w-3" />
        Wrong Tool
      </button>
    </div>
  )
}
