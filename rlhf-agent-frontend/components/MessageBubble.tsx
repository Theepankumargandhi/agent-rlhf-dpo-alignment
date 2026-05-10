'use client'

import { useState } from 'react'
import { ChevronDown } from 'lucide-react'
import { TOOL_BADGE_CLASS, TOOL_LABELS, TOOLS } from '@/lib/tools'
import { FeedbackButtons } from './FeedbackButtons'
import type { Message, Rating, ToolName } from '@/types'

interface Props {
  message: Message
  onFeedback: (message: Message, rating: Rating, correctTool: ToolName) => Promise<void>
}

function rewardScoreClass(score: number): string {
  if (score >= 0.7) return 'text-green-400'
  if (score >= 0.4) return 'text-yellow-400'
  return 'text-red-400'
}

// ── All-tool scores panel ──────────────────────────────────────────────────

function ToolScoresPanel({ scores }: { scores: Record<string, number> }) {
  const values = TOOLS.map((t) => scores[t] ?? 0)
  const maxScore = Math.max(...values)

  return (
    <div className="space-y-1.5 pt-1">
      {TOOLS.map((tool) => {
        const score = scores[tool] ?? 0
        const isHighest = score === maxScore
        const widthPct = `${(score * 100).toFixed(0)}%`

        return (
          <div
            key={tool}
            className={`flex items-center gap-3 px-2 py-1 rounded-lg ${
              isHighest ? 'bg-violet-500/5 border border-violet-500/15' : ''
            }`}
          >
            <span className="text-[11px] text-zinc-500 w-28 shrink-0">
              {TOOL_LABELS[tool as ToolName]}
            </span>
            <div className="flex-1 h-1.5 bg-zinc-800 rounded-full overflow-hidden">
              <div
                className="h-full bg-violet-500 rounded-full transition-all duration-700"
                style={{ width: widthPct }}
              />
            </div>
            <span
              className={`text-[11px] tabular-nums w-8 text-right shrink-0 ${
                isHighest ? 'text-violet-400 font-semibold' : 'text-zinc-500'
              }`}
            >
              {score.toFixed(2)}
            </span>
          </div>
        )
      })}
    </div>
  )
}

// ── Main component ─────────────────────────────────────────────────────────

export function MessageBubble({ message, onFeedback }: Props) {
  const [scoresOpen, setScoresOpen] = useState(false)

  // ── User ────────────────────────────────────────────────────────
  if (message.type === 'user') {
    return (
      <div className="flex justify-end">
        <div className="bg-violet-600 text-white rounded-2xl rounded-br-sm px-4 py-3 max-w-[78%] text-sm leading-relaxed">
          {message.content}
        </div>
      </div>
    )
  }

  // ── Error ───────────────────────────────────────────────────────
  if (message.type === 'error') {
    return (
      <div className="flex justify-start">
        <div className="bg-red-950/40 border border-red-800/50 text-red-400 rounded-2xl px-4 py-3 max-w-[78%] text-sm">
          ⚠️ {message.content}
        </div>
      </div>
    )
  }

  // ── Assistant ───────────────────────────────────────────────────
  const { queryResult } = message

  return (
    <div className="flex items-start gap-3 max-w-[85%]">
      {/* Avatar */}
      <div className="w-7 h-7 rounded-lg bg-violet-600/80 flex items-center justify-center text-xs flex-shrink-0 mt-0.5 select-none">
        🤖
      </div>

      <div className="flex-1 space-y-2.5">
        {/* Response text */}
        <div className="bg-[#1a1a1a] border border-[#2a2a2a] rounded-2xl rounded-tl-sm px-4 py-4">
          <p className="text-sm text-zinc-200 leading-relaxed whitespace-pre-wrap">
            {message.content}
          </p>
        </div>

        {/* Tool metadata + feedback */}
        {queryResult && (
          <div className="space-y-2 px-1">
            {/* Tool badge + confidence + reward score */}
            <div className="flex items-center gap-2.5 flex-wrap">
              <span className="text-xs text-zinc-600">Tool used:</span>
              <span
                className={`text-xs px-2 py-0.5 rounded-full border font-medium ${
                  TOOL_BADGE_CLASS[queryResult.selected_tool]
                }`}
              >
                {TOOL_LABELS[queryResult.selected_tool]}
              </span>
              <span className="text-xs text-zinc-600">
                Confidence: {(queryResult.confidence * 100).toFixed(0)}%
              </span>
              {queryResult.reward_score != null && (
                <>
                  <span className="text-zinc-700" aria-hidden>·</span>
                  <span className="text-xs text-zinc-600">
                    Reward Score:{' '}
                    <span className={rewardScoreClass(queryResult.reward_score)}>
                      {queryResult.reward_score.toFixed(2)}
                    </span>
                  </span>
                </>
              )}
            </div>

            {/* Confidence bar */}
            <div className="h-1 w-40 bg-zinc-800 rounded-full overflow-hidden">
              <div
                className="h-full bg-violet-500 rounded-full transition-all duration-700"
                style={{ width: `${(queryResult.confidence * 100).toFixed(0)}%` }}
              />
            </div>

            {/* Collapsible all-tool scores */}
            {queryResult.all_tool_scores != null && (
              <div>
                <button
                  onClick={() => setScoresOpen((prev) => !prev)}
                  className="flex items-center gap-1 text-[11px] text-zinc-600 hover:text-zinc-400 transition-colors"
                >
                  <ChevronDown
                    className={`h-3 w-3 transition-transform duration-200 ${
                      scoresOpen ? 'rotate-180' : ''
                    }`}
                  />
                  {scoresOpen ? 'Hide tool scores' : 'Show tool scores'}
                </button>

                <div
                  className={`overflow-hidden transition-all duration-300 ease-in-out ${
                    scoresOpen ? 'max-h-48 opacity-100 mt-2' : 'max-h-0 opacity-0'
                  }`}
                >
                  <ToolScoresPanel scores={queryResult.all_tool_scores} />
                </div>
              </div>
            )}

            {/* Feedback row */}
            <FeedbackButtons message={message} onFeedback={onFeedback} />
          </div>
        )}
      </div>
    </div>
  )
}
