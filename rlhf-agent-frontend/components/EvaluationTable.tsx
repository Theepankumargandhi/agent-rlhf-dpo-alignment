import { TOOL_BADGE_CLASS, TOOL_LABELS, TOOLS } from '@/lib/tools'
import type { EvaluationResponse, ToolName } from '@/types'

interface Props {
  data: EvaluationResponse | null
  isLoading: boolean
}

function pct(val: number | null | undefined): string {
  if (val == null) return '—'
  return `${(val * 100).toFixed(1)}%`
}

function delta(sft: number | null | undefined, dpo: number | null | undefined): string {
  if (sft == null || dpo == null) return '—'
  const d = (dpo - sft) * 100
  return `${d >= 0 ? '+' : ''}${d.toFixed(1)}%`
}

function deltaClass(sft: number | null | undefined, dpo: number | null | undefined): string {
  if (sft == null || dpo == null) return 'text-zinc-600'
  const d = dpo - sft
  if (d > 0.001) return 'text-green-400'
  if (d < -0.001) return 'text-red-400'
  return 'text-zinc-400'
}

function SkeletonRow() {
  return (
    <tr className="border-b border-[#1e1e1e]">
      {[1, 2, 3, 4].map((i) => (
        <td key={i} className="py-3.5 px-5">
          <div className="h-4 bg-zinc-800 rounded animate-pulse" />
        </td>
      ))}
    </tr>
  )
}

export function EvaluationTable({ data, isLoading }: Props) {
  const sft = data?.sft_baseline
  const dpo = data?.dpo_model

  const rows: Array<{ key: string; label: React.ReactNode; tool: ToolName | 'overall' }> = [
    ...TOOLS.map((t) => ({
      key: t,
      tool: t as ToolName,
      label: (
        <span className={`text-xs px-2 py-0.5 rounded-full border font-medium ${TOOL_BADGE_CLASS[t]}`}>
          {TOOL_LABELS[t]}
        </span>
      ),
    })),
    { key: 'overall', tool: 'overall', label: <span className="text-sm font-semibold text-white">Overall</span> },
  ]

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-[#2a2a2a]">
            <th className="py-3.5 px-5 text-xs font-medium uppercase tracking-wider text-zinc-500 text-left">
              Tool
            </th>
            <th className="py-3.5 px-5 text-xs font-medium uppercase tracking-wider text-zinc-500 text-right">
              SFT Baseline
            </th>
            <th className="py-3.5 px-5 text-xs font-medium uppercase tracking-wider text-zinc-500 text-right">
              <span className="inline-flex items-center justify-end gap-2">
                DPO Model
                {dpo && (
                  <span className="text-[10px] font-semibold px-1.5 py-0.5 rounded-full bg-violet-600/20 text-violet-400 border border-violet-500/30 normal-case tracking-normal">
                    DPO Trained ✓
                  </span>
                )}
              </span>
            </th>
            <th className="py-3.5 px-5 text-xs font-medium uppercase tracking-wider text-zinc-500 text-right">
              Improvement
            </th>
          </tr>
        </thead>
        <tbody>
          {isLoading
            ? Array.from({ length: 5 }).map((_, i) => <SkeletonRow key={i} />)
            : rows.map(({ key, label, tool }) => {
                const isOverall = tool === 'overall'
                const sftVal = isOverall ? sft?.overall_accuracy : sft?.per_tool[tool as ToolName]
                const dpoVal = isOverall ? dpo?.overall_accuracy : dpo?.per_tool[tool as ToolName]

                return (
                  <tr
                    key={key}
                    className={`border-b border-[#1e1e1e] hover:bg-white/[0.02] transition-colors ${
                      isOverall ? 'bg-white/[0.02]' : ''
                    }`}
                  >
                    <td className="py-3.5 px-5">{label}</td>
                    <td className="py-3.5 px-5 text-right tabular-nums text-zinc-300">{pct(sftVal)}</td>
                    <td className={`py-3.5 px-5 text-right tabular-nums ${dpo ? 'text-zinc-300' : 'text-zinc-500'}`}>{pct(dpoVal)}</td>
                    <td className={`py-3.5 px-5 text-right tabular-nums font-medium ${deltaClass(sftVal, dpoVal)}`}>
                      {delta(sftVal, dpoVal)}
                    </td>
                  </tr>
                )
              })}
        </tbody>
      </table>

      {!isLoading && !dpo && (
        <p className="px-5 py-3 text-xs text-zinc-600 border-t border-[#1e1e1e]">
          DPO model not yet trained — run{' '}
          <code className="text-violet-400 font-mono">python training/run_dpo.py</code> to populate
          the DPO columns.
        </p>
      )}
    </div>
  )
}
