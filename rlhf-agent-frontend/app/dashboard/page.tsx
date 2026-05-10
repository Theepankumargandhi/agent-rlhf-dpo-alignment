'use client'

import { useCallback, useEffect, useState } from 'react'
import { RefreshCw, TrendingUp, ThumbsUp, MessageSquare, Layers, Cpu, Award } from 'lucide-react'
import { api } from '@/lib/api'
import { EvaluationTable } from '@/components/EvaluationTable'
import type { EvaluationResponse, FeedbackStats, RewardModelStats, TrainingInfo } from '@/types'

const REFRESH_MS = 30_000

// ── Stat card ─────────────────────────────────────────────────────────────

interface StatCardProps {
  icon: React.ReactNode
  label: string
  value: string | number
  sub?: string
  isLoading?: boolean
}

function StatCard({ icon, label, value, sub, isLoading }: StatCardProps) {
  return (
    <div className="bg-[#1a1a1a] border border-[#2a2a2a] rounded-xl p-5 space-y-3">
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium uppercase tracking-wider text-zinc-500">{label}</span>
        <div className="w-8 h-8 rounded-lg bg-violet-600/10 border border-violet-500/20
                        flex items-center justify-center text-violet-400">
          {icon}
        </div>
      </div>
      {isLoading ? (
        <div className="h-8 w-24 bg-zinc-800 rounded animate-pulse" />
      ) : (
        <p className="text-3xl font-bold text-white tabular-nums">{value}</p>
      )}
      {sub && <p className="text-xs text-zinc-600">{sub}</p>}
    </div>
  )
}

// ── Accuracy progress bar ──────────────────────────────────────────────────

interface AccuracyBarProps {
  label: string
  value: number
  variant: 'sft' | 'dpo'
}

function AccuracyBar({ label, value, variant }: AccuracyBarProps) {
  const pct = `${(value * 100).toFixed(1)}%`
  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between text-xs">
        <span className="text-zinc-400">{label}</span>
        <span className={`font-semibold tabular-nums ${variant === 'dpo' ? 'text-violet-400' : 'text-zinc-300'}`}>
          {pct}
        </span>
      </div>
      <div className="h-2 rounded-full bg-zinc-800 overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-700 ${
            variant === 'dpo' ? 'bg-violet-500' : 'bg-zinc-500'
          }`}
          style={{ width: pct }}
        />
      </div>
    </div>
  )
}

// ── DPO Training info card ─────────────────────────────────────────────────

function TrainingInfoCard({ info, isLoading }: { info: TrainingInfo | null; isLoading: boolean }) {
  const items: Array<{ label: string; value: string }> = info
    ? [
        { label: 'Preference pairs', value: String(info.preference_pairs_used) },
        { label: 'DPO beta (β)', value: String(info.dpo_beta) },
        { label: 'Epochs', value: String(info.epochs) },
        { label: 'Learning rate', value: info.learning_rate.toExponential(0) },
        {
          label: 'Trained at',
          value: new Date(info.trained_at).toLocaleString(undefined, {
            dateStyle: 'medium',
            timeStyle: 'short',
          }),
        },
      ]
    : []

  return (
    <div className="bg-[#1a1a1a] border border-[#2a2a2a] rounded-xl overflow-hidden">
      <div className="px-5 py-4 border-b border-[#2a2a2a]">
        <h2 className="text-sm font-semibold text-white flex items-center gap-2">
          <Cpu className="h-3.5 w-3.5 text-violet-400" />
          DPO Training Info
        </h2>
        <p className="text-xs text-zinc-500 mt-0.5">Hyperparameters used for the last DPO run</p>
      </div>

      {isLoading ? (
        <div className="px-5 py-4 grid grid-cols-2 sm:grid-cols-3 gap-4">
          {Array.from({ length: 5 }).map((_, i) => (
            <div key={i} className="space-y-1.5">
              <div className="h-3 w-20 bg-zinc-800 rounded animate-pulse" />
              <div className="h-4 w-12 bg-zinc-800 rounded animate-pulse" />
            </div>
          ))}
        </div>
      ) : !info ? (
        <p className="px-5 py-4 text-xs text-zinc-600">
          No DPO training data yet — run{' '}
          <code className="text-violet-400 font-mono">python training/run_dpo.py</code> after collecting preference pairs.
        </p>
      ) : (
        <dl className="px-5 py-4 grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-5">
          {items.map(({ label, value }) => (
            <div key={label}>
              <dt className="text-[11px] text-zinc-500 uppercase tracking-wider mb-1">{label}</dt>
              <dd className="text-sm font-semibold text-white tabular-nums">{value}</dd>
            </div>
          ))}
        </dl>
      )}
    </div>
  )
}

// ── Reward Model card ──────────────────────────────────────────────────────

function RewardModelCard({ stats, isLoading }: { stats: RewardModelStats | null; isLoading: boolean }) {
  const fmt = (v: number | null) => (v == null ? '—' : v.toFixed(4))

  return (
    <div className="bg-[#1a1a1a] border border-[#2a2a2a] rounded-xl overflow-hidden">
      <div className="px-5 py-4 border-b border-[#2a2a2a] flex items-center justify-between">
        <div>
          <h2 className="text-sm font-semibold text-white flex items-center gap-2">
            <Award className="h-3.5 w-3.5 text-violet-400" />
            Reward Model Status
          </h2>
          <p className="text-xs text-zinc-500 mt-0.5">
            DistilBERT regression head scoring (query, tool) pairs
          </p>
        </div>
        {!isLoading && stats && (
          <span
            className={`text-xs font-semibold px-2 py-0.5 rounded-full border ${
              stats.trained
                ? 'bg-green-500/10 text-green-400 border-green-500/25'
                : 'bg-zinc-800 text-zinc-500 border-zinc-700'
            }`}
          >
            {stats.trained ? 'Trained ✅' : 'Not trained'}
          </span>
        )}
      </div>

      {isLoading ? (
        <div className="px-5 py-4 grid grid-cols-2 sm:grid-cols-3 gap-4">
          {Array.from({ length: 4 }).map((_, i) => (
            <div key={i} className="space-y-1.5">
              <div className="h-3 w-20 bg-zinc-800 rounded animate-pulse" />
              <div className="h-4 w-12 bg-zinc-800 rounded animate-pulse" />
            </div>
          ))}
        </div>
      ) : !stats || !stats.trained ? (
        <p className="px-5 py-4 text-xs text-zinc-600">
          {stats?.note ?? 'Reward model not yet trained.'}{' '}
          Run{' '}
          <code className="text-violet-400 font-mono">python reward_model/train_reward.py</code>.
        </p>
      ) : (
        <div className="divide-y divide-[#1e1e1e]">
          <dl className="px-5 py-4 grid grid-cols-2 sm:grid-cols-4 gap-5">
            <div>
              <dt className="text-[11px] text-zinc-500 uppercase tracking-wider mb-1">Pearson r</dt>
              <dd className="text-sm font-semibold text-white tabular-nums">
                {fmt(stats.pearson_correlation)}
              </dd>
            </div>
            <div>
              <dt className="text-[11px] text-zinc-500 uppercase tracking-wider mb-1">Discrim. Gap</dt>
              <dd className="text-sm font-semibold text-white tabular-nums">
                {fmt(stats.discrimination_gap)}
              </dd>
            </div>
            <div>
              <dt className="text-[11px] text-zinc-500 uppercase tracking-wider mb-1">Samples used</dt>
              <dd className="text-sm font-semibold text-white tabular-nums">
                {stats.samples_used}
                <span className="text-zinc-600 font-normal text-[11px] ml-1">
                  ({stats.real_samples} real + {stats.synthetic_samples} synthetic)
                </span>
              </dd>
            </div>
            <div>
              <dt className="text-[11px] text-zinc-500 uppercase tracking-wider mb-1">Trained at</dt>
              <dd className="text-sm font-semibold text-white">
                {stats.trained_at
                  ? new Date(stats.trained_at).toLocaleString(undefined, {
                      dateStyle: 'medium',
                      timeStyle: 'short',
                    })
                  : '—'}
              </dd>
            </div>
          </dl>
          {stats.note && (
            <p className="px-5 py-3 text-xs text-zinc-500 italic">{stats.note}</p>
          )}
        </div>
      )}
    </div>
  )
}

// ── Page ──────────────────────────────────────────────────────────────────

export default function DashboardPage() {
  const [stats, setStats] = useState<FeedbackStats | null>(null)
  const [evaluation, setEvaluation] = useState<EvaluationResponse | null>(null)
  const [rewardStats, setRewardStats] = useState<RewardModelStats | null>(null)
  const [statsLoading, setStatsLoading] = useState(true)
  const [evalLoading, setEvalLoading] = useState(true)
  const [rewardLoading, setRewardLoading] = useState(true)
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [isRefreshing, setIsRefreshing] = useState(false)

  const refresh = useCallback(async () => {
    setError(null)
    setIsRefreshing(true)
    const [sResult, eResult, rResult] = await Promise.allSettled([
      api.feedbackStats(),
      api.evaluation(),
      api.rewardModelStats(),
    ])

    if (sResult.status === 'fulfilled') {
      setStats(sResult.value)
    } else {
      setError((sResult.reason as Error).message)
    }

    if (eResult.status === 'fulfilled') {
      setEvaluation(eResult.value)
    }

    if (rResult.status === 'fulfilled') {
      setRewardStats(rResult.value)
    }

    setStatsLoading(false)
    setEvalLoading(false)
    setRewardLoading(false)
    setLastUpdated(new Date())
    setIsRefreshing(false)
  }, [])

  useEffect(() => {
    refresh()
    const id = setInterval(refresh, REFRESH_MS)
    return () => clearInterval(id)
  }, [refresh])

  const accuracy =
    stats?.accuracy_from_feedback != null
      ? `${(stats.accuracy_from_feedback * 100).toFixed(1)}%`
      : '—'

  const sftOverall = evaluation?.sft_baseline?.overall_accuracy ?? null
  const dpoOverall = evaluation?.dpo_model?.overall_accuracy ?? null
  const showBars = !evalLoading && (sftOverall !== null || dpoOverall !== null)

  return (
    <main className="max-w-5xl mx-auto px-4 py-8 space-y-8">
      {/* ── Header ─────────────────────────────────────────── */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold text-white">Dashboard</h1>
          <p className="text-xs text-zinc-500 mt-1">
            RLHF feedback metrics and model evaluation · auto-refreshes every 30 s
          </p>
        </div>
        <button
          onClick={refresh}
          disabled={isRefreshing}
          className="flex items-center gap-1.5 text-xs text-zinc-500 hover:text-zinc-300
                     px-3 py-1.5 rounded-md hover:bg-zinc-800 transition-colors
                     disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <RefreshCw className={`h-3.5 w-3.5 ${isRefreshing ? 'animate-spin' : ''}`} />
          {lastUpdated ? lastUpdated.toLocaleTimeString() : 'Loading…'}
        </button>
      </div>

      {/* ── Error banner ───────────────────────────────────── */}
      {error && (
        <div className="bg-red-950/40 border border-red-800/50 text-red-400 text-xs
                        rounded-xl px-4 py-3">
          ⚠️ Could not reach backend: {error}
        </div>
      )}

      {/* ── Stat cards ─────────────────────────────────────── */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          icon={<MessageSquare className="h-4 w-4" />}
          label="Total Feedback"
          value={stats?.total_feedback ?? 0}
          sub="responses rated"
          isLoading={statsLoading}
        />
        <StatCard
          icon={<ThumbsUp className="h-4 w-4" />}
          label="Thumbs Up"
          value={stats?.thumbs_up ?? 0}
          sub="correct predictions"
          isLoading={statsLoading}
        />
        <StatCard
          icon={<Layers className="h-4 w-4" />}
          label="Preference Pairs"
          value={stats?.preference_pairs_collected ?? 0}
          sub="ready for DPO"
          isLoading={statsLoading}
        />
        <StatCard
          icon={<TrendingUp className="h-4 w-4" />}
          label="Feedback Accuracy"
          value={statsLoading ? '—' : accuracy}
          sub="from human ratings"
          isLoading={statsLoading}
        />
      </div>

      {/* ── Overall accuracy comparison ─────────────────────── */}
      {showBars && (
        <div className="bg-[#1a1a1a] border border-[#2a2a2a] rounded-xl p-5 space-y-4">
          <div>
            <h2 className="text-sm font-semibold text-white">Overall Accuracy</h2>
            <p className="text-xs text-zinc-500 mt-0.5">Held-out test set (20% of 60 queries)</p>
          </div>
          <div className="space-y-3">
            {sftOverall !== null && <AccuracyBar label="SFT Baseline" value={sftOverall} variant="sft" />}
            {dpoOverall !== null && <AccuracyBar label="DPO Model" value={dpoOverall} variant="dpo" />}
          </div>
        </div>
      )}

      {/* ── DPO Training Info ───────────────────────────────── */}
      <TrainingInfoCard info={evaluation?.training_info ?? null} isLoading={evalLoading} />

      {/* ── Reward Model Status ─────────────────────────────── */}
      <RewardModelCard stats={rewardStats} isLoading={rewardLoading} />

      {/* ── Per-tool accuracy table ─────────────────────────── */}
      <div className="bg-[#1a1a1a] border border-[#2a2a2a] rounded-xl overflow-hidden">
        <div className="px-5 py-4 border-b border-[#2a2a2a]">
          <h2 className="text-sm font-semibold text-white">Per-Tool Accuracy</h2>
          <p className="text-xs text-zinc-500 mt-0.5">
            Per-tool accuracy on the held-out test set (20% of 60 queries)
          </p>
        </div>
        <EvaluationTable data={evaluation} isLoading={evalLoading} />
      </div>
    </main>
  )
}
