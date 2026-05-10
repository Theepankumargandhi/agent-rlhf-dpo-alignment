import type {
  EvaluationResponse,
  FeedbackRequest,
  FeedbackResponse,
  FeedbackStats,
  HealthResponse,
  QueryResponse,
  RewardModelStats,
} from '@/types'

// All calls go through Next.js proxy routes to avoid CORS.
const BASE = '/api'

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })

  if (!res.ok) {
    let detail = res.statusText
    try {
      const body = await res.json()
      detail = body?.detail ?? detail
    } catch {
      // non-JSON error body — keep statusText
    }
    throw new Error(detail || `Request failed with status ${res.status}`)
  }

  return res.json() as Promise<T>
}

export const api = {
  query: (query: string) =>
    request<QueryResponse>('/query', {
      method: 'POST',
      body: JSON.stringify({ query }),
    }),

  feedback: (payload: FeedbackRequest) =>
    request<FeedbackResponse>('/feedback', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  feedbackStats: () => request<FeedbackStats>('/feedback/stats'),

  evaluation: () => request<EvaluationResponse>('/evaluation'),

  health: () => request<HealthResponse>('/health'),

  rewardModelStats: () => request<RewardModelStats>('/reward-model/stats'),
}
