export type ToolName = 'faq_search' | 'order_status' | 'refund_policy' | 'raise_ticket'
export type Rating = 'thumbs_up' | 'thumbs_down'
export type FeedbackState = 'idle' | 'correcting' | 'submitted'

// ── API request / response shapes ─────────────────────────────────────────

export interface QueryRequest {
  query: string
}

export interface QueryResponse {
  query: string
  selected_tool: ToolName
  confidence: number
  response: string
  session_id: string
  reward_score: number | null
  all_tool_scores: Record<string, number> | null
}

export interface FeedbackRequest {
  session_id: string
  query: string
  selected_tool: ToolName
  correct_tool: ToolName
  rating: Rating
}

export interface FeedbackResponse {
  status: string
  preference_pair_created: boolean
}

export interface FeedbackStats {
  total_feedback: number
  thumbs_up: number
  thumbs_down: number
  preference_pairs_collected: number
  accuracy_from_feedback: number | null
}

export interface ToolAccuracy {
  overall_accuracy: number
  per_tool: Record<ToolName, number | null>
}

export interface TrainingInfo {
  preference_pairs_used: number
  dpo_beta: number
  epochs: number
  learning_rate: number
  gradient_accumulation_steps: number
  trained_at: string
}

export interface EvaluationResponse {
  sft_baseline: ToolAccuracy | null
  dpo_model: ToolAccuracy | null
  training_info: TrainingInfo | null
}

export interface HealthResponse {
  status: string
  model_loaded: boolean
}

export interface RewardModelStats {
  trained: boolean
  pearson_correlation: number | null
  discrimination_gap: number | null
  correct_tool_avg_score: number | null
  wrong_tool_avg_score: number | null
  samples_used: number
  real_samples: number
  synthetic_samples: number
  trained_at: string | null
  note: string | null
}

// ── UI-level types ─────────────────────────────────────────────────────────

export interface Message {
  id: string
  type: 'user' | 'assistant' | 'error'
  content: string
  queryResult?: QueryResponse
  feedbackState: FeedbackState
}
