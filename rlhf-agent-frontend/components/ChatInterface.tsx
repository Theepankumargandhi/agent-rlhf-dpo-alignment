'use client'

import { useCallback, useEffect, useRef, useState } from 'react'
import { Send } from 'lucide-react'
import toast from 'react-hot-toast'
import { api } from '@/lib/api'
import { Button } from '@/components/ui/button'
import { MessageBubble } from '@/components/MessageBubble'
import type { FeedbackState, Message, Rating, ToolName } from '@/types'

const EXAMPLE_QUERIES = [
  'Where is my order #1042?',
  'What is your refund policy?',
  'My screen cracked on arrival',
  'What payment methods do you accept?',
]

export function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const bottomRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  // Scroll to latest message
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  const updateMessage = useCallback(
    (id: string, patch: Partial<Message>) =>
      setMessages((prev) => prev.map((m) => (m.id === id ? { ...m, ...patch } : m))),
    [],
  )

  async function submit(query: string) {
    query = query.trim()
    if (!query || loading) return

    const userId = crypto.randomUUID()
    setMessages((prev) => [
      ...prev,
      { id: userId, type: 'user', content: query, feedbackState: 'idle' },
    ])
    setInput('')
    setLoading(true)

    try {
      const result = await api.query(query)
      setMessages((prev) => [
        ...prev,
        {
          id: crypto.randomUUID(),
          type: 'assistant',
          content: result.response,
          queryResult: result,
          feedbackState: 'idle',
        },
      ])
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          id: crypto.randomUUID(),
          type: 'error',
          content: err instanceof Error ? err.message : 'Something went wrong.',
          feedbackState: 'submitted',
        },
      ])
    } finally {
      setLoading(false)
      setTimeout(() => inputRef.current?.focus(), 50)
    }
  }

  async function handleFeedback(
    message: Message,
    rating: Rating,
    correctTool: ToolName,
  ) {
    if (!message.queryResult) return
    try {
      await api.feedback({
        session_id: message.queryResult.session_id,
        query: message.queryResult.query,
        selected_tool: message.queryResult.selected_tool,
        correct_tool: correctTool,
        rating,
      })
      updateMessage(message.id, { feedbackState: 'submitted' as FeedbackState })
      toast.success('Feedback saved!', { duration: 2000 })
    } catch {
      toast.error('Could not save feedback — please try again.')
    }
  }

  return (
    <div className="flex flex-col h-full">
      {/* ── Message list ─────────────────────────────────────────── */}
      <div className="flex-1 overflow-y-auto px-4 py-6 space-y-5">
        {messages.length === 0 && !loading && (
          <div className="flex flex-col items-center justify-center h-full gap-6 text-center select-none">
            <div className="w-14 h-14 rounded-2xl bg-violet-600/15 border border-violet-500/20 flex items-center justify-center text-2xl">
              🤖
            </div>
            <div>
              <p className="text-sm font-medium text-zinc-300">Customer Support Agent</p>
              <p className="text-xs text-zinc-600 mt-1">
                Powered by RLHF · Your ratings improve the model
              </p>
            </div>
            {/* Example query chips */}
            <div className="flex flex-wrap justify-center gap-2 max-w-sm">
              {EXAMPLE_QUERIES.map((q) => (
                <button
                  key={q}
                  onClick={() => submit(q)}
                  className="text-xs px-3 py-1.5 rounded-full border border-zinc-800
                             text-zinc-500 hover:border-violet-500/50 hover:text-zinc-300
                             hover:bg-violet-500/5 transition-colors"
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((msg) => (
          <MessageBubble key={msg.id} message={msg} onFeedback={handleFeedback} />
        ))}

        {loading && <TypingIndicator />}
        <div ref={bottomRef} />
      </div>

      {/* ── Input bar ────────────────────────────────────────────── */}
      <div className="border-t border-[#2a2a2a] px-4 py-4">
        <form
          onSubmit={(e) => { e.preventDefault(); submit(input) }}
          className="flex gap-3 max-w-3xl mx-auto"
        >
          <input
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask about orders, refunds, or get help…"
            disabled={loading}
            className="flex-1 bg-[#1a1a1a] border border-[#2a2a2a] rounded-xl px-4 py-3 text-sm
                       text-white placeholder-zinc-600
                       focus:outline-none focus:border-violet-500/60
                       disabled:opacity-50 transition-colors"
          />
          <Button type="submit" disabled={!input.trim() || loading} size="md">
            <Send className="h-4 w-4" />
          </Button>
        </form>
      </div>
    </div>
  )
}

function TypingIndicator() {
  return (
    <div className="flex items-start gap-3">
      <div className="w-7 h-7 rounded-lg bg-violet-600/80 flex items-center justify-center text-xs flex-shrink-0 select-none">
        🤖
      </div>
      <div className="bg-[#1a1a1a] border border-[#2a2a2a] rounded-2xl rounded-tl-sm px-4 py-3.5">
        <div className="flex gap-1.5 items-center h-4">
          <div className="w-1.5 h-1.5 rounded-full bg-zinc-500 animate-bounce [animation-delay:0ms]" />
          <div className="w-1.5 h-1.5 rounded-full bg-zinc-500 animate-bounce [animation-delay:150ms]" />
          <div className="w-1.5 h-1.5 rounded-full bg-zinc-500 animate-bounce [animation-delay:300ms]" />
        </div>
      </div>
    </div>
  )
}
