import { useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { Plus, Send, Trash2 } from 'lucide-react'
import { api, streamChat } from '../lib/api.js'
import PageHeader from '../components/PageHeader.jsx'

function upsertConversationItem(items, nextItem) {
  const index = items.findIndex((item) => item.id === nextItem.id)
  if (index === -1) {
    return [nextItem, ...items]
  }

  const copy = [...items]
  copy[index] = { ...copy[index], ...nextItem }
  return copy
}

function getActivityCopy(step) {
  if (!step) {
    return {
      title: 'Thinking',
      detail: 'Working through the request',
    }
  }

  const title = {
    thinking: 'Thinking',
    fetching: 'Searching',
    evaluating: 'Reviewing',
    processing: 'Processing',
  }[step.kind] || 'Processing'

  return {
    title,
    detail: step.detail || 'Working through the request',
  }
}

function getActivityPhrases(step) {
  const byKind = {
    thinking: ['thinking', 'working', 'checking'],
    fetching: ['checking', 'working', 'thinking'],
    evaluating: ['checking', 'reviewing', 'thinking'],
    processing: ['working', 'checking', 'thinking'],
  }

  return byKind[step?.kind] || ['thinking', 'working', 'checking']
}

const THOUGHT_TOKEN_PHASES = new Set(['skills_check', 'think', 'reflect'])
const FINAL_TOKEN_PHASES = new Set(['direct_answer', 'answer', 'response_final'])

export default function ChatPage() {
  const { conversationId } = useParams()
  const navigate = useNavigate()
  const [conversations, setConversations] = useState([])
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [steps, setSteps] = useState([])
  const [busy, setBusy] = useState(false)
  const [activityPhraseIndex, setActivityPhraseIndex] = useState(0)
  const [reasoningExpanded, setReasoningExpanded] = useState(false)
  const activeId = conversationId || null
  
  const messagesEndRef = useRef(null)
  const streamingConversationIdRef = useRef(null)
  const streamingMessageIdRef = useRef(null)
  const isNewConversationRef = useRef(false)
  const isStreamingRef = useRef(false)

  const scrollToMessagesBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => { scrollToMessagesBottom() }, [messages])

  const loadConversations = async () => {
    const res = await api.get('/api/conversations')
    setConversations(res.data.items || [])
  }

  const loadConversation = async (id) => {
    if (!id) {
      setMessages([])
      return
    }
    const res = await api.get(`/api/conversations/${id}`)
    const loadedMessages = (res.data.messages || []).map((msg, idx) => ({
      ...msg,
      id: msg.id || `${msg.timestamp}-${idx}`,
    }))
    setMessages(loadedMessages)
  }

  useEffect(() => { loadConversations() }, [])
  useEffect(() => {
    if (!activeId) {
      setMessages([])
      return
    }
    // Skip reloading if this conversation is currently receiving a stream
    if (isStreamingRef.current && streamingConversationIdRef.current === activeId) {
      return
    }
    loadConversation(activeId)
  }, [activeId])

  const newChat = () => {
    setMessages([])
    setSteps([])
    navigate('/chat')
  }

  const removeConversation = async (id) => {
    await api.delete(`/api/conversations/${id}`)
    await loadConversations()
    if (activeId === id) navigate('/chat')
  }

  const send = async () => {
    if (!input.trim() || busy) return
    const outgoing = input.trim()
    const userTimestamp = new Date().toISOString()
    const assistantMessageId = `${userTimestamp}-assistant`
    const userMessageId = `${userTimestamp}-user`
    const userMessage = {
      id: userMessageId,
      role: 'user',
      content: outgoing,
      timestamp: userTimestamp,
    }
    const assistantMessage = {
      id: assistantMessageId,
      role: 'assistant',
      content: '',
      thought_content: '',
      is_streaming: true,
      timestamp: userTimestamp,
      routing_skills: [],
    }
    streamingMessageIdRef.current = assistantMessageId
    isNewConversationRef.current = !activeId
    setMessages((prev) => [...prev, userMessage, assistantMessage])
    isStreamingRef.current = true
    setBusy(true)
    setSteps([])
    setReasoningExpanded(false)
    setActivityPhraseIndex(0)
    setInput('')

    try {
      await streamChat({
        message: outgoing,
        conversationId: activeId,
        onEvent: async (event, payload) => {
          if (event === 'meta' && payload.conversation_id && !activeId) {
            streamingConversationIdRef.current = payload.conversation_id
            setConversations((prev) => upsertConversationItem(prev, {
              id: payload.conversation_id,
              first_question: outgoing,
              preview: outgoing,
              messages: 1,
              timestamp: userTimestamp,
              last_update: userTimestamp,
              created_at: userTimestamp,
            }))
            navigate(`/chat/${payload.conversation_id}`, { replace: true })
          }
          if (event === 'step') {
            setSteps((prev) => [...prev, payload])
          }
          if (event === 'token') {
            const token = String(payload.token || '')
            const phase = String(payload.phase || '')
            const assistantMessageId = streamingMessageIdRef.current
            if (!token || !assistantMessageId) {
              return
            }

            setMessages((prev) => prev.map((message) => {
              if (message.id !== assistantMessageId) {
                return message
              }

              if (FINAL_TOKEN_PHASES.has(phase)) {
                return {
                  ...message,
                  content: `${message.content || ''}${token}`,
                }
              }

              if (THOUGHT_TOKEN_PHASES.has(phase) || phase) {
                return {
                  ...message,
                  thought_content: `${message.thought_content || ''}${token}`,
                }
              }

              return {
                ...message,
                thought_content: `${message.thought_content || ''}${token}`,
              }
            }))
          }
          if (event === 'response') {
            const responseTimestamp = new Date().toISOString()
            const resolvedConversationId = payload.conversation_id || activeId || streamingConversationIdRef.current
            const assistantMessageId = streamingMessageIdRef.current
            if (assistantMessageId) {
              setMessages((prev) => prev.map((message) => {
                if (message.id !== assistantMessageId) {
                  return message
                }

                return {
                  ...message,
                  content: payload.response || message.content,
                  is_streaming: false,
                  timestamp: responseTimestamp,
                  routing_skills: payload.routing?.skills || [],
                }
              }))
            }
            if (resolvedConversationId) {
              setConversations((prev) => upsertConversationItem(prev, {
                id: resolvedConversationId,
                first_question: prev.find((item) => item.id === resolvedConversationId)?.first_question || outgoing,
                preview: outgoing,
                messages: (prev.find((item) => item.id === resolvedConversationId)?.messages || 0) + (isNewConversationRef.current ? 1 : 2),
                timestamp: responseTimestamp,
                last_update: responseTimestamp,
              }))
            }
            await loadConversations()
          }
        },
      })
    } finally {
      isStreamingRef.current = false
      streamingConversationIdRef.current = null
      streamingMessageIdRef.current = null
      isNewConversationRef.current = false
      setBusy(false)
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      send()
    }
  }

  const orderedConversations = useMemo(() => {
    return [...conversations].sort((a, b) => {
      const timeA = new Date(a.timestamp || a.created_at || 0).getTime()
      const timeB = new Date(b.timestamp || b.created_at || 0).getTime()
      return timeB - timeA
    })
  }, [conversations])

  const currentStep = steps[steps.length - 1] || null
  const activity = getActivityCopy(currentStep)
  const activityPhrases = getActivityPhrases(currentStep)
  const activityPhrase = activityPhrases[activityPhraseIndex % activityPhrases.length]

  useEffect(() => {
    if (!busy) {
      setActivityPhraseIndex(0)
      return undefined
    }

    const timer = window.setInterval(() => {
      setActivityPhraseIndex((prev) => prev + 1)
    }, 1400)

    return () => window.clearInterval(timer)
  }, [busy, currentStep])

  return (
    <div className="flex h-full min-h-0 gap-6">
      <div className="panel flex w-80 flex-col overflow-hidden">
        <div className="border-b border-border p-4">
          <button className="btn btn-primary w-full" onClick={newChat}>
            <Plus className="h-4 w-4" /> New Chat
          </button>
        </div>
        <div className="min-h-0 flex-1 overflow-auto p-3 space-y-2">
          {orderedConversations.map((conv) => (
            <div key={conv.id} className={`rounded-xl border p-3 ${activeId === conv.id ? 'border-cyan bg-cyan/10' : 'border-border bg-panel2'}`}>
              <button className="w-full text-left" onClick={() => navigate(`/chat/${conv.id}`)}>
                <div className="truncate font-mono text-xs uppercase tracking-[0.14em] text-cyan">{conv.id}</div>
                <div className="mt-1 line-clamp-2 text-sm text-text">{conv.first_question || conv.preview || 'Conversation'}</div>
                <div className="mt-2 font-mono text-[11px] text-dim">{conv.messages} entries</div>
              </button>
              <button className="mt-3 inline-flex items-center gap-1 text-xs text-danger" onClick={() => removeConversation(conv.id)}>
                <Trash2 className="h-3 w-3" /> delete
              </button>
            </div>
          ))}
        </div>
      </div>

      <div className="flex min-w-0 flex-1 flex-col gap-6">
        <PageHeader title="Chat" subtitle="Supervisor-driven operator console with step-level progress, not raw logs." />

        <div className="panel flex min-h-0 flex-1 flex-col overflow-hidden">
          <div className="border-b border-border px-5 py-3 font-mono text-xs uppercase tracking-[0.18em] text-cyan">Conversation</div>
          <div className="min-h-0 flex-1 space-y-4 overflow-auto p-5">
            {messages.length === 0 ? <div className="font-mono text-dim">Start a new investigation.</div> : null}
            {messages.filter(m => !m.is_streaming).map((message) => (
              <div key={message.id || message.timestamp} className={`rounded-xl border p-4 ${message.role === 'assistant' ? 'border-cyan/20 bg-cyan/5' : 'border-border bg-panel2'}`}>
                <div className="mb-2 flex items-center justify-between gap-3">
                  <div className="font-mono text-xs uppercase tracking-[0.18em] text-dim">{message.role === 'assistant' ? 'SecurityClaw' : 'Operator'}</div>
                  {message.routing_skills?.length ? <div className="flex flex-wrap gap-2">{message.routing_skills.map((skill) => <span key={skill} className="badge badge-green">{skill}</span>)}</div> : null}
                </div>
                {message.role === 'assistant' && message.thought_content ? (
                  <div className="rounded-xl border border-border/70 bg-panel2 p-3">
                    <div className="font-mono text-[11px] uppercase tracking-[0.16em] text-dim">LLM thought</div>
                    <div className="mt-2 whitespace-pre-wrap text-sm text-dim">{message.thought_content}</div>
                  </div>
                ) : null}
                {message.role === 'assistant' ? (
                  <div className={`markdown text-sm text-text ${message.thought_content ? 'mt-3' : ''}`}>
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown>
                  </div>
                ) : (
                  <div className="markdown text-sm text-text">
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown>
                  </div>
                )}
              </div>
            ))}
            {busy ? (
              <div className="rounded-xl border border-cyan/20 bg-cyan/5 p-4">
                <div className="mb-2 flex items-center gap-3 font-mono text-xs uppercase tracking-[0.18em] text-dim">
                  <span>SecurityClaw</span>
                  <span className="inline-flex items-center gap-2 text-cyan">
                    <span>{activityPhrase}</span>
                    <span className="activity-ellipsis" aria-hidden="true">
                      <span />
                      <span />
                      <span />
                    </span>
                  </span>
                </div>
                <div className="text-sm text-text">{activity.detail}</div>
                {steps.length ? (
                  <div className="mt-4 border-t border-border/70 pt-3">
                    <button
                      className="font-mono text-[11px] uppercase tracking-[0.16em] text-dim transition hover:text-cyan"
                      onClick={() => setReasoningExpanded((prev) => !prev)}
                      type="button"
                    >
                      {reasoningExpanded ? 'Hide Reasoning Steps' : 'Show Reasoning Steps'}
                    </button>
                    {reasoningExpanded ? (
                      <div className="mt-3 space-y-3">
                        {steps.map((step, index) => (
                          <div key={`${step.kind}-${index}`} className="rounded-xl border border-border/70 bg-panel2 px-3 py-3">
                            <div className="font-mono text-[11px] uppercase tracking-[0.16em] text-cyan">{step.label}</div>
                            <div className="mt-1 text-sm text-text">{step.detail}</div>
                          </div>
                        ))}
                      </div>
                    ) : null}
                  </div>
                ) : null}
              </div>
            ) : null}
            <div ref={messagesEndRef} />
          </div>
          <div className="border-t border-border p-4">
            <div className="flex flex-col gap-2">
              <textarea
                className="textarea min-h-24 flex-1"
                placeholder="Ask SecurityClaw to investigate, query, compare, or triage... Press Enter to send, Shift+Enter for new line"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
              />
              <button className="btn btn-primary self-start" onClick={send} disabled={busy || !input.trim()}>
                <Send className="h-4 w-4" /> {busy ? activityPhrase.toUpperCase() : 'SEND'}
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
