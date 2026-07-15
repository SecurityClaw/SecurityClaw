import { lazy, Suspense, useEffect, useMemo, useRef, useState } from 'react'
import { useLocation, useNavigate, useParams } from 'react-router-dom'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { BarChart3, Brain, History, ListTree, PanelRightClose, PanelRightOpen, Plus, ScanSearch, Send, Trash2, Wrench } from 'lucide-react'
import { api, streamChat } from '../lib/api.js'

const EndpointInsights = lazy(() => import('../components/EndpointInsights.jsx'))
const EvidenceGraph = lazy(() => import('../components/EvidenceGraph.jsx'))

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
    tool: ['observing', 'checking', 'thinking'],
    processing: ['working', 'checking', 'thinking'],
  }

  return byKind[step?.kind] || ['thinking', 'working', 'checking']
}

const THOUGHT_TOKEN_PHASES = new Set(['skills_check', 'think', 'reflect'])
const FINAL_TOKEN_PHASES = new Set(['direct_answer', 'answer', 'response_final'])

function AgentTimeline({ items = [] }) {
  if (!items.length) return null
  return (
    <div className="mb-4 space-y-3 border-l border-cyan/30 pl-4">
      {items.map((item, index) => (
        <details key={`${item.kind}-${item.step || 0}-${index}`} className="rounded-xl border border-border/70 bg-panel2 p-3" open={item.kind === 'tool'}>
          <summary className="flex cursor-pointer list-none items-center gap-2">
            {item.kind === 'thinking' ? <Brain className="h-4 w-4 text-cyan" /> : <Wrench className="h-4 w-4 text-neon" />}
            <span className="font-mono text-[11px] uppercase tracking-[0.16em] text-cyan">{item.label}</span>
            {item.action ? <span className="badge badge-green">{item.action}</span> : null}
          </summary>
          <div className="mt-2 text-sm text-text">{item.detail}</div>
          {item.reasoning ? <div className="mt-2 text-sm text-dim">Reasoning: {item.reasoning}</div> : null}
          {item.skills?.length ? <div className="mt-2 flex flex-wrap gap-2">{item.skills.map((skill) => <span key={skill} className="badge badge-green">{skill}</span>)}</div> : null}
          {item.debug ? <pre className="mt-3 max-h-80 overflow-auto whitespace-pre-wrap rounded-lg bg-black/20 p-3 text-xs text-dim">{JSON.stringify(item.debug, null, 2)}</pre> : null}
        </details>
      ))}
    </div>
  )
}

export default function ChatPage() {
  const { conversationId } = useParams()
  const navigate = useNavigate()
  const location = useLocation()
  const [conversations, setConversations] = useState([])
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [steps, setSteps] = useState([])
  const [busy, setBusy] = useState(false)
  const [activityPhraseIndex, setActivityPhraseIndex] = useState(0)
  const [reasoningExpanded, setReasoningExpanded] = useState(false)
  const [agentView, setAgentView] = useState('timeline')
  const [agentDrawerOpen, setAgentDrawerOpen] = useState(false)
  const [conversationsOpen, setConversationsOpen] = useState(false)
  const activeId = conversationId || null
  
  const messagesEndRef = useRef(null)
  const messagesContainerRef = useRef(null)
  const shouldAutoScrollRef = useRef(true)
  const streamingConversationIdRef = useRef(null)
  const streamingMessageIdRef = useRef(null)
  const isNewConversationRef = useRef(false)
  const isStreamingRef = useRef(false)
  const isSendingRef = useRef(false)
  const activeConversationRef = useRef(activeId)
  const conversationLoadSequenceRef = useRef(0)

  const scrollToMessagesBottom = (behavior = 'auto') => {
    messagesEndRef.current?.scrollIntoView({ behavior, block: 'end' })
  }

  const handleMessagesScroll = () => {
    const container = messagesContainerRef.current
    if (!container) return
    const distanceFromBottom = container.scrollHeight - container.scrollTop - container.clientHeight
    shouldAutoScrollRef.current = distanceFromBottom < 96
  }

  useEffect(() => {
    if (!shouldAutoScrollRef.current) return undefined
    const frame = window.requestAnimationFrame(() => scrollToMessagesBottom())
    return () => window.cancelAnimationFrame(frame)
  }, [messages])

  const loadConversations = async () => {
    const res = await api.get('/api/conversations')
    setConversations(res.data.items || [])
  }

  const loadConversation = async (id) => {
    const sequence = ++conversationLoadSequenceRef.current
    if (!id) {
      setMessages([])
      return
    }
    const res = await api.get(`/api/conversations/${id}`)
    const loadedMessages = (res.data.messages || []).map((msg, idx) => ({
      ...msg,
      id: msg.id || `${msg.timestamp}-${idx}`,
    }))
    if (sequence === conversationLoadSequenceRef.current && activeConversationRef.current === id) {
      setMessages(loadedMessages)
    }
  }

  useEffect(() => { loadConversations() }, [])
  useEffect(() => {
    activeConversationRef.current = activeId
    conversationLoadSequenceRef.current += 1
    if (!activeId) {
      setMessages([])
      setSteps([])
      return
    }
    // Skip reloading if this conversation is currently receiving a stream
    if (isStreamingRef.current && streamingConversationIdRef.current === activeId) {
      return
    }
    shouldAutoScrollRef.current = true
    setMessages([])
    setSteps([])
    loadConversation(activeId)
  }, [activeId])

  useEffect(() => {
    if (!activeId) return undefined

    const timer = window.setInterval(() => {
      if (!isStreamingRef.current) {
        loadConversation(activeId)
        loadConversations()
      }
    }, 2000)

    return () => window.clearInterval(timer)
  }, [activeId])

  const newChat = () => {
    shouldAutoScrollRef.current = true
    setMessages([])
    setSteps([])
    navigate('/agent')
  }

  const removeConversation = async (id) => {
    await api.delete(`/api/conversations/${id}`)
    await loadConversations()
    if (activeId === id) navigate('/agent')
  }

  const send = async (messageOverride = null) => {
    const requestedMessage = typeof messageOverride === 'string' ? messageOverride.trim() : input.trim()
    if (!requestedMessage || isSendingRef.current) return
    if (busy) {
      const guidance = requestedMessage
      const guidanceConversationId = activeId || streamingConversationIdRef.current
      if (!guidanceConversationId) return
      isSendingRef.current = true
      try {
        await api.post('/api/chat/guidance', {
          conversation_id: guidanceConversationId,
          message: guidance,
        })
        setInput('')
        setSteps((prev) => [...prev, {
          kind: 'guidance',
          label: 'Operator guidance queued',
          detail: guidance,
        }])
      } finally {
        isSendingRef.current = false
      }
      return
    }
    isSendingRef.current = true
    const outgoing = requestedMessage
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
    shouldAutoScrollRef.current = true
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
            navigate(`/agent/${payload.conversation_id}`, { replace: true })
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
                  agent_timeline: payload.agent_timeline || [],
                  trace: payload.trace || [],
                  skill_results: payload.skill_results || {},
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
          if (event === 'error') {
            const assistantMessageId = streamingMessageIdRef.current
            if (assistantMessageId) {
              setMessages((prev) => prev.map((message) => (
                message.id === assistantMessageId
                  ? {
                      ...message,
                      content: payload.message || 'The request could not be completed.',
                      is_streaming: false,
                      error: true,
                    }
                  : message
              )))
            }
            await loadConversations()
          }
        },
      })
    } finally {
      isSendingRef.current = false
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
  const conversationEvidenceResults = useMemo(() => {
    const merged = {}
    messages.forEach((message) => {
      if (message.role === 'assistant' && message.skill_results) Object.assign(merged, message.skill_results)
    })
    return merged
  }, [messages])
  const liveEvidenceResults = useMemo(() => {
    const merged = { ...conversationEvidenceResults }
    steps.forEach((step) => {
      if (step.kind === 'tool' && step.debug && typeof step.debug === 'object') Object.assign(merged, step.debug)
    })
    return merged
  }, [conversationEvidenceResults, steps])
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

  const runEndpointScan = () => {
    if (busy) return
    send('Perform a comprehensive defensive security assessment of this endpoint. Inspect host inventory, installed software and versions, vulnerability and CVE exposure, defensive posture, services, running processes, active connections, ARP/NDP neighbor integrity, routes, gateways, network interfaces, persistence mechanisms, and file integrity. Look for evidence of ARP spoofing or local network manipulation without treating an uncorroborated change as proof. Correlate all evidence, enrich suspicious network entities with the available threat-intelligence capabilities, continue with additional tools when observations create new questions, and provide a detailed risk analysis with coverage gaps and prioritized recommendations. Do not perform containment actions without my explicit authorization.')
  }

  useEffect(() => {
    const initialPrompt = location.state?.initialPrompt
    if (!initialPrompt) return
    setInput(initialPrompt)
    navigate('/agent', { replace: true, state: null })
  }, [location.state, navigate])

  return (
    <div className="relative flex h-full min-h-0 overflow-hidden">
      {conversationsOpen ? <div className="panel absolute inset-y-0 left-0 z-30 flex w-80 flex-col overflow-hidden shadow-2xl">
        <div className="border-b border-border p-4">
          <button className="btn btn-primary w-full" onClick={newChat}>
            <Plus className="h-4 w-4" /> New Chat
          </button>
          <button className="btn mt-2 w-full" onClick={runEndpointScan} disabled={busy}>
            <ScanSearch className="h-4 w-4" /> Scan endpoint
          </button>
        </div>
        <div className="min-h-0 flex-1 overflow-auto p-3 space-y-2">
          {orderedConversations.map((conv) => (
            <div key={conv.id} className={`rounded-xl border p-3 ${activeId === conv.id ? 'border-cyan bg-cyan/10' : 'border-border bg-panel2'}`}>
              <button className="w-full text-left" onClick={() => { setConversationsOpen(false); navigate(`/agent/${conv.id}`) }}>
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
      </div> : null}

      <div className="relative flex min-w-0 flex-1 flex-col overflow-hidden bg-[#070d18]">
        <div className="flex min-h-0 flex-1 flex-col overflow-hidden">
          <Suspense fallback={<div className="p-6 text-sm text-dim">Loading evidence graph…</div>}><EvidenceGraph skillResults={liveEvidenceResults} storageKey={activeId || 'new'} /></Suspense>
        </div>

        <div className="absolute bottom-4 right-4 z-10 flex gap-2">
          <button className={`btn shadow-xl ${conversationsOpen ? 'btn-primary' : ''}`} onClick={() => setConversationsOpen((value) => !value)}><History className="h-4 w-4" /> History</button>
          <button className={`btn shadow-xl ${agentDrawerOpen ? 'btn-primary' : ''}`} onClick={() => setAgentDrawerOpen((value) => !value)}>{agentDrawerOpen ? <PanelRightClose className="h-4 w-4" /> : <PanelRightOpen className="h-4 w-4" />} Chat</button>
        </div>

        {agentDrawerOpen ? <div className="absolute inset-y-0 right-0 z-20 flex w-[min(720px,96%)] min-w-0 flex-col border-l border-border bg-panel shadow-2xl">
          <div className="flex items-center justify-between border-b border-border px-5 py-3">
            <div>
              <div className="font-mono text-xs uppercase tracking-[0.18em] text-cyan">SecurityClaw Agent Chat</div>
              <div className="mt-1 text-xs text-dim">ReAct investigation console · operator-guided · approval-gated actions</div>
            </div>
            <div className="flex gap-2">
              <button className={`btn ${agentView === 'timeline' ? 'btn-primary' : ''}`} onClick={() => setAgentView('timeline')}><ListTree className="h-4 w-4" /> Timeline</button>
              <button className={`btn ${agentView === 'insights' ? 'btn-primary' : ''}`} onClick={() => setAgentView('insights')}><BarChart3 className="h-4 w-4" /> Insights</button>
              <button className="btn" onClick={() => setAgentDrawerOpen(false)} aria-label="Close Agent"><PanelRightClose className="h-4 w-4" /></button>
            </div>
          </div>
          {agentView === 'insights' ? <Suspense fallback={<div className="p-6 text-sm text-dim">Loading insights…</div>}><EndpointInsights messages={messages} liveSteps={steps} /></Suspense> : null}
          <div
            ref={messagesContainerRef}
            className={`min-h-0 flex-1 space-y-4 overflow-auto p-5 ${agentView !== 'timeline' ? 'hidden' : ''}`}
            onScroll={handleMessagesScroll}
          >
            {messages.length === 0 ? <div className="font-mono text-dim">Start a new investigation.</div> : null}
            {messages.map((message) => (
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
                  <div className={`markdown text-base leading-7 text-text ${message.thought_content ? 'mt-3' : ''}`}>
                    <AgentTimeline items={message.agent_timeline || []} />
                    {message.content ? (
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown>
                    ) : message.is_streaming ? (
                      <span className="font-mono text-xs text-dim">Waiting for response…</span>
                    ) : null}
                    {message.is_streaming && message.content ? (
                      <span className="ml-1 inline-block h-4 w-1.5 animate-pulse bg-cyan align-middle" aria-label="Streaming response" />
                    ) : null}
                  </div>
                ) : (
                  <div className="markdown text-base leading-7 text-text">
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
                            {step.skills?.length ? (
                              <div className="mt-2 flex flex-wrap gap-2">
                                {step.skills.map((skill) => <span key={skill} className="badge badge-green">{skill}</span>)}
                              </div>
                            ) : null}
                            {step.debug ? (
                              <details className="mt-3">
                                <summary className="cursor-pointer font-mono text-[11px] uppercase tracking-[0.14em] text-dim">Debug output</summary>
                                <pre className="mt-2 max-h-72 overflow-auto whitespace-pre-wrap rounded-lg bg-black/20 p-3 text-xs text-dim">{JSON.stringify(step.debug, null, 2)}</pre>
                              </details>
                            ) : null}
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
          <div className="border-t border-border p-3">
            <div className="flex items-end gap-2">
              <textarea
                className="textarea min-h-16 flex-1 resize-y"
                placeholder="Ask SecurityClaw to investigate, query, compare, or triage... Press Enter to send, Shift+Enter for new line"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
              />
              <button className="btn btn-primary shrink-0" onClick={send} disabled={!input.trim()}>
                <Send className="h-4 w-4" /> {busy ? 'GUIDE' : 'SEND'}
              </button>
            </div>
          </div>
        </div> : null}
      </div>
    </div>
  )
}
