"""StepFun bidirectional voice calls.

This package is a separate socket from text chat. It does not change the
chat orchestrator, DashScope ASR, or DashScope TTS.

Architecture (v2, 2026-11):
  - Server VAD enabled: StepFun handles turn detection, barge-in, and
    response triggering natively for lowest latency.
  - recall_memory tool: the model calls this when it needs specific facts,
    rather than blocking every turn with memory retrieval.
  - End-of-call extraction: transcripts are persisted as messages and
    extracted via the standard text-chat watermark pipeline at call end.
"""
