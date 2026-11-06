"""Bidirectional voice calls on DashScope's Qwen-Omni realtime model.

A separate socket from text chat. Server VAD finds the end of each turn;
the bridge retrieves memories, filters echo and backchannels, then asks for
the reply. Transcripts are stored as messages and extracted through the
standard text-chat pipeline at call end. Voice messages run on the same
model (speech_output/client.py), so an agent has one voice for both.
"""
