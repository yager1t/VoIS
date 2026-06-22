# Voice-to-Cursor AI Dictation System --- Master Prompt

You are a senior software architect and AI engineer.

Your task is to design and implement a production-grade local-first
application: a "Hotkey Voice-to-Cursor AI Dictation System".

The system allows a user to press a global hotkey, speak, and have their
speech converted into high-quality text that is automatically inserted
at the current cursor position in any application.

------------------------------------------------------------------------

# 🎯 CORE GOAL

Build a system that:

1.  Activates on a global hotkey (e.g., F9 or configurable)
2.  Records microphone audio while the key is pressed (push-to-talk or
    toggle mode)
3.  Performs real-time or near-real-time speech-to-text transcription
4.  Optionally improves output using an LLM (local or cloud)
5.  Injects final text into the active cursor position in the OS
6.  Learns user-specific vocabulary over time (adaptive dictionary)
7.  Works with minimal latency (target: \<500--800ms perceived delay)

------------------------------------------------------------------------

# ⚙️ SYSTEM REQUIREMENTS

## Audio Capture Layer

-   Real-time microphone capture
-   Push-to-talk / toggle mode
-   Low latency streaming buffers

## Hotkey Manager

-   Global system hotkey
-   Configurable bindings

## Voice Activity Detection

-   Speech vs silence detection
-   Segmenting audio efficiently

## ASR Layer

-   Voxtral / Whisper / fallback models
-   Streaming partial + final transcripts

## Post-processing Layer

-   Grammar correction
-   Punctuation restoration
-   Formatting by context (chat/email/code)

## Dictionary System (CRITICAL)

-   Static domain vocabulary
-   Learned user vocabulary
-   Context-aware mode dictionaries
-   ASR biasing + correction layer

## Keyboard Injection

-   OS-level text injection
-   Windows/macOS/Linux support

------------------------------------------------------------------------

# 🧠 LEARNING SYSTEM

-   Store corrections
-   Extract repeated terms
-   Auto-update vocabulary
-   Context-based adaptation

------------------------------------------------------------------------

# 🚀 OUTPUT REQUIREMENTS

-   Architecture diagram
-   Tech stack proposal
-   Module breakdown
-   Data flow design
-   MVP plan
-   Production roadmap

------------------------------------------------------------------------

# 🎯 FINAL GOAL

A system where: Press hotkey → speak → instantly get clean, formatted
text at cursor.
