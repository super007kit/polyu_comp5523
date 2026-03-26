"""Offline text-to-speech via pyttsx3 (works on macOS without network).

All synthesis runs on a background thread so the OpenCV loop never blocks on runAndWait().
"""

from __future__ import annotations

import queue
import threading

import pyttsx3

from intent_semantic import INTENT_ASK_BOTTLE_POSITION, INTENT_GREETING

_tts_queue: queue.Queue[str | None] = queue.Queue()
_tts_worker: threading.Thread | None = None
_tts_worker_lock = threading.Lock()


def _tts_worker_loop() -> None:
    eng = pyttsx3.init()
    eng.setProperty("rate", 170)
    eng.setProperty("volume", 1.0)
    while True:
        item = _tts_queue.get()
        if item is None:
            break
        print("SAY:", item, flush=True)
        try:
            eng.stop()
        except Exception:
            pass
        eng.say(item)
        eng.runAndWait()


def _ensure_tts_worker() -> None:
    global _tts_worker
    with _tts_worker_lock:
        if _tts_worker is not None and _tts_worker.is_alive():
            return
        _tts_worker = threading.Thread(
            target=_tts_worker_loop,
            name="tts-worker",
            daemon=True,
        )
        _tts_worker.start()


def speak(text: str) -> None:
    """Queue text to be spoken (returns immediately; does not block the caller)."""
    _ensure_tts_worker()
    _tts_queue.put(text)


def speak_for_intent(
    transcript: str | None,
    intent: str,
    bottle_answer: str | None,
) -> None:
    """
    Map recognition result + intent to a short spoken reply (queued, non-blocking).
    bottle_answer should be from bottle_position_answer() when intent is ASK_BOTTLE_POSITION.
    """
    if not transcript or not transcript.strip():
        speak("Sorry, I did not understand.")
        return
    if intent == INTENT_ASK_BOTTLE_POSITION and bottle_answer:
        speak(bottle_answer)
    elif intent == INTENT_GREETING:
        speak("Hello, how can I help you?")
    else:
        speak("Sorry, I did not understand.")


def shutdown_tts() -> None:
    """Signal the TTS worker to exit (optional; process exit kills daemon thread anyway)."""
    _tts_queue.put(None)
