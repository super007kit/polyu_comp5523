"""Speech-to-text (Google) + semantic intent. Use from main on keypress or run as CLI."""

import speech_recognition as sr

from intent_semantic import classify_intent_semantic


def listen_transcribe_and_classify() -> tuple[str | None, str]:
    """
    Listen once, transcribe with Google STT, classify intent.

    Returns (transcript, intent). transcript is None if STT failed; intent is still set.
    """
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening… (speak now)")
        r.adjust_for_ambient_noise(source, duration=0.5)
        audio = r.listen(source)

    try:
        text = r.recognize_google(audio, language="en-US")
        intent = classify_intent_semantic(text)
        print("You said:", text)
        print("Intent:", intent)
        return text, intent
    except sr.UnknownValueError:
        print("Could not understand the audio — try again or speak more clearly.")
        return None, "UNKNOWN"
    except sr.RequestError as e:
        print("Recognition service error (check network):", e)
        return None, "UNKNOWN"


def main() -> None:
    listen_transcribe_and_classify()


if __name__ == "__main__":
    main()
