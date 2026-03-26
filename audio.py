"""Capture device lifecycle: open the microphone while the app runs, release it on shutdown."""

from __future__ import annotations

import sounddevice as sd


class MicrophoneStream:
    """
    Opens the default input device with a callback that drains buffers (microphone stays active).

    Use as a context manager or call start() / stop() explicitly.
    """

    def __init__(
        self,
        samplerate: int = 16_000,
        channels: int = 1,
        blocksize: int = 1024,
        device: int | str | None = None,
    ) -> None:
        self._samplerate = samplerate
        self._channels = channels
        self._blocksize = blocksize
        self._device = device
        self._stream: sd.InputStream | None = None

    @staticmethod
    def _callback(indata, frames, time_info, status) -> None:
        if status:
            print(f"[mic] {status}", flush=True)

    def start(self) -> None:
        if self._stream is not None:
            return
        self._stream = sd.InputStream(
            samplerate=self._samplerate,
            channels=self._channels,
            dtype="float32",
            blocksize=self._blocksize,
            device=self._device,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is None:
            return
        self._stream.stop()
        self._stream.close()
        self._stream = None

    def __enter__(self) -> MicrophoneStream:
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()
