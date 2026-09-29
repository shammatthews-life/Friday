from __future__ import annotations

import time
import wave
from pathlib import Path

from piper.voice import PiperVoice


class PiperTTS:
    """Local-only Piper voice wrapper for FRIDAY responses."""

    def __init__(
        self,
        voice_path: str | Path = "models/tts/en_US-amy-medium/en_US-amy-medium.onnx",
    ) -> None:
        self.voice_path = Path(voice_path)
        self.config_path = self.voice_path.with_suffix(".onnx.json")
        self.voice: PiperVoice | None = None
        self.load_time_ms: float | None = None
        self.last_synthesis_latency_ms: float | None = None

    def load(self) -> None:
        if self.voice is not None:
            return
        missing = [
            str(path) for path in (self.voice_path, self.config_path) if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(f"Local Piper voice files missing: {', '.join(missing)}")
        start = time.perf_counter()
        self.voice = PiperVoice.load(
            str(self.voice_path), config_path=str(self.config_path), use_cuda=False
        )
        self.load_time_ms = (time.perf_counter() - start) * 1000

    def synthesize(self, text: str, output_path: str | Path) -> Path:
        if not text.strip():
            raise ValueError("Text to synthesize must not be empty.")
        destination = Path(output_path)
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite existing WAV: {destination}")
        self.load()
        destination.parent.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        with wave.open(str(destination), "wb") as wav_file:
            self.voice.synthesize_wav(text, wav_file)
        self.last_synthesis_latency_ms = (time.perf_counter() - start) * 1000
        return destination
