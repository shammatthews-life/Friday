from __future__ import annotations

import os
import sys
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models/speech/faster-whisper-tiny"
AUDIO_PATH = ROOT / "data/test_audio/microphone_test.wav"
OUTPUT_PATH = ROOT / "data/evaluation/whisper_test_result.txt"
REQUIRED_MODEL_FILES = ("model.bin", "config.json", "tokenizer.json", "vocabulary.txt")


def audio_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as audio_file:
        return audio_file.getnframes() / audio_file.getframerate()


def fail(message: str) -> None:
    print("WHISPER_LOCAL = FAIL")
    print(message)
    raise SystemExit(1)


def main() -> None:
    missing = [name for name in REQUIRED_MODEL_FILES if not (MODEL_PATH / name).is_file()]
    if missing:
        fail(f"LOCAL_MODEL_MISSING_FILES: {', '.join(missing)}")
    if not AUDIO_PATH.is_file():
        fail(f"AUDIO_FILE_MISSING: {AUDIO_PATH}")

    try:
        duration_seconds = audio_duration_seconds(AUDIO_PATH)
    except (OSError, wave.Error) as error:
        fail(f"AUDIO_DURATION_ERROR: {type(error).__name__}: {error}")

    print(f"MODEL PATH: {MODEL_PATH}")
    print("DEVICE: cpu")
    print("COMPUTE TYPE: int8")
    print(f"AUDIO DURATION SECONDS: {duration_seconds:.2f}")

    # The model argument is an existing directory, never a remote model name.
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        from faster_whisper import WhisperModel

        load_start = time.perf_counter()
        model = WhisperModel(str(MODEL_PATH), device="cpu", compute_type="int8")
        model_load_ms = (time.perf_counter() - load_start) * 1000

        transcription_start = time.perf_counter()
        segments, info = model.transcribe(
            str(AUDIO_PATH), language="en", beam_size=1, vad_filter=False
        )
        segment_list = list(segments)
        transcription_ms = (time.perf_counter() - transcription_start) * 1000
    except Exception as error:
        fail(f"LOCAL_TRANSCRIPTION_ERROR: {type(error).__name__}: {error}")

    transcript = " ".join(segment.text.strip() for segment in segment_list).strip()
    language = getattr(info, "language", None)
    language_probability = getattr(info, "language_probability", None)
    result_lines = [
        f"model_path: {MODEL_PATH}",
        "device: cpu",
        "compute_type: int8",
        f"audio_path: {AUDIO_PATH}",
        f"audio_duration_seconds: {duration_seconds:.2f}",
        f"model_load_time_ms: {model_load_ms:.2f}",
        f"transcription_latency_ms: {transcription_ms:.2f}",
        f"language: {language}",
        f"language_probability: {language_probability}",
        f"segment_count: {len(segment_list)}",
        "transcription:",
        transcript,
    ]
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(result_lines) + "\n", encoding="utf-8")

    print(f"MODEL PATH: {MODEL_PATH}")
    print("DEVICE: cpu")
    print("COMPUTE TYPE: int8")
    print(f"AUDIO DURATION SECONDS: {duration_seconds:.2f}")
    print(f"MODEL LOAD TIME MS: {model_load_ms:.2f}")
    print(f"TRANSCRIPTION LATENCY MS: {transcription_ms:.2f}")
    print(f"DETECTED LANGUAGE: {language}")
    print(f"LANGUAGE PROBABILITY: {language_probability}")
    print(f"SEGMENT COUNT: {len(segment_list)}")
    print(f"COMPLETE TRANSCRIPTION: {transcript}")
    print(f"RESULT OUTPUT: {OUTPUT_PATH}")
    print("WHISPER_LOCAL = PASS")


if __name__ == "__main__":
    main()
