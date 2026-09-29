from __future__ import annotations

import sys
import time
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "data/test_audio/microphone_test.wav"
DURATION_SECONDS = 5.0
PREFERRED_SAMPLE_RATE = 16_000
CHANNELS = 1
NON_SILENT_RMS_THRESHOLD = 0.001


def select_capture_settings() -> tuple[dict, float]:
    device = sd.query_devices(kind="input")
    try:
        sd.check_input_settings(
            device=device["name"], channels=CHANNELS, samplerate=PREFERRED_SAMPLE_RATE, dtype="int16"
        )
        return device, float(PREFERRED_SAMPLE_RATE)
    except sd.PortAudioError:
        sample_rate = float(device["default_samplerate"])
        sd.check_input_settings(
            device=device["name"], channels=CHANNELS, samplerate=sample_rate, dtype="int16"
        )
        return device, sample_rate


def main() -> None:
    if OUTPUT_PATH.exists():
        print("MICROPHONE = FAIL")
        print("AUDIO_CAPTURE = FAIL")
        print(f"OUTPUT_EXISTS: refusing to overwrite {OUTPUT_PATH}")
        raise SystemExit(1)

    try:
        device, sample_rate = select_capture_settings()
        sample_count = int(DURATION_SECONDS * sample_rate)
        print(f"MICROPHONE DEVICE: {device['name']}", flush=True)
        print(f"SAMPLE RATE: {sample_rate:.0f} Hz", flush=True)
        print(f"CHANNELS: {CHANNELS}", flush=True)
        print(f"Recording for {DURATION_SECONDS:.1f} seconds. Speak now.", flush=True)
        start = time.perf_counter()
        recording = sd.rec(
            frames=sample_count,
            samplerate=sample_rate,
            channels=CHANNELS,
            dtype="int16",
            device=device["name"],
        )
        sd.wait()
        elapsed_seconds = time.perf_counter() - start
    except (sd.PortAudioError, ValueError) as error:
        print("MICROPHONE = FAIL")
        print("AUDIO_CAPTURE = FAIL")
        print(f"MICROPHONE_ERROR: {type(error).__name__}: {error}")
        raise SystemExit(1)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        with wave.open(str(OUTPUT_PATH), "wb") as output_file:
            output_file.setnchannels(CHANNELS)
            output_file.setsampwidth(2)
            output_file.setframerate(int(sample_rate))
            output_file.writeframes(recording.tobytes())
    except OSError as error:
        print("MICROPHONE = PASS")
        print("AUDIO_CAPTURE = FAIL")
        print(f"WAV_WRITE_ERROR: {error}")
        raise SystemExit(1)

    normalized = recording.astype(np.float32) / np.iinfo(np.int16).max
    rms_level = float(np.sqrt(np.mean(np.square(normalized))))
    non_silent = rms_level >= NON_SILENT_RMS_THRESHOLD
    audio_duration_seconds = recording.shape[0] / sample_rate
    print("MICROPHONE = PASS")
    print("AUDIO_CAPTURE = PASS")
    print(f"RECORDING DURATION SECONDS: {audio_duration_seconds:.2f}")
    print(f"CAPTURE WALL TIME SECONDS: {elapsed_seconds:.2f}")
    print(f"SAMPLE COUNT: {recording.shape[0]}")
    print(f"WAV SIZE BYTES: {OUTPUT_PATH.stat().st_size}")
    print(f"RMS AUDIO LEVEL: {rms_level:.6f}")
    print(f"NON-SILENT AUDIO: {non_silent} (RMS threshold: {NON_SILENT_RMS_THRESHOLD})")
    print(f"WAV OUTPUT: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
