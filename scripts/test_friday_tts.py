from __future__ import annotations

import sys
import wave
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assistant.friday import FridayAssistant
from src.core.scene_state import SceneObject, SceneState
from src.voice.tts.piper_tts import PiperTTS


TEST_TEXT = "Hello. I am FRIDAY. I can see a person on your right."
OUTPUT_DIRECTORY = ROOT / "data/test_audio"
PRIMARY_OUTPUT = OUTPUT_DIRECTORY / "friday_tts_test.wav"
RESPONSE_OUTPUT = OUTPUT_DIRECTORY / "friday_tts_response_test.wav"


def available_path(preferred_path: Path) -> Path:
    if not preferred_path.exists():
        return preferred_path
    for number in range(1, 10_000):
        candidate = preferred_path.with_stem(f"{preferred_path.stem}_{number}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"No available numbered filename for {preferred_path}")


def wav_details(path: Path) -> tuple[float, int]:
    with wave.open(str(path), "rb") as wav_file:
        duration = wav_file.getnframes() / wav_file.getframerate()
    return duration, path.stat().st_size


def main() -> None:
    scene = SceneState(objects=[SceneObject(label="person", position_category="right")])
    friday = FridayAssistant(ROOT / "models/detection/yolo26n.pt")
    friday_response = friday.respond("Where is the person?", scene)
    tts = PiperTTS(ROOT / "models/tts/en_US-amy-medium/en_US-amy-medium.onnx")

    try:
        primary_output = tts.synthesize(TEST_TEXT, available_path(PRIMARY_OUTPUT))
        primary_duration, primary_size = wav_details(primary_output)
        primary_latency_ms = tts.last_synthesis_latency_ms

        response_output = tts.synthesize(friday_response, available_path(RESPONSE_OUTPUT))
        response_duration, response_size = wav_details(response_output)
        response_latency_ms = tts.last_synthesis_latency_ms
    except (FileNotFoundError, FileExistsError, OSError, RuntimeError, ValueError) as error:
        print("PIPER = FAIL")
        print(f"PIPER_ERROR: {type(error).__name__}: {error}")
        raise SystemExit(1)

    print("PIPER = PASS")
    print(f"VOICE PATH: {tts.voice_path}")
    print(f"MODEL LOAD TIME MS: {tts.load_time_ms:.2f}")
    print(f"TEST TEXT: {TEST_TEXT}")
    print(f"TEST AUDIO DURATION SECONDS: {primary_duration:.2f}")
    print(f"TEST WAV SIZE BYTES: {primary_size}")
    print(f"TEST SYNTHESIS LATENCY MS: {primary_latency_ms:.2f}")
    print(f"TEST OUTPUT PATH: {primary_output}")
    print(f"FRIDAY TEXT RESPONSE: {friday_response}")
    print(f"FRIDAY AUDIO DURATION SECONDS: {response_duration:.2f}")
    print(f"FRIDAY WAV SIZE BYTES: {response_size}")
    print(f"FRIDAY SYNTHESIS LATENCY MS: {response_latency_ms:.2f}")
    print(f"FRIDAY OUTPUT PATH: {response_output}")


if __name__ == "__main__":
    main()
