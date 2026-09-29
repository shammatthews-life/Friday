from __future__ import annotations

import queue
import sys
import tempfile
import threading
import time
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assistant.friday import FridayAssistant
from src.core.scene_state import SceneObject, SceneState
from src.voice.tts.piper_tts import PiperTTS


PROCESS_INTERVAL_SECONDS = 0.5
WINDOW_TITLE = "FRIDAY live smoke test (press q to quit)"
LIVE_AUDIO_DIRECTORY = ROOT / "data/test_audio/live"


def input_worker(questions: queue.Queue[str]) -> None:
    while True:
        try:
            question = input("> ").strip()
        except EOFError:
            return
        if question:
            questions.put(question)


def scene_from_live_result(assistant: FridayAssistant, result) -> SceneState:
    """Build the existing SceneState shape from one YOLO result on a camera frame."""
    timestamp = time.time()
    height, width = result.orig_shape
    scene = SceneState(timestamp=timestamp)
    for box, confidence, class_id in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
        left, top, right, bottom = [float(value) for value in box]
        center_x = (left + right) / 2
        center_y = (top + bottom) / 2
        normalized_horizontal = center_x / width if width else 0.5
        position_category = (
            "left" if normalized_horizontal < 1 / 3 else
            "right" if normalized_horizontal > 2 / 3 else "center"
        )
        obj = SceneObject(
            label=str(result.names[int(class_id)]),
            confidence=float(confidence),
            center_x=center_x,
            center_y=center_y,
            normalized_horizontal=normalized_horizontal,
            position_category=position_category,
            timestamp=timestamp,
        )
        scene.add_object(obj)
        assistant._object_boxes[id(obj)] = (left, top, right, bottom)
    return scene


def is_valid_frame(ok: bool, frame) -> bool:
    return bool(ok and frame is not None and frame.ndim == 3 and frame.shape[0] > 0 and frame.shape[1] > 0)


def print_scene(scene: SceneState, latency_ms: float) -> None:
    if scene.objects:
        objects = ", ".join(
            f"{obj.label} ({obj.confidence:.2f}, {obj.position_category})"
            for obj in scene.objects
        )
    else:
        objects = "none"
    fps = 1000 / latency_ms if latency_ms else 0.0
    print(f"DETECTIONS: {objects} | YOLO latency: {latency_ms:.1f} ms | FPS: {fps:.2f}")


def answer_question(
    assistant: FridayAssistant, tts: PiperTTS, question: str, scene: SceneState, latest_frame
) -> None:
    depth_label = assistant._depth_query_label(question)
    if depth_label is None:
        response = assistant.respond(question, scene)
    elif latest_frame is None:
        response = f"I can see the {depth_label}, but I cannot estimate its relative depth."
    else:
        # The existing estimator takes an image path. The temporary frame exists
        # only for this on-demand query and is removed immediately afterwards.
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temporary_file:
                temporary_path = Path(temporary_file.name)
            if not cv2.imwrite(str(temporary_path), latest_frame):
                raise RuntimeError("Could not save the current camera frame for depth estimation.")
            response = assistant.respond(question, scene, temporary_path)
        except (OSError, RuntimeError) as error:
            print(f"DEPTH ERROR: {error}")
            response = f"I can see the {depth_label}, but I cannot estimate its relative depth."
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
    print(f"FRIDAY TEXT: {response}")
    audio_path = LIVE_AUDIO_DIRECTORY / f"friday_response_{time.time_ns()}.wav"
    try:
        tts.synthesize(response, audio_path)
        print(f"TTS LATENCY MS: {tts.last_synthesis_latency_ms:.2f}")
        print(f"AUDIO OUTPUT: {audio_path}")
    except (FileNotFoundError, FileExistsError, OSError, RuntimeError, ValueError) as error:
        print(f"TTS ERROR: {type(error).__name__}: {error}")


def main() -> None:
    assistant = FridayAssistant(ROOT / "models/detection/yolo26n.pt")
    tts = PiperTTS(ROOT / "models/tts/en_US-amy-medium/en_US-amy-medium.onnx")
    try:
        tts.load()
        print(f"PIPER TTS: loaded once in {tts.load_time_ms:.2f} ms")
    except (FileNotFoundError, OSError, RuntimeError) as error:
        print(f"TEST: FAIL - could not load local Piper voice: {error}")
        raise SystemExit(1)
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera.release()
        print("TEST: FAIL - could not open webcam index 0 with CAP_DSHOW.")
        raise SystemExit(1)

    warmup_deadline = time.perf_counter() + 1.0
    warmup_frame = None
    try:
        while time.perf_counter() < warmup_deadline:
            ok, frame = camera.read()
            if is_valid_frame(ok, frame):
                warmup_frame = frame
    except cv2.error as error:
        camera.release()
        print(f"TEST: FAIL - CAP_DSHOW warm-up read failed: {error}")
        raise SystemExit(1)
    if warmup_frame is None:
        camera.release()
        print("TEST: FAIL - CAP_DSHOW returned no valid frame during warm-up.")
        raise SystemExit(1)
    warmup_height, warmup_width = warmup_frame.shape[:2]
    print("CAMERA BACKEND: CAP_DSHOW")
    print(f"CAMERA RESOLUTION: {warmup_width}x{warmup_height}")
    print("CAMERA WARM-UP: complete")

    questions: queue.Queue[str] = queue.Queue()
    threading.Thread(target=input_worker, args=(questions,), daemon=True).start()
    latest_scene = SceneState()
    latest_frame = None
    last_processed = 0.0
    model_initialized = False

    print("Camera running. Type a question below; press q in the camera window or Ctrl+C to quit.")
    try:
        while True:
            ok, frame = camera.read()
            if not is_valid_frame(ok, frame):
                print("TEST: FAIL - CAP_DSHOW returned an invalid camera frame.")
                break
            now = time.perf_counter()
            if now - last_processed >= PROCESS_INTERVAL_SECONDS:
                if not model_initialized:
                    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temporary_file:
                        detection_path = Path(temporary_file.name)
                    try:
                        if not cv2.imwrite(str(detection_path), frame):
                            raise RuntimeError("Could not save the current camera frame for detection.")
                        start = time.perf_counter()
                        latest_scene = assistant.detect(detection_path)
                        latency_ms = (time.perf_counter() - start) * 1000
                    finally:
                        detection_path.unlink(missing_ok=True)
                    model_initialized = True
                else:
                    start = time.perf_counter()
                    result = assistant.model.predict(source=frame, device="cpu", verbose=False, imgsz=640)[0]
                    latency_ms = (time.perf_counter() - start) * 1000
                    latest_scene = scene_from_live_result(assistant, result)
                latest_frame = frame.copy()
                last_processed = now
                print_scene(latest_scene, latency_ms)

            while not questions.empty():
                question = questions.get_nowait()
                if question.lower() == "q":
                    return
                answer_question(assistant, tts, question, latest_scene, latest_frame)

            cv2.imshow(WINDOW_TITLE, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return
    except KeyboardInterrupt:
        print("\nStopped by Ctrl+C.")
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
