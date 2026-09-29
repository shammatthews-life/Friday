from __future__ import annotations

import time
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import cv2


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MODEL_PATH = ROOT / "models/detection/yolo26n.pt"
VOICE_PATH = ROOT / "models/tts/en_US-amy-medium/en_US-amy-medium.onnx"
AUDIO_DIRECTORY = ROOT / "data/test_audio/live_automated"
ANNOTATED_IMAGE = ROOT / "data/test_images/friday_live_automated.jpg"
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
WARMUP_SECONDS = 1.0
CAPTURE_SECONDS = 9.0
INFERENCE_INTERVAL_SECONDS = 0.5
QUERIES = (
    "What is around me?",
    "Where is the person?",
    "How far is the person?",
    "Find person",
)


def valid_frame(ok: bool, frame) -> bool:
    return bool(
        ok
        and frame is not None
        and frame.ndim == 3
        and frame.shape[0] > 0
        and frame.shape[1] > 0
    )


def detections_text(scene) -> str:
    if not scene.objects:
        return "none"
    return ", ".join(
        f"{obj.label} ({obj.confidence:.2f}, {obj.position_category})"
        for obj in scene.objects
    )


def annotate_frame(frame, scene, assistant):
    annotated = frame.copy()
    for obj in scene.objects:
        box = assistant.bounding_box_for(obj)
        if box is None:
            continue
        left, top, right, bottom = (int(value) for value in box)
        label = f"{obj.label} {obj.confidence:.2f}"
        cv2.rectangle(annotated, (left, top), (right, bottom), (40, 210, 80), 2)
        cv2.putText(
            annotated,
            label,
            (left, max(18, top - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (40, 210, 80),
            2,
            cv2.LINE_AA,
        )
    return annotated


def run_queries(assistant, tts, scene, frame_path: Path, latest_yolo_latency_ms: float | None):
    results = []
    for index, question in enumerate(QUERIES, start=1):
        print(f"\nQUERY: {question}")
        response = ""
        tts_latency_ms = None
        audio_path = None
        try:
            response = assistant.respond(question, scene, frame_path)
        except Exception as error:
            response = f"ERROR: {type(error).__name__}: {error}"

        print(f"FRIDAY TEXT: {response}")
        print(f"DETECTIONS USED: {detections_text(scene)}")
        if latest_yolo_latency_ms is None:
            print("YOLO LATENCY: unavailable")
        else:
            print(f"YOLO LATENCY: {latest_yolo_latency_ms:.2f} ms (latest prediction)")

        if response and not response.startswith("ERROR:"):
            filename = f"friday_response_{time.time_ns()}_{index}.wav"
            audio_path = AUDIO_DIRECTORY / filename
            try:
                tts.synthesize(response, audio_path)
                tts_latency_ms = tts.last_synthesis_latency_ms
                print(f"TTS LATENCY: {tts_latency_ms:.2f} ms")
                print(f"AUDIO OUTPUT: {audio_path}")
            except Exception as error:
                audio_path = None
                print(f"TTS LATENCY: unavailable")
                print(f"AUDIO OUTPUT: FAIL - {type(error).__name__}: {error}")
        else:
            print("TTS LATENCY: unavailable")
            print("AUDIO OUTPUT: not generated")

        results.append(
            {
                "question": question,
                "response": response,
                "tts_ok": audio_path is not None and audio_path.is_file(),
                "audio_path": audio_path,
            }
        )
    return results


def main() -> None:
    from src.assistant.friday import FridayAssistant
    from src.voice.tts.piper_tts import PiperTTS

    assistant = FridayAssistant(MODEL_PATH)
    tts = PiperTTS(VOICE_PATH)
    camera = None
    live_camera_ok = False
    inference_ok = False
    annotated_ok = False
    query_results = []
    model_load_times_ms: list[float] = []
    prediction_latencies_ms: list[float] = []
    detected_labels: dict[str, float] = {}
    camera_frame_count = 0
    runtime_start = time.perf_counter()
    failure_reason = None

    try:
        try:
            tts.load()
            print(f"PIPER LOAD: PASS ({tts.load_time_ms:.2f} ms)")
        except Exception as error:
            print(f"PIPER LOAD: FAIL - {type(error).__name__}: {error}")

        camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not camera.isOpened():
            failure_reason = "CAP_DSHOW could not open webcam index 0."
            print(f"CAMERA/INFERENCE FAILURE: {failure_reason}")
        else:
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            warmup_deadline = time.perf_counter() + WARMUP_SECONDS
            warmup_frame_seen = False
            while time.perf_counter() < warmup_deadline:
                ok, frame = camera.read()
                if valid_frame(ok, frame):
                    warmup_frame_seen = True
            live_camera_ok = warmup_frame_seen
            actual_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"CAMERA BACKEND: CAP_DSHOW ({actual_width}x{actual_height})")
            print(f"CAMERA WARM-UP: {'PASS' if warmup_frame_seen else 'FAIL'}")

            if not warmup_frame_seen:
                failure_reason = "No valid camera frame arrived during warm-up."
                print(f"CAMERA/INFERENCE FAILURE: {failure_reason}")
            else:
                import ultralytics

                original_yolo = ultralytics.YOLO

                def timed_yolo(*args, **kwargs):
                    load_start = time.perf_counter()
                    try:
                        model = original_yolo(*args, **kwargs)
                    finally:
                        model_load_times_ms.append(
                            (time.perf_counter() - load_start) * 1000
                        )
                    original_predict = model.predict

                    def timed_predict(*predict_args, **predict_kwargs):
                        predict_start = time.perf_counter()
                        try:
                            return original_predict(*predict_args, **predict_kwargs)
                        finally:
                            prediction_latencies_ms.append(
                                (time.perf_counter() - predict_start) * 1000
                            )

                    model.predict = timed_predict
                    return model

                query_batch_done = False
                next_inference = 0.0
                capture_deadline = time.perf_counter() + CAPTURE_SECONDS
                with tempfile.TemporaryDirectory(prefix="friday_live_auto_") as temporary_dir:
                    inference_frame_path = Path(temporary_dir) / "latest_frame.jpg"
                    while time.perf_counter() < capture_deadline:
                        ok, frame = camera.read()
                        if not valid_frame(ok, frame):
                            continue
                        camera_frame_count += 1
                        now = time.perf_counter()
                        if now < next_inference:
                            continue
                        next_inference = now + INFERENCE_INTERVAL_SECONDS
                        if not cv2.imwrite(str(inference_frame_path), frame):
                            failure_reason = "Could not save a captured frame for YOLO inference."
                            continue

                        try:
                            if assistant.model is None:
                                with patch.object(ultralytics, "YOLO", timed_yolo):
                                    scene = assistant.detect(inference_frame_path)
                            else:
                                scene = assistant.detect(inference_frame_path)
                        except Exception as error:
                            failure_reason = f"YOLO inference failed: {type(error).__name__}: {error}"
                            print(f"INFERENCE ERROR: {failure_reason}")
                            continue

                        inference_ok = True
                        if len(prediction_latencies_ms) > 1:
                            latest_yolo_latency_ms = prediction_latencies_ms[-1]
                        elif prediction_latencies_ms:
                            latest_yolo_latency_ms = prediction_latencies_ms[0]
                        else:
                            latest_yolo_latency_ms = None
                        for obj in scene.objects:
                            detected_labels[obj.label] = max(
                                obj.confidence, detected_labels.get(obj.label, 0.0)
                            )

                        annotated = annotate_frame(frame, scene, assistant)
                        ANNOTATED_IMAGE.parent.mkdir(parents=True, exist_ok=True)
                        annotated_ok = cv2.imwrite(str(ANNOTATED_IMAGE), annotated)
                        print(
                            f"YOLO FRAME: {detections_text(scene)} | "
                            f"YOLO latency: {latest_yolo_latency_ms:.2f} ms"
                            if latest_yolo_latency_ms is not None
                            else f"YOLO FRAME: {detections_text(scene)} | YOLO latency: unavailable"
                        )

                        if not query_batch_done:
                            query_start = time.perf_counter()
                            query_results = run_queries(
                                assistant,
                                tts,
                                scene,
                                inference_frame_path,
                                latest_yolo_latency_ms,
                            )
                            capture_deadline += time.perf_counter() - query_start
                            query_batch_done = True

                    if not inference_ok:
                        failure_reason = failure_reason or "No valid real frame completed YOLO inference."
                        print(f"CAMERA/INFERENCE FAILURE: {failure_reason}")

    except Exception as error:
        failure_reason = f"{type(error).__name__}: {error}"
        print(f"LIVE TEST ERROR: {failure_reason}")
    finally:
        if camera is not None:
            camera.release()
        cv2.destroyAllWindows()
        tts.voice = None

    warm_latencies = prediction_latencies_ms[1:]
    average_warm_latency_ms = (
        sum(warm_latencies) / len(warm_latencies) if warm_latencies else None
    )
    approximate_fps = (
        1000 / average_warm_latency_ms if average_warm_latency_ms else 0.0
    )
    response_ok = len(query_results) == len(QUERIES) and all(
        item["response"] and not item["response"].startswith("ERROR:")
        for item in query_results
    )
    piper_ok = len(query_results) == len(QUERIES) and all(
        item["tts_ok"] for item in query_results
    )
    end_to_end_ok = live_camera_ok and inference_ok and response_ok and piper_ok and annotated_ok
    total_runtime = time.perf_counter() - runtime_start

    print("\nFINAL SUMMARY")
    print(f"LIVE_CAMERA = {'PASS' if live_camera_ok else 'FAIL'}")
    print(f"YOLO_REAL_FRAME = {'PASS' if inference_ok else 'FAIL'}")
    print(f"FRIDAY_RESPONSE = {'PASS' if response_ok else 'FAIL'}")
    print(f"PIPER_TTS = {'PASS' if piper_ok else 'FAIL'}")
    print(f"END_TO_END = {'PASS' if end_to_end_ok else 'FAIL'}")
    print(f"DETECTED LABELS: {detected_labels if detected_labels else 'none'}")
    print(f"MODEL LOAD TIME: {model_load_times_ms[0]:.2f} ms" if model_load_times_ms else "MODEL LOAD TIME: unavailable")
    print(
        f"AVERAGE WARM YOLO LATENCY: {average_warm_latency_ms:.2f} ms"
        if average_warm_latency_ms is not None
        else "AVERAGE WARM YOLO LATENCY: unavailable"
    )
    print(f"APPROXIMATE FPS: {approximate_fps:.2f}")
    print(f"CAMERA FRAMES CAPTURED: {camera_frame_count}")
    print(f"TOTAL RUNTIME: {total_runtime:.2f} s")
    print(f"ANNOTATED IMAGE: {ANNOTATED_IMAGE if annotated_ok else 'not saved'}")
    if query_results:
        for result in query_results:
            print(f"SAVED WAV: {result['audio_path'] if result['audio_path'] else 'not saved'}")
    if failure_reason:
        print(f"FAILURE DETAIL: {failure_reason}")


if __name__ == "__main__":
    main()