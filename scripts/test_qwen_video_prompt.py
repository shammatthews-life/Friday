from __future__ import annotations

import copy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.llm.qwen_adapter import _compact_video_response_evidence


def test_visual_detail_is_referenced_without_repeating_nested_records() -> None:
    original = {
        "available": True,
        "data": {
            "evidence_status": "grounded",
            "context": {
                "visual_objects": [
                    {
                        "track_id": 7,
                        "label": "sports ball",
                        "confidence": 0.91,
                        "bbox": [10, 20, 30, 40],
                        "events": [{"track_id": 7, "timestamp": 1.0}],
                    }
                ],
                "visual_events": [
                    {
                        "track_id": 7,
                        "label": "sports ball",
                        "timestamp": 1.0,
                        "frame_index": 4,
                        "confidence": 0.91,
                        "position": [0.4, 0.5],
                    }
                ],
                "episodes": [
                    {
                        "track_id": 7,
                        "label": "sports ball",
                        "start_timestamp": 1.0,
                        "event_types": ["OBJECT_APPEARED"],
                        "events": [{"track_id": 7, "timestamp": 1.0}],
                        "evidence": [{"track_id": 7, "timestamp": 1.0}],
                    }
                ],
                "evidence_references": [
                    {
                        "track_id": 7,
                        "timestamp": 1.0,
                        "source_frame_index": 4,
                    }
                ],
            },
        },
    }
    original_snapshot = copy.deepcopy(original)

    compacted = _compact_video_response_evidence(original)
    compact_context = compacted["data"]["context"]

    assert "events" not in compact_context["visual_objects"][0]
    assert compact_context["visual_objects"][0]["event_details_in"] == "visual_events"
    assert "events" not in compact_context["episodes"][0]
    assert "evidence" not in compact_context["episodes"][0]
    assert compact_context["episodes"][0]["event_details_in"] == "visual_events"
    assert (
        compact_context["episodes"][0]["evidence_details_in"]
        == "evidence_references"
    )
    assert compact_context["visual_events"] == original["data"]["context"][
        "visual_events"
    ]
    assert compact_context["evidence_references"] == original["data"]["context"][
        "evidence_references"
    ]
    assert compact_context["visual_objects"][0]["bbox"] == [10, 20, 30, 40]
    assert compact_context["visual_objects"][0]["confidence"] == 0.91
    assert original == original_snapshot
    print("VIDEO PROMPT RETAINS GROUNDED DETAIL WITHOUT DUPLICATE RECORDS: PASS")


if __name__ == "__main__":
    test_visual_detail_is_referenced_without_repeating_nested_records()
