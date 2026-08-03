"""
Test script for TransNetV2 Shot Boundary Detector & Keyframe Extractor.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from models.shot_detection.transnetv2 import TransNetV2Detector


def test_transnetv2_mock():
    print("Testing TransNetV2 Detector logic...")

    detector = TransNetV2Detector(threshold=0.5, keyframes_per_shot=1)

    # Mock predictions: 100 frames with 2 distinct cuts at frame 30 and frame 70
    mock_preds = np.zeros(100, dtype=np.float32)
    mock_preds[30] = 0.85
    mock_preds[71] = 0.92

    scenes = detector.predictions_to_scenes(mock_preds)
    print(f"[OK] Predictions converted to {len(scenes)} scenes:")
    for i, (s, e) in enumerate(scenes):
        print(f"  Scene {i+1}: frames {s} to {e} (length {e-s+1})")

    assert len(scenes) == 3, f"Expected 3 scenes, got {len(scenes)}"

    # Test keyframe selection (1 per shot -> middle frame)
    keyframes = detector.select_keyframes(scenes, keyframes_per_shot=1)
    print(f"[OK] Selected Keyframes (1 per shot): {keyframes}")
    assert len(keyframes) == 3, "Expected 3 keyframes"

    # Test keyframe selection (3 per shot -> start, middle, end)
    keyframes_3 = detector.select_keyframes(scenes, keyframes_per_shot=3)
    print(f"[OK] Selected Keyframes (3 per shot): {keyframes_3}")
    assert len(keyframes_3) > 3, "Expected multiple keyframes per shot"

    print("\nALL TRANSNETV2 DETECTOR TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    test_transnetv2_mock()
