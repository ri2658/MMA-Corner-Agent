"""Validate the vision pipeline components with synthetic data.

Tests the rule-based action classifier and pipeline logic without
requiring video files, MediaPipe, or YOLO -- just pure Python + numpy.
"""
import sys
sys.path.insert(0, ".")

import numpy as np
from src.vision.pose_estimator import PoseResult, LandmarkIndex as LM
from src.vision.action_classifier import (
    ActionClassifier,
    RuleBasedClassifier,
    _angle_3pts,
    _distance,
    _velocity,
)


def make_pose(
    fighter_id: str = "fighter_a",
    frame_index: int = 0,
    timestamp_s: float = 0.0,
    overrides: dict = None,
) -> PoseResult:
    """Create a synthetic PoseResult with a neutral standing pose.

    Keypoints are in a ~500px tall coordinate space.
    Override specific landmarks by passing {landmark_index: (x, y, vis)}.
    """
    # Neutral standing pose (approximate MediaPipe layout)
    base_keypoints = [
        (250, 50, 0.99),    # 0: NOSE
        (245, 45, 0.95),    # 1: LEFT_EYE_INNER
        (243, 44, 0.95),    # 2: LEFT_EYE
        (241, 45, 0.95),    # 3: LEFT_EYE_OUTER
        (255, 45, 0.95),    # 4: RIGHT_EYE_INNER
        (257, 44, 0.95),    # 5: RIGHT_EYE
        (259, 45, 0.95),    # 6: RIGHT_EYE_OUTER
        (235, 50, 0.90),    # 7: LEFT_EAR
        (265, 50, 0.90),    # 8: RIGHT_EAR
        (245, 58, 0.90),    # 9: MOUTH_LEFT
        (255, 58, 0.90),    # 10: MOUTH_RIGHT
        (220, 120, 0.95),   # 11: LEFT_SHOULDER
        (280, 120, 0.95),   # 12: RIGHT_SHOULDER
        (210, 200, 0.90),   # 13: LEFT_ELBOW
        (290, 200, 0.90),   # 14: RIGHT_ELBOW
        (220, 160, 0.90),   # 15: LEFT_WRIST (guard position)
        (280, 160, 0.90),   # 16: RIGHT_WRIST (guard position)
        (218, 158, 0.85),   # 17: LEFT_PINKY
        (282, 158, 0.85),   # 18: RIGHT_PINKY
        (222, 155, 0.85),   # 19: LEFT_INDEX
        (278, 155, 0.85),   # 20: RIGHT_INDEX
        (220, 156, 0.85),   # 21: LEFT_THUMB
        (280, 156, 0.85),   # 22: RIGHT_THUMB
        (230, 280, 0.95),   # 23: LEFT_HIP
        (270, 280, 0.95),   # 24: RIGHT_HIP
        (225, 380, 0.90),   # 25: LEFT_KNEE
        (275, 380, 0.90),   # 26: RIGHT_KNEE
        (220, 480, 0.90),   # 27: LEFT_ANKLE
        (280, 480, 0.90),   # 28: RIGHT_ANKLE
        (215, 490, 0.85),   # 29: LEFT_HEEL
        (285, 490, 0.85),   # 30: RIGHT_HEEL
        (225, 495, 0.85),   # 31: LEFT_FOOT_INDEX
        (275, 495, 0.85),   # 32: RIGHT_FOOT_INDEX
    ]

    kps = list(base_keypoints)
    if overrides:
        for idx, val in overrides.items():
            kps[idx] = val

    return PoseResult(
        fighter_id=fighter_id,
        frame_index=frame_index,
        timestamp_s=timestamp_s,
        keypoints=kps,
        confidence=0.9,
    )


def test_geometry_helpers():
    """Test angle and distance calculations."""
    # Right angle
    angle = _angle_3pts((0, 0), (0, 1), (1, 1))
    assert 88 < angle < 92, f"Expected ~90, got {angle}"

    # Straight line
    angle = _angle_3pts((0, 0), (1, 0), (2, 0))
    assert angle > 175, f"Expected ~180, got {angle}"

    # Distance
    d = _distance((0, 0), (3, 4))
    assert abs(d - 5.0) < 0.01, f"Expected 5.0, got {d}"

    print("[OK] Geometry helpers")


def test_velocity():
    """Test velocity computation from pose sequence."""
    poses = [
        make_pose(timestamp_s=0.0, overrides={LM.LEFT_WRIST: (220, 160, 0.9)}),
        make_pose(timestamp_s=0.1, overrides={LM.LEFT_WRIST: (270, 160, 0.9)}),
    ]
    vx, vy = _velocity(poses, LM.LEFT_WRIST)
    assert abs(vx - 500.0) < 1.0, f"Expected vx~500, got {vx}"
    assert abs(vy) < 1.0, f"Expected vy~0, got {vy}"
    print("[OK] Velocity computation")


def test_detect_jab():
    """Test jab detection: fast lead hand extension.

    A real jab covers ~300px in 3-4 frames at 15fps (~0.2s).
    Body height ~430px, so normalized speed = ~300/(430*0.2) = ~3.5.
    """
    fps = 15.0
    dt = 1.0 / fps

    # Build a window: 3 frames idle guard, then 4 frames of fast extension
    window = []
    # Idle guard frames
    for i in range(3):
        window.append(make_pose(timestamp_s=i * dt, frame_index=i))

    # Jab extension over 4 frames (rapid)
    for i in range(4):
        t = (3 + i) * dt
        progress = i / 3
        # Wrist moves 300px forward, 40px up over 4 frames
        wrist_x = 220 - 300 * progress
        wrist_y = 160 - 40 * progress
        # Elbow straightens
        elbow_x = 210 - 150 * progress
        elbow_y = 200 - 80 * progress

        window.append(make_pose(
            timestamp_s=t,
            frame_index=3 + i,
            overrides={
                LM.LEFT_WRIST: (wrist_x, wrist_y, 0.9),
                LM.LEFT_ELBOW: (elbow_x, elbow_y, 0.9),
                LM.LEFT_INDEX: (wrist_x + 2, wrist_y - 3, 0.85),
            },
        ))

    classifier = RuleBasedClassifier()
    result = classifier.classify(window)

    assert result is not None, "Expected jab detection, got None"
    action_id, conf = result
    assert action_id == "jab_head", f"Expected jab_head, got {action_id}"
    assert conf > 0.4, f"Expected confidence > 0.4, got {conf}"
    print(f"[OK] Jab detection: {action_id} (conf={conf:.2f})")


def test_detect_high_guard():
    """Test high guard detection: hands near face, low velocity."""
    window = []
    for i in range(8):
        window.append(make_pose(
            timestamp_s=i * 0.067,
            frame_index=i,
            overrides={
                LM.LEFT_WRIST: (242, 60, 0.9),   # Near nose
                LM.RIGHT_WRIST: (258, 60, 0.9),   # Near nose
            },
        ))

    classifier = RuleBasedClassifier()
    result = classifier.classify(window)

    assert result is not None, "Expected high_block detection"
    action_id, conf = result
    assert action_id == "high_block", f"Expected high_block, got {action_id}"
    print(f"[OK] High guard detection: {action_id} (conf={conf:.2f})")


def test_detect_level_change():
    """Test level change detection: hips drop significantly."""
    window = []
    for i in range(10):
        t = i * 0.067
        progress = i / 9
        hip_drop = 80 * progress  # Hips drop 80px

        window.append(make_pose(
            timestamp_s=t,
            frame_index=i,
            overrides={
                LM.LEFT_HIP: (230, 280 + hip_drop, 0.95),
                LM.RIGHT_HIP: (270, 280 + hip_drop, 0.95),
                # Hands go low (reaching for legs)
                LM.LEFT_WRIST: (220, 280 + hip_drop + 10, 0.9),
                LM.RIGHT_WRIST: (280, 280 + hip_drop + 10, 0.9),
                LM.LEFT_KNEE: (225, 380 + hip_drop * 0.3, 0.9),
                LM.RIGHT_KNEE: (275, 380 + hip_drop * 0.3, 0.9),
            },
        ))

    classifier = RuleBasedClassifier()
    result = classifier.classify(window)

    assert result is not None, "Expected level change detection"
    action_id, conf = result
    assert action_id == "double_leg", f"Expected double_leg, got {action_id}"
    print(f"[OK] Level change detection: {action_id} (conf={conf:.2f})")


def test_detect_head_kick():
    """Test head kick detection: foot rises to head height rapidly.

    A head kick takes ~6-8 frames at 15fps. The ankle covers ~400px.
    """
    dt = 1.0 / 15.0
    window = []

    # 3 idle frames, then 5 frames of fast kick
    for i in range(3):
        window.append(make_pose(timestamp_s=i * dt, frame_index=i))

    for i in range(5):
        t = (3 + i) * dt
        progress = i / 4
        ankle_y = 480 - 420 * progress  # Rises to head level (60)
        ankle_x = 280 + 80 * progress   # Lateral arc

        window.append(make_pose(
            timestamp_s=t,
            frame_index=3 + i,
            overrides={
                LM.RIGHT_ANKLE: (ankle_x, ankle_y, 0.9),
                LM.RIGHT_KNEE: (275, 380 - 250 * progress, 0.9),
                LM.RIGHT_FOOT_INDEX: (ankle_x + 5, ankle_y + 5, 0.85),
            },
        ))

    classifier = RuleBasedClassifier()
    result = classifier.classify(window)

    assert result is not None, "Expected head kick detection"
    action_id, conf = result
    assert "head_kick" in action_id or "body_kick" in action_id, f"Expected kick, got {action_id}"
    print(f"[OK] Head kick detection: {action_id} (conf={conf:.2f})")


def test_action_classifier_stream():
    """Test the full ActionClassifier with stream processing."""
    # Create a sequence: idle -> jab -> idle -> guard
    poses = []
    t = 0.0
    dt = 1.0 / 15.0

    # 15 frames idle
    for i in range(15):
        poses.append(make_pose(timestamp_s=t, frame_index=len(poses)))
        t += dt

    # 5 frames fast jab (realistic speed)
    for i in range(5):
        progress = i / 4
        poses.append(make_pose(
            timestamp_s=t,
            frame_index=len(poses),
            overrides={
                LM.LEFT_WRIST: (220 - 300 * progress, 160 - 40 * progress, 0.9),
                LM.LEFT_ELBOW: (210 - 150 * progress, 200 - 80 * progress, 0.9),
            },
        ))
        t += dt

    # 15 frames idle
    for i in range(15):
        poses.append(make_pose(timestamp_s=t, frame_index=len(poses)))
        t += dt

    classifier = ActionClassifier(window_size=8, stride=4)
    predictions = classifier.classify_stream(poses)

    print(f"[OK] Stream classifier: {len(predictions)} predictions")
    for pred in predictions:
        print(f"     t={pred.start_timestamp_s:.2f}-{pred.end_timestamp_s:.2f}: "
              f"{pred.action_id} (conf={pred.confidence:.2f})")


def test_idle_detection():
    """Test that a stationary pose is classified as idle."""
    window = [make_pose(timestamp_s=i * 0.067, frame_index=i) for i in range(8)]

    classifier = RuleBasedClassifier()
    result = classifier.classify(window)

    assert result is not None
    action_id, conf = result
    # Should be idle or high_block depending on hand position
    assert action_id in ("idle", "high_block"), f"Got {action_id}"
    print(f"[OK] Idle/neutral detection: {action_id} (conf={conf:.2f})")


if __name__ == "__main__":
    test_geometry_helpers()
    test_velocity()
    test_detect_jab()
    test_detect_high_guard()
    test_detect_level_change()
    test_detect_head_kick()
    test_idle_detection()
    test_action_classifier_stream()

    print("\n=== ALL VISION TESTS PASSED ===")
