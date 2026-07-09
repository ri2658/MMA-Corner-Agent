"""Action classifier.

Classifies MMA actions from sequences of pose keypoints. Provides:

1. A **rule-based geometric classifier** that uses joint angles,
   velocities, and spatial relationships to identify common MMA actions.
   This works immediately with no training data.

2. Infrastructure for a **learned temporal model** (Transformer/TCN)
   that can be trained on annotated pose sequences for higher accuracy.

The rule-based classifier serves as:
  - A working baseline for the full pipeline
  - A label generator for bootstrapping training data
  - A fallback when the learned model's confidence is low
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .pose_estimator import LandmarkIndex as LM, PoseResult


@dataclass
class ActionPrediction:
    """Predicted action from a sequence of poses."""

    action_id: str                    # ID from action_taxonomy.json
    confidence: float                 # [0, 1]
    start_timestamp_s: float
    end_timestamp_s: float
    fighter_id: str

    @property
    def duration_s(self) -> float:
        return self.end_timestamp_s - self.start_timestamp_s


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _angle_3pts(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
) -> float:
    """Compute the angle at point B formed by segments BA and BC (degrees)."""
    ba = np.array([a[0] - b[0], a[1] - b[1]])
    bc = np.array([c[0] - b[0], c[1] - b[1]])
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-7)
    return float(np.degrees(np.arccos(np.clip(cos_angle, -1, 1))))


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Euclidean distance between two 2D points."""
    return float(np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2))


def _velocity(
    poses: list[PoseResult],
    landmark_idx: int,
) -> tuple[float, float]:
    """Compute average velocity of a landmark across a pose window.

    Returns (vx, vy) in pixels per second.
    """
    if len(poses) < 2:
        return (0.0, 0.0)

    first = poses[0].get_point_2d(landmark_idx)
    last = poses[-1].get_point_2d(landmark_idx)
    dt = poses[-1].timestamp_s - poses[0].timestamp_s

    if dt < 1e-6:
        return (0.0, 0.0)

    return (
        (last[0] - first[0]) / dt,
        (last[1] - first[1]) / dt,
    )


def _max_displacement(
    poses: list[PoseResult],
    landmark_idx: int,
) -> float:
    """Maximum displacement of a landmark from its starting position."""
    if not poses:
        return 0.0
    start = np.array(poses[0].get_point_2d(landmark_idx))
    max_disp = 0.0
    for pose in poses[1:]:
        pt = np.array(pose.get_point_2d(landmark_idx))
        disp = float(np.linalg.norm(pt - start))
        max_disp = max(max_disp, disp)
    return max_disp


def _body_height(pose: PoseResult) -> float:
    """Estimate body height from nose to ankle midpoint."""
    nose = pose.get_point_2d(LM.NOSE)
    l_ankle = pose.get_point_2d(LM.LEFT_ANKLE)
    r_ankle = pose.get_point_2d(LM.RIGHT_ANKLE)
    ankle_mid = ((l_ankle[0] + r_ankle[0]) / 2, (l_ankle[1] + r_ankle[1]) / 2)
    return _distance(nose, ankle_mid)


# ---------------------------------------------------------------------------
# Rule-based classifier
# ---------------------------------------------------------------------------

class RuleBasedClassifier:
    """Classifies MMA actions using geometric heuristics on pose keypoints.

    Analyzes joint angles, limb velocities, and spatial relationships
    across a temporal window of poses to identify actions. Designed
    for the most common and visually distinct MMA techniques.

    This classifier uses normalized coordinates (relative to body height)
    to be scale-invariant across different camera distances.
    """

    # Velocity thresholds (normalized by body height per second)
    FAST_HAND_VELOCITY = 2.0      # Striking speed
    FAST_FOOT_VELOCITY = 1.5      # Kicking speed
    LEVEL_CHANGE_THRESHOLD = 0.15  # Hip drop as fraction of body height

    def classify(
        self, window: list[PoseResult]
    ) -> Optional[tuple[str, float]]:
        """Classify the action in a pose window.

        Args:
            window: List of PoseResult for a single fighter (chronological).

        Returns:
            (action_id, confidence) or None if no action detected.
        """
        if len(window) < 3:
            return None

        last_pose = window[-1]
        body_h = _body_height(last_pose)
        if body_h < 10:  # Too small / unreliable
            return None

        # Check actions from most specific to most general
        checks = [
            self._check_head_kick(window, body_h),
            self._check_body_kick(window, body_h),
            self._check_low_kick(window, body_h),
            self._check_teep(window, body_h),
            self._check_lead_hook(window, body_h),
            self._check_uppercut(window, body_h),
            self._check_cross(window, body_h),
            self._check_jab(window, body_h),
            self._check_level_change(window, body_h),
            self._check_high_guard(window, body_h),
        ]

        for result in checks:
            if result is not None:
                return result

        return ("idle", 0.3)

    # ------------------------------------------------------------------
    # Individual action detectors
    # ------------------------------------------------------------------

    def _check_jab(
        self, window: list[PoseResult], body_h: float
    ) -> Optional[tuple[str, float]]:
        """Detect a jab: lead hand extends rapidly forward."""
        # Lead wrist velocity
        v = _velocity(window, LM.LEFT_WRIST)
        speed = np.sqrt(v[0] ** 2 + v[1] ** 2) / body_h

        # Lead arm extension
        last = window[-1]
        shoulder = last.get_point_2d(LM.LEFT_SHOULDER)
        elbow = last.get_point_2d(LM.LEFT_ELBOW)
        wrist = last.get_point_2d(LM.LEFT_WRIST)
        arm_angle = _angle_3pts(shoulder, elbow, wrist)

        # Jab: fast lead hand, arm relatively straight, hand near head height
        if speed > self.FAST_HAND_VELOCITY and arm_angle > 140:
            # Check hand is roughly at shoulder/head height
            if wrist[1] < shoulder[1] + body_h * 0.15:
                conf = min(0.9, 0.5 + (speed - self.FAST_HAND_VELOCITY) * 0.1)
                return ("jab_head", conf)

        return None

    def _check_cross(
        self, window: list[PoseResult], body_h: float
    ) -> Optional[tuple[str, float]]:
        """Detect a cross: rear hand extends rapidly with hip rotation."""
        v = _velocity(window, LM.RIGHT_WRIST)
        speed = np.sqrt(v[0] ** 2 + v[1] ** 2) / body_h

        last = window[-1]
        shoulder = last.get_point_2d(LM.RIGHT_SHOULDER)
        elbow = last.get_point_2d(LM.RIGHT_ELBOW)
        wrist = last.get_point_2d(LM.RIGHT_WRIST)
        arm_angle = _angle_3pts(shoulder, elbow, wrist)

        # Check for hip rotation (shoulders rotating relative to hips)
        l_shoulder = last.get_point_2d(LM.LEFT_SHOULDER)
        r_shoulder = last.get_point_2d(LM.RIGHT_SHOULDER)
        l_hip = last.get_point_2d(LM.LEFT_HIP)
        r_hip = last.get_point_2d(LM.RIGHT_HIP)

        shoulder_angle = np.arctan2(
            r_shoulder[1] - l_shoulder[1],
            r_shoulder[0] - l_shoulder[0],
        )
        hip_angle = np.arctan2(
            r_hip[1] - l_hip[1],
            r_hip[0] - l_hip[0],
        )
        rotation = abs(shoulder_angle - hip_angle)

        if speed > self.FAST_HAND_VELOCITY and arm_angle > 130 and rotation > 0.15:
            if wrist[1] < shoulder[1] + body_h * 0.15:
                conf = min(0.9, 0.5 + (speed - self.FAST_HAND_VELOCITY) * 0.1)
                return ("cross_head", conf)

        return None

    def _check_lead_hook(
        self, window: list[PoseResult], body_h: float
    ) -> Optional[tuple[str, float]]:
        """Detect a lead hook: lead hand moves laterally with bent elbow."""
        v = _velocity(window, LM.LEFT_WRIST)
        speed = np.sqrt(v[0] ** 2 + v[1] ** 2) / body_h

        last = window[-1]
        shoulder = last.get_point_2d(LM.LEFT_SHOULDER)
        elbow = last.get_point_2d(LM.LEFT_ELBOW)
        wrist = last.get_point_2d(LM.LEFT_WRIST)
        arm_angle = _angle_3pts(shoulder, elbow, wrist)

        # Hook: fast hand, bent elbow (60-120°), lateral movement
        lateral_v = abs(v[0]) / body_h
        if speed > self.FAST_HAND_VELOCITY * 0.8 and 60 < arm_angle < 130:
            if lateral_v > self.FAST_HAND_VELOCITY * 0.5:
                conf = min(0.85, 0.5 + (speed - self.FAST_HAND_VELOCITY * 0.8) * 0.1)
                return ("lead_hook_head", conf)

        return None

    def _check_uppercut(
        self, window: list[PoseResult], body_h: float
    ) -> Optional[tuple[str, float]]:
        """Detect an uppercut: hand moves rapidly upward with bent elbow."""
        for wrist_idx, action_id in [
            (LM.LEFT_WRIST, "lead_uppercut"),
            (LM.RIGHT_WRIST, "rear_uppercut"),
        ]:
            v = _velocity(window, wrist_idx)
            vertical_speed = -v[1] / body_h  # Negative y = upward

            if vertical_speed > self.FAST_HAND_VELOCITY * 0.8:
                last = window[-1]
                shoulder_idx = (
                    LM.LEFT_SHOULDER if wrist_idx == LM.LEFT_WRIST
                    else LM.RIGHT_SHOULDER
                )
                elbow_idx = (
                    LM.LEFT_ELBOW if wrist_idx == LM.LEFT_WRIST
                    else LM.RIGHT_ELBOW
                )
                shoulder = last.get_point_2d(shoulder_idx)
                elbow = last.get_point_2d(elbow_idx)
                wrist = last.get_point_2d(wrist_idx)
                arm_angle = _angle_3pts(shoulder, elbow, wrist)

                if 60 < arm_angle < 140:
                    conf = min(0.8, 0.5 + vertical_speed * 0.1)
                    return (action_id, conf)

        return None

    def _check_head_kick(
        self, window: list[PoseResult], body_h: float
    ) -> Optional[tuple[str, float]]:
        """Detect a head kick: foot rises rapidly to head height."""
        for ankle_idx, action_id in [
            (LM.LEFT_ANKLE, "lead_head_kick"),
            (LM.RIGHT_ANKLE, "rear_head_kick"),
        ]:
            v = _velocity(window, ankle_idx)
            speed = np.sqrt(v[0] ** 2 + v[1] ** 2) / body_h

            if speed > self.FAST_FOOT_VELOCITY:
                last = window[-1]
                ankle = last.get_point_2d(ankle_idx)
                nose = last.get_point_2d(LM.NOSE)
                shoulder_mid_y = (
                    last.get_point_2d(LM.LEFT_SHOULDER)[1]
                    + last.get_point_2d(LM.RIGHT_SHOULDER)[1]
                ) / 2

                # Foot must be at or above shoulder height
                if ankle[1] < shoulder_mid_y:
                    conf = min(0.85, 0.5 + speed * 0.15)
                    return (action_id, conf)

        return None

    def _check_body_kick(
        self, window: list[PoseResult], body_h: float
    ) -> Optional[tuple[str, float]]:
        """Detect a body kick: foot at torso height with lateral velocity."""
        for ankle_idx, action_id in [
            (LM.LEFT_ANKLE, "lead_body_kick"),
            (LM.RIGHT_ANKLE, "rear_body_kick"),
        ]:
            v = _velocity(window, ankle_idx)
            speed = np.sqrt(v[0] ** 2 + v[1] ** 2) / body_h

            if speed > self.FAST_FOOT_VELOCITY * 0.8:
                last = window[-1]
                ankle = last.get_point_2d(ankle_idx)
                hip_mid_y = (
                    last.get_point_2d(LM.LEFT_HIP)[1]
                    + last.get_point_2d(LM.RIGHT_HIP)[1]
                ) / 2
                shoulder_mid_y = (
                    last.get_point_2d(LM.LEFT_SHOULDER)[1]
                    + last.get_point_2d(LM.RIGHT_SHOULDER)[1]
                ) / 2

                # Foot between hip and shoulder height
                if shoulder_mid_y < ankle[1] < hip_mid_y:
                    conf = min(0.8, 0.5 + speed * 0.1)
                    return (action_id, conf)

        return None

    def _check_low_kick(
        self, window: list[PoseResult], body_h: float
    ) -> Optional[tuple[str, float]]:
        """Detect a low kick: foot moves laterally at or below hip height."""
        for ankle_idx, action_id in [
            (LM.LEFT_ANKLE, "lead_calf_kick"),
            (LM.RIGHT_ANKLE, "rear_leg_kick"),
        ]:
            v = _velocity(window, ankle_idx)
            lateral_speed = abs(v[0]) / body_h

            if lateral_speed > self.FAST_FOOT_VELOCITY * 0.6:
                last = window[-1]
                ankle = last.get_point_2d(ankle_idx)
                hip_mid_y = (
                    last.get_point_2d(LM.LEFT_HIP)[1]
                    + last.get_point_2d(LM.RIGHT_HIP)[1]
                ) / 2

                # Foot at or below hip height
                if ankle[1] >= hip_mid_y:
                    conf = min(0.75, 0.4 + lateral_speed * 0.1)
                    return (action_id, conf)

        return None

    def _check_teep(
        self, window: list[PoseResult], body_h: float
    ) -> Optional[tuple[str, float]]:
        """Detect a teep/front kick: foot extends forward (along z/x axis)."""
        for ankle_idx, knee_idx, action_id in [
            (LM.LEFT_ANKLE, LM.LEFT_KNEE, "lead_teep"),
            (LM.RIGHT_ANKLE, LM.RIGHT_KNEE, "rear_teep"),
        ]:
            v = _velocity(window, ankle_idx)
            forward_speed = abs(v[0]) / body_h

            if forward_speed > self.FAST_FOOT_VELOCITY * 0.5:
                last = window[-1]
                hip_idx = (
                    LM.LEFT_HIP if ankle_idx == LM.LEFT_ANKLE
                    else LM.RIGHT_HIP
                )
                hip = last.get_point_2d(hip_idx)
                knee = last.get_point_2d(knee_idx)
                ankle = last.get_point_2d(ankle_idx)

                # Leg should be relatively straight (teep is a push)
                leg_angle = _angle_3pts(hip, knee, ankle)
                if leg_angle > 130:
                    conf = min(0.7, 0.4 + forward_speed * 0.1)
                    return (action_id, conf)

        return None

    def _check_level_change(
        self, window: list[PoseResult], body_h: float
    ) -> Optional[tuple[str, float]]:
        """Detect a level change (potential takedown setup)."""
        if len(window) < 3:
            return None

        # Track hip height change across window
        first_hip_y = (
            window[0].get_point_2d(LM.LEFT_HIP)[1]
            + window[0].get_point_2d(LM.RIGHT_HIP)[1]
        ) / 2
        last_hip_y = (
            window[-1].get_point_2d(LM.LEFT_HIP)[1]
            + window[-1].get_point_2d(LM.RIGHT_HIP)[1]
        ) / 2

        hip_drop = (last_hip_y - first_hip_y) / body_h  # Positive = downward

        if hip_drop > self.LEVEL_CHANGE_THRESHOLD:
            # Check if hands are reaching forward (takedown) vs. punching down
            last = window[-1]
            l_wrist = last.get_point_2d(LM.LEFT_WRIST)
            r_wrist = last.get_point_2d(LM.RIGHT_WRIST)
            l_hip = last.get_point_2d(LM.LEFT_HIP)
            r_hip = last.get_point_2d(LM.RIGHT_HIP)

            # Hands at or below hip level suggests takedown
            hands_low = (
                l_wrist[1] > l_hip[1] - body_h * 0.05
                and r_wrist[1] > r_hip[1] - body_h * 0.05
            )

            if hands_low:
                conf = min(0.75, 0.4 + hip_drop * 2)
                return ("double_leg", conf)

        return None

    def _check_high_guard(
        self, window: list[PoseResult], body_h: float
    ) -> Optional[tuple[str, float]]:
        """Detect a high guard / defensive shell."""
        last = window[-1]
        l_wrist = last.get_point_2d(LM.LEFT_WRIST)
        r_wrist = last.get_point_2d(LM.RIGHT_WRIST)
        nose = last.get_point_2d(LM.NOSE)

        # Both hands near the face, low velocity
        l_dist = _distance(l_wrist, nose) / body_h
        r_dist = _distance(r_wrist, nose) / body_h

        l_v = _velocity(window, LM.LEFT_WRIST)
        r_v = _velocity(window, LM.RIGHT_WRIST)
        l_speed = np.sqrt(l_v[0] ** 2 + l_v[1] ** 2) / body_h
        r_speed = np.sqrt(r_v[0] ** 2 + r_v[1] ** 2) / body_h

        if l_dist < 0.15 and r_dist < 0.15 and l_speed < 0.5 and r_speed < 0.5:
            return ("high_block", 0.6)

        return None


# ---------------------------------------------------------------------------
# Main classifier (wraps rule-based + future learned model)
# ---------------------------------------------------------------------------

class ActionClassifier:
    """Classifies MMA actions from pose keypoint sequences.

    Supports two modes:
    - Rule-based (default): Uses geometric heuristics. No training needed.
    - Learned model: Uses a trained temporal model. Requires model_path.

    Usage:
        classifier = ActionClassifier()  # Rule-based
        prediction = classifier.classify(pose_window)

        classifier = ActionClassifier(model_path="models/action_classifier.pt")
        predictions = classifier.classify_stream(pose_stream)
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        window_size: int = 15,           # ~1 second at 15fps
        stride: int = 5,                 # Prediction every ~0.33s
        confidence_threshold: float = 0.4,
    ):
        self.model_path = model_path
        self.window_size = window_size
        self.stride = stride
        self.confidence_threshold = confidence_threshold

        self._rule_classifier = RuleBasedClassifier()
        self._learned_model = None

        if model_path is not None:
            self._init_model()

    def _init_model(self):
        """Load a trained action classification model."""
        import torch

        if self.model_path and self.model_path != "rules":
            self._learned_model = torch.load(
                self.model_path, map_location="cpu"
            )
            self._learned_model.eval()

    def classify(
        self,
        pose_window: list[PoseResult],
    ) -> Optional[ActionPrediction]:
        """Classify the action in a window of pose results.

        Uses the learned model if available, otherwise falls back to
        the rule-based classifier.

        Args:
            pose_window: List of PoseResult objects (chronological).

        Returns:
            ActionPrediction if confidence exceeds threshold, None otherwise.
        """
        if not pose_window:
            return None

        fighter_id = pose_window[0].fighter_id
        start_t = pose_window[0].timestamp_s
        end_t = pose_window[-1].timestamp_s

        # Try learned model first
        if self._learned_model is not None:
            result = self._classify_learned(pose_window)
            if result is not None:
                return result

        # Fall back to rule-based
        result = self._rule_classifier.classify(pose_window)
        if result is None:
            return None

        action_id, confidence = result
        if confidence < self.confidence_threshold:
            return None

        return ActionPrediction(
            action_id=action_id,
            confidence=confidence,
            start_timestamp_s=start_t,
            end_timestamp_s=end_t,
            fighter_id=fighter_id,
        )

    def _classify_learned(
        self, pose_window: list[PoseResult]
    ) -> Optional[ActionPrediction]:
        """Classify using the learned temporal model.

        Converts pose keypoints to tensor, runs inference, and returns
        the top prediction.
        """
        # TODO: Implement once model architecture is defined and trained
        # This would:
        # 1. Stack keypoints into tensor of shape (T, 33, 3)
        # 2. Run through the model
        # 3. Apply softmax
        # 4. Map class index to action_id
        return None

    def classify_stream(
        self,
        pose_stream: list[PoseResult],
    ) -> list[ActionPrediction]:
        """Classify actions across an entire stream of poses.

        Applies a sliding window with the configured stride to produce
        a sequence of action predictions.

        Args:
            pose_stream: Full chronological list of PoseResult objects
                         for a single fighter.

        Returns:
            List of ActionPrediction objects.
        """
        predictions: list[ActionPrediction] = []

        for i in range(0, max(1, len(pose_stream) - self.window_size + 1), self.stride):
            end = min(i + self.window_size, len(pose_stream))
            window = pose_stream[i:end]

            if len(window) < 3:
                continue

            pred = self.classify(window)
            if pred is not None:
                # Deduplicate: don't repeat the same action back-to-back
                if predictions and predictions[-1].action_id == pred.action_id:
                    # Extend the previous prediction's duration
                    predictions[-1] = ActionPrediction(
                        action_id=pred.action_id,
                        confidence=max(predictions[-1].confidence, pred.confidence),
                        start_timestamp_s=predictions[-1].start_timestamp_s,
                        end_timestamp_s=pred.end_timestamp_s,
                        fighter_id=pred.fighter_id,
                    )
                else:
                    predictions.append(pred)

        return predictions
