"""Video processing pipeline.

Orchestrates the full vision pipeline: video I/O -> fighter tracking ->
pose estimation -> action classification -> combat state generation.

This is the main entry point for processing fight footage.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Iterator

import numpy as np

from .fighter_tracker import FighterTracker, TrackedFighter
from .pose_estimator import PoseEstimator, PoseResult
from .action_classifier import ActionClassifier, ActionPrediction
from ..analysis.state_vector import (
    CombatState, Stance, RingPosition, Distance, Momentum,
)


@dataclass
class FrameResult:
    """Complete analysis result for a single video frame."""

    frame_index: int
    timestamp_s: float
    fighters: list[TrackedFighter]
    poses: dict[str, Optional[PoseResult]]   # fighter_id -> PoseResult
    actions: dict[str, Optional[ActionPrediction]]  # fighter_id -> latest action


@dataclass
class PipelineConfig:
    """Configuration for the video processing pipeline."""

    # Video processing
    target_fps: float = 15.0           # Process at this FPS (skip frames)
    max_frames: Optional[int] = None   # Stop after this many frames

    # Fighter tracker
    detector_model: str = "yolov8n.pt"
    min_detection_confidence: float = 0.5

    # Pose estimation
    pose_model_complexity: int = 1
    min_pose_confidence: float = 0.5

    # Action classification
    action_model_path: Optional[str] = None  # None = rule-based
    action_window_size: int = 15
    action_stride: int = 5
    action_confidence_threshold: float = 0.4


class VideoPipeline:
    """Full video processing pipeline.

    Processes a video file or live feed through:
    1. Frame extraction at target FPS
    2. Fighter detection and tracking (YOLO)
    3. Pose estimation per fighter (MediaPipe)
    4. Action classification from pose sequences
    5. Combat state vector generation

    Usage:
        pipeline = VideoPipeline()
        for states in pipeline.process_video("fight.mp4"):
            # states = (CombatState for fighter_a, CombatState for fighter_b)
            print(states)
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()

        self.tracker = FighterTracker(
            detector_model=self.config.detector_model,
            min_detection_confidence=self.config.min_detection_confidence,
        )
        self.pose_estimator = PoseEstimator(
            model_complexity=self.config.pose_model_complexity,
            min_detection_confidence=self.config.min_pose_confidence,
        )
        self.classifier_a = ActionClassifier(
            model_path=self.config.action_model_path,
            window_size=self.config.action_window_size,
            stride=self.config.action_stride,
            confidence_threshold=self.config.action_confidence_threshold,
        )
        self.classifier_b = ActionClassifier(
            model_path=self.config.action_model_path,
            window_size=self.config.action_window_size,
            stride=self.config.action_stride,
            confidence_threshold=self.config.action_confidence_threshold,
        )

        # Pose history buffers for sliding-window classification
        self._pose_buffer_a: list[PoseResult] = []
        self._pose_buffer_b: list[PoseResult] = []
        self._max_buffer = self.config.action_window_size * 3

    def process_video(
        self,
        video_path: str,
        round_number: int = 1,
    ) -> Iterator[tuple[Optional[CombatState], Optional[CombatState]]]:
        """Process a video file and yield combat states per frame.

        Args:
            video_path: Path to the video file.
            round_number: Round number for tagging states.

        Yields:
            (CombatState for fighter_a, CombatState for fighter_b) per processed frame.
            Either may be None if the fighter isn't detected in that frame.
        """
        import cv2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        source_fps = cap.get(cv2.CAP_PROP_FPS)
        if source_fps <= 0:
            source_fps = 30.0

        frame_skip = max(1, int(source_fps / self.config.target_fps))
        frame_idx = 0
        processed = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Skip frames to match target FPS
                if frame_idx % frame_skip != 0:
                    frame_idx += 1
                    continue

                timestamp = frame_idx / source_fps

                # Check frame limit
                if (
                    self.config.max_frames is not None
                    and processed >= self.config.max_frames
                ):
                    break

                # Run pipeline on this frame
                state_a, state_b = self.process_frame(
                    frame, frame_idx, timestamp
                )

                yield (state_a, state_b)

                processed += 1
                frame_idx += 1

        finally:
            cap.release()

    def process_frame(
        self,
        frame: np.ndarray,
        frame_index: int,
        timestamp_s: float,
    ) -> tuple[Optional[CombatState], Optional[CombatState]]:
        """Process a single frame through the full pipeline.

        Args:
            frame: BGR image (H, W, 3).
            frame_index: Frame number.
            timestamp_s: Timestamp in seconds from round start.

        Returns:
            (CombatState for fighter_a, CombatState for fighter_b).
        """
        # Step 1: Detect and track fighters
        fighters = self.tracker.update(frame, frame_index)

        fighter_map = {f.fighter_id: f for f in fighters}
        fa = fighter_map.get("fighter_a")
        fb = fighter_map.get("fighter_b")

        # Step 2: Pose estimation per fighter
        pose_a = None
        pose_b = None

        if fa is not None:
            pose_a = self.pose_estimator.estimate(
                frame, fa.bbox, "fighter_a", frame_index, timestamp_s
            )
            if pose_a is not None:
                self._pose_buffer_a.append(pose_a)
                if len(self._pose_buffer_a) > self._max_buffer:
                    self._pose_buffer_a = self._pose_buffer_a[-self._max_buffer:]

        if fb is not None:
            pose_b = self.pose_estimator.estimate(
                frame, fb.bbox, "fighter_b", frame_index, timestamp_s
            )
            if pose_b is not None:
                self._pose_buffer_b.append(pose_b)
                if len(self._pose_buffer_b) > self._max_buffer:
                    self._pose_buffer_b = self._pose_buffer_b[-self._max_buffer:]

        # Step 3: Action classification from pose buffer
        action_a = None
        action_b = None

        if len(self._pose_buffer_a) >= 3:
            window = self._pose_buffer_a[-self.config.action_window_size:]
            action_a = self.classifier_a.classify(window)

        if len(self._pose_buffer_b) >= 3:
            window = self._pose_buffer_b[-self.config.action_window_size:]
            action_b = self.classifier_b.classify(window)

        # Step 4: Build combat states
        inter_fighter_dist = self.tracker.get_distance_between_fighters()
        frame_h = frame.shape[0]

        state_a = self._build_combat_state(
            "fighter_a", timestamp_s, fa, pose_a, action_a,
            inter_fighter_dist, frame_h, frame.shape[1],
        )
        state_b = self._build_combat_state(
            "fighter_b", timestamp_s, fb, pose_b, action_b,
            inter_fighter_dist, frame_h, frame.shape[1],
        )

        return state_a, state_b

    def _build_combat_state(
        self,
        fighter_id: str,
        timestamp_s: float,
        tracked: Optional[TrackedFighter],
        pose: Optional[PoseResult],
        action: Optional[ActionPrediction],
        inter_distance: Optional[float],
        frame_h: int,
        frame_w: int,
    ) -> Optional[CombatState]:
        """Build a CombatState from all pipeline outputs."""
        if tracked is None:
            return None

        # Determine stance from pose (if available)
        stance = self._infer_stance(pose)

        # Determine ring position from bbox location
        ring_pos = self._infer_ring_position(tracked, frame_w)

        # Determine distance between fighters
        distance = self._infer_distance(inter_distance, frame_h)

        # Determine momentum from tracker velocity
        momentum = self._infer_momentum(tracked, frame_h)

        return CombatState(
            fighter_id=fighter_id,
            timestamp_s=timestamp_s,
            action_id=action.action_id if action else None,
            action_confidence=action.confidence if action else 0.0,
            stance=stance,
            ring_position=ring_pos,
            distance=distance,
            momentum=momentum,
            keypoints=(
                pose.keypoints if pose else None
            ),
        )

    @staticmethod
    def _infer_stance(pose: Optional[PoseResult]) -> Stance:
        """Infer stance (orthodox/southpaw) from foot positions."""
        if pose is None:
            return Stance.ORTHODOX

        from .pose_estimator import LandmarkIndex as LM

        l_ankle = pose.get_point_2d(LM.LEFT_ANKLE)
        r_ankle = pose.get_point_2d(LM.RIGHT_ANKLE)
        l_hip = pose.get_point_2d(LM.LEFT_HIP)
        r_hip = pose.get_point_2d(LM.RIGHT_HIP)

        # In orthodox: left foot forward. In broadcast view,
        # "forward" depends on which side they're facing.
        # Use hip-to-ankle angles to determine which foot is leading.
        hip_mid_y = (l_hip[1] + r_hip[1]) / 2

        # Check if hips are very low (ground position)
        body_h = pose.to_numpy()[:, 1].ptp()
        if body_h > 0 and (hip_mid_y - pose.get_point_2d(LM.NOSE)[1]) / body_h < 0.15:
            return Stance.GROUND

        # Determine lead foot by which ankle is further from the body center
        hip_mid_x = (l_hip[0] + r_hip[0]) / 2
        l_forward = abs(l_ankle[0] - hip_mid_x)
        r_forward = abs(r_ankle[0] - hip_mid_x)

        if l_forward > r_forward * 1.15:
            return Stance.ORTHODOX  # Left foot forward
        elif r_forward > l_forward * 1.15:
            return Stance.SOUTHPAW  # Right foot forward
        else:
            return Stance.OPEN  # Feet roughly even

    @staticmethod
    def _infer_ring_position(
        tracked: TrackedFighter,
        frame_w: int,
    ) -> RingPosition:
        """Infer ring position from bbox location relative to frame edges."""
        cx, _ = tracked.center
        rel_x = cx / max(frame_w, 1)

        edge_margin = 0.12  # ~12% from edge = near cage

        if rel_x < edge_margin or rel_x > (1 - edge_margin):
            return RingPosition.ON_CAGE
        elif rel_x < 0.25 or rel_x > 0.75:
            return RingPosition.NEAR_CAGE
        elif 0.35 < rel_x < 0.65:
            return RingPosition.CENTER
        else:
            return RingPosition.MID_RANGE

    @staticmethod
    def _infer_distance(
        inter_distance: Optional[float],
        frame_h: int,
    ) -> Distance:
        """Infer distance between fighters from pixel distance."""
        if inter_distance is None:
            return Distance.BOXING

        # Normalize by frame height for scale invariance
        rel_dist = inter_distance / max(frame_h, 1)

        if rel_dist > 0.45:
            return Distance.OUT_OF_RANGE
        elif rel_dist > 0.25:
            return Distance.KICKING
        elif rel_dist > 0.12:
            return Distance.BOXING
        else:
            return Distance.CLINCH

    @staticmethod
    def _infer_momentum(
        tracked: TrackedFighter,
        frame_h: int,
    ) -> Momentum:
        """Infer fighter's momentum from velocity estimates."""
        vx, vy = tracked.velocity
        speed = np.sqrt(vx ** 2 + vy ** 2)

        # Normalize by frame height
        rel_speed = speed / max(frame_h, 1) * 30  # Scale to be meaningful

        if rel_speed < 0.01:
            return Momentum.STATIONARY

        # Determine primary direction
        if abs(vy) > abs(vx) * 1.5 and vy > 0:
            return Momentum.LEVEL_CHANGING

        if abs(vx) > abs(vy):
            # Lateral movement
            return Momentum.CIRCLING_LEFT if vx < 0 else Momentum.CIRCLING_RIGHT
        else:
            # Forward/backward
            return Momentum.ADVANCING if vy > 0 else Momentum.RETREATING

    def reset(self) -> None:
        """Reset all pipeline state."""
        self.tracker.reset()
        self._pose_buffer_a.clear()
        self._pose_buffer_b.clear()

    def close(self) -> None:
        """Release all resources."""
        self.pose_estimator.close()
        self.reset()
