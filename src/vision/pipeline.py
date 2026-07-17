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


import cv2

@dataclass
class FrameResult:
    """Complete analysis result for a single video frame."""

    frame_index: int
    timestamp_s: float
    fighters: list[TrackedFighter]
    poses: dict[str, Optional[PoseResult]]   # fighter_id -> PoseResult
    actions: dict[str, Optional[ActionPrediction]]  # fighter_id -> latest action


@dataclass
class KeyFrame:
    """A captured video frame with annotation for display."""
    frame_index: int
    timestamp_s: float
    image: np.ndarray  # BGR image (original resolution)
    fighter_a_action: Optional[str] = None
    fighter_b_action: Optional[str] = None
    fighter_a_confidence: float = 0.0
    fighter_b_confidence: float = 0.0
    reason: str = ""  # Why this frame was captured
    bboxes: list[tuple[int, int, int, int]] = field(default_factory=list)


@dataclass
class PipelineConfig:
    """Configuration for the video processing pipeline."""

    # Video processing
    target_fps: float = 15.0           # Process at this FPS (skip frames)
    max_frames: Optional[int] = None   # Stop after this many frames
    inference_width: int = 640         # Resize to this width for inference (0 or None to disable)
    capture_key_frames: bool = True     # Capture key action frames

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

        # Key frame capture tracking
        self.key_frames: list[KeyFrame] = []
        self._last_capture_ts: float = -999.0
        self._capture_cooldown: float = 1.0  # Capture at most once per second
        self._max_key_frames: int = 20

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
                # Skip frames to match target FPS using cap.grab() (fast skipping)
                if frame_idx % frame_skip != 0:
                    if not cap.grab():
                        break
                    frame_idx += 1
                    continue

                ret, frame = cap.read()
                if not ret:
                    break

                timestamp = frame_idx / source_fps

                # Check frame limit
                if (
                    self.config.max_frames is not None
                    and processed >= self.config.max_frames
                ):
                    break

                # Resize for faster inference while preserving aspect ratio
                h, w = frame.shape[:2]
                inf_w = self.config.inference_width
                if inf_w and w > inf_w:
                    scale = inf_w / w
                    inference_frame = cv2.resize(
                        frame,
                        (inf_w, int(h * scale)),
                        interpolation=cv2.INTER_LINEAR
                    )
                else:
                    inference_frame = frame

                # Run pipeline on this frame, passing original frame for key frame capture
                state_a, state_b = self.process_frame(
                    inference_frame, frame_idx, timestamp, original_frame=frame
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
        original_frame: Optional[np.ndarray] = None,
    ) -> tuple[Optional[CombatState], Optional[CombatState]]:
        """Process a single frame through the full pipeline.

        Args:
            frame: BGR image (H, W, 3) (potentially resized for inference).
            frame_index: Frame number.
            timestamp_s: Timestamp in seconds from round start.
            original_frame: Original full-resolution BGR frame.

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

        # Compute scaling between inference frame and original frame
        if original_frame is not None:
            orig_w = original_frame.shape[1]
            inf_w = frame.shape[1]
            scale = orig_w / inf_w if inf_w > 0 else 1.0
        else:
            original_frame = frame
            scale = 1.0

        # Maybe capture key frame for visualization
        self._maybe_capture_frame(
            original_frame, frame_index, timestamp_s,
            state_a, state_b, fa, fb, scale
        )

        return state_a, state_b

    def _maybe_capture_frame(
        self,
        original_frame: np.ndarray,
        frame_index: int,
        timestamp_s: float,
        state_a: Optional[CombatState],
        state_b: Optional[CombatState],
        fa: Optional[TrackedFighter],
        fb: Optional[TrackedFighter],
        scale: float,
    ) -> None:
        """Capture a key frame if a high-confidence action occurred."""
        if not self.config.capture_key_frames:
            return
        if len(self.key_frames) >= self._max_key_frames:
            return
        if timestamp_s - self._last_capture_ts < self._capture_cooldown:
            return

        # Check if either fighter performed a significant action
        a_act = state_a.action_id if state_a else None
        b_act = state_b.action_id if state_b else None
        a_conf = state_a.action_confidence if state_a else 0.0
        b_conf = state_b.action_confidence if state_b else 0.0

        # Don't capture idle/neutral/simple block poses
        def is_significant(act, conf):
            return act and act not in ("idle", "high_block") and conf >= self.config.action_confidence_threshold

        has_a = is_significant(a_act, a_conf)
        has_b = is_significant(b_act, b_conf)

        if not (has_a or has_b):
            return

        # Make copy of original frame to draw annotation overlays
        annotated = original_frame.copy()
        h, w = annotated.shape[:2]

        bboxes = []

        # Helper to format action names cleanly
        def clean_name(name):
            return name.replace("_", " ").title() if name else "Idle"

        # Draw Fighter A (Blue-ish)
        if fa and fa.bbox:
            x1, y1, x2, y2 = [int(coord * scale) for coord in fa.bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            bboxes.append((x1, y1, x2, y2))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (225, 153, 66), 2)
            label = f"A: {clean_name(a_act)} ({a_conf:.0%})"
            cv2.putText(
                annotated,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (225, 153, 66),
                2,
                cv2.LINE_AA,
            )

        # Draw Fighter B (Red-ish)
        if fb and fb.bbox:
            x1, y1, x2, y2 = [int(coord * scale) for coord in fb.bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            bboxes.append((x1, y1, x2, y2))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (66, 62, 225), 2)
            label = f"B: {clean_name(b_act)} ({b_conf:.0%})"
            cv2.putText(
                annotated,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (66, 62, 225),
                2,
                cv2.LINE_AA,
            )

        # Global time overlay
        time_text = f"Round Time: {timestamp_s:.1f}s | Frame: {frame_index}"
        cv2.putText(
            annotated,
            time_text,
            (15, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        reason = ""
        if has_a and has_b:
            reason = f"Exchange: A({clean_name(a_act)}) & B({clean_name(b_act)})"
        elif has_a:
            reason = f"A Threw: {clean_name(a_act)}"
        else:
            reason = f"B Threw: {clean_name(b_act)}"

        kf = KeyFrame(
            frame_index=frame_index,
            timestamp_s=timestamp_s,
            image=annotated,
            fighter_a_action=a_act,
            fighter_b_action=b_act,
            fighter_a_confidence=a_conf,
            fighter_b_confidence=b_conf,
            reason=reason,
            bboxes=bboxes,
        )
        self.key_frames.append(kf)
        self._last_capture_ts = timestamp_s

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
        body_h = np.ptp(pose.to_numpy()[:, 1])
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
