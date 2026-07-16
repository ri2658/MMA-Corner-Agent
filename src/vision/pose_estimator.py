"""Pose estimation pipeline.

Extracts skeletal keypoints from video frames using MediaPipe Pose.
Handles multi-person detection by running pose estimation within
bounding boxes provided by the fighter tracker.
"""

from dataclasses import dataclass
import warnings
from typing import Optional

import numpy as np


# MediaPipe Pose landmark indices (33 landmarks total).
# We define the subset most relevant for MMA action recognition.
class LandmarkIndex:
    """MediaPipe Pose landmark indices for quick reference."""

    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32

    # Grouped for MMA-relevant analysis
    UPPER_BODY = [
        NOSE, LEFT_SHOULDER, RIGHT_SHOULDER,
        LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST,
    ]
    LOWER_BODY = [
        LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE,
        LEFT_ANKLE, RIGHT_ANKLE,
    ]
    HANDS = [LEFT_WRIST, RIGHT_WRIST, LEFT_INDEX, RIGHT_INDEX]
    FEET = [LEFT_ANKLE, RIGHT_ANKLE, LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX]


@dataclass
class PoseResult:
    """Result of pose estimation for a single fighter in a single frame."""

    fighter_id: str
    frame_index: int
    timestamp_s: float
    keypoints: list[tuple[float, float, float]]  # (x, y, visibility) per joint
    confidence: float
    bbox: Optional[tuple[int, int, int, int]] = None  # Source bounding box

    @property
    def num_keypoints(self) -> int:
        return len(self.keypoints)

    def to_numpy(self) -> np.ndarray:
        """Convert keypoints to a numpy array of shape (N, 3)."""
        return np.array(self.keypoints, dtype=np.float32)

    def get_landmark(self, idx: int) -> tuple[float, float, float]:
        """Get a specific landmark by its MediaPipe index."""
        if idx < len(self.keypoints):
            return self.keypoints[idx]
        return (0.0, 0.0, 0.0)

    def get_point_2d(self, idx: int) -> tuple[float, float]:
        """Get just the (x, y) coordinates of a landmark."""
        kp = self.get_landmark(idx)
        return (kp[0], kp[1])

    def mean_visibility(self, indices: Optional[list[int]] = None) -> float:
        """Average visibility across all or specified landmarks."""
        if indices is None:
            indices = list(range(len(self.keypoints)))
        vis = [self.keypoints[i][2] for i in indices if i < len(self.keypoints)]
        return sum(vis) / max(len(vis), 1)


class PoseEstimator:
    """Extracts pose keypoints from video frames using MediaPipe.

    Uses the MediaPipe Tasks API (PoseLandmarker) which replaced the
    legacy ``mediapipe.solutions.pose`` module in MediaPipe >= 0.10.

    Runs within bounding boxes provided by the fighter tracker,
    producing per-fighter keypoint results for each frame.

    Usage:
        estimator = PoseEstimator()
        result = estimator.estimate(frame, bbox, fighter_id="fighter_a", timestamp_s=12.5)
    """

    # URL for the MediaPipe pose model
    _MODEL_URL = (
        "https://storage.googleapis.com/mediapipe-models/"
        "pose_landmarker/pose_landmarker_lite/float16/latest/"
        "pose_landmarker_lite.task"
    )

    def __init__(
        self,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        static_image_mode: bool = False,
        model_path: str | None = None,
    ):
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.static_image_mode = static_image_mode
        self._model_path = model_path
        self._landmarker = None  # Lazy init
        self._pose_disabled = False

    def _find_model_path(self) -> str:
        """Locate or download the pose landmarker model file."""
        import os
        from pathlib import Path

        # Check explicit path
        if self._model_path and os.path.isfile(self._model_path):
            return self._model_path

        # Check common locations relative to project root
        project_root = Path(__file__).resolve().parent.parent.parent
        candidates = [
            project_root / "models" / "pose_landmarker_lite.task",
            project_root / "models" / "pose_landmarker_full.task",
            project_root / "models" / "pose_landmarker_heavy.task",
            Path.home() / ".mediapipe" / "pose_landmarker_lite.task",
        ]

        for p in candidates:
            if p.exists():
                return str(p)

        # Download the model
        model_dir = project_root / "models"
        model_dir.mkdir(exist_ok=True)
        model_file = model_dir / "pose_landmarker_lite.task"

        print(f"Downloading pose model to {model_file} ...")
        import urllib.request
        urllib.request.urlretrieve(self._MODEL_URL, str(model_file))
        print("Download complete.")

        return str(model_file)

    def _init_model(self):
        """Initialize the MediaPipe PoseLandmarker on first use."""
        try:
            from mediapipe.tasks.python import BaseOptions
            from mediapipe.tasks.python.vision import (
                PoseLandmarker,
                PoseLandmarkerOptions,
                RunningMode,
            )

            model_path = self._find_model_path()

            options = PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            )
            self._landmarker = PoseLandmarker.create_from_options(options)

        except Exception as exc:
            self._landmarker = None
            self._pose_disabled = True
            warnings.warn(
                f"Pose estimation disabled: {exc}. "
                f"Video analysis will continue without pose keypoints.",
                RuntimeWarning,
                stacklevel=2,
            )

    def estimate(
        self,
        frame: np.ndarray,
        bbox: tuple[int, int, int, int] | None = None,
        fighter_id: str = "unknown",
        frame_index: int = 0,
        timestamp_s: float = 0.0,
    ) -> PoseResult | None:
        """Run pose estimation on a single frame (or cropped region).

        Args:
            frame: BGR image as numpy array (H, W, 3).
            bbox: Optional (x1, y1, x2, y2) bounding box to crop to.
                  If provided, pose estimation runs only within this region
                  and keypoints are mapped back to full-frame coordinates.
            fighter_id: ID string for this fighter.
            frame_index: Frame number in the video.
            timestamp_s: Timestamp in seconds.

        Returns:
            PoseResult if a pose is detected, None otherwise.
        """
        if self._landmarker is None and not self._pose_disabled:
            self._init_model()

        if self._landmarker is None:
            return None

        import cv2
        import mediapipe as mp

        # Crop to bounding box if provided
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                return None
            crop_h, crop_w = crop.shape[:2]
        else:
            crop = frame
            x1, y1 = 0, 0
            crop_h, crop_w = frame.shape[:2]

        # Convert BGR to RGB and create MediaPipe Image
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Run pose detection
        result = self._landmarker.detect(mp_image)

        if not result.pose_landmarks or len(result.pose_landmarks) == 0:
            return None

        # Use the first detected pose
        landmarks = result.pose_landmarks[0]

        # Extract keypoints and map back to full-frame coords
        keypoints: list[tuple[float, float, float]] = []
        for lm in landmarks:
            # Tasks API gives normalized coords [0, 1] relative to the crop
            abs_x = lm.x * crop_w + x1
            abs_y = lm.y * crop_h + y1
            vis = lm.visibility if hasattr(lm, "visibility") else 0.5
            keypoints.append((abs_x, abs_y, vis))

        # Compute overall confidence from visibility of key landmarks
        key_indices = LandmarkIndex.UPPER_BODY + LandmarkIndex.LOWER_BODY
        avg_vis = sum(
            keypoints[i][2] for i in key_indices if i < len(keypoints)
        ) / len(key_indices)

        return PoseResult(
            fighter_id=fighter_id,
            frame_index=frame_index,
            timestamp_s=timestamp_s,
            keypoints=keypoints,
            confidence=avg_vis,
            bbox=bbox,
        )

    def estimate_both(
        self,
        frame: np.ndarray,
        bbox_a: tuple[int, int, int, int],
        bbox_b: tuple[int, int, int, int],
        frame_index: int = 0,
        timestamp_s: float = 0.0,
    ) -> tuple[PoseResult | None, PoseResult | None]:
        """Run pose estimation on both fighters in a single frame.

        Args:
            frame: Full BGR frame.
            bbox_a: Bounding box for fighter A.
            bbox_b: Bounding box for fighter B.
            frame_index: Frame number.
            timestamp_s: Timestamp in seconds.

        Returns:
            Tuple of (PoseResult for A, PoseResult for B). Either may be None.
        """
        pose_a = self.estimate(
            frame, bbox_a, "fighter_a", frame_index, timestamp_s
        )
        pose_b = self.estimate(
            frame, bbox_b, "fighter_b", frame_index, timestamp_s
        )
        return pose_a, pose_b

    def close(self):
        """Release MediaPipe resources."""
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def __del__(self):
        self.close()

