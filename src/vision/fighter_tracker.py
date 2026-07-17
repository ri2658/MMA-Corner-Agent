"""Fighter tracker.

Tracks and identifies individual fighters across video frames using
YOLO for person detection and IoU-based association for consistent
identity tracking. Maintains exactly two fighter IDs ("fighter_a" and
"fighter_b") throughout the fight.

Re-identification uses a combination of:
  1. IoU (Intersection over Union) between predicted and detected boxes
  2. Color histogram similarity (shorts/glove colors) for recovery
     after occlusion or camera cuts
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class TrackedFighter:
    """State of a tracked fighter across frames."""

    fighter_id: str                                   # "fighter_a" or "fighter_b"
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)   # (x1, y1, x2, y2)
    confidence: float = 0.0
    last_seen_frame: int = 0
    velocity: tuple[float, float] = (0.0, 0.0)       # Estimated (dx, dy) per frame
    color_histogram: Optional[np.ndarray] = None      # HSV histogram for re-id
    _prev_center: Optional[tuple[float, float]] = field(
        default=None, repr=False
    )

    @property
    def center(self) -> tuple[float, float]:
        """Center point of the bounding box."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def area(self) -> int:
        """Area of the bounding box in pixels."""
        x1, y1, x2, y2 = self.bbox
        return max(0, x2 - x1) * max(0, y2 - y1)

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]

    def predicted_center(self) -> tuple[float, float]:
        """Predict next center using simple linear velocity model."""
        cx, cy = self.center
        return (cx + self.velocity[0], cy + self.velocity[1])


def _iou(box_a: tuple, box_b: tuple) -> float:
    """Compute Intersection over Union between two (x1, y1, x2, y2) boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def _compute_color_histogram(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
    bins: int = 32,
) -> np.ndarray:
    """Compute an HSV color histogram for the torso region of a bbox.

    Focuses on the middle third vertically (torso/shorts area) which
    is most distinctive between fighters.
    """
    import cv2

    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    # Focus on the middle third (torso/shorts)
    box_h = y2 - y1
    torso_y1 = y1 + box_h // 3
    torso_y2 = y2 - box_h // 6

    crop = frame[torso_y1:torso_y2, x1:x2]
    if crop.size == 0:
        return np.zeros(bins * 2, dtype=np.float32)

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # Hue and Saturation histograms (ignore Value for lighting robustness)
    hist_h = cv2.calcHist([hsv], [0], None, [bins], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [bins], [0, 256])

    hist = np.concatenate([hist_h, hist_s]).flatten()
    hist = hist / (hist.sum() + 1e-7)  # Normalize
    return hist.astype(np.float32)


def _histogram_distance(h1: np.ndarray, h2: np.ndarray) -> float:
    """Compute Bhattacharyya-like distance between two histograms.

    Returns a value in [0, 1] where 0 = identical, 1 = completely different.
    """
    if h1 is None or h2 is None:
        return 1.0
    bc = np.sum(np.sqrt(h1 * h2))  # Bhattacharyya coefficient
    return 1.0 - bc


class FighterTracker:
    """Tracks two fighters across video frames.

    Uses YOLO for person detection, then associates detections with
    tracked fighter IDs using IoU overlap and color histogram similarity.

    Handles:
    - Consistent identity across frames
    - Brief occlusions (clinch, camera cuts)
    - Re-identification after track loss using color appearance

    Usage:
        tracker = FighterTracker()
        fighters = tracker.update(frame, frame_index=42)
    """

    def __init__(
        self,
        detector_model: str = "yolov8n.pt",
        person_class_id: int = 0,
        min_detection_confidence: float = 0.5,
        max_lost_frames: int = 30,
        iou_threshold: float = 0.3,
        histogram_bins: int = 32,
    ):
        self.detector_model = detector_model
        self.person_class_id = person_class_id
        self.min_detection_confidence = min_detection_confidence
        self.max_lost_frames = max_lost_frames
        self.iou_threshold = iou_threshold
        self.histogram_bins = histogram_bins

        self._fighters: dict[str, TrackedFighter] = {}
        self._detector = None  # Lazy init
        self._initialized = False
        self._frame_count = 0

    def _init_detector(self):
        """Initialize the YOLO person detector on first use."""
        from ultralytics import YOLO

        self._detector = YOLO(self.detector_model)

    def _is_referee_or_official(
        self, frame: np.ndarray, bbox: tuple[int, int, int, int]
    ) -> bool:
        """Detect if a detection is a referee or official based on low torso HSV saturation.

        Referees typically wear black, grey, or black/white shirts, which have
        extremely low color saturation compared to fighters' skin tone or colored shorts.
        """
        import cv2

        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        # Torso area is roughly the upper-middle vertical segment
        box_h = y2 - y1
        torso_y1 = y1 + box_h // 5
        torso_y2 = y1 + box_h // 2

        crop = frame[torso_y1:torso_y2, x1:x2]
        if crop.size == 0:
            return False

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        # Average Saturation (S channel is index 1)
        mean_s = np.mean(hsv[:, :, 1])

        # Dark shirts/stripes have saturation typically < 25.
        # Skin tones have saturation > 35.
        if mean_s < 25:
            return True
        return False

    def _detect_persons(
        self, frame: np.ndarray
    ) -> list[tuple[tuple[int, int, int, int], float]]:
        """Detect all persons in a frame using YOLO, filtering out referees/officials.

        Returns:
            List of ((x1, y1, x2, y2), confidence) tuples, sorted by area descending.
        """
        if self._detector is None:
            self._init_detector()

        results = self._detector(frame, verbose=False)

        detections: list[tuple[tuple[int, int, int, int], float]] = []
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                if cls == self.person_class_id and conf >= self.min_detection_confidence:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    bbox = (int(x1), int(y1), int(x2), int(y2))

                    # Filter out referee/officials based on torso clothing colors
                    if self._is_referee_or_official(frame, bbox):
                        continue

                    detections.append((bbox, conf))

        # Sort by area descending — fighters are typically the largest persons
        detections.sort(key=lambda d: (d[0][2] - d[0][0]) * (d[0][3] - d[0][1]), reverse=True)

        # Keep up to 5 detections for track association
        return detections[:5]

    def update(
        self,
        frame: np.ndarray,
        frame_index: int = 0,
    ) -> list[TrackedFighter]:
        """Detect and track fighters in a new frame.

        On the first frame with 2 detections, assigns fighter_a (left) and
        fighter_b (right). On subsequent frames, uses IoU-based association
        with color histogram fallback.

        Args:
            frame: BGR image as numpy array (H, W, 3).
            frame_index: Frame number in the video.

        Returns:
            List of TrackedFighter objects (up to 2).
        """
        self._frame_count = frame_index
        detections = self._detect_persons(frame)

        if not detections:
            return list(self._fighters.values())

        if not self._initialized:
            return self._initialize_tracks(frame, detections, frame_index)

        return self._associate_tracks(frame, detections, frame_index)

    def _initialize_tracks(
        self,
        frame: np.ndarray,
        detections: list[tuple[tuple[int, int, int, int], float]],
        frame_index: int,
    ) -> list[TrackedFighter]:
        """Initialize fighter tracks from the first frame with detections.

        Assigns fighter_a to the leftmost detection and fighter_b to the
        rightmost. This matches the typical broadcast convention where
        fighters are introduced from their respective corners.
        """
        if len(detections) < 2:
            # Need both fighters to initialize
            return []

        # Sort by x-center: leftmost is fighter_a, rightmost is fighter_b
        sorted_dets = sorted(detections, key=lambda d: (d[0][0] + d[0][2]) / 2)

        # Initialize fighter_a (leftmost) and fighter_b (rightmost)
        # This naturally skips the referee if they are standing in the middle
        targets = [("fighter_a", sorted_dets[0]), ("fighter_b", sorted_dets[-1])]

        for fighter_id, (bbox, conf) in targets:
            hist = _compute_color_histogram(frame, bbox, self.histogram_bins)
            self._fighters[fighter_id] = TrackedFighter(
                fighter_id=fighter_id,
                bbox=bbox,
                confidence=conf,
                last_seen_frame=frame_index,
                color_histogram=hist,
            )

        self._initialized = True
        return list(self._fighters.values())

    def _associate_tracks(
        self,
        frame: np.ndarray,
        detections: list[tuple[tuple[int, int, int, int], float]],
        frame_index: int,
    ) -> list[TrackedFighter]:
        """Associate new detections with existing tracked fighters.

        Strategy:
        1. Compute IoU between each detection and each tracked fighter's
           predicted position.
        2. If IoU is above threshold, assign directly.
        3. If IoU fails (occlusion recovery), fall back to color histogram
           distance.
        4. Update velocity estimates for next-frame prediction.
        """
        fighter_ids = list(self._fighters.keys())
        det_bboxes = [d[0] for d in detections]
        det_confs = [d[1] for d in detections]

        # Build IoU cost matrix
        n_tracks = len(fighter_ids)
        n_dets = len(detections)
        iou_matrix = np.zeros((n_tracks, n_dets))

        for i, fid in enumerate(fighter_ids):
            fighter = self._fighters[fid]
            # Use predicted center to build predicted bbox
            pred_cx, pred_cy = fighter.predicted_center()
            hw, hh = fighter.width / 2, fighter.height / 2
            pred_bbox = (
                int(pred_cx - hw), int(pred_cy - hh),
                int(pred_cx + hw), int(pred_cy + hh),
            )
            for j, det_bbox in enumerate(det_bboxes):
                iou_matrix[i, j] = _iou(pred_bbox, det_bbox)

        # Greedy assignment by highest IoU
        assigned_dets: set[int] = set()
        assigned_tracks: set[int] = set()

        while True:
            if iou_matrix.size == 0:
                break
            best_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            best_iou = iou_matrix[best_idx]

            if best_iou < self.iou_threshold:
                break

            track_i, det_j = best_idx
            self._update_fighter(
                fighter_ids[track_i], det_bboxes[det_j],
                det_confs[det_j], frame, frame_index,
            )
            assigned_dets.add(det_j)
            assigned_tracks.add(track_i)

            # Remove assigned row/col from consideration
            iou_matrix[track_i, :] = -1
            iou_matrix[:, det_j] = -1

        # Handle unassigned detections via color histogram fallback
        unassigned_tracks = set(range(n_tracks)) - assigned_tracks
        unassigned_dets = set(range(n_dets)) - assigned_dets

        if unassigned_tracks and unassigned_dets:
            for track_i in unassigned_tracks:
                fid = fighter_ids[track_i]
                fighter = self._fighters[fid]
                if fighter.color_histogram is None:
                    continue

                best_det = None
                best_dist = float("inf")

                for det_j in unassigned_dets:
                    det_hist = _compute_color_histogram(
                        frame, det_bboxes[det_j], self.histogram_bins
                    )
                    dist = _histogram_distance(fighter.color_histogram, det_hist)
                    if dist < best_dist:
                        best_dist = dist
                        best_det = det_j

                if best_det is not None and best_dist < 0.6:
                    self._update_fighter(
                        fid, det_bboxes[best_det],
                        det_confs[best_det], frame, frame_index,
                    )
                    unassigned_dets.discard(best_det)

        # Prune tracks that haven't been seen for too long
        lost = [
            fid for fid, f in self._fighters.items()
            if frame_index - f.last_seen_frame > self.max_lost_frames
        ]
        for fid in lost:
            del self._fighters[fid]

        return list(self._fighters.values())

    def _update_fighter(
        self,
        fighter_id: str,
        bbox: tuple[int, int, int, int],
        confidence: float,
        frame: np.ndarray,
        frame_index: int,
    ) -> None:
        """Update a tracked fighter with a new detection."""
        fighter = self._fighters[fighter_id]

        # Compute velocity from center shift
        old_cx, old_cy = fighter.center
        new_cx = (bbox[0] + bbox[2]) / 2.0
        new_cy = (bbox[1] + bbox[3]) / 2.0
        dt = max(1, frame_index - fighter.last_seen_frame)
        fighter.velocity = (
            (new_cx - old_cx) / dt,
            (new_cy - old_cy) / dt,
        )

        fighter.bbox = bbox
        fighter.confidence = confidence
        fighter.last_seen_frame = frame_index

        # Update color histogram periodically (every ~30 frames)
        if frame_index % 30 == 0:
            fighter.color_histogram = _compute_color_histogram(
                frame, bbox, self.histogram_bins
            )

    def get_fighter(self, fighter_id: str) -> Optional[TrackedFighter]:
        """Get the current state of a specific fighter."""
        return self._fighters.get(fighter_id)

    def get_distance_between_fighters(self) -> Optional[float]:
        """Compute pixel distance between the two fighters' centers.

        Returns None if both fighters aren't currently tracked.
        """
        fa = self._fighters.get("fighter_a")
        fb = self._fighters.get("fighter_b")
        if fa is None or fb is None:
            return None

        ax, ay = fa.center
        bx, by = fb.center
        return float(np.sqrt((ax - bx) ** 2 + (ay - by) ** 2))

    def reset(self) -> None:
        """Reset all tracking state."""
        self._fighters.clear()
        self._initialized = False
        self._frame_count = 0
