"""MMA Corner Agent — core orchestrator.

Wires the full pipeline end-to-end:

  Video → FighterTracker → PoseEstimator → ActionClassifier
        → CombatState → PairLinker → PatternAggregator
        → AdjustmentEngine → CornerAdvice

Supports three modes:
  1. **Offline video**: Process a recorded fight clip
  2. **Live feed**: Process frames from a camera/stream in real-time
  3. **Synthetic**: Run with generated combat states (no video/GPU needed)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .analysis.state_vector import CombatState, RoundSummary, Momentum, Distance
from .analysis.pair_linker import PairLinker, ActionCounterPair
from .analysis.pattern_aggregator import PatternAggregator, CounterPattern
from .strategy.knowledge_base import KnowledgeBase
from .strategy.adjustment_engine import AdjustmentEngine, CornerAdvice


@dataclass
class RoundReport:
    """Complete analysis report for a single round."""

    round_number: int
    duration_s: float = 0.0
    frames_processed: int = 0

    # Per-fighter state counts
    fighter_a_actions: dict[str, int] = field(default_factory=dict)
    fighter_b_actions: dict[str, int] = field(default_factory=dict)

    # Counter patterns detected
    pairs_detected: int = 0
    top_patterns: list[CounterPattern] = field(default_factory=list)

    # Generated advice
    advice: list[CornerAdvice] = field(default_factory=list)

    # Safe actions ("more of this")
    safe_actions: list[str] = field(default_factory=list)

    # Captured key frames for visualization
    key_frames: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "round_number": self.round_number,
            "duration_s": round(self.duration_s, 2),
            "frames_processed": self.frames_processed,
            "fighter_a_actions": self.fighter_a_actions,
            "fighter_b_actions": self.fighter_b_actions,
            "pairs_detected": self.pairs_detected,
            "top_patterns": [p.to_dict() for p in self.top_patterns],
            "advice": [a.to_display_dict() for a in self.advice],
            "safe_actions": self.safe_actions,
        }


class CornerAgent:
    """AI-powered MMA corner coach.

    The main orchestrator that connects all pipeline components and
    generates actionable corner advice from fight footage.

    Usage (offline video)::

        agent = CornerAgent()
        report = agent.analyze_video("fight_rd1.mp4", round_number=1)
        print(agent.format_report(report))

    Usage (frame-by-frame)::

        agent = CornerAgent()
        for frame in video_frames:
            agent.ingest_frame(frame, frame_idx, timestamp)
        report = agent.end_round()

    Usage (synthetic / pre-computed states)::

        agent = CornerAgent()
        agent.ingest_states(fighter_a_states, fighter_b_states, round_number=1)
        report = agent.end_round()
    """

    def __init__(
        self,
        knowledge_base: Optional[KnowledgeBase] = None,
        counter_window_s: float = 0.6,
        min_pattern_occurrences: int = 2,
        max_advice_items: int = 3,
        top_patterns_count: int = 5,
    ):
        self.kb = knowledge_base or KnowledgeBase()
        self.counter_window_s = counter_window_s
        self.min_pattern_occurrences = min_pattern_occurrences
        self.max_advice_items = max_advice_items
        self.top_patterns_count = top_patterns_count

        # Analysis components
        self.pair_linker = PairLinker(counter_window_s=counter_window_s)
        self.aggregator = PatternAggregator()
        self.adjustment_engine = AdjustmentEngine(knowledge_base=self.kb)

        # State accumulators (per round)
        self._states_a: list[CombatState] = []
        self._states_b: list[CombatState] = []
        self._current_round: int = 1
        self._round_start_time: float = 0.0
        self._frames_processed: int = 0

        # Fight-level accumulators
        self._fight_aggregator = PatternAggregator()
        self._round_reports: list[RoundReport] = []

        # Vision pipeline (lazy init — only needed for video mode)
        self._pipeline = None

    # ------------------------------------------------------------------
    # Video mode
    # ------------------------------------------------------------------

    def analyze_video(
        self,
        video_path: str,
        round_number: int = 1,
        target_fps: float = 15.0,
        max_frames: Optional[int] = None,
        progress_callback=None,
    ) -> RoundReport:
        """Analyze a video file and generate a round report.

        Args:
            video_path: Path to the video file.
            round_number: Which round this video represents.
            target_fps: Process at this frame rate.
            max_frames: Stop after this many frames.
            progress_callback: Called with (frames_processed, timestamp_s).

        Returns:
            RoundReport with detected patterns and corner advice.
        """
        from .vision.pipeline import VideoPipeline, PipelineConfig

        config = PipelineConfig(
            target_fps=target_fps,
            max_frames=max_frames,
        )
        pipeline = VideoPipeline(config=config)
        self._pipeline = pipeline

        self.start_round(round_number)

        for state_a, state_b in pipeline.process_video(video_path, round_number):
            if state_a is not None:
                self._states_a.append(state_a)
            if state_b is not None:
                self._states_b.append(state_b)
            self._frames_processed += 1

            if progress_callback:
                ts = state_a.timestamp_s if state_a else (
                    state_b.timestamp_s if state_b else 0
                )
                progress_callback(self._frames_processed, ts)

        pipeline.close()
        report = self.end_round()
        report.key_frames = getattr(pipeline, "key_frames", [])
        return report

    # ------------------------------------------------------------------
    # Frame-by-frame mode
    # ------------------------------------------------------------------

    def ingest_frame(
        self,
        frame,
        frame_index: int,
        timestamp_s: float,
    ) -> tuple[Optional[CombatState], Optional[CombatState]]:
        """Process a single frame and accumulate combat states.

        Args:
            frame: BGR numpy array (H, W, 3).
            frame_index: Frame number.
            timestamp_s: Seconds from round start.

        Returns:
            (CombatState for fighter_a, CombatState for fighter_b).
        """
        from .vision.pipeline import VideoPipeline

        if self._pipeline is None:
            self._pipeline = VideoPipeline()

        state_a, state_b = self._pipeline.process_frame(
            frame, frame_index, timestamp_s
        )

        if state_a is not None:
            self._states_a.append(state_a)
        if state_b is not None:
            self._states_b.append(state_b)
        self._frames_processed += 1

        return state_a, state_b

    # ------------------------------------------------------------------
    # Synthetic / pre-computed mode
    # ------------------------------------------------------------------

    def ingest_states(
        self,
        fighter_a_states: list[CombatState],
        fighter_b_states: list[CombatState],
        round_number: int = 1,
    ) -> None:
        """Ingest pre-computed combat states directly (no video needed).

        Args:
            fighter_a_states: Chronological states for "your" fighter.
            fighter_b_states: Chronological states for the opponent.
            round_number: Round number for tagging.
        """
        self._current_round = round_number
        self._states_a.extend(fighter_a_states)
        self._states_b.extend(fighter_b_states)
        self._frames_processed += max(len(fighter_a_states), len(fighter_b_states))

    # ------------------------------------------------------------------
    # Round management
    # ------------------------------------------------------------------

    def start_round(self, round_number: int = 1) -> None:
        """Start a new round, resetting per-round accumulators."""
        self._current_round = round_number
        self._states_a.clear()
        self._states_b.clear()
        self._frames_processed = 0
        self._round_start_time = time.time()
        self.pair_linker.reset()
        self.aggregator.reset()

    def end_round(self) -> RoundReport:
        """Finalize the current round and generate the report.

        Runs pair linking, pattern aggregation, and advice generation
        on the accumulated combat states.

        Returns:
            RoundReport with all analysis results.
        """
        # Smooth the combat state sequences to group frame-level predictions into distinct events
        smoothed_a = self._smooth_states(self._states_a)
        smoothed_b = self._smooth_states(self._states_b)

        # Step 1: Link action→counter pairs
        pairs = self.pair_linker.link(
            smoothed_a,
            smoothed_b,
            round_number=self._current_round,
        )

        # Step 2: Aggregate patterns
        self.aggregator.ingest(pairs)
        self._fight_aggregator.ingest(pairs)

        top_patterns = self.aggregator.get_top_patterns(
            n=self.top_patterns_count,
            min_occurrences=self.min_pattern_occurrences,
        )

        # Step 3: Generate corner advice
        advice = self.adjustment_engine.generate_advice(
            top_patterns,
            max_items=self.max_advice_items,
        )

        # Step 4: Identify safe actions
        safe_actions = self.aggregator.get_safe_actions()

        # Step 5: Count actions per fighter
        a_actions = self._count_actions(smoothed_a)
        b_actions = self._count_actions(smoothed_b)

        # Step 6: Compute duration
        timestamps = [s.timestamp_s for s in self._states_a + self._states_b]
        duration = max(timestamps) - min(timestamps) if timestamps else 0.0

        report = RoundReport(
            round_number=self._current_round,
            duration_s=duration,
            frames_processed=self._frames_processed,
            fighter_a_actions=a_actions,
            fighter_b_actions=b_actions,
            pairs_detected=len(pairs),
            top_patterns=top_patterns,
            advice=advice,
            safe_actions=safe_actions,
        )

        self._round_reports.append(report)
        return report

    # ------------------------------------------------------------------
    # Fight-level analysis
    # ------------------------------------------------------------------

    def get_fight_patterns(self) -> list[CounterPattern]:
        """Get accumulated patterns across all rounds of the fight."""
        return self._fight_aggregator.get_top_patterns(
            n=10, min_occurrences=self.min_pattern_occurrences
        )

    def get_fight_advice(self) -> list[CornerAdvice]:
        """Generate advice based on the entire fight so far."""
        patterns = self.get_fight_patterns()
        return self.adjustment_engine.generate_advice(
            patterns, max_items=self.max_advice_items
        )

    def get_all_reports(self) -> list[RoundReport]:
        """Return all round reports generated so far."""
        return list(self._round_reports)

    # ------------------------------------------------------------------
    # Display formatting
    # ------------------------------------------------------------------

    def format_report(self, report: RoundReport) -> str:
        """Format a round report as a rich text block for terminal display."""
        lines: list[str] = []
        w = 64

        # Header
        lines.append("")
        lines.append("\u2554" + "\u2550" * (w - 2) + "\u2557")
        lines.append(
            "\u2551"
            + f"  CORNER ADVICE  |  Round {report.round_number}".center(w - 2)
            + "\u2551"
        )
        lines.append("\u2560" + "\u2550" * (w - 2) + "\u2563")

        # Stats bar
        stats = (
            f"  {report.frames_processed} frames  |  "
            f"{report.duration_s:.1f}s  |  "
            f"{report.pairs_detected} action-counter pairs"
        )
        lines.append("\u2551" + stats.ljust(w - 2) + "\u2551")
        lines.append("\u2560" + "\u2550" * (w - 2) + "\u2563")

        # Advice items
        if report.advice:
            for advice in report.advice:
                d = advice.to_display_dict()
                severity_line = f"  {d['icon']} {d['severity'].upper()}"
                lines.append("\u2551" + severity_line.ljust(w - 2) + "\u2551")

                headline = f"    {d['headline']}"
                lines.append("\u2551" + headline.ljust(w - 2) + "\u2551")

                stat_line = f"    {d['stats']}"
                lines.append("\u2551" + stat_line.ljust(w - 2) + "\u2551")

                adj_line = f"    {d['adjustment']}"
                # Wrap long adjustment lines
                while len(adj_line) > w - 3:
                    lines.append("\u2551" + adj_line[:w - 3].ljust(w - 2) + "\u2551")
                    adj_line = "      " + adj_line[w - 3:]
                lines.append("\u2551" + adj_line.ljust(w - 2) + "\u2551")

                lines.append("\u2551" + " " * (w - 2) + "\u2551")
        else:
            lines.append(
                "\u2551" + "  No significant counter patterns detected.".ljust(w - 2) + "\u2551"
            )
            lines.append("\u2551" + " " * (w - 2) + "\u2551")

        # Safe actions
        if report.safe_actions:
            lines.append("\u2560" + "\u2550" * (w - 2) + "\u2563")
            lines.append(
                "\u2551"
                + "  \u2705 MORE OF THIS:".ljust(w - 2)
                + "\u2551"
            )
            for action_id in report.safe_actions:
                action = self.kb.get_action(action_id)
                name = action["name"] if action else action_id
                lines.append(
                    "\u2551" + f"    \u2022 {name}".ljust(w - 2) + "\u2551"
                )

        # Action breakdown
        lines.append("\u2560" + "\u2550" * (w - 2) + "\u2563")
        lines.append(
            "\u2551" + "  ACTION BREAKDOWN:".ljust(w - 2) + "\u2551"
        )

        if report.fighter_a_actions:
            lines.append(
                "\u2551" + "  Your fighter:".ljust(w - 2) + "\u2551"
            )
            for action_id, count in sorted(
                report.fighter_a_actions.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:6]:
                action = self.kb.get_action(action_id)
                name = action["name"] if action else action_id
                bar = "\u2588" * min(count, 20)
                line = f"    {name:<24} {bar} {count}"
                lines.append("\u2551" + line.ljust(w - 2) + "\u2551")

        if report.fighter_b_actions:
            lines.append(
                "\u2551" + "  Opponent:".ljust(w - 2) + "\u2551"
            )
            for action_id, count in sorted(
                report.fighter_b_actions.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:6]:
                action = self.kb.get_action(action_id)
                name = action["name"] if action else action_id
                bar = "\u2588" * min(count, 20)
                line = f"    {name:<24} {bar} {count}"
                lines.append("\u2551" + line.ljust(w - 2) + "\u2551")

        # Footer
        lines.append("\u255a" + "\u2550" * (w - 2) + "\u255d")
        lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _count_actions(states: list[CombatState]) -> dict[str, int]:
        """Count occurrences of each action in a state list."""
        counts: dict[str, int] = {}
        prev_action: Optional[str] = None

        for state in states:
            if state.action_id and state.action_id != prev_action:
                counts[state.action_id] = counts.get(state.action_id, 0) + 1
            prev_action = state.action_id

        # Remove idle/None from the counts
        counts.pop("idle", None)
        return counts

    @staticmethod
    def _smooth_states(
        states: list[CombatState],
        min_event_duration_s: float = 0.25,
        max_gap_to_merge_s: float = 0.6,
    ) -> list[CombatState]:
        """Smooth a sequence of CombatStates to remove frame-level jitter/flutter.

        Fills in brief gaps in contiguous actions and filters out short transient
        action spikes (noise). Returns a new copy of the state sequence.
        """
        if not states:
            return []

        # Estimate average FPS from timestamps
        if len(states) >= 2:
            time_gaps = [
                states[i].timestamp_s - states[i - 1].timestamp_s
                for i in range(1, min(100, len(states)))
            ]
            mean_gap = sum(time_gaps) / len(time_gaps)
            fps = 1.0 / mean_gap if mean_gap > 0 else 15.0
        else:
            fps = 15.0

        min_frames = max(1, int(min_event_duration_s * fps))
        gap_frames = max(1, int(max_gap_to_merge_s * fps))

        import copy
        smoothed = [copy.copy(s) for s in states]

        # Step 1: Fill short gaps in contiguous actions
        i = 0
        while i < len(smoothed):
            act = smoothed[i].action_id
            if not act or act == "idle":
                i += 1
                continue

            # Look ahead for the next occurrence of the same action
            next_occ_idx = -1
            for j in range(i + 1, min(i + gap_frames + 2, len(smoothed))):
                if smoothed[j].action_id == act:
                    next_occ_idx = j
                    break

            if next_occ_idx != -1:
                # Fill the gap with the action
                for k in range(i + 1, next_occ_idx):
                    smoothed[k].action_id = act
                    smoothed[k].action_confidence = max(
                        smoothed[k].action_confidence,
                        smoothed[i].action_confidence,
                        smoothed[next_occ_idx].action_confidence,
                    )
                i = next_occ_idx
            else:
                i += 1

        # Step 2: Remove short transient spikes (less than min_frames duration)
        i = 0
        while i < len(smoothed):
            act = smoothed[i].action_id
            if not act or act == "idle":
                i += 1
                continue

            start_idx = i
            while i < len(smoothed) and smoothed[i].action_id == act:
                i += 1
            duration_frames = i - start_idx

            if duration_frames < min_frames:
                for k in range(start_idx, i):
                    smoothed[k].action_id = None
                    smoothed[k].action_confidence = 0.0

        return smoothed
