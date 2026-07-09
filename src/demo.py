"""MMA Corner Agent — command-line interface.

End-to-end demo that processes fight footage (or runs a synthetic
simulation) and outputs corner advice.

Usage:

  # Analyze a video file
  python -m src.demo --video fight_rd1.mp4 --round 1

  # Run the synthetic demo (no video/GPU needed)
  python -m src.demo --synthetic

  # Analyze multiple rounds
  python -m src.demo --video rd1.mp4 rd2.mp4 rd3.mp4

  # Save report to JSON
  python -m src.demo --synthetic --output report.json
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import random
from pathlib import Path

# Force UTF-8 output on Windows (box-drawing characters, emojis)
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace"
    )
    sys.stderr = io.TextIOWrapper(
        sys.stderr.buffer, encoding="utf-8", errors="replace"
    )

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.corner_agent import CornerAgent
from src.analysis.state_vector import (
    CombatState, Stance, RingPosition, Distance, Momentum,
)


# ======================================================================
# Synthetic fight simulation
# ======================================================================

class SyntheticFight:
    """Generates realistic synthetic combat states for demo purposes.

    Simulates a 3-minute round with a narrative arc:
    - Opening minute: feeling out, jabs and leg kicks
    - Middle minute: more aggressive exchanges, counters appear
    - Final minute: fatigue, patterns become more exploitable

    This lets you see the full pipeline in action without any
    video files, GPU, MediaPipe, or YOLO.
    """

    # Action pools with rough frequencies
    FIGHTER_A_ACTIONS = [
        # (action_id, base_weight, phases_active)
        ("jab_head", 5.0, [1, 2, 3]),
        ("cross_head", 3.0, [2, 3]),
        ("lead_hook_head", 2.0, [2, 3]),
        ("rear_body_kick", 2.5, [1, 2, 3]),
        ("lead_calf_kick", 2.0, [1, 2]),
        ("rear_leg_kick", 1.5, [2, 3]),
        ("rear_head_kick", 0.8, [3]),
        ("lead_teep", 1.5, [1, 2]),
        ("rear_uppercut", 1.0, [3]),
        ("double_leg", 0.5, [2, 3]),
        ("jab_body", 1.5, [1, 2, 3]),
    ]

    FIGHTER_B_COUNTERS = {
        # action_id -> [(counter_id, probability, damage_level)]
        "jab_head": [
            ("cross_head", 0.35, "moderate"),
            ("lead_hook_head", 0.15, "significant"),
            ("jab_head", 0.20, None),
        ],
        "cross_head": [
            ("lead_hook_head", 0.25, "significant"),
            ("rear_body_kick", 0.15, "moderate"),
        ],
        "lead_hook_head": [
            ("cross_head", 0.30, "moderate"),
            ("rear_uppercut", 0.10, "significant"),
        ],
        "rear_body_kick": [
            ("lead_hook_head", 0.20, "significant"),
            ("cross_head", 0.15, "moderate"),
        ],
        "lead_calf_kick": [
            ("cross_head", 0.25, "moderate"),
        ],
        "rear_leg_kick": [
            ("lead_teep", 0.20, "glancing"),
        ],
        "lead_teep": [
            ("lead_calf_kick", 0.15, "glancing"),
        ],
        "double_leg": [
            ("lead_uppercut", 0.20, "significant"),
            ("knee_clinch", 0.15, "significant"),
        ],
    }

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def generate_round(
        self,
        round_number: int = 1,
        duration_s: float = 180.0,
        events_per_second: float = 0.8,
    ) -> tuple[list[CombatState], list[CombatState]]:
        """Generate a full round of synthetic combat states.

        Args:
            round_number: Round number (affects fatigue/aggression).
            duration_s: Round duration in seconds.
            events_per_second: Average events per second.

        Returns:
            (fighter_a_states, fighter_b_states) as chronological lists.
        """
        a_states: list[CombatState] = []
        b_states: list[CombatState] = []

        t = 0.0
        fatigue_factor = 1.0 + (round_number - 1) * 0.15

        while t < duration_s:
            # Time between events (with some randomness)
            gap = self.rng.expovariate(events_per_second) * fatigue_factor
            t += gap

            if t >= duration_s:
                break

            # Determine phase (1=opening, 2=middle, 3=closing)
            phase = 1 if t < 60 else (2 if t < 120 else 3)

            # Pick fighter A's action
            action_a = self._pick_action(phase)
            if action_a is None:
                continue

            # Distance and momentum context
            distance = self.rng.choice(
                [Distance.BOXING, Distance.BOXING, Distance.KICKING, Distance.CLINCH]
            )
            momentum_a = self.rng.choice(
                [Momentum.ADVANCING, Momentum.STATIONARY, Momentum.CIRCLING_LEFT]
            )

            a_state = CombatState(
                fighter_id="fighter_a",
                timestamp_s=round(t, 3),
                action_id=action_a,
                action_confidence=0.6 + self.rng.random() * 0.3,
                stance=Stance.ORTHODOX,
                ring_position=self.rng.choice(list(RingPosition)),
                distance=distance,
                momentum=momentum_a,
            )
            a_states.append(a_state)

            # Check if opponent counters
            counter = self._pick_counter(action_a, phase)
            if counter is not None:
                counter_id, damage = counter
                counter_delay = 0.1 + self.rng.random() * 0.4
                landed = self.rng.random() < 0.7  # 70% of counters land

                b_state = CombatState(
                    fighter_id="fighter_b",
                    timestamp_s=round(t + counter_delay, 3),
                    action_id=counter_id,
                    action_confidence=0.5 + self.rng.random() * 0.4,
                    stance=Stance.ORTHODOX,
                    ring_position=a_state.ring_position,
                    distance=distance,
                    momentum=Momentum.ADVANCING if momentum_a == Momentum.RETREATING else Momentum.STATIONARY,
                    strike_landed=landed,
                    damage_level=damage if landed else None,
                )
                b_states.append(b_state)

        return a_states, b_states

    def _pick_action(self, phase: int) -> str | None:
        """Pick a random action for fighter A based on current phase."""
        eligible = [
            (aid, w) for aid, w, phases in self.FIGHTER_A_ACTIONS
            if phase in phases
        ]
        if not eligible:
            return None

        actions, weights = zip(*eligible)
        return self.rng.choices(actions, weights=weights, k=1)[0]

    def _pick_counter(
        self, action_id: str, phase: int
    ) -> tuple[str, str | None] | None:
        """Determine if and how the opponent counters an action.

        Counter probability increases in later phases (opponent reads
        the pattern).
        """
        counters = self.FIGHTER_B_COUNTERS.get(action_id, [])
        if not counters:
            return None

        # Increase counter probability in later phases
        phase_multiplier = 0.8 + (phase - 1) * 0.3

        for counter_id, base_prob, damage in counters:
            prob = base_prob * phase_multiplier
            if self.rng.random() < prob:
                return (counter_id, damage)

        return None


# ======================================================================
# CLI
# ======================================================================

def run_synthetic_demo(
    rounds: int = 3,
    output_path: str | None = None,
    verbose: bool = True,
) -> list[dict]:
    """Run the full pipeline with synthetic data.

    Simulates a multi-round fight and generates corner advice
    between each round.
    """
    agent = CornerAgent(
        min_pattern_occurrences=2,
        max_advice_items=3,
    )
    sim = SyntheticFight(seed=42)
    all_reports: list[dict] = []

    if verbose:
        print()
        print("\u250c" + "\u2500" * 62 + "\u2510")
        print(
            "\u2502"
            + "  MMA CORNER AGENT  \u2014  Synthetic Fight Demo".center(62)
            + "\u2502"
        )
        print("\u2514" + "\u2500" * 62 + "\u2518")
        print()
        print("  Simulating a 3-round fight with synthetic combat states...")
        print("  No video, GPU, or ML models needed for this demo.\n")

    for rd in range(1, rounds + 1):
        if verbose:
            print(f"  {'=' * 58}")
            print(f"  ROUND {rd} {'— FIGHT!' if rd == 1 else '— CONTINUE'}")
            print(f"  {'=' * 58}\n")

        # Generate synthetic round data
        a_states, b_states = sim.generate_round(round_number=rd)

        if verbose:
            print(f"  Generated {len(a_states)} fighter A events, "
                  f"{len(b_states)} fighter B counters")

        # Feed into the corner agent
        agent.start_round(rd)
        agent.ingest_states(a_states, b_states, round_number=rd)
        report = agent.end_round()

        if verbose:
            print(agent.format_report(report))

        all_reports.append(report.to_dict())

    # Fight-level summary
    if verbose and rounds > 1:
        print("\n" + "=" * 64)
        print("  FIGHT SUMMARY — Accumulated patterns across all rounds")
        print("=" * 64)

        fight_patterns = agent.get_fight_patterns()
        fight_advice = agent.get_fight_advice()

        if fight_patterns:
            print(f"\n  Top {len(fight_patterns)} recurring counter patterns:\n")
            for i, p in enumerate(fight_patterns, 1):
                action = agent.kb.get_action(p.your_action)
                counter = agent.kb.get_action(p.their_counter)
                a_name = action["name"] if action else p.your_action
                c_name = counter["name"] if counter else p.their_counter
                print(
                    f"    {i}. {a_name} \u2192 {c_name}  "
                    f"({p.occurrences}x across R{sorted(p.rounds_seen)}, "
                    f"threat={p.threat_score:.1f})"
                )

        if fight_advice:
            print(f"\n  Fight-level adjustments:\n")
            for advice in fight_advice:
                d = advice.to_display_dict()
                print(f"    {d['icon']} {d['headline']}")
                print(f"       {d['adjustment']}")
                print()

    # Save to file if requested
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                {"rounds": all_reports, "rounds_count": rounds},
                f, indent=2, ensure_ascii=False,
            )
        if verbose:
            print(f"\n  Report saved to: {output_path}")

    return all_reports


def run_video_analysis(
    video_paths: list[str],
    output_path: str | None = None,
    target_fps: float = 15.0,
    max_frames: int | None = None,
    verbose: bool = True,
) -> list[dict]:
    """Run the full pipeline on video file(s)."""
    agent = CornerAgent()
    all_reports: list[dict] = []

    for rd, video_path in enumerate(video_paths, 1):
        if not Path(video_path).exists():
            print(f"  ERROR: Video file not found: {video_path}", file=sys.stderr)
            continue

        if verbose:
            print(f"\n  Processing Round {rd}: {video_path}")
            print(f"  {'=' * 50}")

        def progress_cb(frames, ts):
            if frames % 50 == 0:
                print(f"    Frame {frames} | t={ts:.1f}s", end="\r")

        report = agent.analyze_video(
            video_path,
            round_number=rd,
            target_fps=target_fps,
            max_frames=max_frames,
            progress_callback=progress_cb if verbose else None,
        )

        if verbose:
            print()  # Clear progress line
            print(agent.format_report(report))

        all_reports.append(report.to_dict())

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({"rounds": all_reports}, f, indent=2, ensure_ascii=False)
        if verbose:
            print(f"\n  Report saved to: {output_path}")

    return all_reports


def main():
    parser = argparse.ArgumentParser(
        description="MMA Corner Agent — AI-powered corner coach",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.demo --synthetic
  python -m src.demo --synthetic --rounds 5 --output report.json
  python -m src.demo --video fight_rd1.mp4 fight_rd2.mp4
  python -m src.demo --video fight.mp4 --fps 10 --max-frames 500
        """,
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Run synthetic fight simulation (no video/GPU needed)",
    )
    parser.add_argument(
        "--video", nargs="+", metavar="FILE",
        help="Video file(s) to analyze (one per round)",
    )
    parser.add_argument(
        "--rounds", type=int, default=3,
        help="Number of rounds for synthetic demo (default: 3)",
    )
    parser.add_argument(
        "--output", "-o", metavar="FILE",
        help="Save report as JSON to this file",
    )
    parser.add_argument(
        "--fps", type=float, default=15.0,
        help="Target processing FPS for video (default: 15)",
    )
    parser.add_argument(
        "--max-frames", type=int, default=None,
        help="Maximum frames to process per video",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress terminal output",
    )

    args = parser.parse_args()

    if not args.synthetic and not args.video:
        # Default to synthetic
        args.synthetic = True

    if args.synthetic:
        run_synthetic_demo(
            rounds=args.rounds,
            output_path=args.output,
            verbose=not args.quiet,
        )
    elif args.video:
        run_video_analysis(
            video_paths=args.video,
            output_path=args.output,
            target_fps=args.fps,
            max_frames=args.max_frames,
            verbose=not args.quiet,
        )


if __name__ == "__main__":
    main()
