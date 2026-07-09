"""Temporal pair linker.

Associates Fighter A's actions with Fighter B's counters based on
temporal proximity. This is the component that detects action->reaction
pairs from raw combat state sequences.
"""

from dataclasses import dataclass
from typing import Optional

from .state_vector import CombatState


# Default timing window: a counter must start within this many seconds
# of the initial action for it to be considered a response.
_DEFAULT_COUNTER_WINDOW_S = 0.6

# Minimum gap between an action ending and a counter starting.
_MIN_GAP_S = 0.05


@dataclass
class ActionCounterPair:
    """A detected (action -> counter -> outcome) event."""

    trigger_state: CombatState             # The initiating action
    counter_state: CombatState             # The opponent's counter
    time_delta_s: float                    # Gap between action and counter
    outcome: Optional[str] = None          # "landed", "blocked", "missed"
    damage: Optional[str] = None           # "significant", "moderate", "glancing"
    round_number: int = 0

    @property
    def pair_key(self) -> str:
        """A string key representing this action->counter type."""
        return f"{self.trigger_state.action_id}->{self.counter_state.action_id}"

    def to_dict(self) -> dict:
        """Serialize for logging and aggregation."""
        return {
            "your_action": self.trigger_state.action_id,
            "their_counter": self.counter_state.action_id,
            "timestamp": self.trigger_state.timestamp_s,
            "time_delta_s": self.time_delta_s,
            "outcome": self.outcome,
            "damage": self.damage,
            "round": self.round_number,
        }


class PairLinker:
    """Links fighter actions to opponent counters using temporal proximity.

    Given two streams of CombatState objects (one per fighter), identifies
    moments where Fighter B's action is a direct response to Fighter A's
    action within a configurable time window.
    """

    def __init__(
        self,
        counter_window_s: float = _DEFAULT_COUNTER_WINDOW_S,
        min_gap_s: float = _MIN_GAP_S,
    ):
        self.counter_window_s = counter_window_s
        self.min_gap_s = min_gap_s
        self.detected_pairs: list[ActionCounterPair] = []

    def link(
        self,
        fighter_a_states: list[CombatState],
        fighter_b_states: list[CombatState],
        round_number: int = 0,
    ) -> list[ActionCounterPair]:
        """Find all action->counter pairs between two fighters.

        For each action by Fighter A, search for the nearest subsequent
        action by Fighter B that falls within the counter window.

        Args:
            fighter_a_states: Chronological combat states for "your" fighter.
            fighter_b_states: Chronological combat states for the opponent.
            round_number: Current round number for tagging.

        Returns:
            List of detected ActionCounterPair objects.
        """
        pairs: list[ActionCounterPair] = []
        b_idx = 0  # Pointer into fighter_b_states for efficiency

        for a_state in fighter_a_states:
            if a_state.action_id is None:
                continue

            # Advance b_idx to the first B-state that could be a counter
            while (
                b_idx < len(fighter_b_states)
                and fighter_b_states[b_idx].timestamp_s
                < a_state.timestamp_s + self.min_gap_s
            ):
                b_idx += 1

            # Search the window for matching B-actions
            for j in range(b_idx, len(fighter_b_states)):
                b_state = fighter_b_states[j]
                delta = b_state.timestamp_s - a_state.timestamp_s

                if delta > self.counter_window_s:
                    break  # Past the window

                if b_state.action_id is None:
                    continue

                pair = ActionCounterPair(
                    trigger_state=a_state,
                    counter_state=b_state,
                    time_delta_s=round(delta, 4),
                    outcome=(
                        "landed" if b_state.strike_landed else
                        "blocked" if b_state.strike_landed is False else
                        None
                    ),
                    damage=b_state.damage_level,
                    round_number=round_number,
                )
                pairs.append(pair)
                break  # Only link the first counter per action

        self.detected_pairs.extend(pairs)
        return pairs

    def reset(self) -> None:
        """Clear all detected pairs."""
        self.detected_pairs.clear()
