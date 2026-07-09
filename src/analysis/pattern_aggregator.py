"""Pattern aggregator.

Accumulates action->counter pairs over a round or fight, ranks them by
frequency and damage, and outputs the top patterns that need addressing.
"""

from dataclasses import dataclass, field
from typing import Optional

from .pair_linker import ActionCounterPair


_DAMAGE_WEIGHTS = {
    "significant": 3.0,
    "moderate": 1.5,
    "glancing": 0.5,
    None: 1.0,
}


@dataclass
class CounterPattern:
    """An aggregated counter pattern with frequency and damage score."""

    your_action: str
    their_counter: str
    occurrences: int = 0
    landed_count: int = 0
    total_damage_score: float = 0.0
    timestamps: list[float] = field(default_factory=list)
    rounds_seen: set[int] = field(default_factory=set)

    @property
    def threat_score(self) -> float:
        """Combined score: frequency x damage. Higher = more urgent."""
        return self.occurrences * (1 + self.total_damage_score)

    @property
    def pair_key(self) -> str:
        return f"{self.your_action}->{self.their_counter}"

    def to_dict(self) -> dict:
        return {
            "your_action": self.your_action,
            "their_counter": self.their_counter,
            "occurrences": self.occurrences,
            "landed_count": self.landed_count,
            "threat_score": round(self.threat_score, 2),
            "total_damage_score": round(self.total_damage_score, 2),
            "rounds_seen": sorted(self.rounds_seen),
            "timestamps": self.timestamps,
        }


class PatternAggregator:
    """Aggregates action->counter pairs into ranked threat patterns."""

    def __init__(self):
        self._patterns: dict[str, CounterPattern] = {}

    def ingest(self, pairs: list[ActionCounterPair]) -> None:
        """Add a batch of detected pairs to the aggregator."""
        for pair in pairs:
            key = pair.pair_key
            if key not in self._patterns:
                self._patterns[key] = CounterPattern(
                    your_action=pair.trigger_state.action_id,
                    their_counter=pair.counter_state.action_id,
                )

            pattern = self._patterns[key]
            pattern.occurrences += 1
            pattern.timestamps.append(pair.trigger_state.timestamp_s)
            pattern.rounds_seen.add(pair.round_number)

            if pair.outcome == "landed":
                pattern.landed_count += 1

            pattern.total_damage_score += _DAMAGE_WEIGHTS.get(
                pair.damage, _DAMAGE_WEIGHTS[None]
            )

    def get_top_patterns(
        self,
        n: int = 5,
        min_occurrences: int = 2,
    ) -> list[CounterPattern]:
        """Return the top-N most threatening counter patterns.

        Args:
            n: Maximum number of patterns to return.
            min_occurrences: Minimum times a pattern must appear to be reported.

        Returns:
            Patterns sorted by threat_score descending.
        """
        eligible = [
            p for p in self._patterns.values()
            if p.occurrences >= min_occurrences
        ]
        eligible.sort(key=lambda p: p.threat_score, reverse=True)
        return eligible[:n]

    def get_safe_actions(self, min_thrown: int = 2) -> list[str]:
        """Return action IDs that are NOT being consistently countered.

        These are actions the fighter is throwing that opponents are not
        successfully countering -- i.e. 'more of this'.
        """
        countered_actions = {
            p.your_action
            for p in self._patterns.values()
            if p.landed_count >= 2
        }

        all_actions = {p.your_action for p in self._patterns.values()}
        return sorted(all_actions - countered_actions)

    def reset(self) -> None:
        """Clear all aggregated patterns."""
        self._patterns.clear()

    def summary(self) -> list[dict]:
        """Return all patterns as dicts, sorted by threat score."""
        patterns = sorted(
            self._patterns.values(),
            key=lambda p: p.threat_score,
            reverse=True,
        )
        return [p.to_dict() for p in patterns]
