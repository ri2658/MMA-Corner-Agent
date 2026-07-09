"""Adjustment engine.

Takes ranked counter patterns from the PatternAggregator and looks up
the appropriate adjustments from the KnowledgeBase. Formats the output
for the corner coach display.
"""

from dataclasses import dataclass
from typing import Optional

from ..analysis.pattern_aggregator import CounterPattern
from .knowledge_base import KnowledgeBase


@dataclass
class CornerAdvice:
    """A single piece of corner advice to display between rounds."""

    severity: str                          # "critical", "warning", "working"
    your_action: str                       # Human-readable action name
    their_counter: str                     # Human-readable counter name
    occurrences: int
    landed_count: int
    adjustment_name: str                   # Short name for the adjustment
    adjustment_detail: str                 # Full technical explanation
    threat_score: float

    def to_display_dict(self) -> dict:
        """Format for UI rendering."""
        icon = {
            "critical": "\U0001f534",
            "warning": "\U0001f7e1",
            "working": "\U0001f7e2",
        }.get(self.severity, "\u26aa")

        return {
            "icon": icon,
            "severity": self.severity,
            "headline": f"{self.your_action} -> countered by {self.their_counter}",
            "stats": f"{self.occurrences}x this round | {self.landed_count} landed",
            "adjustment": f">> {self.adjustment_name}: {self.adjustment_detail}",
            "threat_score": self.threat_score,
        }


class AdjustmentEngine:
    """Generates corner advice from detected counter patterns."""

    def __init__(self, knowledge_base: Optional[KnowledgeBase] = None):
        self.kb = knowledge_base or KnowledgeBase()

    def generate_advice(
        self,
        patterns: list[CounterPattern],
        max_items: int = 3,
    ) -> list[CornerAdvice]:
        """Generate prioritized corner advice from the top counter patterns.

        Args:
            patterns: Ranked counter patterns from PatternAggregator.
            max_items: Maximum number of advice items to return.

        Returns:
            List of CornerAdvice objects ready for UI display.
        """
        advice_list: list[CornerAdvice] = []

        for pattern in patterns[:max_items]:
            # Look up the counter pair in the knowledge base
            counters = self.kb.get_counters_for_action(pattern.your_action)

            # Try to find a matching counter name
            adjustment = None
            counter_display = pattern.their_counter

            for counter_entry in counters:
                # Match by counter action ID or name
                entry_counter_id = counter_entry.get(
                    "opponent_counter", {}
                ).get("attack", "")
                if entry_counter_id == pattern.their_counter:
                    counter_display = counter_entry["opponent_counter"]["name"]
                    if counter_entry.get("adjustments"):
                        adjustment = counter_entry["adjustments"][0]
                    break

            # Determine severity based on threat score
            if pattern.threat_score >= 8:
                severity = "critical"
            elif pattern.threat_score >= 4:
                severity = "warning"
            else:
                severity = "working"

            # Get human-readable action name
            action_entry = self.kb.get_action(pattern.your_action)
            action_display = (
                action_entry["name"] if action_entry else pattern.your_action
            )

            advice = CornerAdvice(
                severity=severity,
                your_action=action_display,
                their_counter=counter_display,
                occurrences=pattern.occurrences,
                landed_count=pattern.landed_count,
                adjustment_name=(
                    adjustment["name"]
                    if adjustment
                    else "No specific adjustment found"
                ),
                adjustment_detail=(
                    adjustment["description"]
                    if adjustment
                    else "Vary your timing and entry angle."
                ),
                threat_score=pattern.threat_score,
            )
            advice_list.append(advice)

        return advice_list

    def format_for_display(self, advice_list: list[CornerAdvice]) -> str:
        """Format advice as a text block for terminal/simple display."""
        lines = ["=" * 60, "  CORNER ADVICE", "=" * 60, ""]

        for advice in advice_list:
            d = advice.to_display_dict()
            lines.append(
                f"  {d['icon']} {d['severity'].upper()}: {d['headline']}"
            )
            lines.append(f"     {d['stats']}")
            lines.append(f"     {d['adjustment']}")
            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)
