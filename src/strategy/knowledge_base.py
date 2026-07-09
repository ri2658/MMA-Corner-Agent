"""Knowledge base for MMA counter pairs and action taxonomy.

Loads and queries the structured JSON data files that define the action
taxonomy and counter-pair adjustment database.
"""

import json
from pathlib import Path
from typing import Optional


_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


class KnowledgeBase:
    """Loads the action taxonomy and counter-pair database for querying."""

    def __init__(
        self,
        taxonomy_path: Optional[Path] = None,
        counter_pairs_path: Optional[Path] = None,
    ):
        taxonomy_path = taxonomy_path or _DATA_DIR / "action_taxonomy.json"
        counter_pairs_path = counter_pairs_path or _DATA_DIR / "counter_pairs.json"

        with open(taxonomy_path, "r", encoding="utf-8") as f:
            taxonomy_data = json.load(f)
        with open(counter_pairs_path, "r", encoding="utf-8") as f:
            counter_pairs_data = json.load(f)

        self.actions = self._flatten_actions(taxonomy_data["actions"])
        self.defensive_actions = taxonomy_data.get("defensive_actions", {})
        self.counter_pairs = counter_pairs_data["counter_pairs"]

        # Build lookup indices
        self._action_index: dict[str, dict] = {a["id"]: a for a in self.actions}
        self._counter_by_action: dict[str, list[dict]] = {}
        for pair in self.counter_pairs:
            action_id = pair["your_action"]
            self._counter_by_action.setdefault(action_id, []).append(pair)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_action(self, action_id: str) -> Optional[dict]:
        """Return the taxonomy entry for a single action by ID."""
        return self._action_index.get(action_id)

    def get_counters_for_action(self, action_id: str) -> list[dict]:
        """Return all counter pairs where *your* action matches ``action_id``."""
        return self._counter_by_action.get(action_id, [])

    def get_pairs_by_severity(self, severity: str) -> list[dict]:
        """Return all counter pairs matching the given severity level."""
        return [p for p in self.counter_pairs if p["severity"] == severity]

    def get_adjustment(
        self, action_id: str, counter_name: str
    ) -> Optional[list[dict]]:
        """Return adjustment list for a specific action->counter combination."""
        for pair in self.get_counters_for_action(action_id):
            if pair["opponent_counter"]["name"] == counter_name:
                return pair["adjustments"]
        return None

    def suggest_top_adjustment(
        self, action_id: str, counter_name: str
    ) -> Optional[dict]:
        """Return the highest-priority adjustment for a specific counter."""
        adjustments = self.get_adjustment(action_id, counter_name)
        if adjustments:
            return min(adjustments, key=lambda a: a["priority"])
        return None

    def all_action_ids(self) -> list[str]:
        """Return a sorted list of every action ID in the taxonomy."""
        return sorted(self._action_index.keys())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _flatten_actions(actions_dict: dict) -> list[dict]:
        """Recursively collect all action entries from the nested taxonomy."""
        result: list[dict] = []

        for key, value in actions_dict.items():
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict) and "id" in item:
                        result.append(item)
            elif isinstance(value, dict):
                result.extend(KnowledgeBase._flatten_actions(value))

        return result
