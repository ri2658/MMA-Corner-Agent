"""Combat state representation.

Defines the structured state vector that captures a fighter's current
combat state at any given moment. This is the core data structure that
flows through the entire pipeline.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Stance(Enum):
    """Fighter's current stance."""
    ORTHODOX = "orthodox"
    SOUTHPAW = "southpaw"
    OPEN = "open"          # Switching or transitional
    GROUND = "ground"


class RingPosition(Enum):
    """Fighter's position relative to the octagon."""
    CENTER = "center"
    MID_RANGE = "mid_range"
    NEAR_CAGE = "near_cage"
    ON_CAGE = "on_cage"
    AGAINST_CAGE = "against_cage"  # Pressed against cage by opponent


class Distance(Enum):
    """Distance between fighters."""
    OUT_OF_RANGE = "out_of_range"   # No strikes can land
    KICKING = "kicking"              # Kicks and long punches
    BOXING = "boxing"                # Punches land cleanly
    CLINCH = "clinch"                # Grappling range
    GROUND = "ground"                # On the mat


class Momentum(Enum):
    """Fighter's current directional pressure."""
    ADVANCING = "advancing"
    RETREATING = "retreating"
    CIRCLING_LEFT = "circling_left"
    CIRCLING_RIGHT = "circling_right"
    STATIONARY = "stationary"
    LEVEL_CHANGING = "level_changing"


@dataclass
class CombatState:
    """Snapshot of a fighter's state at a single moment in time.

    This vector is the fundamental unit flowing through the pipeline:
    Video -> Pose -> ActionClassifier -> **CombatState** -> PairLinker -> Aggregator
    """

    # Identity
    fighter_id: str                           # "fighter_a" or "fighter_b"
    timestamp_s: float                        # Seconds from round start

    # Action
    action_id: Optional[str] = None           # From action_taxonomy.json
    action_confidence: float = 0.0            # Classifier confidence [0, 1]

    # Context
    stance: Stance = Stance.ORTHODOX
    ring_position: RingPosition = RingPosition.CENTER
    distance: Distance = Distance.BOXING
    momentum: Momentum = Momentum.STATIONARY

    # Pose data (raw keypoints if available)
    keypoints: Optional[list[tuple[float, float, float]]] = None

    # Outcome (filled in by pair linker after the fact)
    strike_landed: Optional[bool] = None
    damage_level: Optional[str] = None        # "clean", "partial", "blocked"

    def to_dict(self) -> dict:
        """Serialize to a plain dictionary."""
        return {
            "fighter_id": self.fighter_id,
            "timestamp_s": self.timestamp_s,
            "action_id": self.action_id,
            "action_confidence": self.action_confidence,
            "stance": self.stance.value,
            "ring_position": self.ring_position.value,
            "distance": self.distance.value,
            "momentum": self.momentum.value,
            "strike_landed": self.strike_landed,
            "damage_level": self.damage_level,
        }


@dataclass
class RoundSummary:
    """Aggregated statistics for a single round."""

    round_number: int
    fighter_id: str

    # Time distribution (proportions, sum to 1.0)
    time_at_range: float = 0.0
    time_in_clinch: float = 0.0
    time_on_ground: float = 0.0

    # Positional tendencies
    time_circling: float = 0.0
    time_pressuring: float = 0.0
    time_retreating: float = 0.0
    time_on_cage: float = 0.0

    # Action counts
    total_strikes_thrown: int = 0
    total_strikes_landed: int = 0
    significant_strikes_landed: int = 0
    takedowns_attempted: int = 0
    takedowns_landed: int = 0

    # Counter patterns detected
    counter_patterns: list[dict] = field(default_factory=list)
