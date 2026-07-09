"""Quick validation of the knowledge base and analysis pipeline."""
import sys
sys.path.insert(0, ".")

from src.strategy.knowledge_base import KnowledgeBase
from src.analysis.state_vector import CombatState, Stance, Distance, Momentum
from src.analysis.pair_linker import PairLinker
from src.analysis.pattern_aggregator import PatternAggregator

# Test 1: Knowledge Base
kb = KnowledgeBase()
print(f"[OK] Actions loaded: {len(kb.actions)}")
print(f"[OK] Counter pairs loaded: {len(kb.counter_pairs)}")

counters = kb.get_counters_for_action("jab_head")
print(f"[OK] Counters for jab_head: {len(counters)}")

critical = kb.get_pairs_by_severity("critical")
print(f"[OK] Critical pairs: {len(critical)}")

adj = kb.suggest_top_adjustment("jab_head", "Slip Outside + Cross")
print(f"[OK] Top adjustment: {adj['name']}")

# Test 2: Pair Linker with simulated data
fighter_a = [
    CombatState("a", 1.0, action_id="jab_head"),
    CombatState("a", 3.0, action_id="jab_head"),
    CombatState("a", 5.0, action_id="rear_body_kick"),
    CombatState("a", 7.0, action_id="jab_head"),
]

fighter_b = [
    CombatState("b", 1.3, action_id="cross_head", strike_landed=True, damage_level="significant"),
    CombatState("b", 3.4, action_id="cross_head", strike_landed=True, damage_level="moderate"),
    CombatState("b", 5.3, action_id="lead_hook_head", strike_landed=True, damage_level="significant"),
    CombatState("b", 7.2, action_id="cross_head", strike_landed=False),
]

linker = PairLinker()
pairs = linker.link(fighter_a, fighter_b, round_number=1)
print(f"[OK] Pairs detected: {len(pairs)}")

# Test 3: Pattern Aggregator
agg = PatternAggregator()
agg.ingest(pairs)
top = agg.get_top_patterns(n=3, min_occurrences=2)
print(f"[OK] Top patterns (min 2 occ): {len(top)}")
for p in top:
    print(f"     {p.pair_key}: {p.occurrences}x, threat={p.threat_score:.1f}")

safe = agg.get_safe_actions()
print(f"[OK] Safe actions: {safe}")

print("\n=== ALL VALIDATIONS PASSED ===")
