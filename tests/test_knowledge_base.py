"""Tests for the knowledge base module."""
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.strategy.knowledge_base import KnowledgeBase


def test_load_taxonomy():
    kb = KnowledgeBase()
    assert len(kb.actions) > 0


def test_load_counter_pairs():
    kb = KnowledgeBase()
    assert len(kb.counter_pairs) > 0


def test_get_counters_for_action():
    kb = KnowledgeBase()
    counters = kb.get_counters_for_action("jab_head")
    assert len(counters) > 0
    for counter in counters:
        assert "opponent_counter" in counter
        assert "adjustments" in counter


def test_get_adjustments_by_severity():
    kb = KnowledgeBase()
    critical = kb.get_pairs_by_severity("critical")
    assert all(p["severity"] == "critical" for p in critical)


def test_all_action_ids():
    kb = KnowledgeBase()
    ids = kb.all_action_ids()
    assert len(ids) > 0
    assert "jab_head" in ids
    assert "cross_head" in ids


def test_suggest_top_adjustment():
    kb = KnowledgeBase()
    adj = kb.suggest_top_adjustment("jab_head", "Slip Outside + Cross")
    assert adj is not None
    assert "name" in adj
    assert adj["priority"] == 1
