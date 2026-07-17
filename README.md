# MMA Corner Agent 🥊

AI-powered MMA corner coach that identifies counter patterns in real time and suggests technical adjustments between rounds.

> Are you too poor to afford a corner man? Do you wish you had a coach to tell you what adjustments to make in-between rounds? This agentic cornerman will (try to) tell you exactly that!

## What It Does

1. **Watches the fight** — Pose estimation extracts fighter body positions from video
2. **Classifies actions** — Identifies what each fighter is doing (jabbing, kicking, sprawling, etc.)
3. **Detects counter patterns** — Links your attacks to the opponent's successful counters
4. **Ranks threats** — Identifies the most frequent and damaging counters being used against you
5. **Suggests adjustments** — Provides specific technical adjustments to break the pattern

## Architecture

```
Video Feed → Pose Estimation → Action Classifier → Combat State Vectors
                                                          ↓
                                                    Pair Linker
                                              (temporal association)
                                                          ↓
                                                Pattern Aggregator
                                              (frequency × damage ranking)
                                                          ↓
                                                Adjustment Engine
                                              (knowledge base lookup)
                                                          ↓
                                                Corner Coach Display
```

## Project Structure

```
MMA-Corner-Agent/
├── data/
│   ├── action_taxonomy.json      # 40+ MMA actions with timing/properties
│   └── counter_pairs.json        # 20 counter-pair entries with adjustments
├── src/
│   ├── vision/                   # Video processing pipeline
│   │   ├── pose_estimator.py     # MediaPipe/MMPose pose extraction
│   │   ├── fighter_tracker.py    # Multi-fighter tracking
│   │   └── action_classifier.py  # Pose → action classification
│   ├── analysis/                 # Core analysis logic
│   │   ├── state_vector.py       # Combat state representation
│   │   ├── pair_linker.py        # Temporal action→counter linking
│   │   └── pattern_aggregator.py # Pattern frequency/damage ranking
│   ├── strategy/                 # Adjustment generation
│   │   ├── knowledge_base.py     # Counter-pair database queries
│   │   └── adjustment_engine.py  # Corner advice generation
│   └── ui/                       # Display layer
│       └── corner_display.py     # Between-round coach display
├── models/                       # Trained model weights (git-ignored)
├── notebooks/                    # Exploration and prototyping
├── tests/                        # Test suite
├── pyproject.toml                # Project configuration
└── README.md
```

## Quick Start

To clone this repository, run

```bash
# Clone
git clone https://github.com/ri2658/MMA-Corner-Agent.git
cd MMA-Corner-Agent

# Install
pip install -e ".[dev]"

# Run tests
pytest
```

Then, to boot up the website locally, run

```bash
streamlit run src/ui/app.py
```

## Status

🚧 **Early development** — Building the foundation.

- [x] Action taxonomy (40+ MMA actions with properties)
- [x] Counter-pair database (20 entries with adjustments)
- [x] Knowledge base module
- [x] Combat state representation
- [x] Pair linker (temporal association)
- [x] Pattern aggregator
- [x] Adjustment engine
- [x] Pose estimation pipeline
- [x] Action classifier model
- [x] Fighter tracking
- [x] Corner coach UI
- [x] Real-time video processing

## Disclaimer

This project is FULLY vibe-coded; it was created as a fun summer side-project.

## License

MIT
