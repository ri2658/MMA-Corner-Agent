# MMA Corner Agent 🥊

AI-powered MMA corner coach that identifies counter patterns in real time and suggests technical adjustments between rounds.

> Are you too poor to afford a corner man? Do you wish you had a coach to tell you what adjustments to make in-between rounds? This agentic cornerman will (try to) tell you exactly that!

## What It Does

1. **Watches the fight:** Processes local video using YOLO for tracking and MediaPipe for pose estimation.
2. **Classifies actions:** Translates 2D skeletal data into actionable combat states (jabbing, kicking, high-blocking).
3. **Smooths temporal data:** Groups frame-by-frame tracker jitter into distinct event-level strikes. 
4. **Detects counter patterns:** Links your attacks to the opponent's successful counters within sliding temporal windows.
5. **Suggests adjustments:** Queries an internal knowledge base to generate actionable corner advice to break the opponent's reads.
6. **Key Frame Extraction:** Captures heavily annotated original-resolution frames during action events for visual explainability.

## Architecture

```
Video Feed → YOLO Tracker + Referee Filtering → MediaPipe Pose → 2D Joint Coordinates
                                                                       ↓
                                                      Geometric Action Classifier
                                                                       ↓
                                                        Temporal State Smoother
                                                                       ↓
                                                                  Pair Linker
                                                            (temporal association)
                                                                       ↓
                                                              Pattern Aggregator
                                                                       ↓
                                                              Adjustment Engine
                                                                       ↓
                                                             Corner Coach Display
```

## Quick Start

To clone this repository, run:

```bash
# Clone
git clone https://github.com/ri2658/MMA-Corner-Agent.git
cd MMA-Corner-Agent

# Install
pip install -e ".[dev]"

# Run tests
pytest
```

Then, to boot up the website locally, run:

```bash
streamlit run src/ui/app.py
```

## Status

🚧 **Alpha Prototype** — The core architecture is fully wired and functional, capable of processing videos and extracting counter patterns, but bounded by limitations in current computer vision approaches for combat sports.

- [x] Action taxonomy (40+ MMA actions with properties)
- [x] Counter-pair database (20 entries with adjustments)
- [x] Knowledge base module
- [x] Combat state representation
- [x] Pair linker (temporal association)
- [x] Pattern aggregator
- [x] Adjustment engine
- [x] Pose estimation pipeline (YOLO + MediaPipe)
- [x] Action classifier model (Geometric rule-based + relative velocity)
- [x] Fighter tracking (with referee/official appearance filtering)
- [x] Corner coach UI (Streamlit)
- [x] Real-time video processing (with downscaled inference optimization)

## Critical Bugs & Limitations

While the pipeline works effectively on clean, standing striking (e.g. Yan vs. Dvalishvili), it currently runs into fundamental computer vision bottlenecks:

1. **The Intelligence Bottleneck (Brittle Heuristics):** The system relies on 2D geometric rules (e.g., wrist velocity, knee angle). This breaks on depth-based strikes (e.g., a teep moving along the Z-axis) and struggles with full-body kinetic chains. For instance, a heavy lunging step for an overhand can falsely trigger the "lead calf kick" heuristic due to lateral ankle velocity.
2. **The Occlusion & Grappling Problem:** MediaPipe is a top-down pose estimator trained on upright individuals. Once fighters enter the clinch, pull guard, or scramble on the ground, their bounding boxes merge. This causes massive occlusion and limb hallucination, rendering the tracker blind during grappling exchanges.
3. **The Taxonomy Problem:** MMA contains infinite unorthodox techniques (e.g. Dricus Du Plessis's shifting blitzes). Hardcoding rules to bucket these into finite labels (Jab, Hook, Cross) is structurally flawed.
4. **The Performance Bottleneck:** Running YOLO and two instances of MediaPipe sequentially on a CPU caps throughput at ~10-15 FPS, preventing true real-time analysis.

## Future Aspirations & Roadmap

To elevate this from a prototype to a production-grade tool, the architecture will need to pivot towards modern deep learning approaches:

1. **Single-Pass YOLO-Pose:** Replace the YOLO + MediaPipe stack with a custom-trained YOLO-Pose model. A single-pass model trained specifically on BJJ and MMA datasets can natively handle multi-person occlusion, tracking grappling limbs accurately without bounding box confusion, while drastically increasing FPS.
2. **Vulnerability & Opening Detector:** Shift the AI paradigm away from the *Taxonomy Problem* (trying to perfectly label every chaotic strike). Instead, redesign the classifier to detect *Structural Vulnerabilities* (e.g. dropped hands, chin high, crossed feet, weight leaning heavily on the lead leg). Tracking these universal geometric truths is vastly more reliable and provides much higher tactical value for Corner Advice.
3. **Active Learning Engine:** Build an automated pipeline to detect low-confidence frames (e.g., clinch knees, ground-and-pound) and output them for manual annotation to cheaply build a proprietary dataset of complex MMA transitions.

## Disclaimer

This project is FULLY vibe-coded; it was created as a fun summer side-project to explore the intersection of Agentic AI and MMA combat analytics.

## License

MIT
