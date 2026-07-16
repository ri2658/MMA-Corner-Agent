"""Quick test: run video pipeline on the bundled fight clip."""
import sys
sys.path.insert(0, ".")

from src.corner_agent import CornerAgent

agent = CornerAgent(min_pattern_occurrences=1, max_advice_items=4)

print("Processing first 100 frames of Yan vs Dvalishvili...")


def on_progress(frames, ts):
    if frames % 25 == 0:
        print(f"  frame {frames}, t={ts:.1f}s")


report = agent.analyze_video(
    "data/yan_vs_dvalishvili_2_clips.mp4",
    round_number=1,
    target_fps=10.0,
    max_frames=100,
    progress_callback=on_progress,
)

print(f"\nResults:")
print(f"  Frames processed: {report.frames_processed}")
print(f"  Duration: {report.duration_s:.1f}s")
print(f"  Pairs detected: {report.pairs_detected}")
print(f"  Fighter A actions: {report.fighter_a_actions}")
print(f"  Fighter B actions: {report.fighter_b_actions}")
print(f"  Advice items: {len(report.advice)}")
for a in report.advice:
    d = a.to_display_dict()
    print(f"    {d['icon']} {d['headline']}")
print(f"  Safe actions: {report.safe_actions}")
print("\n=== VIDEO PIPELINE TEST PASSED ===")
