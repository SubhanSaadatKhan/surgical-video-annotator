"""Analysis script: generates detailed metrics from pipeline results.

Run this after the pipeline completes to get a clear picture of what worked
and what didn't. Produces a text report and a comparison-friendly summary.

Usage in Colab:
    from scripts.analyze_results import analyze_run
    analyze_run(results, output_path='analysis_report.txt')
"""

import json
import os
from collections import Counter, defaultdict


def analyze_run(results: dict, output_path: str = "analysis_report.txt") -> dict:
    """Generate a detailed analysis report from pipeline results.

    Args:
        results: The dict returned by pipeline.run()
        output_path: Where to save the text report

    Returns:
        Dict of metrics that can be compared across runs.
    """
    stats = results["stats"]
    annotations = results["annotations"]
    quality_report = results["quality_report"]
    review_queue = results.get("review_queue", [])

    # ============================================================
    # Basic counts
    # ============================================================
    total_frames = len(annotations)
    frames_with_detections = sum(1 for a in annotations if a.bboxes)
    frames_without_detections = total_frames - frames_with_detections
    total_detections = sum(len(a.bboxes) for a in annotations)

    detection_rate = (frames_with_detections / total_frames * 100) if total_frames > 0 else 0
    avg_detections_per_frame = (total_detections / total_frames) if total_frames > 0 else 0

    # ============================================================
    # Confidence analysis
    # ============================================================
    all_confidences = []
    for a in annotations:
        for bbox in a.bboxes:
            all_confidences.append(bbox.confidence)

    avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0
    min_confidence = min(all_confidences) if all_confidences else 0
    max_confidence = max(all_confidences) if all_confidences else 0

    high_conf_count = sum(1 for c in all_confidences if c >= 0.5)
    medium_conf_count = sum(1 for c in all_confidences if 0.3 <= c < 0.5)
    low_conf_count = sum(1 for c in all_confidences if c < 0.3)

    # ============================================================
    # Label analysis (detect merging issues)
    # ============================================================
    all_labels = []
    for a in annotations:
        for bbox in a.bboxes:
            all_labels.append(bbox.label)

    label_counts = Counter(all_labels)

    # Detect merged labels (multi-word or containing our prompts)
    merged_labels = {
        label: count for label, count in label_counts.items()
        if len(label.split()) > 1
    }

    # ============================================================
    # Bounding box size analysis
    # ============================================================
    frame_area = 1280 * 720  # From config (default resize)
    box_sizes = []
    for a in annotations:
        for bbox in a.bboxes:
            area_ratio = bbox.area / frame_area if frame_area > 0 else 0
            box_sizes.append(area_ratio)

    if box_sizes:
        avg_box_ratio = sum(box_sizes) / len(box_sizes)
        tiny_boxes = sum(1 for r in box_sizes if r < 0.01)  # < 1% of frame
        small_boxes = sum(1 for r in box_sizes if 0.01 <= r < 0.05)
        medium_boxes = sum(1 for r in box_sizes if 0.05 <= r < 0.2)
        large_boxes = sum(1 for r in box_sizes if 0.2 <= r < 0.5)
        huge_boxes = sum(1 for r in box_sizes if r >= 0.5)
    else:
        avg_box_ratio = tiny_boxes = small_boxes = medium_boxes = large_boxes = huge_boxes = 0

    # ============================================================
    # Tracking analysis
    # ============================================================
    track_ids = set()
    for a in annotations:
        for bbox in a.bboxes:
            if bbox.track_id is not None:
                track_ids.add(bbox.track_id)

    unique_tracks = len(track_ids)

    # ============================================================
    # Quality flag breakdown
    # ============================================================
    flag_types = quality_report.get("by_type", {})
    flag_severities = quality_report.get("by_severity", {})

    # ============================================================
    # Build report text
    # ============================================================
    lines = []
    lines.append("=" * 70)
    lines.append(f"PIPELINE ANALYSIS REPORT: {stats.get('video', 'unknown')}")
    lines.append("=" * 70)

    lines.append("\n## SAMPLING & PROCESSING")
    lines.append(f"  Total frames in source video: {stats.get('total_frames_in_video', 'unknown')}")
    lines.append(f"  Frames sampled and processed: {total_frames}")
    lines.append(f"  Processing time: {stats.get('execution_time_sec', 0):.1f} sec")
    lines.append(f"  Speed: {stats.get('fps_processed', 0):.1f} frames/sec")

    lines.append("\n## DETECTION RECALL (THE BIG QUESTION)")
    lines.append(f"  Frames WITH detections: {frames_with_detections}/{total_frames} ({detection_rate:.1f}%)")
    lines.append(f"  Frames WITHOUT detections: {frames_without_detections}/{total_frames} ({100-detection_rate:.1f}%)")
    lines.append(f"  Total detections: {total_detections}")
    lines.append(f"  Average detections per frame: {avg_detections_per_frame:.2f}")
    if frames_without_detections > total_frames * 0.3:
        lines.append(f"  ISSUE: >{frames_without_detections} frames have zero detections. Recall is low.")

    lines.append("\n## DETECTION CONFIDENCE")
    lines.append(f"  Average confidence: {avg_confidence:.3f}")
    lines.append(f"  Range: {min_confidence:.3f} to {max_confidence:.3f}")
    lines.append(f"  High confidence (>=0.5): {high_conf_count} detections")
    lines.append(f"  Medium confidence (0.3-0.5): {medium_conf_count} detections")
    lines.append(f"  Low confidence (<0.3): {low_conf_count} detections")
    if avg_confidence < 0.4:
        lines.append(f"  ISSUE: Average confidence is very low. Model is unsure about most detections.")

    lines.append("\n## LABEL QUALITY")
    lines.append(f"  Unique labels found: {len(label_counts)}")
    lines.append(f"  All labels: {dict(label_counts)}")
    if merged_labels:
        lines.append(f"  MERGED LABELS DETECTED (label merging bug):")
        for label, count in merged_labels.items():
            lines.append(f"    '{label}': {count} times")

    lines.append("\n## BOUNDING BOX SIZE DISTRIBUTION")
    lines.append(f"  Average box size (as % of frame): {avg_box_ratio*100:.1f}%")
    lines.append(f"  Tiny boxes (<1% of frame): {tiny_boxes} (likely noise)")
    lines.append(f"  Small boxes (1-5%): {small_boxes} (probably good)")
    lines.append(f"  Medium boxes (5-20%): {medium_boxes} (typical for instruments)")
    lines.append(f"  Large boxes (20-50%): {large_boxes} (likely too big)")
    lines.append(f"  Huge boxes (>50%): {huge_boxes} (definitely wrong)")
    if huge_boxes + large_boxes > total_detections * 0.3:
        lines.append(f"  ISSUE: Many bounding boxes are too large. Model is not localizing tightly.")

    lines.append("\n## TRACKING")
    lines.append(f"  Unique tracks assigned: {unique_tracks}")
    if unique_tracks > total_detections * 0.7:
        lines.append(f"  ISSUE: Almost every detection gets a new track ID. Tracking is not connecting objects across frames.")

    lines.append("\n## QUALITY VALIDATION")
    lines.append(f"  Quality score: {quality_report.get('quality_score', 0):.2f}")
    lines.append(f"  Total issues flagged: {quality_report.get('total_issues', 0)}")
    lines.append(f"  Errors: {flag_severities.get('error', 0)}")
    lines.append(f"  Warnings: {flag_severities.get('warning', 0)}")
    lines.append(f"  Breakdown by type:")
    for flag_type, count in flag_types.items():
        lines.append(f"    {flag_type}: {count}")

    lines.append("\n## ACTIVE LEARNING")
    lines.append(f"  Samples flagged for human review: {len(review_queue)}")

    lines.append("\n## OVERALL VERDICT")
    verdicts = []
    if detection_rate < 60:
        verdicts.append(f"- Recall is the biggest problem ({detection_rate:.0f}% of frames have detections)")
    if avg_confidence < 0.4:
        verdicts.append(f"- Model has low confidence overall ({avg_confidence:.2f})")
    if merged_labels:
        verdicts.append(f"- Label merging bug is present ({len(merged_labels)} merged labels)")
    if huge_boxes + large_boxes > total_detections * 0.3:
        verdicts.append(f"- Bounding boxes are often too large")

    if not verdicts:
        verdicts.append("- Pipeline is working reasonably well!")

    for v in verdicts:
        lines.append(v)

    lines.append("\n" + "=" * 70)

    report_text = "\n".join(lines)

    # Save report
    with open(output_path, "w") as f:
        f.write(report_text)

    print(report_text)

    # Return metrics dict for comparison across runs
    metrics = {
        "video": stats.get("video"),
        "total_frames": total_frames,
        "frames_with_detections": frames_with_detections,
        "detection_rate_pct": round(detection_rate, 1),
        "total_detections": total_detections,
        "avg_detections_per_frame": round(avg_detections_per_frame, 2),
        "avg_confidence": round(avg_confidence, 3),
        "num_unique_labels": len(label_counts),
        "num_merged_labels": len(merged_labels),
        "avg_box_ratio_pct": round(avg_box_ratio * 100, 1),
        "large_or_huge_boxes": large_boxes + huge_boxes,
        "unique_tracks": unique_tracks,
        "quality_score": quality_report.get("quality_score", 0),
        "total_quality_issues": quality_report.get("total_issues", 0),
    }

    return metrics


def compare_runs(metrics_list: list) -> None:
    """Compare metrics across multiple pipeline runs.

    Args:
        metrics_list: List of metric dicts from analyze_run()
    """
    if not metrics_list:
        print("No runs to compare.")
        return

    print("\n" + "=" * 90)
    print("COMPARISON ACROSS RUNS")
    print("=" * 90)

    # Print header
    header = f"{'Metric':<30}"
    for m in metrics_list:
        header += f"{m['video'][:15]:>18}"
    print(header)
    print("-" * len(header))

    # Print each metric
    metric_keys = [
        "total_frames", "frames_with_detections", "detection_rate_pct",
        "total_detections", "avg_detections_per_frame", "avg_confidence",
        "num_unique_labels", "num_merged_labels", "avg_box_ratio_pct",
        "large_or_huge_boxes", "unique_tracks", "quality_score",
        "total_quality_issues",
    ]

    for key in metric_keys:
        row = f"{key:<30}"
        for m in metrics_list:
            val = m.get(key, "N/A")
            row += f"{str(val):>18}"
        print(row)

    print("=" * 90)
