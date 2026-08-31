"""Streamlit review app: human-in-the-loop annotation review interface.

This is the final step in the pipeline. After automated annotation,
a human reviewer uses this interface to:
    - Browse annotated frames
    - Approve or reject individual annotations
    - Filter by quality flags (focus on problematic annotations)
    - View tracking consistency across frames
    - Export the reviewed, cleaned dataset

Usage:
    streamlit run app.py -- --annotations output/video_coco.json
"""

import streamlit as st
import json
import os
import cv2
import numpy as np
from pathlib import Path


def load_coco_annotations(coco_path: str) -> dict:
    """Load COCO annotations from JSON file."""
    with open(coco_path, "r") as f:
        return json.load(f)


def draw_annotations_on_frame(
    image_path: str,
    annotations: list,
    categories: dict,
    alpha: float = 0.4,
) -> np.ndarray:
    """Draw bounding boxes and labels on a frame."""
    frame = cv2.imread(image_path)
    if frame is None:
        return np.zeros((480, 640, 3), dtype=np.uint8)

    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
    ]

    for ann in annotations:
        cat_id = ann["category_id"]
        color = colors[(cat_id - 1) % len(colors)]

        # Draw bbox
        x, y, w, h = [int(v) for v in ann["bbox"]]
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Draw label
        cat_name = categories.get(cat_id, "unknown")
        score = ann.get("score", 0)
        track_id = ann.get("track_id", "")
        label = f"{cat_name} {score:.2f}"
        if track_id:
            label += f" [T{track_id}]"

        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(
            frame, (x, y - label_size[1] - 5),
            (x + label_size[0], y), color, -1
        )
        cv2.putText(
            frame, label, (x, y - 3),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )

    # Convert BGR to RGB for Streamlit
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def main():
    st.set_page_config(
        page_title="Surgical Video Annotation Review",
        page_icon="🔬",
        layout="wide",
    )

    st.title("🔬 Surgical Video Annotation Review")
    st.caption("Human-in-the-loop review for automated surgical annotations")

    # ---- Sidebar: Load annotations ----
    st.sidebar.header("Load Annotations")

    coco_path = st.sidebar.text_input(
        "COCO JSON path",
        value="output/video_coco.json",
    )

    if not os.path.exists(coco_path):
        st.warning(f"File not found: {coco_path}")
        st.info(
            "Run the pipeline first:\n\n"
            "```\npython -m src.pipeline video.mp4\n```"
        )
        return

    coco = load_coco_annotations(coco_path)

    # Build lookup maps
    categories = {c["id"]: c["name"] for c in coco["categories"]}
    images_by_id = {img["id"]: img for img in coco["images"]}
    anns_by_image = {}
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        if img_id not in anns_by_image:
            anns_by_image[img_id] = []
        anns_by_image[img_id].append(ann)

    # ---- Sidebar: Filters ----
    st.sidebar.header("Filters")

    # Filter by category
    selected_categories = st.sidebar.multiselect(
        "Instrument types",
        options=list(categories.values()),
        default=list(categories.values()),
    )
    selected_cat_ids = {
        k for k, v in categories.items() if v in selected_categories
    }

    # Filter by confidence
    min_conf = st.sidebar.slider(
        "Minimum confidence", 0.0, 1.0, 0.3, 0.05
    )

    # Filter by review status
    status_filter = st.sidebar.selectbox(
        "Review status",
        ["All", "Pending", "Approved", "Rejected"],
    )

    # ---- Sidebar: Stats ----
    st.sidebar.header("Dataset Stats")
    st.sidebar.metric("Total frames", len(coco["images"]))
    st.sidebar.metric("Total annotations", len(coco["annotations"]))
    st.sidebar.metric("Instrument types", len(categories))

    # Count by category
    cat_counts = {}
    for ann in coco["annotations"]:
        cat_name = categories.get(ann["category_id"], "unknown")
        cat_counts[cat_name] = cat_counts.get(cat_name, 0) + 1
    for cat, count in sorted(cat_counts.items()):
        st.sidebar.text(f"  {cat}: {count}")

    # ---- Main area: Frame browser ----
    image_ids = sorted(images_by_id.keys())

    if not image_ids:
        st.warning("No images found in annotations.")
        return

    # Initialize review state in session
    if "review_status" not in st.session_state:
        st.session_state.review_status = {}

    # Frame navigation
    col_nav1, col_nav2, col_nav3 = st.columns([1, 3, 1])

    with col_nav1:
        if st.button("⬅ Previous"):
            if "frame_idx" not in st.session_state:
                st.session_state.frame_idx = 0
            st.session_state.frame_idx = max(0, st.session_state.frame_idx - 1)

    with col_nav3:
        if st.button("Next ➡"):
            if "frame_idx" not in st.session_state:
                st.session_state.frame_idx = 0
            st.session_state.frame_idx = min(
                len(image_ids) - 1, st.session_state.frame_idx + 1
            )

    with col_nav2:
        frame_slider = st.slider(
            "Frame",
            min_value=0,
            max_value=len(image_ids) - 1,
            value=st.session_state.get("frame_idx", 0),
            key="frame_slider",
        )
        st.session_state.frame_idx = frame_slider

    current_img_id = image_ids[st.session_state.frame_idx]
    current_image = images_by_id[current_img_id]
    current_anns = anns_by_image.get(current_img_id, [])

    # Apply filters
    filtered_anns = [
        a for a in current_anns
        if a["category_id"] in selected_cat_ids
        and a.get("score", 1.0) >= min_conf
    ]

    # ---- Display frame with annotations ----
    col_img, col_details = st.columns([2, 1])

    with col_img:
        st.subheader(
            f"Frame {current_img_id} | "
            f"Time: {current_image.get('timestamp_sec', 0):.1f}s | "
            f"{len(filtered_anns)} annotations"
        )

        frames_dir = os.path.join(os.path.dirname(coco_path), "frames")
        image_path = os.path.join(frames_dir, current_image["file_name"])

        if os.path.exists(image_path):
            annotated_frame = draw_annotations_on_frame(
                image_path, filtered_anns, categories
            )
            st.image(annotated_frame, use_container_width=True)
        else:
            st.error(f"Frame image not found: {image_path}")

    with col_details:
        st.subheader("Annotations")

        for i, ann in enumerate(filtered_anns):
            cat_name = categories.get(ann["category_id"], "unknown")
            score = ann.get("score", 0)
            track_id = ann.get("track_id", "N/A")

            with st.expander(
                f"**{cat_name}** (conf: {score:.2f}, track: {track_id})",
                expanded=False,
            ):
                st.text(f"BBox: {[round(v, 1) for v in ann['bbox']]}")
                st.text(f"Area: {ann.get('area', 0):.0f} px")

                # Review buttons
                ann_key = f"{current_img_id}_{i}"
                current_status = st.session_state.review_status.get(
                    ann_key, "pending"
                )

                col_a, col_r = st.columns(2)
                with col_a:
                    if st.button(
                        "✅ Approve", key=f"approve_{ann_key}",
                        type="primary" if current_status == "approved" else "secondary",
                    ):
                        st.session_state.review_status[ann_key] = "approved"
                        st.rerun()
                with col_r:
                    if st.button(
                        "❌ Reject", key=f"reject_{ann_key}",
                        type="primary" if current_status == "rejected" else "secondary",
                    ):
                        st.session_state.review_status[ann_key] = "rejected"
                        st.rerun()

                st.caption(f"Status: {current_status}")

    # ---- Review progress ----
    st.divider()
    total_anns = len(coco["annotations"])
    reviewed = len(st.session_state.review_status)
    approved = sum(
        1 for s in st.session_state.review_status.values() if s == "approved"
    )
    rejected = sum(
        1 for s in st.session_state.review_status.values() if s == "rejected"
    )

    col_p1, col_p2, col_p3, col_p4 = st.columns(4)
    col_p1.metric("Total annotations", total_anns)
    col_p2.metric("Reviewed", reviewed)
    col_p3.metric("Approved", approved)
    col_p4.metric("Rejected", rejected)

    if total_anns > 0:
        st.progress(reviewed / total_anns, text=f"Review progress: {reviewed}/{total_anns}")

    # ---- Export reviewed annotations ----
    st.divider()
    if st.button("📥 Export Reviewed Annotations"):
        # Create a new COCO file with review status
        reviewed_coco = coco.copy()
        for ann in reviewed_coco["annotations"]:
            ann_key = f"{ann['image_id']}_{coco['annotations'].index(ann)}"
            ann["review_status"] = st.session_state.review_status.get(
                ann_key, "pending"
            )

        # Remove rejected annotations
        reviewed_coco["annotations"] = [
            a for a in reviewed_coco["annotations"]
            if a.get("review_status") != "rejected"
        ]

        export_path = coco_path.replace(".json", "_reviewed.json")
        with open(export_path, "w") as f:
            json.dump(reviewed_coco, f, indent=2)

        st.success(f"Exported to {export_path} ({len(reviewed_coco['annotations'])} annotations kept)")


if __name__ == "__main__":
    main()
