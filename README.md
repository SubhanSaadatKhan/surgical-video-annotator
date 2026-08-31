# Surgical Video Annotator

Automated surgical video annotation pipeline combining fine-tuned specialized detectors with foundation models, active learning, and rigorous evaluation.

## The Problem

Training AI models for surgical video analysis requires large, high-quality annotated datasets. Manual annotation is extremely slow: a single laparoscopic video contains 30,000+ frames, each requiring bounding boxes and pixel-level segmentation masks.

## The Solution

A multi-model pipeline that combines the strengths of specialized and general-purpose models:

**Primary detector: Fine-tuned YOLOv8** for accurate detection of known surgical instruments and organs (trained on CholecSeg8k)

**Fallback: Grounding DINO** for open-vocabulary detection of unusual items via text prompts

**Segmentation: SAM 2** for pixel-level masks with native video propagation

## Architecture

```
                  SURGICAL VIDEO
                        |
                        v
              Smart Temporal Sampling
              (motion + scene detection)
                        |
                        v
                 Selected Frames
                        |
        ┌───────────────┴───────────────┐
        v                               v
Fine-tuned YOLOv8          Grounding DINO (fallback)
(specialized surgical      (open-vocabulary for
 instrument detector)       rare/new items)
        |                               |
        └───────────────┬───────────────┘
                        v
             Ensemble Merger (NMS)
                        |
                        v
              SAM 2 Video Propagation
              (pixel masks + tracking)
                        |
                        v
              Quality Validation
                        |
                        v
              Uncertainty Sampling
                        |
                        v
                COCO Export
                        |
        ┌───────────────┴───────────────┐
        v                               v
Streamlit Review              Active Learning Loop
```

## Evaluation Results

Tested on 200 images from CholecSeg8k (13 classes: 2 instruments + 10 organs/tissues).

**Baseline (Grounding DINO zero-shot):**
- Overall mAP@0.5: 4.5%
- Only grasper detection worked (54.5% AP)
- All other classes: 0% AP

**Fine-tuned YOLOv8 (after training on 2000 CholecSeg8k images):**
- See notebook `03_train_and_evaluate.ipynb` for measured results
- Expected significant improvement across all classes

The evaluation script produces direct side-by-side comparison metrics.

## Key Features

1. **Multi-model detection**: Fine-tuned YOLO for accuracy + Grounding DINO for coverage
2. **Smart temporal sampling**: Adaptive frame selection based on motion/scene changes
3. **SAM 2 video propagation**: Native temporal tracking, not per-frame + IoU
4. **Rigorous evaluation**: mAP, precision, recall, F1 on labeled datasets
5. **Active learning**: Uncertainty-based sampling for human review
6. **Streamlit review interface**: Human corrections feed back to next training round
7. **Modular design**: Swap detectors, tune thresholds, disable features via config
8. **Full test coverage**: 42 unit tests for all non-ML components

## Setup

### Requirements
- Python 3.10+
- CUDA-capable GPU (8GB+ VRAM)
- ~5GB disk space (models + dataset)

### Installation

```bash
pip install -r requirements.txt

# For SAM 2 video propagation:
pip install git+https://github.com/facebookresearch/sam2.git
```

## Usage

### Complete workflow (recommended for first-time users)

Open `notebooks/03_train_and_evaluate.ipynb` in Colab and run all cells. This:
1. Downloads CholecSeg8k
2. Trains YOLOv8 on surgical data
3. Evaluates against Grounding DINO baseline
4. Shows side-by-side comparison

### Training YOLO on surgical data

```bash
python scripts/train_yolo.py --n-samples 2000 --epochs 50
```

### Comparing baseline vs fine-tuned

```bash
python scripts/compare_detectors.py --yolo-weights yolo_models/surgical_yolov8/weights/best.pt
```

### Running the pipeline on a video

```bash
python -m src.pipeline path/to/surgical_video.mp4
```

### With ground truth evaluation

```bash
python -m src.pipeline video.mp4 --ground-truth gt_annotations.json
```

### Review interface

```bash
streamlit run app.py
```

## Configuration

All behavior controlled via `config.yaml`. Key settings:

```yaml
# Which detector to use
detector_type: "yolo"           # "yolo" | "grounding_dino" | "ensemble"

# YOLO settings (primary)
yolo_detection:
  weights_path: "yolo_models/surgical_yolov8/weights/best.pt"
  confidence_threshold: 0.25

# Grounding DINO (fallback)
detection:
  prompts: ["grasper", "hook", ...]
  box_threshold: 0.25

# Ensemble mode
ensemble:
  use_dino_fallback: true
  merge_iou_threshold: 0.5
```

## Project Structure

```
surgical-video-annotator/
├── README.md
├── requirements.txt
├── config.yaml
├── app.py                          # Streamlit review interface
├── src/
│   ├── pipeline.py                 # Main video pipeline
│   ├── image_pipeline.py           # Image-mode for evaluation
│   ├── dataset_loader.py           # CholecSeg8k loader + YOLO export
│   ├── detector.py                 # Grounding DINO wrapper
│   ├── yolo_detector.py            # YOLOv8 wrapper + Ensemble
│   ├── segmentor.py                # SAM 2 per-frame segmentation
│   ├── video_propagator.py         # SAM 2 video tracking
│   ├── temporal_sampler.py         # Smart sampling
│   ├── validator.py                # Quality validation
│   ├── evaluator.py                # mAP/precision/recall metrics
│   ├── active_learning.py          # Uncertainty sampling
│   ├── exporter.py                 # COCO export
│   └── utils.py
├── scripts/
│   ├── train_yolo.py               # Fine-tune YOLOv8
│   ├── compare_detectors.py        # Baseline vs fine-tuned
│   └── analyze_results.py          # Result analysis
├── notebooks/
│   ├── 01_video_testing.ipynb
│   ├── 02_cholecseg8k_baseline.ipynb
│   └── 03_train_and_evaluate.ipynb
└── tests/                          # 42 unit tests
```

## Testing

```bash
pytest tests/ -v
```

## Author

Subhan Saadat Khan
M.Sc. Data Science, FAU Erlangen-Nuremberg

Built to demonstrate end-to-end ML engineering: data pipelines, model training, evaluation, and production integration.
