"""
EcoSort — YOLOv8 Model Training Script
Run this to fine-tune a YOLOv8 model on waste classification datasets.

Requirements:
  pip install ultralytics roboflow

Usage:
  python train_model.py
"""

import os
from pathlib import Path

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
MODEL_BASE   = "yolov8n.pt"    # Start from YOLOv8 nano (lightest). Use yolov8s.pt for better accuracy.
OUTPUT_DIR   = "models"
EPOCHS       = 50
IMAGE_SIZE   = 640
BATCH_SIZE   = 16
DATA_YAML    = "dataset/data.yaml"   # See below for structure


def create_sample_data_yaml():
    """
    Creates a sample data.yaml for training.
    Replace paths with your actual TrashNet/TACO dataset location.
    """
    yaml_content = """
# EcoSort Dataset Config
# Download TrashNet: https://github.com/garwalar/TrashNet
# Download TACO:     http://taco-dataset.net

path: ./dataset          # Root directory
train: images/train      # Train images (relative to 'path')
val:   images/val        # Validation images
test:  images/test       # Test images (optional)

# Class names (map to EcoSort's 4 categories)
nc: 4
names:
  0: Plastic
  1: Paper
  2: Metal
  3: Organic
"""
    os.makedirs("dataset", exist_ok=True)
    with open(DATA_YAML, "w") as f:
        f.write(yaml_content.strip())
    print(f"[✔] Created {DATA_YAML}")
    print("    → Replace with actual dataset paths before training.")


def train():
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[✗] ultralytics not installed.")
        print("    Run: pip install ultralytics")
        return

    print("\n" + "="*55)
    print("  EcoSort — YOLOv8 Training")
    print("="*55)

    if not Path(DATA_YAML).exists():
        print(f"[!] {DATA_YAML} not found. Creating template...")
        create_sample_data_yaml()
        print("\n[!] Add your dataset images and update data.yaml, then re-run.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load base model
    model = YOLO(MODEL_BASE)
    print(f"[✔] Loaded base model: {MODEL_BASE}")

    # Train
    print(f"[→] Training for {EPOCHS} epochs on {IMAGE_SIZE}px images...")
    results = model.train(
        data        = DATA_YAML,
        epochs      = EPOCHS,
        imgsz       = IMAGE_SIZE,
        batch       = BATCH_SIZE,
        name        = "ecosort_run",
        project     = OUTPUT_DIR,
        pretrained  = True,
        optimizer   = "AdamW",
        lr0         = 0.001,
        weight_decay= 0.0005,
        # Data Augmentation (as per EcoSort spec)
        fliplr      = 0.5,     # Horizontal flip
        degrees     = 15.0,    # Rotation
        hsv_v       = 0.4,     # Brightness jitter
        hsv_s       = 0.4,     # Saturation jitter
        translate   = 0.1,
        scale       = 0.5,
        mosaic      = 1.0,
    )

    # Export best weights
    best_weights = Path(OUTPUT_DIR) / "ecosort_run" / "weights" / "best.pt"
    if best_weights.exists():
        import shutil
        dest = Path(OUTPUT_DIR) / "ecosort_yolov8.pt"
        shutil.copy(best_weights, dest)
        print(f"\n[✔] Best model saved to: {dest}")
        print("[→] Restart app.py — it will auto-load the trained model.")
    else:
        print("[!] Training complete but best.pt not found. Check training output.")

    print("\n[✔] Training complete!")
    print(f"    Results saved to: {OUTPUT_DIR}/ecosort_run/")


if __name__ == "__main__":
    train()
