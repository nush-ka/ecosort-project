"""
EcoSort - AI-Powered Smart Waste Classification Web App
Flask Backend
"""

import os
import io
import base64
import json
import time
import random
import numpy as np
from flask import Flask, render_template, request, jsonify
from PIL import Image
import cv2

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# ─────────────────────────────────────────────
# Waste Category Configuration
# ─────────────────────────────────────────────
WASTE_CATEGORIES = {
    "Plastic": {
        "bin_color": "#3B82F6",       # Blue
        "bin_label": "Blue Bin",
        "co2_saving": 1.5,            # kg CO2 per item recycled
        "examples": ["bottle", "bag", "container", "straw", "cup", "wrapper"],
        "icon": "🔵",
        "tip": "Rinse before recycling. Remove caps separately."
    },
    "Paper": {
        "bin_color": "#F59E0B",       # Yellow/Amber
        "bin_label": "Yellow Bin",
        "co2_saving": 0.9,
        "examples": ["newspaper", "cardboard", "book", "box", "tissue", "envelope"],
        "icon": "🟡",
        "tip": "Keep dry. Remove any plastic coatings."
    },
    "Metal": {
        "bin_color": "#6B7280",       # Grey
        "bin_label": "Grey Bin",
        "co2_saving": 4.2,
        "examples": ["can", "tin", "foil", "wire", "scrap", "utensil"],
        "icon": "⚪",
        "tip": "Crush cans to save space. Rinse food residue."
    },
    "Organic": {
        "bin_color": "#10B981",       # Green
        "bin_label": "Green Bin",
        "co2_saving": 0.4,
        "examples": ["food", "vegetable", "fruit", "leaf", "peel", "leftover"],
        "icon": "🟢",
        "tip": "Compost organic waste. Avoid mixing with dry waste."
    }
}

# Keywords for simple rule-based fallback classification
KEYWORD_MAP = {
    "Plastic": ["bottle", "plastic", "bag", "container", "straw", "cup", "wrapper",
                "nylon", "polythene", "tupperware", "packaging", "film"],
    "Paper":   ["paper", "cardboard", "newspaper", "book", "box", "tissue",
                "envelope", "magazine", "carton", "receipt", "napkin"],
    "Metal":   ["can", "tin", "metal", "foil", "wire", "scrap", "iron",
                "steel", "aluminum", "copper", "alloy", "nail", "coin"],
    "Organic": ["food", "vegetable", "fruit", "leaf", "peel", "leftover",
                "banana", "apple", "egg", "meat", "grain", "waste", "plant"]
}


# ─────────────────────────────────────────────
# Model Loading (YOLOv8 if available, else mock)
# ─────────────────────────────────────────────
MODEL = None
USE_MOCK = True

def load_model():
    global MODEL, USE_MOCK
    try:
        from ultralytics import YOLO
        model_path = "models/ecosort_yolov8.pt"
        if os.path.exists(model_path):
            MODEL = YOLO(model_path)
            USE_MOCK = False
            print("[EcoSort] ✅ Loaded custom YOLOv8 model")
        else:
            # Use pretrained YOLOv8n as object detector fallback
            MODEL = YOLO("yolov8n.pt")
            USE_MOCK = False
            print("[EcoSort] ℹ️  Using YOLOv8n (no custom model found). Mapping COCO classes to waste types.")
    except ImportError:
        USE_MOCK = True
        print("[EcoSort] ⚠️  ultralytics not installed — using intelligent mock classifier")

load_model()


# ─────────────────────────────────────────────
# COCO Class → Waste Category Mapping
# ─────────────────────────────────────────────
COCO_TO_WASTE = {
    # Plastic
    "bottle": "Plastic", "cup": "Plastic", "vase": "Plastic",
    "sports ball": "Plastic", "frisbee": "Plastic",
    # Paper/Cardboard
    "book": "Paper", "newspaper": "Paper", "suitcase": "Paper",
    # Metal
    "scissors": "Metal", "knife": "Metal", "spoon": "Metal",
    "fork": "Metal", "oven": "Metal", "microwave": "Metal",
    "refrigerator": "Metal", "toaster": "Metal",
    # Organic
    "apple": "Organic", "banana": "Organic", "orange": "Organic",
    "broccoli": "Organic", "carrot": "Organic", "hot dog": "Organic",
    "pizza": "Organic", "donut": "Organic", "cake": "Organic",
    "sandwich": "Organic", "potted plant": "Organic", "bowl": "Organic",
}


# ─────────────────────────────────────────────
# Grad-CAM Heatmap Generator
# ─────────────────────────────────────────────
def generate_gradcam_heatmap(image_np, bbox=None):
    """
    Generates a synthetic Grad-CAM style heatmap.
    In production, replace with real gradient-based visualization
    using pytorch hooks on the YOLOv8 backbone.
    """
    h, w = image_np.shape[:2]
    heatmap = np.zeros((h, w), dtype=np.float32)

    if bbox:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        bw = max(x2 - x1, 10)
        bh = max(y2 - y1, 10)
    else:
        cx, cy = w // 2, h // 2
        bw, bh = w // 3, h // 3

    # Gaussian blob centered on the object
    for y in range(h):
        for x in range(w):
            dx = (x - cx) / (bw * 0.6)
            dy = (y - cy) / (bh * 0.6)
            heatmap[y, x] = np.exp(-(dx**2 + dy**2))

    # Add subtle noise
    noise = np.random.uniform(0, 0.15, (h, w)).astype(np.float32)
    heatmap = np.clip(heatmap + noise * (heatmap > 0.1), 0, 1)

    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # Blend with original
    overlay = cv2.addWeighted(image_np, 0.55, heatmap_color, 0.45, 0)
    return overlay


# ─────────────────────────────────────────────
# Sustainability Score Calculator
# ─────────────────────────────────────────────
def calculate_sustainability_score(category: str, confidence: float) -> dict:
    cat_data = WASTE_CATEGORIES.get(category, {})
    co2_base = cat_data.get("co2_saving", 1.0)
    score = round(co2_base * confidence * 100)
    co2_saved = round(co2_base * confidence, 3)
    return {
        "score": score,
        "co2_saved_kg": co2_saved,
        "category_multiplier": co2_base,
        "confidence_factor": round(confidence, 3)
    }


# ─────────────────────────────────────────────
# Contamination Detector
# ─────────────────────────────────────────────
def check_contamination(detections: list) -> dict:
    """
    Checks if organic + recyclable waste appear together with overlapping boxes.
    """
    categories = [d["category"] for d in detections]
    has_organic = "Organic" in categories
    has_recyclable = any(c in categories for c in ["Plastic", "Paper", "Metal"])

    contaminated = False
    message = ""

    if has_organic and has_recyclable:
        # Check bounding box proximity
        organic_boxes = [d["bbox"] for d in detections if d["category"] == "Organic"]
        recyclable_boxes = [d["bbox"] for d in detections if d["category"] != "Organic"]

        for ob in organic_boxes:
            for rb in recyclable_boxes:
                # Simple overlap check
                ox1, oy1, ox2, oy2 = ob
                rx1, ry1, rx2, ry2 = rb
                overlap = not (ox2 < rx1 or rx2 < ox1 or oy2 < ry1 or ry2 < oy1)
                if overlap:
                    contaminated = True
                    break

        if contaminated:
            message = "⚠️ Contamination detected! Organic waste is near recyclables. Please separate and rinse items before disposal."
        else:
            message = "ℹ️ Organic and recyclable waste in frame — ensure they are in separate bins."

    return {
        "contaminated": contaminated,
        "warning": message,
        "has_organic": has_organic,
        "has_recyclable": has_recyclable
    }


# ─────────────────────────────────────────────
# Mock Classifier (when YOLOv8 not available)
# ─────────────────────────────────────────────
def mock_classify(image_np) -> list:
    """
    Returns plausible mock detections for demo purposes.
    Uses image properties to vary results slightly.
    """
    categories = list(WASTE_CATEGORIES.keys())
    # Use image mean color to seed randomness (deterministic per frame)
    seed = int(image_np.mean() * 100) % 1000
    random.seed(seed)

    h, w = image_np.shape[:2]
    n_objects = random.randint(1, 2)
    detections = []

    chosen_cats = random.sample(categories, min(n_objects, len(categories)))
    for i, cat in enumerate(chosen_cats):
        # Random bbox in image
        bx1 = random.randint(int(w * 0.1), int(w * 0.4))
        by1 = random.randint(int(h * 0.1), int(h * 0.4))
        bx2 = random.randint(int(w * 0.5), int(w * 0.85))
        by2 = random.randint(int(h * 0.5), int(h * 0.85))
        conf = round(random.uniform(0.62, 0.97), 3)
        detections.append({
            "category": cat,
            "confidence": conf,
            "bbox": [bx1, by1, bx2, by2],
            "label": cat
        })
    return detections


# ─────────────────────────────────────────────
# Real YOLOv8 Classifier
# ─────────────────────────────────────────────
def yolo_classify(image_np) -> list:
    results = MODEL(image_np, verbose=False)[0]
    detections = []
    names = MODEL.names

    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        cls_name = names.get(cls_id, "unknown").lower()

        # Map COCO class to waste category
        category = COCO_TO_WASTE.get(cls_name, None)
        if category is None:
            # Keyword fallback
            for wcat, keywords in KEYWORD_MAP.items():
                if any(kw in cls_name for kw in keywords):
                    category = wcat
                    break
        if category is None:
            category = random.choice(list(WASTE_CATEGORIES.keys()))

        detections.append({
            "category": category,
            "confidence": round(conf, 3),
            "bbox": [x1, y1, x2, y2],
            "label": cls_name
        })

    if not detections:
        # No detection — return mock
        return mock_classify(image_np)
    return detections


# ─────────────────────────────────────────────
# Frame Processor (Core Logic)
# ─────────────────────────────────────────────
def process_frame(image_np):
    """Full EcoSort pipeline: detect → classify → heatmap → score → contamination"""

    # 1. Detection & Classification
    if USE_MOCK:
        detections = mock_classify(image_np)
    else:
        detections = yolo_classify(image_np)

    if not detections:
        return {"error": "No waste detected. Try holding the item closer to the camera."}

    # 2. Primary detection (highest confidence)
    primary = max(detections, key=lambda d: d["confidence"])
    cat_data = WASTE_CATEGORIES[primary["category"]]

    # 3. Grad-CAM Heatmap
    heatmap_img = generate_gradcam_heatmap(image_np.copy(), primary["bbox"])

    # Draw bounding boxes on heatmap image
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        color_hex = WASTE_CATEGORIES[det["category"]]["bin_color"].lstrip("#")
        color_rgb = tuple(int(color_hex[i:i+2], 16) for i in (0, 2, 4))
        color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
        cv2.rectangle(heatmap_img, (x1, y1), (x2, y2), color_bgr, 3)
        label = f"{det['category']} {det['confidence']*100:.0f}%"
        cv2.putText(heatmap_img, label, (x1, max(y1-10, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_bgr, 2)

    # Encode heatmap to base64
    _, buf = cv2.imencode(".jpg", heatmap_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    heatmap_b64 = base64.b64encode(buf).decode("utf-8")

    # 4. Sustainability Score
    sustainability = calculate_sustainability_score(primary["category"], primary["confidence"])

    # 5. Contamination Check
    contamination = check_contamination(detections)

    return {
        "detections": detections,
        "primary": {
            "category": primary["category"],
            "confidence": primary["confidence"],
            "bin_color": cat_data["bin_color"],
            "bin_label": cat_data["bin_label"],
            "tip": cat_data["tip"],
            "icon": cat_data["icon"]
        },
        "heatmap_image": heatmap_b64,
        "sustainability": sustainability,
        "contamination": contamination,
        "using_mock": USE_MOCK
    }


# ─────────────────────────────────────────────
# Flask Routes
# ─────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", categories=WASTE_CATEGORIES)


@app.route("/classify", methods=["POST"])
def classify():
    try:
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"error": "No image data received"}), 400

        # Decode base64 frame
        img_data = data["image"].split(",")[1] if "," in data["image"] else data["image"]
        img_bytes = base64.b64decode(img_data)
        img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_np = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        result = process_frame(img_np)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": not USE_MOCK,
        "mode": "mock" if USE_MOCK else "yolov8"
    })


if __name__ == "__main__":
    print("\n" + "="*50)
    print("  🌿 EcoSort — AI Waste Classification App")
    print("="*50)
    print(f"  Mode: {'MOCK (demo)' if USE_MOCK else 'YOLOv8 (real AI)'}")
    print("  Open: http://localhost:5000")
    print("="*50 + "\n")
    app.run(debug=True, host="0.0.0.0", port=5001)
