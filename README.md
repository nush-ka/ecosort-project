# 🌿 EcoSort — AI-Powered Smart Waste Classification

**Detect. Classify. Sustain.**  
A fully browser-based web app using YOLOv8, Grad-CAM heatmaps, contamination detection, and a live sustainability score.

---

## 📁 Project Structure

```
ecosort/
├── app.py               # Flask backend (main server)
├── train_model.py       # YOLOv8 fine-tuning script
├── requirements.txt     # Python dependencies
├── models/              # Put your trained .pt file here
│   └── ecosort_yolov8.pt  (auto-loaded if present)
└── templates/
    └── index.html       # Full frontend UI
```

---

## 🚀 Setup & Run (Mac)

### Step 1 — Install Python 3.10+

Check if you already have it:
```bash
python3 --version
```

If not, install via Homebrew:
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install python@3.11
```

---

### Step 2 — Create a Virtual Environment

```bash
cd ~/Desktop         # or wherever you saved the project
cd ecosort
python3 -m venv venv
source venv/bin/activate
```

You'll see `(venv)` in your terminal — good!

---

### Step 3 — Install Dependencies

```bash
pip install --upgrade pip
pip install flask opencv-python-headless Pillow numpy
```

**Optional — for real YOLOv8 AI (recommended):**
```bash
pip install ultralytics
```

---

### Step 4 — Run the App

```bash
python app.py
```

You'll see:
```
==================================================
  🌿 EcoSort — AI Waste Classification App
==================================================
  Mode: MOCK (demo)   ← or YOLOv8 (real AI) if ultralytics installed
  Open: http://localhost:5000
==================================================
```

---

### Step 5 — Open in Browser

Open **Safari** or **Chrome** and go to:
```
http://localhost:5000
```

> ⚠️ **For webcam access on Mac:** Use Chrome or Firefox. Safari may block localhost camera. If prompted, click **Allow** for camera permission.

---

## 🤖 Modes

| Mode | What it does |
|------|-------------|
| **Demo / Mock** | Works without any model. Generates realistic mock detections using image properties. Great for UI testing. |
| **YOLOv8 (pretrained)** | Uses YOLOv8n pretrained on COCO. Maps 80 COCO classes to 4 waste categories. Installs with `pip install ultralytics`. |
| **YOLOv8 (fine-tuned)** | Best accuracy. Place `ecosort_yolov8.pt` in the `models/` folder. Auto-detected on startup. |

---

## 🧠 Training Your Own Model

1. Download datasets:
   - **TrashNet**: https://github.com/garwalar/TrashNet
   - **TACO**: http://taco-dataset.net

2. Organize into YOLO format (images + labels in `dataset/`)

3. Run training:
```bash
pip install ultralytics
python train_model.py
```

4. The trained model is auto-saved to `models/ecosort_yolov8.pt`  
5. Restart `app.py` — it loads automatically

---

## 🛠 Troubleshooting (Mac)

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: flask` | Run `source venv/bin/activate` first |
| Camera not showing | Use Chrome, not Safari. Allow camera in System Settings → Privacy → Camera |
| Port 5000 in use | Change port in app.py: `app.run(port=5001)` — then open `http://localhost:5001` |
| `cv2` error | Run `pip install opencv-python-headless` |
| Slow on M1/M2 Mac | Normal — ultralytics runs on CPU by default. Consider `yolov8n.pt` (nano) for speed. |

---

## ✅ Features

- 🎥 **Live webcam feed** — browser-based, no plugins
- 🔍 **Real-time classification** — auto-classifies every 3 seconds
- 🗂 **4 categories** — Plastic (Blue), Paper (Yellow), Metal (Grey), Organic (Green)
- 🌡 **Grad-CAM heatmaps** — visual AI explainability
- ⚠️ **Contamination detection** — bounding box overlap analysis
- 🌱 **Sustainability Score** — CO₂-weighted, scientifically grounded
- 📊 **Live dashboard** — items classified, CO₂ saved, confidence, alerts
- 📋 **Session history** — full log of all detections

---

## 📚 References

- YOLOv8: https://docs.ultralytics.com
- TrashNet: https://github.com/garwalar/TrashNet
- TACO: http://taco-dataset.net
- India SWM Rules 2016: https://moef.gov.in
