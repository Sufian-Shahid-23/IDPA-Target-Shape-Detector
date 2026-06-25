# 🎯 IDPA Target Shape Detector

> An intelligent computer vision pipeline that automatically detects, classifies, and outlines all scored zones on IDPA shooting targets — from any angle, any lighting, any camera.

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=flat&logo=opencv&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat&logo=jupyter&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.21+-013243?style=flat&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.4+-11557C?style=flat)

---

## 📌 Overview

IDPA (International Defensive Pistol Association) paper targets have multiple scored zones that need to be tracked after a shooting session. This project automates the detection of every printed shape on those targets using classical computer vision — no neural networks, no training data required.

Given a folder of target photos — including heavily angled shots, varying resolutions, images with dozens of bullet holes, and targets that are small in the frame — the pipeline corrects the perspective, finds all shapes, classifies them, and displays annotated results in a single grid view.

---

## ✨ Features

- **Automatic perspective correction** — handles side-angle shots using a 3-stage deskew system
- **Auto-zoom** — detects when the target is far/small in the frame and crops to it automatically
- **Bullet hole tolerance** — shapes with punched-through edges are still detected cleanly
- **Concave shape detection** — correctly outlines the body silhouette including its neck cutout without collapsing it to a convex shape
- **Resolution-independent** — all thresholds scale with image size, works from 444px to 1200px+
- **Dual threshold detection** — detects both light-outline shapes and dark/black filled shapes
- **Shape classification** — labels each detection as circle, ellipse, rectangle, square, or polygon (n pts)
- **Batch processing** — processes an entire folder of images and displays all results in one click
- **Noise rejection** — filters text, numbers, bullet holes, background walls, and tape artifacts

---

## 🗂 Target Shapes Detected

| Shape | Description |
|---|---|
| **Outer body silhouette** | Large concave polygon with neck cutout and angled corners |
| **torso ring** | big octagon inside the body outline |
| **Inner torso ring** | Smaller octagon inside the torso |
| **Head rectangle** | Rectangle at the top connected to the neck |
| **Torso circle** | Large circle in the center chest zone (black or green fill) |
| **Head circle** | Small circle inside the head rectangle |

---

## 🧠 How It Works — Pipeline

```
┌─────────────┐    ┌──────────────┐    ┌─────────────────────────────────────┐
│  Load Image │───▶│    Resize    │───▶│         deskew_if_needed()          │
│             │    │ (keep ratio) │    │  A: Rotation via polygon edge angle  │
└─────────────┘    └──────────────┘    │  B: Auto-zoom via polygon fill ratio │
                                       │  C: Perspective via circle ellipse   │
                                       └─────────────────┬───────────────────┘
                                                         │
                        ┌────────────────────────────────▼──────────────────┐
                        │                  Preprocessing                     │
                        │  CLAHE → GaussianBlur → AdaptiveThreshold + Otsu  │
                        │  bitwise_OR merge → MORPH_CLOSE → findContours    │
                        └────────────────────────────────┬──────────────────┘
                                                         │
                        ┌────────────────────────────────▼──────────────────┐
                        │               Filter Loop (per contour)            │
                        │  area · compactness · aspect · solidity · dedup   │
                        │  border noise check · scale-relative thresholds   │
                        └────────────────────────────────┬──────────────────┘
                                                         │
                   ┌─────────────────────────────────────▼──────────────────────┐
                   │                    Classify + Clean                         │
                   │  Circle? → ellipse2Poly smooth ring                         │
                   │  Large polygon? → isolated canvas morph + spike buster      │
                   │  Small polygon? → convexHull + approxPolyDP                 │
                   │  Classify on FINAL contour (after cleaning)                 │
                   └─────────────────────────────────────┬──────────────────────┘
                                                         │
                        ┌────────────────────────────────▼──────────────────┐
                        │           Post-loop Noise Removal                  │
                        │  Drop shapes < 2% of largest detected shape area  │
                        └────────────────────────────────┬──────────────────┘
                                                         │
                        ┌────────────────────────────────▼──────────────────┐
                        │           Draw & Display                           │
                        │  Red outlines + labels on original image           │
                        │  All images in one matplotlib grid                 │
                        └───────────────────────────────────────────────────┘
```

### Deskew System — 3 Substages

The deskew function is the most complex part of the pipeline. It runs before any shape detection and uses the **inner torso polygon** as a reliable anchor — it is the only shape detected consistently under every condition including heavy angles and bullet holes.

**Stage A — Rotation correction**
Finds the inner torso polygon, extracts all its edges, takes the longest edge, and computes its angle from horizontal via `arctan2`. Applies `warpAffine` rotation if tilt exceeds 5°.

**Stage B — Auto-zoom**
Measures how much of the frame the polygon fills. If it fills less than 20% (target is far away), the image is cropped and zoomed centered on the polygon with a 2.5× scale factor to include the full outer silhouette.

**Stage C — Perspective correction**
After zooming, detects the torso circle using Otsu thresholding. A circle viewed at an angle becomes an ellipse — the ratio of ellipse axes (`min/max`) is exactly the cosine of the viewing angle. Stretches the short axis back to match the long axis, correcting the perspective distortion mathematically.

---

## 📁 Project Structure

```
idpa-target-detector/
│
├── target_detector.ipynb       # Main Jupyter notebook
├── requirements.txt            # Python dependencies
├── README.md                   # This file
---

## ⚙️ Installation & Setup

### Prerequisites

- Python 3.8 or higher
- pip
- Jupyter Notebook or JupyterLab

### Step 1 — Clone the repository

```bash
git clone https://github.com/yourusername/idpa-target-detector.git
cd idpa-target-detector
```

### Step 2 — Create a virtual environment (recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Add your images

```bash
Place your target photos in the image path at the top. Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.webp`
```

### Step 5 — Run the notebook

```bash
jupyter notebook target_detector.ipynb
```

Run all cells. All detected shapes will be displayed in one grid at the end.

---

## 🔧 Requirements

Create a `requirements.txt` with:

```
opencv-python>=4.5.0
numpy>=1.21.0
matplotlib>=3.4.0
jupyter>=1.0.0
```

---

## 🎛 Tuning Parameters

If detection isn't perfect for your specific images, these are the key values to adjust:

| Parameter | Default | What it controls |
|---|---|---|
| `target_width` | `666` | Working resolution — increase for more detail |
| `ZOOM_THRESHOLD` | `0.20` | How small the target must be to trigger auto-zoom |
| `ZOOM_SCALE` | `2.5` | How much to scale up when zooming |
| `angle_threshold` | `5°` | Minimum tilt to trigger rotation correction |
| `PERSP_THRESHOLD` | `0.92` | How elliptical the circle must be to trigger perspective fix |
| `epsilon` (large poly) | `0.005 × arcLength` | Smoothing of outer silhouette — higher = smoother |
| `spike angle` | `115°` | Max angle for bullet hole removal — higher removes more |
| `MIN_RELATIVE_AREA` | `largest × 0.02` | Post-loop noise floor — raise to remove more noise |
| `compactness limit` | `500` | Jagged fragment filter — lower to be more aggressive |

---

## 🖼 Example Results

The pipeline draws red outlines and labels directly onto the original image for each detected shape:

```
#1 polygon        — outer body silhouette
#2 polygon (8 pts)— inner torso ring  
#3 rectangle      — head box
#4 circle         — torso scoring zone
#5 circle         — head scoring zone
```

All images in a batch are displayed together in a 4-column grid with the filename and shape count as the title for each.

---

## 🧩 Technical Challenges Solved

**Concave polygon detection**
Standard `convexHull` collapses the neck indentation of the body silhouette into a straight line. The solution uses an isolated canvas approach: draw the filled contour, apply morphological smoothing, re-extract the outer contour, then use `approxPolyDP` without hull.

**Bullet hole tolerance**
Bullet holes create sharp inward dents on shape edges. A spike buster algorithm removes these by identifying points where both adjacent edges are very short (< 40% of a threshold) AND the angle is sharp (< 115°). This condition protects real corners like shoulders which always have at least one long edge.

**Resolution independence**
All pixel-based thresholds are derived from `IMG_DIAG` and `IMG_AREA` as ratios, so the same code works identically on a 444px phone photo and a 1200px DSLR shot.

**Perspective correction from circle geometry**
A circle viewed at an angle θ from perpendicular becomes an ellipse with axis ratio = cos(θ). This means the correction scale factor is exactly `long_axis / short_axis` — no approximations needed.

---

## 📄 License

MIT License — feel free to use, modify, and distribute.

---

## 🙋 Author: Sufian Shahid

Built as a computer vision portfolio project demonstrating classical CV techniques including adaptive thresholding, morphological operations, contour analysis, perspective transforms, and geometric shape classification — without any machine learning or neural networks.
