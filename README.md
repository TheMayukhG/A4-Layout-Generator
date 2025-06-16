# 🖼️ A4 Layout Designer

A smart, PyQt6-based desktop application to arrange multiple images efficiently on an A4 canvas — with precision editing tools, layout intelligence, and export-ready rendering.

## ✨ Features

- 📐 **Smart Layout Optimization**  
  Automatically arranges multiple images on an A4 sheet to minimize whitespace and avoid overlaps — including support for rotation and scaling.

- 🎚️ **Per-Image Adjustments**  
  Each image supports individual:
  - Scale
  - Contrast
  - Exposure
  - Saturation
  - Rotation (manual and auto)

- 🧠 **Automatic Scaling & Rotation**  
  Uses heuristics to determine optimal image orientation and size for best fit.

- 🔍 **Zoom Modes**  
  - Fit to Screen  
  - Fill Screen  
  - Manual Zoom (% control)

- 🧱 **Layout Modes** *(Planned)*  
  - Grid Layout  
  - Collage Layout  
  - Row/Column-based arrangements  
  - Preset Templates (CVs, posters, portfolios)

- 🖨️ **Export-Ready Output**  
  Render your layout at 300 DPI (print resolution) and save to high-quality images or PDFs (upcoming).

- 🖼️ **Visual Debug Tools**  
  - Bounding box display
  - Layer labels and index indicators
  - Smart error handling during layout failures

---

## 📦 Installation

### Requirements

- Python 3.8+
- `PyQt6`
- `Pillow`
- `NumPy`
- `scikit-learn`

### Install Dependencies

```bash
pip install PyQt6 Pillow numpy scikit-learn
