import sys
import os
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageDraw
from sklearn.cluster import KMeans
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout,
    QFileDialog, QScrollArea, QMessageBox, QSlider, QHBoxLayout,
    QComboBox, QListWidget, QListWidgetItem, QDoubleSpinBox,
    QTabWidget, QSplitter, QFrame, QAbstractItemView
)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt

# Constants
DPI = 300
A4_WIDTH, A4_HEIGHT = int(8.27 * DPI), int(11.69 * DPI)
PADDING = 10
TARGET_AREA_RATIO = 0.95
STEP_SIZE = 10

class ImageAdjustment:
    def __init__(self):
        self.scale = 1.0
        self.exposure = 1.0
        self.contrast = 1.0
        self.saturation = 1.0
        self.rotation = 0  # degrees


class ImageLayoutApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("A4 Smart Layout (AI Enhanced)")
        self.resize(1400, 900)

        self.images = []
        self.layout_data = {}
        self.composite_image = None
        self.scale_factor = 100
        self.adjustments = {}

        self.label = QLabel("Select images to begin.")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setMinimumSize(1, 1)  # Helps ScrollArea size negotiation
        self.label.setScaledContents(False)  # Prevent QLabel auto-stretching


        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.label)

        self.select_button = QPushButton("Select Images")
        self.select_button.clicked.connect(self.select_images)

        self.save_button = QPushButton("Save Layout as PNG/PDF")
        self.save_button.clicked.connect(self.save_output)
        self.save_button.setEnabled(False)

        self.scale_slider = QSlider(Qt.Orientation.Horizontal)
        self.scale_slider.setRange(50, 150)
        self.scale_slider.setValue(100)
        self.scale_slider.setTickInterval(10)
        self.scale_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.scale_slider.valueChanged.connect(self.update_scale)

        self.scale_label = QLabel("Scale: 100%")
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(self.scale_label)
        slider_layout.addWidget(self.scale_slider)

        self.zoom_box = QComboBox()
        self.zoom_box.addItems(["Fit to Screen", "Fill Screen", "25%", "50%", "75%", "100%", "150%"])
        self.zoom_box.currentTextChanged.connect(self.recalculate_layout)
        zoom_layout = QHBoxLayout()
        zoom_layout.addWidget(QLabel("Zoom:"))
        zoom_layout.addWidget(self.zoom_box)

        self.layout_box = QComboBox()
        self.layout_box.addItems(["Smart (AI)", "Grid", "Rows", "Columns"])
        self.layout_box.currentTextChanged.connect(self.recalculate_layout)

        self.layer_list = QListWidget()
        self.layer_list.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.layer_list.model().rowsMoved.connect(self.sync_layer_order)
        self.layer_list.currentRowChanged.connect(self.update_layer_controls)

        self.controls_frame = QFrame()
        self.controls_layout = QVBoxLayout()
        self.controls_frame.setLayout(self.controls_layout)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.addWidget(self.scroll)

        right_panel = QWidget()
        right_layout = QVBoxLayout()
        right_layout.addWidget(QLabel("Layers"))
        right_layout.addWidget(self.layer_list)
        right_layout.addWidget(QLabel("Adjustments"))
        right_layout.addWidget(self.controls_frame)
        right_layout.addStretch()
        right_layout.addWidget(QLabel("Layout Mode:"))
        right_layout.addWidget(self.layout_box)
        right_panel.setLayout(right_layout)

        self.splitter.addWidget(right_panel)
        self.splitter.setStretchFactor(0, 3)
        self.splitter.setStretchFactor(1, 1)

        layout = QVBoxLayout()
        layout.addWidget(self.select_button)
        layout.addLayout(slider_layout)
        layout.addLayout(zoom_layout)
        layout.addWidget(self.splitter)
        layout.addWidget(self.save_button)

        self.setLayout(layout)

    def update_layer_controls(self, index):
    # Safely clear all existing widgets/layouts
        for i in reversed(range(self.controls_layout.count())):
            item = self.controls_layout.itemAt(i)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
            else:
                self.controls_layout.removeItem(item)

        if index < 0 or index >= len(self.images):
            return

        adj = self.adjustments[index]
        self.add_spinbox("Scale", adj.scale, lambda val: self.set_adj(index, 'scale', val))
        self.add_spinbox("Exposure", adj.exposure, lambda val: self.set_adj(index, 'exposure', val))
        self.add_spinbox("Contrast", adj.contrast, lambda val: self.set_adj(index, 'contrast', val))
        self.add_spinbox("Saturation", adj.saturation, lambda val: self.set_adj(index, 'saturation', val))
        self.add_spinbox("Rotation", getattr(adj, 'rotation', 0), lambda val: self.set_adj(index, 'rotation', val), -180, 180)


    def add_spinbox(self, label, value, callback, minv=0.1, maxv=3.0, step=0.1):
        row = QHBoxLayout()
        row.addWidget(QLabel(label))
        spin = QDoubleSpinBox()
        spin.setRange(minv, maxv)
        spin.setSingleStep(step)
        spin.setValue(value)
        spin.valueChanged.connect(callback)
        row.addWidget(spin)
        self.controls_layout.addLayout(row)

    def set_adj(self, index, attr, val):
        setattr(self.adjustments[index], attr, val)
        self.recalculate_layout()

    def select_images(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Select Images", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if not paths:
            return

        self.images = self.load_images(paths)
        self.adjustments = {i: ImageAdjustment() for i in range(len(self.images))}
        self.layer_list.clear()
        for name, _ in self.images:
            self.layer_list.addItem(QListWidgetItem(name))

        self.recalculate_layout()
        self.save_button.setEnabled(True)

    def load_images(self, paths):
        return [(os.path.basename(p), Image.open(p).convert("RGB")) for p in paths]

    def update_scale(self, value):
        self.scale_factor = value
        self.scale_label.setText(f"Scale: {value}%")
        if self.images:
            self.recalculate_layout()

    def sync_layer_order(self):
        new_order = [self.layer_list.item(i).text() for i in range(self.layer_list.count())]
        name_to_data = {name: (name, img) for name, img in self.images}
        name_to_adj = {name2: adj for i, adj in self.adjustments.items() for name2, _ in [self.images[i]] if name2 == name2}
        self.images = [name_to_data[name] for name in new_order]
        self.adjustments = {i: name_to_adj[name] for i, name in enumerate(new_order)}
        self.recalculate_layout()

    def recalculate_layout(self):
        layout_type = self.layout_box.currentText()
        if layout_type == "Smart (AI)":
            self.layout_data = self.smart_optimized_layout(self.images, self.scale_factor / 100)
        else:
            self.layout_data = self.manual_layout(self.images, layout_type)
        self.composite_image = self.generate_mockup(self.images, self.layout_data)
        self.display_image(self.composite_image)

    def apply_adjustments(self, img: Image.Image, adj: ImageAdjustment):
        img = img.resize((int(img.width * adj.scale), int(img.height * adj.scale)))
        img = ImageEnhance.Brightness(img).enhance(adj.exposure)
        img = ImageEnhance.Contrast(img).enhance(adj.contrast)
        img = ImageEnhance.Color(img).enhance(adj.saturation)
        if adj.rotation:
            img = img.rotate(adj.rotation, expand=True)
        return img

    def smart_optimized_layout(self, images, master_scale):
        n = len(images)
        canvas_w, canvas_h = A4_WIDTH, A4_HEIGHT
        usable_area = canvas_w * canvas_h * TARGET_AREA_RATIO
        target_area = usable_area / n

        areas = np.array([img.size[0] * img.size[1] for _, img in images]).reshape(-1, 1)
        clusters = KMeans(n_clusters=2, random_state=42).fit_predict(areas)
        means = [np.mean(areas[clusters == i]) for i in range(2)]
        avg_area = np.mean(means)

        layout = {}
        used_grid = np.zeros((canvas_h, canvas_w), dtype=bool)
        sorted_images = sorted(enumerate(images), key=lambda x: -x[1][1].size[1] * x[1][1].size[0])

        for idx, (name, img) in sorted_images:
            w, h = img.size
            area = w * h
            adj = self.adjustments[idx]
            img_adj = self.apply_adjustments(img, adj)
            w, h = img_adj.size
            base_scale = (avg_area / area) ** 0.5 * master_scale
            w_scaled, h_scaled = int(w * base_scale), int(h * base_scale)

            if w_scaled > canvas_w or h_scaled > canvas_h:
                factor = min(canvas_w / w_scaled, canvas_h / h_scaled)
                w_scaled = int(w_scaled * factor)
                h_scaled = int(h_scaled * factor)

            for y in range(0, canvas_h - h_scaled, STEP_SIZE):
                for x in range(0, canvas_w - w_scaled, STEP_SIZE):
                    region = used_grid[y:y+h_scaled+PADDING, x:x+w_scaled+PADDING]
                    if region.shape != (h_scaled+PADDING, w_scaled+PADDING):
                        continue
                    if not region.any():
                        layout[idx] = (x, y, w_scaled, h_scaled)
                        used_grid[y:y+h_scaled+PADDING, x:x+w_scaled+PADDING] = True
                        break
                if idx in layout:
                    break
        return layout

    def manual_layout(self, images, mode):
        layout = {}
        x, y = 0, 0
        row_h = 0
        col_w = A4_WIDTH // len(images) if mode == "Columns" else 0
        for idx, (_, img) in enumerate(images):
            img = self.apply_adjustments(img, self.adjustments[idx])
            w, h = img.size
            if mode == "Grid":
                w, h = A4_WIDTH // 3, A4_HEIGHT // 3
                x = (idx % 3) * w
                y = (idx // 3) * h
            elif mode == "Rows":
                h = A4_HEIGHT // len(images)
                w = int(img.width * (h / img.height))
                x, y = 0, idx * h
            elif mode == "Columns":
                x = idx * col_w
                h = A4_HEIGHT
                w = int(img.width * (h / img.height))
            layout[idx] = (x, y, w, h)
        return layout

    def generate_mockup(self, images, layout):
        canvas = Image.new("RGB", (A4_WIDTH, A4_HEIGHT), "white")
        draw = ImageDraw.Draw(canvas)
        for idx, (_, img) in enumerate(images):
            if idx in layout:
                x, y, w, h = layout[idx]
                adj = self.adjustments[idx]
                img = self.apply_adjustments(img, adj).resize((w, h))
                canvas.paste(img, (x, y))
                draw.rectangle([x, y, x + w, y + h], outline="red", width=2)
                draw.text((x + 4, y + 4), str(idx), fill="blue")
        return canvas

    def display_image(self, img: Image.Image):
        try:
            zoom = self.zoom_box.currentText()

            # Original A4 dimensions
            base_w, base_h = img.width, img.height
            img_ratio = base_w / base_h

            view_w = max(1, self.scroll.viewport().width())
            view_h = max(1, self.scroll.viewport().height())

            if zoom.endswith("%"):
                factor = int(zoom[:-1]) / 100
                target_w = int(base_w * factor)
                target_h = int(base_h * factor)

            elif zoom == "Fit to Screen":
                view_ratio = view_w / view_h
                if view_ratio > img_ratio:
                    target_h = view_h - 20
                    target_w = int(target_h * img_ratio)
                else:
                    target_w = view_w - 20
                    target_h = int(target_w / img_ratio)

            elif zoom == "Fill Screen":
                # Fill while preserving aspect
                view_ratio = view_w / view_h
                if view_ratio > img_ratio:
                    target_w = view_w
                    target_h = int(view_w / img_ratio)
                else:
                    target_h = view_h
                    target_w = int(view_h * img_ratio)

            else:
                # Default fallback
                target_w = base_w // 2
                target_h = base_h // 2

            # Clamp minimums
            target_w = max(1, target_w)
            target_h = max(1, target_h)

            # Resize
            img_resized = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
            img_bytes = img_resized.tobytes("raw", "RGB")
            qim = QImage(img_bytes, img_resized.width, img_resized.height, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qim)

            self.label.setPixmap(pixmap)
            self.label.setFixedSize(pixmap.size())  # THIS is critical
            self.label.update()

        except Exception as e:
            print(f"[Zoom Error]: {e}")
            self.label.setText("Image failed to render.")





    def save_output(self):
        if not self.composite_image:
            QMessageBox.warning(self, "Warning", "No image to save.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Layout", "layout", "PNG Files (*.png);;PDF Files (*.pdf)")
        if path:
            if path.endswith(".pdf"):
                self.composite_image.save(path, "PDF", resolution=300.0)
            else:
                self.composite_image.save(path)
            QMessageBox.information(self, "Saved", f"Layout saved to:\n{path}")

def main():
    app = QApplication(sys.argv)
    window = ImageLayoutApp()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
