import sys
import os
import base64
import json
import tempfile
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTextEdit, QLineEdit, QFileDialog,
    QFrame, QScrollArea, QProgressBar, QSplitter, QGraphicsView, 
    QGraphicsScene, QGraphicsPixmapItem, QGraphicsRectItem, QGraphicsTextItem
)
from PySide6.QtCore import Qt, QThread, Signal, QSize, QTimer, QElapsedTimer
from PySide6.QtGui import QPixmap, QFont, QColor, QImage, QPen, QBrush

from app.violation_detector import detect_violations

# ── Modern Theme Stylesheet ───────────────────────────
MODERN_STYLESHEET = """
QMainWindow, QWidget {
    background: #0F172A;
    color: #F8FAFC;
    font-family: 'Inter', 'Segoe UI', sans-serif;
}

QFrame#Sidebar {
    background: #1E293B;
    border-right: 1px solid #334155;
}

QFrame#ResultCard {
    background: #1E293B;
    border-radius: 12px;
    border: 1px solid #334155;
    padding: 16px;
    margin-bottom: 12px;
}

QLabel#Title {
    font-size: 20px;
    font-weight: 800;
    color: #38BDF8;
}

QLabel#Header {
    font-size: 12px;
    font-weight: 700;
    color: #94A3B8;
    text-transform: uppercase;
}

QLabel#Verdict_VIOLATION {
    color: #EF4444;
    font-weight: 700;
}

QLabel#Verdict_COMPLIANT {
    color: #10B981;
    font-weight: 700;
}

QLabel#Verdict_UNCERTAIN {
    color: #F59E0B;
    font-weight: 700;
}

QPushButton#PrimaryBtn {
    background: #38BDF8;
    color: #0F172A;
    border-radius: 8px;
    font-weight: 700;
    padding: 12px;
    border: none;
}

QPushButton#PrimaryBtn:hover {
    background: #7DD3FC;
}

QPushButton#PrimaryBtn:disabled {
    background: #334155;
    color: #94A3B8;
}

QLineEdit, QTextEdit {
    background: #0F172A;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 10px;
    color: #F8FAFC;
}

QScrollBar:vertical {
    border: none;
    background: #0F172A;
    width: 10px;
    margin: 0px;
}

QScrollBar::handle:vertical {
    background: #334155;
    min-height: 20px;
    border-radius: 5px;
}

QProgressBar {
    background: #1E293B;
    border: none;
    border-radius: 4px;
    height: 4px;
    text-align: center;
}

QProgressBar::chunk {
    background: #38BDF8;
    border-radius: 4px;
}

QPushButton#SecondaryBtn {
    background: transparent;
    color: #94A3B8;
    border-radius: 8px;
    font-weight: 600;
    padding: 8px;
    border: 1px solid #334155;
}
QPushButton#SecondaryBtn:hover {
    background: #1E293B;
    color: #F8FAFC;
    border-color: #64748B;
}
QPushButton#SecondaryBtn:disabled {
    color: #334155;
    border-color: #1E293B;
}
"""

class ResizableGraphicsView(QGraphicsView):
    """QGraphicsView with auto-fit on resize, scroll-wheel zoom, and drag-to-pan."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self._fit_on_next = True   # fit when image is first loaded

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._fit_on_next and self.scene() and self.scene().items():
            self.fitInView(self.scene().itemsBoundingRect(), Qt.KeepAspectRatio)

    def wheelEvent(self, event):
        """Scroll wheel zooms in/out; hold Ctrl to pan."""
        factor = 1.18 if event.angleDelta().y() > 0 else 1 / 1.18
        self._fit_on_next = False   # disable auto-fit once user zooms
        self.scale(factor, factor)

class AnalysisWorker(QThread):
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, media_path, custom_prompt, force_fresh=False):
        super().__init__()
        self.media_path = media_path
        self.custom_prompt = custom_prompt
        self.force_fresh = force_fresh

    def run(self):
        try:
            if self.force_fresh:
                from app.violation_detector import _RESULT_CACHE, _image_cache_key
                key = _image_cache_key(self.media_path, self.custom_prompt or "")
                _RESULT_CACHE.pop(key, None)

            result = detect_violations(
                media_path_or_url=self.media_path,
                is_video=False,
                custom_prompt=self.custom_prompt
            )
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))

class ViolationCard(QFrame):
    def __init__(self, verdict):
        super().__init__()
        self.setObjectName("ResultCard")
        layout = QVBoxLayout(self)
        
        # Header Row
        header_layout = QHBoxLayout()
        ref = QLabel(verdict.get("sbc_reference", "Unknown Ref"))
        ref.setStyleSheet("font-weight: bold; color: #38BDF8;")
        
        status = verdict.get("verdict", "UNCERTAIN")
        status_lbl = QLabel(status)
        status_lbl.setObjectName(f"Verdict_{status}")
        
        header_layout.addWidget(ref)
        header_layout.addStretch()
        header_layout.addWidget(status_lbl)
        layout.addLayout(header_layout)
        
        # Priority
        prio = verdict.get("priority", "LOW")
        prio_lbl = QLabel(f"Priority: {prio}")
        prio_lbl.setStyleSheet("font-size: 10px; color: #94A3B8;")
        layout.addWidget(prio_lbl)
        
        # Rule Text
        rule = QLabel(verdict.get("rule_text", ""))
        rule.setWordWrap(True)
        rule.setStyleSheet("font-size: 13px; margin-top: 4px;")
        layout.addWidget(rule)
        
        # Evidence
        evidence = verdict.get("evidence", "")
        if evidence:
            ev_lbl = QLabel(f"Evidence: {evidence}")
            ev_lbl.setWordWrap(True)
            ev_lbl.setStyleSheet("font-size: 12px; color: #CBD5E1; font-style: italic; margin-top: 4px;")
            layout.addWidget(ev_lbl)

class ModernVisionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SBC Compliance AI Inspector")
        self.resize(1280, 800)
        self.selected_file = None
        self._elapsed = QElapsedTimer()    # measures wall time
        self._tick_timer = QTimer(self)    # fires every second for live display
        self._tick_timer.timeout.connect(self._update_timer_label)
        
        self.init_ui()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ── Sidebar ──
        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(350)
        side_layout = QVBoxLayout(sidebar)
        side_layout.setContentsMargins(24, 32, 24, 32)
        side_layout.setSpacing(20)

        title = QLabel("SBC Inspector")
        title.setObjectName("Title")
        side_layout.addWidget(title)

        subtitle = QLabel("Saudi Building Code Compliance")
        subtitle.setStyleSheet("color: #94A3B8; font-size: 12px;")
        side_layout.addWidget(subtitle)

        side_layout.addSpacing(20)

        lbl_media = QLabel("INSPECTION MEDIA")
        lbl_media.setObjectName("Header")
        side_layout.addWidget(lbl_media)
        
        self.upload_btn = QPushButton("Select Site Image")
        self.upload_btn.setObjectName("PrimaryBtn")
        self.upload_btn.clicked.connect(self.open_file)
        side_layout.addWidget(self.upload_btn)

        self.file_label = QLabel("No file selected")
        self.file_label.setStyleSheet("font-size: 11px; color: #64748B;")
        self.file_label.setWordWrap(True)
        side_layout.addWidget(self.file_label)

        side_layout.addSpacing(10)

        lbl_prompt = QLabel("CUSTOM PROMPT")
        lbl_prompt.setObjectName("Header")
        side_layout.addWidget(lbl_prompt)
        self.prompt_edit = QTextEdit()
        self.prompt_edit.setPlaceholderText("Add specific focus areas (optional)...")
        self.prompt_edit.setMaximumHeight(100)
        side_layout.addWidget(self.prompt_edit)

        side_layout.addStretch()

        self.analyze_btn = QPushButton("Start Analysis")
        self.analyze_btn.setObjectName("PrimaryBtn")
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.clicked.connect(self.run_analysis)
        side_layout.addWidget(self.analyze_btn)

        self.reanalyze_btn = QPushButton("Re-analyze (Fresh)")
        self.reanalyze_btn.setObjectName("SecondaryBtn")
        self.reanalyze_btn.setEnabled(False)
        self.reanalyze_btn.setToolTip("Clear cache and run a fresh analysis on this image")
        self.reanalyze_btn.clicked.connect(self.run_fresh_analysis)
        side_layout.addWidget(self.reanalyze_btn)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.hide()
        side_layout.addWidget(self.progress)

        # ── Timing Section ──
        side_layout.addSpacing(8)
        lbl_time_header = QLabel("ANALYSIS TIME")
        lbl_time_header.setObjectName("Header")
        side_layout.addWidget(lbl_time_header)

        self.time_label = QLabel("--")
        self.time_label.setStyleSheet(
            "font-size: 22px; font-weight: 800; color: #38BDF8; letter-spacing: 1px;"
        )
        side_layout.addWidget(self.time_label)

        self.time_sub = QLabel("")
        self.time_sub.setStyleSheet("font-size: 10px; color: #64748B;")
        side_layout.addWidget(self.time_sub)

        main_layout.addWidget(sidebar)

        # ── Main Content Area ──
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(24, 32, 24, 32)
        
        # Splitter for Image and Results
        splitter = QSplitter(Qt.Vertical)
        
        # Image Preview
        self.image_view = ResizableGraphicsView()
        self.image_view.setStyleSheet("background: #020617; border-radius: 12px; border: 1px solid #1E293B;")
        self.scene = QGraphicsScene()
        self.image_view.setScene(self.scene)
        self.image_view.setRenderHint(self.image_view.renderHints().SmoothPixmapTransform)
        splitter.addWidget(self.image_view)

        # Results Area
        results_widget = QWidget()
        results_layout = QVBoxLayout(results_widget)
        results_layout.setContentsMargins(0, 10, 0, 0)
        
        lbl_verdicts = QLabel("COMPLIANCE VERDICTS")
        lbl_verdicts.setObjectName("Header")
        results_layout.addWidget(lbl_verdicts)
        
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setStyleSheet("background: transparent; border: none;")
        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        self.scroll_layout.setAlignment(Qt.AlignTop)
        self.scroll.setWidget(self.scroll_content)
        results_layout.addWidget(self.scroll)
        
        splitter.addWidget(results_widget)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        
        content_layout.addWidget(splitter)
        main_layout.addWidget(content, 1)

    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.jpeg *.webp)")
        if path:
            self.selected_file = path
            self.file_label.setText(os.path.basename(path))
            self.analyze_btn.setEnabled(True)
            self.reanalyze_btn.setEnabled(True)
            self.display_image(path)

    def display_image(self, path_or_b64: str, is_base64: bool = False):
        self.scene.clear()
        if is_base64:
            if "," in path_or_b64:
                path_or_b64 = path_or_b64.split(",", 1)[1]
            import base64
            img_bytes = base64.b64decode(path_or_b64)
            pixmap = QPixmap()
            pixmap.loadFromData(img_bytes, "JPEG")
        else:
            pixmap = QPixmap(path_or_b64)

        if not pixmap.isNull():
            self.current_pixmap = pixmap
            item = QGraphicsPixmapItem(pixmap)
            self.scene.addItem(item)
            self.image_view._fit_on_next = True   # re-enable auto-fit for new image
            QTimer.singleShot(0, lambda: self.image_view.fitInView(
                self.scene.itemsBoundingRect(), Qt.KeepAspectRatio
            ))

    def _update_timer_label(self):
        """Called every second while analysis is running — shows live elapsed time."""
        secs = self._elapsed.elapsed() // 1000
        mins, s = divmod(secs, 60)
        self.time_label.setText(f"{mins}:{s:02d}")
        self.time_sub.setText("Analyzing...")

    def run_analysis(self, force_fresh: bool = False):
        if not self.selected_file:
            return

        self.analyze_btn.setEnabled(False)
        self.upload_btn.setEnabled(False)
        self.reanalyze_btn.setEnabled(False)
        self.progress.show()

        # Start live timer
        self._elapsed.start()
        self._tick_timer.start(1000)
        self.time_label.setText("0:00")
        self.time_sub.setText("Analyzing...")
        
        # Clear previous results
        for i in reversed(range(self.scroll_layout.count())): 
            widget = self.scroll_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)

        self.worker = AnalysisWorker(self.selected_file, self.prompt_edit.toPlainText(), force_fresh=force_fresh)
        self.worker.finished.connect(self.on_success)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def run_fresh_analysis(self):
        """Clear cache for current image and run a completely fresh analysis."""
        self.reanalyze_btn.setEnabled(False)
        self.run_analysis(force_fresh=True)

    def on_success(self, result):
        self._tick_timer.stop()
        secs = self._elapsed.elapsed() // 1000
        mins, s = divmod(secs, 60)
        self.time_label.setText(f"{mins}:{s:02d}")
        self.time_sub.setText(f"Total analysis time")

        self.progress.hide()
        self.analyze_btn.setEnabled(True)
        self.upload_btn.setEnabled(True)
        self.reanalyze_btn.setEnabled(True)

        # ── Show annotated image from backend (OpenCV-drawn bounding boxes) ──
        annotated_b64 = result.get("annotated_image")
        if annotated_b64:
            # Backend returned an annotated image — display it directly
            self.display_image(annotated_b64, is_base64=True)
        else:
            # No annotations (no violations with bbox) — show original
            self.display_image(self.selected_file)

        verdicts = result.get("verdicts", [])
        violations_only = [v for v in verdicts if v.get("verdict") == "VIOLATION"]

        # Add verdict cards — VIOLATION only
        if not violations_only:
            no_results = QLabel("No violations detected in this image.")
            no_results.setStyleSheet("color: #94A3B8; font-style: italic; padding: 20px;")
            self.scroll_layout.addWidget(no_results)
        else:
            for v in violations_only:
                self.scroll_layout.addWidget(ViolationCard(v))

        self.scroll_layout.addStretch()

    def on_error(self, message):
        self._tick_timer.stop()
        secs = self._elapsed.elapsed() // 1000
        mins, s = divmod(secs, 60)
        self.time_label.setText(f"{mins}:{s:02d}")
        self.time_sub.setText("Failed")

        self.progress.hide()
        self.analyze_btn.setEnabled(True)
        self.upload_btn.setEnabled(True)
        self.reanalyze_btn.setEnabled(True)
        err_lbl = QLabel(f"Error: {message}")
        err_lbl.setStyleSheet("color: #EF4444; font-weight: bold; padding: 20px;")
        err_lbl.setWordWrap(True)
        self.scroll_layout.addWidget(err_lbl)

if __name__ == "__main__":
    # Ensure app directory is in path for imports
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(current_dir)
    
    app = QApplication(sys.argv)
    app.setStyleSheet(MODERN_STYLESHEET)
    
    window = ModernVisionApp()
    window.show()
    sys.exit(app.exec())
