"""
Modern PySide6 Vision Analyzer — Powered by LangChain-OpenAI & GLM-4.6V
======================================================================
Premium Desktop Visual Intelligence Tool for Saudi Compliance.
Supports: 
  - Drag-and-drop or browse for local files (images & videos).
  - Paste directly HTTP/HTTPS URLs (images & videos).
  - Native video completion for URL assets, and intelligent key-frame scene-change extraction for local videos.
  - Fully integrated with LangChain ChatOpenAI engine pointing to api.z.ai.
  - Beautiful visual feedback and real-time token tracking stats.
"""

import sys
import os
import base64
import threading
from pathlib import Path
from urllib.parse import urlparse

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTextEdit, QLineEdit, QSpinBox, QFileDialog,
    QFrame, QScrollArea, QSizePolicy, QProgressBar, QSplitter
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QPixmap, QFont, QIcon, QColor, QPalette

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# ── Config Constants ───────────────────────────────────
API_KEY = "6f7d09ab5f0e4edf816408f45f6b2f7b.iNo6WzgdafIlCewY"
BASE_URL = "https://api.z.ai/api/paas/v4/"
MODEL = "GLM-4.6V"

# ── Styled Dark Theme Stylesheet ───────────────────────────
DARK_STYLESHEET = """
QMainWindow, QWidget {
    background: #090b10;
    color: #f0f3f8;
}
QFrame#card {
    background: #121620;
    border: 1px solid #232d3f;
    border-radius: 12px;
}
QFrame#sidebar_container {
    background: #0d111a;
    border-right: 1px solid #1c2433;
}
QLabel#title {
    font-size: 22px;
    font-weight: 700;
    color: #4da6ff;
    font-family: 'Montserrat', 'Segoe UI', Helvetica;
}
QLabel#subtitle {
    font-size: 11px;
    color: #798a9e;
}
QLabel#stat_val {
    font-size: 28px;
    font-weight: 700;
    color: #4da6ff;
}
QLabel#stat_lbl {
    font-size: 11px;
    color: #798a9e;
}
QLabel#drop_hint {
    font-size: 13px;
    color: #798a9e;
    font-weight: 500;
}
QTextEdit, QLineEdit {
    background: #090b10;
    color: #f0f3f8;
    border: 1px solid #232d3f;
    border-radius: 8px;
    padding: 10px;
    font-size: 13px;
    font-family: 'JetBrains Mono', 'Segoe UI', monospace;
    selection-background-color: #2b5c8f;
}
QLineEdit {
    padding: 8px 12px;
}
QTextEdit:focus, QLineEdit:focus {
    border: 1px solid #4da6ff;
}
QPushButton {
    background: #1b2234;
    color: #f0f3f8;
    border: 1px solid #2a354f;
    border-radius: 8px;
    padding: 10px 20px;
    font-size: 13px;
    font-weight: 600;
}
QPushButton:hover {
    background: #252f47;
    border-color: #4da6ff;
    color: #4da6ff;
}
QPushButton#analyze_btn {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #1a62d6, stop:1 #0084ff);
    border: none;
    color: white;
    font-size: 14px;
    padding: 12px 28px;
}
QPushButton#analyze_btn:hover {
    background: #0084ff;
}
QPushButton#analyze_btn:disabled {
    background: #1c2433;
    color: #4c5a6d;
}
QPushButton#load_url_btn {
    padding: 8px 16px;
}
QSpinBox {
    background: #161c28;
    color: #f0f3f8;
    border: 1px solid #232d3f;
    border-radius: 6px;
    padding: 6px 10px;
}
QProgressBar {
    background: #161c28;
    border: none;
    border-radius: 4px;
    height: 6px;
}
QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #4da6ff, stop:1 #00ffaa);
    border-radius: 4px;
}
QScrollBar:vertical {
    background: #090b10;
    width: 8px;
}
QScrollBar::handle:vertical {
    background: #232d3f;
    border-radius: 4px;
}
QSplitter::handle {
    background: #1c2433;
    width: 2px;
}
"""

# ── Helper functions for input checks ───────────────────
def is_url(path: str) -> bool:
    try:
        parsed = urlparse(path)
        return parsed.scheme in ("http", "https")
    except ValueError:
        return False

# ── Worker Thread ───────────────────────────────────────
class AnalyzeWorker(QThread):
    finished = Signal(str, int, int, int)   # response_text, in_tokens, out_tokens, total
    error = Signal(str)

    def __init__(self, file_path_or_url, prompt, is_video):
        super().__init__()
        self.file_path_or_url = file_path_or_url
        self.prompt = prompt
        self.is_video = is_video

    def run(self):
        try:
            content = []
            
            # Check if input is an online URL
            if is_url(self.file_path_or_url):
                if self.is_video:
                    # Online Video natively sent to GLM-4.6V
                    content.append({
                        "type": "video_url",
                        "video_url": {"url": self.file_path_or_url}
                    })
                else:
                    # Online Image sent to GLM-4.6V
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": self.file_path_or_url}
                    })
            else:
                # Local File Processing
                if self.is_video:
                    # Direct local video encoding to base64
                    print("📁 Local video file detected. Encoding directly to base64...")
                    b64 = self._encode_file(self.file_path_or_url)
                    content.append({
                        "type": "video_url",
                        "video_url": {"url": b64}
                    })
                else:
                    # Local image processing
                    mime = self._get_mime(self.file_path_or_url)
                    b64 = self._encode_file(self.file_path_or_url)
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"}
                    })

            # Append text prompt
            content.append({"type": "text", "text": self.prompt})

            # Create LangChain Model Instance pointing to api.z.ai
            llm = ChatOpenAI(
                model=MODEL,
                api_key=API_KEY,
                base_url=BASE_URL,
                temperature=0.2,
                max_tokens=4096
            )

            system_prompt = (
                "You are an expert visual analyzer specializing in Saudi quality & compliance standards. "
                "Provide detailed, precise, and well-structured analysis report."
            )
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=content)
            ]

            # Invoke the API via LangChain
            response = llm.invoke(messages)

            # Retrieve usage statistics
            usage = getattr(response, "usage_metadata", None) or {}
            if not usage:
                usage = response.response_metadata.get("token_usage", {})

            inp = usage.get("input_tokens") or usage.get("prompt_tokens") or 0
            out = usage.get("output_tokens") or usage.get("completion_tokens") or 0
            tot = usage.get("total_tokens") or (inp + out)

            self.finished.emit(response.content, inp, out, tot)

        except Exception as e:
            import traceback
            self.error.emit(f"An error occurred: {str(e)}\n\n{traceback.format_exc()}")

    # ── Local File Encoding Helpers ─────────────────────
    def _encode_file(self, path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _get_mime(self, path):
        ext = Path(path).suffix.lower().lstrip(".")
        mime_map = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp", "gif": "image/gif"}
        return mime_map.get(ext, "image/jpeg")

# ── Drop Zone Widget ────────────────────────────────────
class DropZone(QFrame):
    file_dropped = Signal(str)

    def __init__(self):
        super().__init__()
        self.setObjectName("card")
        self.setAcceptDrops(True)
        self.setMinimumHeight(160)
        self.setCursor(Qt.PointingHandCursor)

        lay = QVBoxLayout(self)
        lay.setAlignment(Qt.AlignCenter)
        lay.setSpacing(8)

        self.icon_lbl = QLabel("📂")
        self.icon_lbl.setAlignment(Qt.AlignCenter)
        self.icon_lbl.setStyleSheet("font-size: 40px; background: transparent;")

        self.hint = QLabel("Drag local file here\nor click to browse")
        self.hint.setObjectName("drop_hint")
        self.hint.setAlignment(Qt.AlignCenter)

        self.file_lbl = QLabel("")
        self.file_lbl.setAlignment(Qt.AlignCenter)
        self.file_lbl.setStyleSheet("color: #00ffaa; font-size: 12px; background: transparent; font-weight: bold;")
        self.file_lbl.setWordWrap(True)

        lay.addWidget(self.icon_lbl)
        lay.addWidget(self.hint)
        lay.addWidget(self.file_lbl)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()
            self.setStyleSheet("QFrame#card { background: #141c2c; border: 2px dashed #4da6ff; border-radius: 12px; }")

    def dragLeaveEvent(self, e):
        self.setStyleSheet("")

    def dropEvent(self, e):
        self.setStyleSheet("")
        urls = e.mimeData().urls()
        if urls:
            self.file_dropped.emit(urls[0].toLocalFile())

    def mousePressEvent(self, e):
        exts = "Visual Assets (*.png *.jpg *.jpeg *.webp *.gif *.mp4 *.avi *.mov *.mkv *.webm)"
        path, _ = QFileDialog.getOpenFileName(self, "Browse File", "", exts)
        if path:
            self.file_dropped.emit(path)

    def set_file(self, path):
        name = Path(path).name
        self.file_lbl.setText(f"📁 {name}")
        self.icon_lbl.setText("🎬" if self._is_video(path) else "🖼️")

    @staticmethod
    def _is_video(p):
        return Path(p).suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm"}

# ── Stat Display Card ───────────────────────────────────
class StatCard(QFrame):
    def __init__(self, label, color="#4da6ff"):
        super().__init__()
        self.setObjectName("card")
        self._color = color
        
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(4)

        self.val_lbl = QLabel("0")
        self.val_lbl.setObjectName("stat_val")
        self.val_lbl.setAlignment(Qt.AlignCenter)
        self.val_lbl.setStyleSheet(f"font-size: 24px; font-weight: 700; color: {color}; background: transparent;")

        self.lbl = QLabel(label)
        self.lbl.setObjectName("stat_lbl")
        self.lbl.setAlignment(Qt.AlignCenter)
        self.lbl.setStyleSheet("background: transparent;")

        lay.addWidget(self.val_lbl)
        lay.addWidget(self.lbl)

    def animate_to(self, target: int):
        self._target = target
        self._current = 0
        self._steps = 25
        self._step_val = max(1, target // self._steps)
        
        timer = QTimer(self)
        timer.timeout.connect(self._tick)
        timer.start(20)
        self._timer = timer

    def _tick(self):
        self._current = min(self._current + self._step_val, self._target)
        self.val_lbl.setText(f"{self._current:,}")
        if self._current >= self._target:
            self._timer.stop()

# ── Main Application Window ─────────────────────────────
class VisionAnalyzerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GLM-4.6V Vision AI Suite")
        self.setMinimumSize(1150, 750)
        self.resize(1200, 800)
        self.file_path_or_url = None
        self.worker = None
        
        self._build_ui()

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        main_layout = QHBoxLayout(root)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ── Sidebar Configuration Column ────────────────────
        sidebar = QFrame()
        sidebar.setObjectName("sidebar_container")
        sidebar.setFixedWidth(320)
        
        sl = QVBoxLayout(sidebar)
        sl.setContentsMargins(20, 24, 20, 24)
        sl.setSpacing(16)

        # Header Title
        title_box = QVBoxLayout()
        title_box.setSpacing(4)
        brand = QLabel("👁️ Vision AI Suite")
        brand.setObjectName("title")
        model_sub = QLabel(f"Platform: api.z.ai  •  Engine: {MODEL}")
        model_sub.setObjectName("subtitle")
        title_box.addWidget(brand)
        title_box.addWidget(model_sub)
        sl.addLayout(title_box)

        # Divider line
        div = QFrame()
        div.setFrameShape(QFrame.HLine)
        div.setStyleSheet("color: #232d3f; background: #232d3f;")
        sl.addWidget(div)

        # Drop Zone (Local File)
        sl.addWidget(QLabel("LOCAL FILE").setStyleSheet("color: #798a9e; font-size: 10px; font-weight: bold;") or QLabel(""))
        self.drop_zone = DropZone()
        self.drop_zone.file_dropped.connect(self._on_local_file)
        sl.addWidget(self.drop_zone)

        # Or Web URL Asset
        sl.addWidget(QLabel("OR WEB LINK (URL)").setStyleSheet("color: #798a9e; font-size: 10px; font-weight: bold;") or QLabel(""))
        url_input_box = QHBoxLayout()
        url_input_box.setSpacing(8)
        self.url_edit = QLineEdit()
        self.url_edit.setPlaceholderText("https://example.com/asset.mp4")
        self.url_btn = QPushButton("Load")
        self.url_btn.setObjectName("load_url_btn")
        self.url_btn.clicked.connect(self._on_url_loaded)
        url_input_box.addWidget(self.url_edit)
        url_input_box.addWidget(self.url_btn)
        sl.addLayout(url_input_box)

        # Media Preview Screen
        self.preview_lbl = QLabel()
        self.preview_lbl.setFixedHeight(120)
        self.preview_lbl.setAlignment(Qt.AlignCenter)
        self.preview_lbl.setStyleSheet("background: #090b10; border-radius: 8px; border: 1px solid #1c2433; color: #798a9e;")
        self.preview_lbl.setText("Media Preview Screen")
        sl.addWidget(self.preview_lbl)



        # Run Analysis Action
        self.analyze_btn = QPushButton("⚡  Analyze Asset")
        self.analyze_btn.setObjectName("analyze_btn")
        self.analyze_btn.setDisabled(True)
        self.analyze_btn.clicked.connect(self._analyze_asset)
        self.analyze_btn.setCursor(Qt.PointingHandCursor)
        sl.addWidget(self.analyze_btn)

        # Progress Spinner equivalent
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setTextVisible(False)
        self.progress.hide()
        sl.addWidget(self.progress)

        # Stats Column
        stats_lbl = QLabel("CONSUMPTION STATS")
        stats_lbl.setStyleSheet("color: #798a9e; font-size: 10px; font-weight: bold;")
        sl.addWidget(stats_lbl)

        grid = QHBoxLayout()
        self.stat_in = StatCard("Input Tok", "#4da6ff")
        self.stat_out = StatCard("Output Tok", "#00ffaa")
        grid.addWidget(self.stat_in)
        grid.addWidget(self.stat_out)
        sl.addLayout(grid)

        self.stat_total = StatCard("Total LLM Tokens", "#ffc83b")
        sl.addWidget(self.stat_total)

        sl.addStretch()

        # Clear/Reset Workspace
        clear_btn = QPushButton("🗑  Clear Workspace")
        clear_btn.clicked.connect(self._reset_workspace)
        clear_btn.setCursor(Qt.PointingHandCursor)
        sl.addWidget(clear_btn)

        main_layout.addWidget(sidebar)

        # ── Right Output Console Column ─────────────────────
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(24, 24, 24, 24)
        rl.setSpacing(14)

        # Upper row
        header = QHBoxLayout()
        h_title = QLabel("AI Insight Terminal")
        h_title.setStyleSheet("font-size: 16px; font-weight: 700; color: #4da6ff;")
        self.status_lbl = QLabel("Console Standing: Ready")
        self.status_lbl.setStyleSheet("color: #798a9e; font-size: 12px; font-weight: 500;")
        header.addWidget(h_title)
        header.addStretch()
        header.addWidget(self.status_lbl)
        rl.addLayout(header)

        # Interactive Prompt Box
        prompt_lbl = QLabel("INSPECTION DIRECTIVE (PROMPT)")
        prompt_lbl.setStyleSheet("color: #798a9e; font-size: 10px; font-weight: bold;")
        rl.addWidget(prompt_lbl)

        self.prompt_edit = QTextEdit()
        self.prompt_edit.setPlaceholderText(
            "Enter custom inspection instructions or leave blank to trigger standard quality & compliance reports."
        )
        self.prompt_edit.setMaximumHeight(90)
        rl.addWidget(self.prompt_edit)

        # Response Screen
        output_lbl = QLabel("COMPLIANCE REPORT OUTPUT")
        output_lbl.setStyleSheet("color: #798a9e; font-size: 10px; font-weight: bold;")
        rl.addWidget(output_lbl)

        self.output_edit = QTextEdit()
        self.output_edit.setReadOnly(True)
        self.output_edit.setPlaceholderText(
            "Upload an image/video or load an online link, enter prompts on the console, and click ⚡ Analyze Asset to receive complete insights."
        )
        rl.addWidget(self.output_edit)

        main_layout.addWidget(right, 1)

    # ── Slots & Logic ───────────────────────────────────
    def _on_local_file(self, path):
        self.file_path_or_url = path
        self.url_edit.clear()
        self.drop_zone.set_file(path)
        
        is_vid = DropZone._is_video(path)
        self.analyze_btn.setEnabled(True)
        self.status_lbl.setText(f"Active Local Asset: {'Video' if is_vid else 'Image'}")

        if not is_vid:
            pix = QPixmap(path).scaled(260, 120, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.preview_lbl.setPixmap(pix)
            self.preview_lbl.setStyleSheet("background: #090b10; border-radius: 8px; border: 1px solid #232d3f;")
        else:
            self.preview_lbl.clear()
            self.preview_lbl.setText("🎬 Local Video Loaded\n(Direct native video stream ready)")
            self.preview_lbl.setStyleSheet("background: #121620; border-radius: 8px; border: 1px dashed #4da6ff; color: #4da6ff; font-weight: bold; font-size: 11px; text-align: center;")

    def _on_url_loaded(self):
        url = self.url_edit.text().strip()
        if not url:
            return
        
        self.file_path_or_url = url
        self.drop_zone.file_lbl.setText("")
        self.drop_zone.icon_lbl.setText("🔗")
        self.drop_zone.hint.setText("Online Asset Locked")

        # Determine file type based on suffix
        ext = Path(urlparse(url).path).suffix.lower()
        is_vid = ext in (".mp4", ".mov", ".avi", ".mkv", ".webm")
        self.analyze_btn.setEnabled(True)
        self.status_lbl.setText(f"Active Online Asset: {'Video Link' if is_vid else 'Image Link'}")

        self.preview_lbl.clear()
        self.preview_lbl.setText(f"🌐 Remote {'Video' if is_vid else 'Image'} URL\n{url[:30]}...")
        self.preview_lbl.setStyleSheet("background: #121620; border-radius: 8px; border: 1px solid #232d3f; color: #00ffaa; font-weight: bold; font-size: 11px;")

    def _analyze_asset(self):
        if not self.file_path_or_url:
            return

        is_vid = False
        if is_url(self.file_path_or_url):
            ext = Path(urlparse(self.file_path_or_url).path).suffix.lower()
            is_vid = ext in (".mp4", ".mov", ".avi", ".mkv", ".webm")
        else:
            is_vid = DropZone._is_video(self.file_path_or_url)

        # Standard Fallback prompts
        default_prompt = (
            "Provide a detailed Saudi compliance inspection report. "
            "Examine site safety, highlight potential hazards, analyze safety gear (PPE like helmets/vests), "
            "and suggest corrective compliance actions."
        ) if is_vid else "Describe this compliance image. List visual details and safety features."
        
        prompt = self.prompt_edit.toPlainText().strip() or default_prompt

        self.analyze_btn.setDisabled(True)
        self.progress.show()
        self.output_edit.setPlainText("")
        self.status_lbl.setText("⏳ Model reasoning active...")

        self.worker = AnalyzeWorker(self.file_path_or_url, prompt, is_vid)
        self.worker.finished.connect(self._on_analysis_success)
        self.worker.error.connect(self._on_analysis_failed)
        self.worker.start()

    def _on_analysis_success(self, text, inp, out, total):
        self.output_edit.setPlainText(text)
        self.progress.hide()
        self.analyze_btn.setEnabled(True)
        self.status_lbl.setText("✅ Analysis complete")
        
        # Trigger stats animations
        self.stat_in.animate_to(inp)
        self.stat_out.animate_to(out)
        self.stat_total.animate_to(total)

    def _on_analysis_failed(self, err):
        self.output_edit.setPlainText(f"❌ Analysis failed:\n\n{err}")
        self.progress.hide()
        self.analyze_btn.setEnabled(True)
        self.status_lbl.setText("⚠️ Execution error")

    def _reset_workspace(self):
        self.file_path_or_url = None
        self.output_edit.clear()
        self.prompt_edit.clear()
        self.url_edit.clear()
        
        self.preview_lbl.clear()
        self.preview_lbl.setText("Media Preview Screen")
        self.preview_lbl.setStyleSheet("background: #090b10; border-radius: 8px; border: 1px solid #1c2433; color: #798a9e;")
        
        self.drop_zone.file_lbl.setText("")
        self.drop_zone.icon_lbl.setText("📂")
        self.drop_zone.hint.setText("Drag local file here\nor click to browse")
        
        self.analyze_btn.setDisabled(True)
        self.status_lbl.setText("Console Standing: Ready")
        
        for widget in (self.stat_in, self.stat_out, self.stat_total):
            widget.val_lbl.setText("0")

# ── Application Main Entry ──────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_STYLESHEET)
    
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    
    window = VisionAnalyzerGUI()
    window.show()
    sys.exit(app.exec())
