import sys
import os
from pathlib import Path
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QTextEdit,
    QFileDialog,
    QMessageBox,
    QLabel,
    QCheckBox,
    QComboBox,
)
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QIcon

# 导入自定义模块
from core.audio_handler import AudioHandler
from core.asr_engine import LegalASREngine, FunASREngine

class TranscribeThread(QThread):
    finished = Signal(str)
    error = Signal(str)

    def __init__(self, engine, file_path, punct_enabled, spk_enabled, hotword_enabled):
        super().__init__()
        self.engine = engine
        self.file_path = file_path
        self.punct_enabled = punct_enabled
        self.spk_enabled = spk_enabled
        self.hotword_enabled = hotword_enabled

    def run(self):
        temp_wav = None
        cleanup = False
        try:
            temp_wav, cleanup = AudioHandler.convert_to_wav(self.file_path)
            self.engine.set_punct(self.punct_enabled)
            if hasattr(self.engine, "set_speaker_diarization"):
                self.engine.set_speaker_diarization(self.spk_enabled)
            if hasattr(self.engine, "set_hotwords"):
                self.engine.set_hotwords(self.hotword_enabled)
            result_text = self.engine.transcribe(temp_wav)
            self.finished.emit(result_text)
        except Exception as e:
            self.error.emit(str(e))
        finally:
            if cleanup and temp_wav and os.path.exists(temp_wav):
                try:
                    os.remove(temp_wav)
                except Exception:
                    pass


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("录音转文字 BY LZF")
        icon_path = self._resolve_icon_path()
        if icon_path and os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        self.resize(800, 600)

        self.asr = None
        self.model_dir = None

        layout = QVBoxLayout()
        self.info_label = QLabel("就绪。请导入通话录音。")

        self.model_combo = QComboBox()
        self._populate_model_combo()
        self.model_combo.currentIndexChanged.connect(self.on_model_change)

        self.btn_pick_model = QPushButton("选择模型文件夹")
        self.btn_pick_model.clicked.connect(self.on_pick_model)

        self.btn_open = QPushButton("导入录音文件")
        self.btn_open.clicked.connect(self.on_open_file)

        self.text_display = QTextEdit()

        self.chk_punct = QCheckBox("自动断句/标点")
        self.chk_punct.setChecked(True)
        self.chk_spk = QCheckBox("说话人区分")
        self.chk_spk.setChecked(False)
        self.chk_hotwords = QCheckBox("热词纠错")
        self.chk_hotwords.setChecked(True)

        self.btn_copy = QPushButton("复制到剪贴板")
        self.btn_copy.clicked.connect(self.on_copy_text)

        layout.addWidget(self.info_label)
        layout.addWidget(self.model_combo)
        layout.addWidget(self.btn_pick_model)
        layout.addWidget(self.btn_open)
        layout.addWidget(self.text_display)
        switch_row = QHBoxLayout()
        switch_row.addWidget(self.chk_punct)
        switch_row.addWidget(self.chk_spk)
        switch_row.addWidget(self.chk_hotwords)
        switch_row.addStretch(1)
        layout.addLayout(switch_row)
        layout.addWidget(self.btn_copy)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.on_model_change()

    def _resolve_model_dir(self):
        def is_model_dir(path: Path) -> bool:
            return (
                (path / "encoder.onnx").exists()
                and (path / "decoder.onnx").exists()
                and (path / "multilingual.tiktoken").exists()
            )

        def normalize(path: Path) -> str:
            path = path.resolve()
            if is_model_dir(path):
                return str(path)
            try:
                for child in path.iterdir():
                    if child.is_dir() and is_model_dir(child):
                        return str(child.resolve())
            except Exception:
                pass
            return str(path)

        env_model_dir = os.environ.get("LEGAL_ASR_MODEL_DIR")
        if env_model_dir:
            env_model_dir_path = Path(env_model_dir).expanduser()
            if env_model_dir_path.exists():
                return normalize(env_model_dir_path)

        candidates = []
        if getattr(sys, "frozen", False):
            candidates.append(Path(sys.executable).resolve().parent)
            meipass = getattr(sys, "_MEIPASS", None)
            if meipass:
                candidates.append(Path(meipass).resolve())
        candidates.append(Path(__file__).resolve().parent)

        for base_dir in candidates:
            model_dir = (base_dir / "models" / "sensevoice-small").resolve()
            if model_dir.exists():
                return normalize(model_dir)

        return normalize((candidates[0] / "models" / "sensevoice-small").resolve())

    def _resolve_icon_path(self):
        env_icon = os.environ.get("LEGAL_ASR_ICON")
        if env_icon:
            return str(Path(env_icon).expanduser())
        candidates = []
        if getattr(sys, "frozen", False):
            candidates.append(Path(sys.executable).resolve().parent)
            meipass = getattr(sys, "_MEIPASS", None)
            if meipass:
                candidates.append(Path(meipass).resolve())
        candidates.append(Path(__file__).resolve().parent)
        for base_dir in candidates:
            icon = (base_dir / "legal-asr.png").resolve()
            if icon.exists():
                return str(icon)
        return None

    def _populate_model_combo(self):
        self.model_combo.clear()
        base = Path(__file__).resolve().parent / "models"
        if base.exists() and base.is_dir():
            for p in sorted(base.iterdir()):
                if p.is_dir() and self._is_asr_model_dir(p):
                    self.model_combo.addItem(str(p))

        env_model_dir = os.environ.get("LEGAL_ASR_MODEL_DIR")
        if (
            env_model_dir
            and self._is_asr_model_dir(Path(env_model_dir))
            and env_model_dir not in [self.model_combo.itemText(i) for i in range(self.model_combo.count())]
        ):
            self.model_combo.addItem(env_model_dir)

        if self.model_combo.count() == 0:
            self.model_combo.addItem(self._resolve_model_dir())

    def _is_asr_model_dir(self, p: Path) -> bool:
        if not p.exists() or not p.is_dir():
            return False
        # ONNX CTC/SenseVoice-style
        if (p / "encoder.onnx").exists() and (p / "decoder.onnx").exists():
            return True
        if (p / "model.onnx").exists() and ((p / "tokens.txt").exists() or (p / "tokens.json").exists()):
            return True
        # FunASR PT-style
        if (p / "model.pt").exists() and ((p / "tokens.json").exists() or (p / "config.yaml").exists()):
            return True
        return False

    def on_pick_model(self):
        path = QFileDialog.getExistingDirectory(self, "选择模型文件夹")
        if path:
            if not self._is_asr_model_dir(Path(path)):
                QMessageBox.warning(self, "提示", "该目录不是可识别的 ASR 模型目录，请选择包含 model.pt 或 onnx ASR 文件的目录。")
                return
            if path not in [self.model_combo.itemText(i) for i in range(self.model_combo.count())]:
                self.model_combo.addItem(path)
            self.model_combo.setCurrentText(path)

    def on_model_change(self):
        model_dir = self.model_combo.currentText().strip()
        if not model_dir:
            return
        if not self._is_asr_model_dir(Path(model_dir)):
            self.info_label.setText("当前目录不是 ASR 模型目录，请重新选择。")
            self.asr = None
            return
        self.info_label.setText("正在加载模型，请稍候...")
        QApplication.processEvents()
        try:
            # Choose engine based on model contents
            p = Path(model_dir)
            if (p / "encoder.onnx").exists() and (p / "decoder.onnx").exists():
                self.asr = LegalASREngine(model_dir)
            else:
                self.asr = FunASREngine(model_dir)
            self.model_dir = model_dir
            desc = self.asr.describe() if hasattr(self.asr, "describe") else "已加载"
            self.info_label.setText(f"模型已加载：{Path(model_dir).name} | {desc}")
        except Exception as e:
            QMessageBox.critical(
                self,
                "初始化失败",
                f"模型加载失败。\n"
                f"当前模型目录：{model_dir}\n\n"
                f"错误: {e}",
            )
            self.asr = None

    def on_open_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择录音", "", "音频文件 (*.m4a *.mp3 *.mp4 *.wav)")
        if path:
            if not self.asr:
                QMessageBox.warning(self, "提示", "模型尚未加载完成，请先选择模型文件夹。")
                return
            self.info_label.setText("正在转写，请稍候...")
            self.btn_open.setEnabled(False)
            self.thread = TranscribeThread(
                self.asr,
                path,
                self.chk_punct.isChecked(),
                self.chk_spk.isChecked(),
                self.chk_hotwords.isChecked(),
            )
            self.thread.finished.connect(self.on_transcribe_finished)
            self.thread.error.connect(self.on_error)
            self.thread.start()

    def on_copy_text(self):
        text = self.text_display.toPlainText().strip()
        if not text:
            QMessageBox.information(self, "提示", "当前没有可复制的内容。")
            return
        QApplication.clipboard().setText(text)
        QMessageBox.information(self, "完成", "已复制到剪贴板。")

    def on_transcribe_finished(self, text):
        self.text_display.setPlainText(text)
        self.info_label.setText("转写完成")
        self.btn_open.setEnabled(True)

    def on_error(self, err):
        QMessageBox.warning(self, "错误", err)
        self.btn_open.setEnabled(True)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
