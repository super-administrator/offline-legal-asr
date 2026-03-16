from pydub import AudioSegment
from pydub import effects
import os
import tempfile
from pathlib import Path

class AudioHandler:
    @staticmethod
    def convert_to_wav(input_path):
        """支持 mp3, m4a, mp4, aac 转换"""
        input_path = str(input_path)

        ffmpeg_path = os.environ.get("FFMPEG_PATH")
        if not ffmpeg_path:
            try:
                import shutil
                ffmpeg_path = shutil.which("ffmpeg")
            except Exception:
                ffmpeg_path = None
        if ffmpeg_path:
            AudioSegment.converter = ffmpeg_path

        try:
            audio = AudioSegment.from_file(input_path)
        except Exception as e:
            msg = (
                f"无法读取音频文件：{e}\n"
                "如导入的是 mp3/m4a/mp4 等格式，请确认已安装 ffmpeg，"
                "并可通过环境变量 FFMPEG_PATH 指定 ffmpeg 路径。"
            )
            raise RuntimeError(msg) from e

        audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        if os.environ.get("LEGAL_ASR_NORMALIZE") == "1":
            audio = effects.normalize(audio)

        tmp_dir = os.environ.get("LEGAL_ASR_TMPDIR") or tempfile.gettempdir()
        tmp_dir = str(Path(tmp_dir).resolve())
        os.makedirs(tmp_dir, exist_ok=True)

        fd, temp_wav = tempfile.mkstemp(prefix="legal_asr_", suffix=".wav", dir=tmp_dir)
        os.close(fd)
        audio.export(temp_wav, format="wav")
        if os.environ.get("LEGAL_ASR_DEBUG") == "1":
            try:
                print(f"[ASR] temp_wav={temp_wav} duration={audio.duration_seconds:.2f}s dBFS={audio.dBFS:.2f}")
            except Exception:
                pass

        cleanup = os.environ.get("LEGAL_ASR_KEEP_WAV") != "1"
        return temp_wav, cleanup
