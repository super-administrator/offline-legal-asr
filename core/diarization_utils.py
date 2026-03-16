import sherpa_onnx
import os

class LegalDiarizer:
    def __init__(self, model_dir):
        # 针对离线环境，初始化 VAD (语音活动检测)
        vad_path = os.path.join(model_dir, "silero_vad.onnx")
        if not os.path.exists(vad_path):
            raise FileNotFoundError(f"缺失 VAD 模型文件: {vad_path}")
        config = sherpa_onnx.SileroVadModelConfig(
            model=vad_path,
            min_speech_duration_ms=500,
            min_silence_duration_ms=500,
        )
        self.vad = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_frames=512)

    # 这里可以预留声纹聚类的接口，但 36 秒的短音频，建议直接 ASR 全文转写
