import sherpa_onnx
import wave
import numpy as np
import os
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple
import inspect
import sys
from core.speaker_diarizer import SpeakerDiarizer
from core.hotword_correction import apply_hotwords, load_hotwords, resolve_hotword_file

try:
    from funasr import AutoModel
except Exception:
    AutoModel = None


class ParaformerSpeakerHelper:
    def __init__(self, model_dir: str):
        if AutoModel is None:
            raise RuntimeError("FunASR 不可用")
        self.model_dir = str(Path(model_dir).resolve())
        kwargs = {
            "model": self.model_dir,
            "device": "cpu",
        }
        self.model = AutoModel(**self._with_supported_automodel_kwargs(kwargs))

    def infer_segments(self, wav_path: str) -> List[Tuple[int, Optional[float], Optional[float], str]]:
        gen_kwargs = {"input": wav_path, "batch_size_s": 300}
        gen_sig = inspect.signature(self.model.generate)
        if "disable_pbar" in gen_sig.parameters:
            gen_kwargs["disable_pbar"] = True
        res = self.model.generate(**gen_kwargs)
        if not isinstance(res, list) or not res:
            return []
        item = res[0] if isinstance(res[0], dict) else {}
        sentences = item.get("sentence_info", [])
        if not isinstance(sentences, list):
            return []

        out = []
        for sentence in sentences:
            if not isinstance(sentence, dict):
                continue
            spk = sentence.get("spk", None)
            if spk is None:
                continue
            text = str(sentence.get("text", "")).strip()
            if not text:
                continue
            start_sec, end_sec = self._extract_time_range(sentence)
            out.append((int(spk), start_sec, end_sec, text))
        return out

    def _extract_time_range(self, sentence: dict) -> Tuple[Optional[float], Optional[float]]:
        def to_sec(val):
            if val is None:
                return None
            try:
                x = float(val)
            except Exception:
                return None
            return x / 1000.0 if x > 1000 else x

        for sk, ek in [
            ("start", "end"),
            ("begin", "end"),
            ("start_time", "end_time"),
            ("start_ms", "end_ms"),
            ("begin_time", "end_time"),
        ]:
            if sk in sentence and ek in sentence:
                s = to_sec(sentence.get(sk))
                e = to_sec(sentence.get(ek))
                if s is not None and e is not None and e > s:
                    return s, e

        ts = sentence.get("timestamp", None)
        if isinstance(ts, (list, tuple)) and len(ts) > 0:
            first = ts[0]
            last = ts[-1]
            if isinstance(first, (list, tuple)) and len(first) >= 2 and isinstance(last, (list, tuple)) and len(last) >= 2:
                s = to_sec(first[0])
                e = to_sec(last[1])
                if s is not None and e is not None and e > s:
                    return s, e
            elif len(ts) >= 2 and not isinstance(first, (list, tuple)):
                s = to_sec(ts[0])
                e = to_sec(ts[1])
                if s is not None and e is not None and e > s:
                    return s, e

        return None, None

    @staticmethod
    def _with_supported_automodel_kwargs(kwargs: dict) -> dict:
        sig = inspect.signature(AutoModel)
        if "disable_update" in sig.parameters:
            kwargs["disable_update"] = True
        if "disable_log" in sig.parameters:
            kwargs["disable_log"] = True
        return kwargs

class LegalASREngine:
    def __init__(self, model_dir):
        """
        离线 ASR 引擎（sherpa-onnx）

        兼容两种常见模型目录：
        1) sherpa-onnx 官方 SenseVoice 导出：model.onnx + tokens.txt
        2) sherpa-onnx 官方 FunASR-nano：encoder_adaptor.onnx + llm.onnx + embedding.onnx + tokenizer_dir
        3) 第三方导出 CTC：encoder.onnx + decoder.onnx + multilingual.tiktoken（使用 onnxruntime 自行解码）
        """
        model_dir = os.path.abspath(model_dir)
        self.punct_enabled = os.environ.get("LEGAL_ASR_PUNCT") == "1"
        self.hotword_enabled = os.environ.get("LEGAL_ASR_HOTWORDS", "1") != "0"
        self.enable_speaker_diarization = False
        self.hybrid_speaker_enabled = os.environ.get("LEGAL_ASR_HYBRID_SPK", "1") == "1"
        cluster_cnt = int(os.environ.get("LEGAL_ASR_SPK_NUM", "2"))
        base_dir = self._resolve_base_dir()
        self.speaker_diarizer = SpeakerDiarizer.from_default_paths(
            base_dir=base_dir,
            num_clusters=max(1, cluster_cnt),
        )
        self.hotword_pairs = self._load_hotwords(base_dir)
        self.hybrid_speaker_helper = self._init_hybrid_speaker_helper(base_dir)
        env_threads = os.environ.get("LEGAL_ASR_NUM_THREADS")
        if env_threads:
            try:
                num_threads = max(1, int(env_threads))
            except Exception:
                num_threads = max(1, (os.cpu_count() or 4))
        else:
            num_threads = max(1, (os.cpu_count() or 4))

        # FunASR-nano (sherpa-onnx) structure
        encoder_adaptor = os.path.join(model_dir, "encoder_adaptor.onnx")
        llm = os.path.join(model_dir, "llm.onnx")
        embedding = os.path.join(model_dir, "embedding.onnx")
        tokenizer_dir = os.path.join(model_dir, "tokenizer")
        if os.path.exists(encoder_adaptor) and os.path.exists(llm) and os.path.exists(embedding) and os.path.isdir(tokenizer_dir):
            if not hasattr(sherpa_onnx.OfflineRecognizer, "from_funasr_nano"):
                raise RuntimeError(
                    "当前 sherpa-onnx 版本不支持 FunASR-nano（缺少 OfflineRecognizer.from_funasr_nano）。\n"
                    "请升级 sherpa-onnx 到支持 FunASR-nano 的版本（本项目建议 1.12.x）。"
                )
            try:
                self.recognizer = sherpa_onnx.OfflineRecognizer.from_funasr_nano(
                    encoder_adaptor=encoder_adaptor,
                    llm=llm,
                    embedding=embedding,
                    tokenizer=tokenizer_dir,
                    num_threads=num_threads,
                    provider="cpu",
                    language="",
                    itn=True,
                )
                return
            except Exception as e:
                raise RuntimeError(
                    f"ASR 引擎初始化失败（FunASR-nano）：{e}\n"
                    f"模型目录：{model_dir}"
                ) from e

        model_onnx = os.path.join(model_dir, "model.onnx")
        tokens_txt = os.path.join(model_dir, "tokens.txt")

        if os.path.exists(model_onnx) and os.path.exists(tokens_txt):
            try:
                self.recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
                    model=model_onnx,
                    tokens=tokens_txt,
                    num_threads=num_threads,
                    provider="cpu",
                    language="auto",
                    use_itn=True,
                )
                return
            except Exception as e:
                raise RuntimeError(
                    f"ASR 引擎初始化失败：{e}\n"
                    f"模型目录：{model_dir}\n"
                    "请确认 model.onnx/tokens.txt 与当前 sherpa-onnx 版本匹配。"
                ) from e

        encoder_path = os.path.join(model_dir, "encoder.onnx")
        decoder_path = os.path.join(model_dir, "decoder.onnx")
        tiktoken_path = os.path.join(model_dir, "multilingual.tiktoken")
        if os.path.exists(encoder_path) and os.path.exists(decoder_path) and os.path.exists(tiktoken_path):
            from core.funasr_ctc_onnx import FunASRCTCOnnxConfig, FunASRCTCOnnxRecognizer

            cfg = FunASRCTCOnnxConfig(
                encoder_path=encoder_path,
                decoder_path=decoder_path,
                tiktoken_path=tiktoken_path,
                num_threads=num_threads,
            )
            self._ctc_onnx = FunASRCTCOnnxRecognizer(cfg)
            self.recognizer = None
            return

        raise FileNotFoundError(
            "未找到可用的模型文件。\n"
            "支持的结构：\n"
            "  - model.onnx + tokens.txt\n"
            "  - encoder_adaptor.onnx + llm.onnx + embedding.onnx + tokenizer/\n"
            "  - encoder.onnx + decoder.onnx + multilingual.tiktoken\n"
            f"当前模型目录：{model_dir}"
        )

    def _init_hybrid_speaker_helper(self, base_dir: str):
        if AutoModel is None or not self.hybrid_speaker_enabled:
            return None
        model_dir = os.environ.get("LEGAL_ASR_HYBRID_SPK_MODEL_DIR")
        if model_dir:
            candidate = Path(model_dir).expanduser().resolve()
        else:
            candidate = Path(base_dir).resolve() / "models" / "speech_paraformer"
        if not candidate.exists():
            return None
        if not (candidate / "model.pt").exists():
            return None
        try:
            return ParaformerSpeakerHelper(str(candidate))
        except Exception:
            return None

    @staticmethod
    def _resolve_base_dir() -> str:
        if getattr(sys, "frozen", False):
            return str(Path(sys.executable).resolve().parent)
        return str(Path(__file__).resolve().parent.parent)

    def _load_hotwords(self, base_dir: str):
        try:
            hotword_file = resolve_hotword_file(base_dir)
            return load_hotwords(hotword_file)
        except Exception:
            return []

    def transcribe(self, wav_path):
        """
        将 WAV 文件转写为文本
        """
        if not os.path.exists(wav_path):
            raise FileNotFoundError("音频文件不存在")

        if self.enable_speaker_diarization and self.speaker_diarizer is not None:
            try:
                text = self._transcribe_with_speakers(wav_path)
                if text.strip():
                    return text
            except Exception as e:
                if os.environ.get("LEGAL_ASR_DEBUG") == "1":
                    print(f"[ASR] diarization failed: {e}")

        return self._transcribe_plain(wav_path)

    def _transcribe_plain(self, wav_path):
        if getattr(self, "_ctc_onnx", None) is not None:
            raw_text = self._ctc_onnx.transcribe_wav(wav_path)
            return self._post_process(raw_text)

        # 打开经过 audio_handler 处理后的 16k 采样率 WAV
        with wave.open(wav_path, "rb") as f:
            # 校验采样率（sherpa-onnx 必须要求 16000）
            if f.getframerate() != 16000:
                raise ValueError("采样率不匹配：必须为 16000Hz（请先转换为 16k 单声道 WAV）")

            chunk_sec = float(os.environ.get("LEGAL_ASR_CHUNK_SEC", "0"))
            if chunk_sec > 0:
                frames_per_chunk = max(16000, int(16000 * chunk_sec))
                stream = self.recognizer.create_stream()
                while True:
                    frames = f.readframes(frames_per_chunk)
                    if not frames:
                        break
                    samples_float = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                    stream.accept_waveform(16000, samples_float)
                stream.input_finished()
                self.recognizer.decode_streams([stream])
                raw_text = stream.result.text
            else:
                num_samples = f.getnframes()
                samples = f.readframes(num_samples)
                # 转为 float32 归一化数据
                samples_float = np.frombuffer(samples, dtype=np.int16).astype(np.float32) / 32768.0

                # 创建流并解码
                stream = self.recognizer.create_stream()
                stream.accept_waveform(16000, samples_float)
                self.recognizer.decode_streams([stream])
                
                # 获取原始文本
                raw_text = stream.result.text
        
        # 清理 SenseVoice 特有的标签（如 <|zh|><|Speech|> 等）
        clean_text = self._post_process(raw_text)
        
        return clean_text

    def _post_process(self, text):
        """
        过滤模型自带的特殊标签
        """
        import re
        # 匹配 <|...|> 格式的标签并替换为空
        pattern = r"<\|.*?\|>"
        text = re.sub(pattern, "", text).strip()
        if self.punct_enabled:
            text = self._punctuate_zh(text)
        if self.hotword_enabled and self.hotword_pairs:
            text = apply_hotwords(text, self.hotword_pairs)
        return text

    def set_punct(self, enabled: bool):
        self.punct_enabled = bool(enabled)

    def set_hotwords(self, enabled: bool):
        self.hotword_enabled = bool(enabled)

    def set_speaker_diarization(self, enabled: bool):
        self.enable_speaker_diarization = bool(enabled)

    def _transcribe_with_speakers(self, wav_path: str) -> str:
        dur = self._get_wav_duration(wav_path)
        max_sec = float(os.environ.get("LEGAL_ASR_SPK_MAX_SEC", "1800"))
        if max_sec > 0 and dur and dur > max_sec:
            if os.environ.get("LEGAL_ASR_DEBUG") == "1":
                print(f"[ASR] diarization disabled for long audio ({dur:.1f}s > {max_sec:.1f}s)")
            return self._transcribe_plain(wav_path)

        if self.hybrid_speaker_enabled and self.hybrid_speaker_helper is not None:
            try:
                text = self._transcribe_with_hybrid_speakers(wav_path)
                if text and text.strip():
                    return text
            except Exception:
                pass

        segments = self.speaker_diarizer.diarize(wav_path)
        if not segments:
            return self._transcribe_plain(wav_path)

        with wave.open(wav_path, "rb") as f:
            sr = f.getframerate()
            n_channels = f.getnchannels()
            sampwidth = f.getsampwidth()
            pcm = f.readframes(f.getnframes())
        if sr != 16000:
            return self._transcribe_plain(wav_path)
        all_samples = np.frombuffer(pcm, dtype=np.int16)
        if n_channels > 1:
            all_samples = all_samples.reshape(-1, n_channels)[:, 0]

        lines = []
        min_dur = float(os.environ.get("LEGAL_ASR_SPK_MIN_SEG", "0.35"))
        for seg in segments:
            if seg.end - seg.start < min_dur:
                continue
            s = max(0, int(seg.start * sr))
            e = min(len(all_samples), int(seg.end * sr))
            if e <= s:
                continue
            chunk = all_samples[s:e]
            fd, tmp = tempfile.mkstemp(prefix="legal_asr_spk_", suffix=".wav")
            os.close(fd)
            try:
                with wave.open(tmp, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(sampwidth)
                    wf.setframerate(sr)
                    wf.writeframes(chunk.tobytes())
                part = self._transcribe_plain(tmp)
            finally:
                try:
                    os.remove(tmp)
                except Exception:
                    pass
            if part and part.strip():
                lines.append(f"说话人{seg.speaker + 1}：{part.strip()}")

        if not lines:
            return self._transcribe_plain(wav_path)
        return self._merge_adjacent_speaker_lines(lines)

    def _transcribe_with_hybrid_speakers(self, wav_path: str) -> str:
        items = self.hybrid_speaker_helper.infer_segments(wav_path)
        if not items:
            return ""

        with wave.open(wav_path, "rb") as f:
            sr = f.getframerate()
            n_channels = f.getnchannels()
            sampwidth = f.getsampwidth()
            pcm = f.readframes(f.getnframes())
        if sr != 16000:
            return ""
        all_samples = np.frombuffer(pcm, dtype=np.int16)
        if n_channels > 1:
            all_samples = all_samples.reshape(-1, n_channels)[:, 0]

        lines = []
        min_dur = float(os.environ.get("LEGAL_ASR_SPK_MIN_SEG", "0.35"))
        for spk_raw, start_sec, end_sec, para_text in items:
            if start_sec is None or end_sec is None:
                text = para_text
                if text:
                    lines.append(f"说话人{spk_raw + 1}：{text}")
                continue

            if end_sec - start_sec < min_dur:
                continue
            s = max(0, int(start_sec * sr))
            e = min(len(all_samples), int(end_sec * sr))
            if e <= s:
                continue
            chunk = all_samples[s:e]
            fd, tmp = tempfile.mkstemp(prefix="legal_asr_hybrid_spk_", suffix=".wav")
            os.close(fd)
            try:
                with wave.open(tmp, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(sampwidth)
                    wf.setframerate(sr)
                    wf.writeframes(chunk.tobytes())
                text = self._transcribe_plain(tmp)
            finally:
                try:
                    os.remove(tmp)
                except Exception:
                    pass
            if text and text.strip():
                lines.append(f"说话人{spk_raw + 1}：{text.strip()}")
            elif para_text:
                lines.append(f"说话人{spk_raw + 1}：{para_text}")

        if not lines:
            return ""
        return self._merge_adjacent_speaker_lines(lines)

    def _merge_adjacent_speaker_lines(self, lines):
        merged = []
        for line in lines:
            if "：" not in line:
                if merged:
                    merged[-1] += line
                else:
                    merged.append(line)
                continue
            spk, text = line.split("：", 1)
            text = text.strip()
            if merged and "：" in merged[-1]:
                prev_spk, prev_text = merged[-1].split("：", 1)
                if prev_spk == spk:
                    joiner = "" if prev_text.endswith(("。", "！", "？")) else "。"
                    merged[-1] = f"{prev_spk}：{prev_text}{joiner}{text}"
                    continue
            merged.append(f"{spk}：{text}")
        return "\n".join(merged)

    @staticmethod
    def _get_wav_duration(wav_path: str) -> float:
        try:
            with wave.open(wav_path, "rb") as f:
                sr = f.getframerate()
                if sr <= 0:
                    return 0.0
                return f.getnframes() / float(sr)
        except Exception:
            return 0.0

    def describe(self) -> str:
        return (
            f"ONNX ASR；diarization={'on' if self.speaker_diarizer else 'off'}；"
            f"hybrid_spk={'on' if self.hybrid_speaker_helper else 'off'}"
        )

    def _punctuate_zh(self, text):
        """
        简单中文断句/标点恢复（启发式）
        说明：不依赖额外模型，适合离线快速可读性提升。
        """
        import re

        text = re.sub(r"\s+", " ", text).strip()
        # Remove spaces between CJK to avoid per-char separators.
        text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", text)

        # Strip comma-like punctuation from raw ASR to avoid per-char noise.
        text = re.sub(r"[，,、､﹐﹑]", "", text)

        fillers = ["嗯", "呃", "额", "啊", "哦", "哎", "欸", "嗯嗯", "好好好"]
        filler_pat = "|".join(sorted(set(fillers), key=len, reverse=True))
        text = re.sub(rf"(?:{filler_pat})+", "。", text)

        # Insert a small number of commas around multi-char discourse markers.
        markers = [
            "那现在",
            "所以",
            "但是",
            "不过",
            "然后",
            "因为",
            "如果",
            "只是",
            "确认了",
            "行吧",
        ]
        marker_pat = "|".join(sorted(set(markers), key=len, reverse=True))
        text = re.sub(rf"([\u4e00-\u9fff])({marker_pat})([\u4e00-\u9fff])", r"\1\2，\3", text)

        # 语气词问句处理
        text = re.sub(r"(吗|呢|吧)(?![，。！？；：、])", r"\1？", text)

        # 长句按长度强制断句
        max_len = int(os.environ.get("LEGAL_ASR_PUNCT_MAXLEN", "32"))
        chunks = []
        buf = ""
        for ch in text:
            buf += ch
            if ch in "。！？；":
                chunks.append(buf)
                buf = ""
            elif len(buf) >= max_len:
                buf += "。"
                chunks.append(buf)
                buf = ""
        if buf:
            chunks.append(buf)
        text = "".join(chunks)

        if text and text[-1] not in "。！？":
            text += "。"
        return text


class FunASREngine:
    def __init__(self, model_dir: str):
        if AutoModel is None:
            raise RuntimeError("FunASR 未安装或不可用，请确认已安装 funasr/torch/torchaudio。")

        model_dir = str(Path(model_dir).resolve())

        kwargs = {
            "model": model_dir,
            "device": "cpu",
        }
        self.model_dir = model_dir

        # Use local model.py if present
        model_py = Path(model_dir) / "model.py"
        if model_py.exists():
            kwargs["trust_remote_code"] = True
            kwargs["remote_code"] = model_dir

        # Optional components (VAD/Punc/Spk)
        vad_model = self._guess_component(model_dir, env_key="LEGAL_ASR_VAD_MODEL_DIR", keywords=("vad", "fsmn"))
        punc_model = self._guess_component(model_dir, env_key="LEGAL_ASR_PUNC_MODEL_DIR", keywords=("punc", "ct"))
        spk_model = self._guess_component(model_dir, env_key="LEGAL_ASR_SPK_MODEL_DIR", keywords=("spk", "cam", "campplus", "speaker"))

        if vad_model:
            kwargs["vad_model"] = vad_model
        if punc_model:
            kwargs["punc_model"] = punc_model
        if spk_model:
            kwargs["spk_model"] = spk_model

        self.loaded_components = {
            "vad_model": vad_model,
            "punc_model": punc_model,
            "spk_model": spk_model,
        }
        self.model = AutoModel(**self._with_supported_automodel_kwargs(kwargs))
        self.enable_speaker_diarization = False
        cluster_cnt = int(os.environ.get("LEGAL_ASR_SPK_NUM", "2"))
        base_dir = LegalASREngine._resolve_base_dir()
        self.speaker_diarizer = SpeakerDiarizer.from_default_paths(
            base_dir=base_dir,
            num_clusters=max(1, cluster_cnt),
        )
        self.hotword_enabled = os.environ.get("LEGAL_ASR_HOTWORDS", "1") != "0"

    def _guess_component(self, model_dir: str, env_key: str, keywords):
        env = os.environ.get(env_key)
        if env and Path(env).exists():
            return str(Path(env).resolve())

        base = Path(model_dir).resolve()
        candidates = []
        # 1) search current model dir children
        candidates.extend([p for p in base.iterdir() if p.is_dir()])
        # 2) search sibling model dirs
        parent = base.parent
        if parent.exists() and parent.is_dir():
            candidates.extend([p for p in parent.iterdir() if p.is_dir() and p != base])

        for p in candidates:
            name = p.name.lower()
            if any(k in name for k in keywords):
                return str(p.resolve())
        return None

    def set_punct(self, enabled: bool):
        # Punctuation handled by FunASR pipeline if punc_model is set.
        pass

    def set_hotwords(self, enabled: bool):
        self.hotword_enabled = bool(enabled)

    def set_speaker_diarization(self, enabled: bool):
        self.enable_speaker_diarization = bool(enabled)

    def describe(self) -> str:
        parts = []
        for key, value in self.loaded_components.items():
            if value:
                parts.append(f"{key}={value}")
        parts.append(f"diarization={'on' if self.speaker_diarizer else 'off'}")
        if not parts:
            return "FunASR(pt)；未检测到额外 vad/punc/spk 组件"
        return "FunASR(pt)；" + "；".join(parts)

    def transcribe(self, wav_path: str) -> str:
        gen_kwargs = {"input": wav_path, "batch_size_s": 300}
        gen_sig = inspect.signature(self.model.generate)
        if "disable_pbar" in gen_sig.parameters:
            gen_kwargs["disable_pbar"] = True
        res = self.model.generate(**gen_kwargs)
        if isinstance(res, list) and res:
            item = res[0]
            if isinstance(item, dict):
                if "sentence_info" in item and isinstance(item["sentence_info"], list):
                    lines = []
                    for s in item["sentence_info"]:
                        spk = s.get("spk", None)
                        text = s.get("text", "")
                        if spk is None:
                            lines.append(text)
                        else:
                            lines.append(f"说话人{spk}：{text}")
                    return self._apply_hotwords("\n".join([l for l in lines if l]))
                if (
                    self.enable_speaker_diarization
                    and self.speaker_diarizer is not None
                    and "timestamp" in item
                    and "text" in item
                ):
                    try:
                        labeled = self._apply_speaker_labels(
                            text=item["text"],
                            timestamps=item["timestamp"],
                            wav_path=wav_path,
                        )
                        return self._apply_hotwords(labeled)
                    except Exception:
                        pass
                if "text" in item:
                    return self._apply_hotwords(item["text"])
        return ""

    def _apply_hotwords(self, text: str) -> str:
        if not text:
            return text
        if not self.hotword_enabled:
            return text
        base_dir = LegalASREngine._resolve_base_dir()
        pairs = load_hotwords(resolve_hotword_file(base_dir))
        return apply_hotwords(text, pairs) if pairs else text

    def _apply_speaker_labels(self, text: str, timestamps, wav_path: str) -> str:
        segments = self.speaker_diarizer.diarize(wav_path)
        if not segments:
            return text

        tokens = text.split()
        if len(tokens) == len(timestamps):
            units = tokens
        else:
            units = list(text.replace(" ", ""))
        if len(units) != len(timestamps):
            summary = "\n".join(
                [f"说话人{s.speaker + 1} [{s.start:.2f}-{s.end:.2f}s]" for s in segments]
            )
            return text + "\n\n" + summary

        def find_speaker(mid_sec: float) -> int:
            for s in segments:
                if s.start <= mid_sec <= s.end:
                    return s.speaker + 1
            nearest = min(segments, key=lambda x: min(abs(mid_sec - x.start), abs(mid_sec - x.end)))
            return nearest.speaker + 1

        labeled = []
        for token, ts in zip(units, timestamps):
            if not isinstance(ts, (list, tuple)) or len(ts) < 2:
                continue
            start_ms = float(ts[0])
            end_ms = float(ts[1])
            mid_sec = (start_ms + end_ms) / 2000.0
            spk = find_speaker(mid_sec)
            labeled.append((spk, token))

        if not labeled:
            return text

        lines = []
        cur_spk = labeled[0][0]
        buf = []
        for spk, token in labeled:
            if spk != cur_spk:
                lines.append(f"说话人{cur_spk}：{''.join(buf)}")
                cur_spk = spk
                buf = [token]
            else:
                buf.append(token)
        if buf:
            lines.append(f"说话人{cur_spk}：{''.join(buf)}")
        return "\n".join(lines)

    @staticmethod
    def _with_supported_automodel_kwargs(kwargs: dict) -> dict:
        sig = inspect.signature(AutoModel)
        if "disable_update" in sig.parameters:
            kwargs["disable_update"] = True
        if "disable_log" in sig.parameters:
            kwargs["disable_log"] = True
        return kwargs
