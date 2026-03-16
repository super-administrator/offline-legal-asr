import base64
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def _read_tiktoken_table(path: str) -> List[bytes]:
    """
    Read a tiktoken-like vocab file where each line is:
        <base64_token_bytes> <int_id>
    """
    table: Dict[int, bytes] = {}
    with open(path, "rb") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                tok_b64, idx_b = raw.split(maxsplit=1)
                idx = int(idx_b.decode("utf-8", errors="strict"))
                table[idx] = base64.b64decode(tok_b64)
            except Exception:
                # Ignore malformed lines
                continue

    if not table:
        raise RuntimeError(f"无法解析词表文件: {path}")

    max_id = max(table.keys())
    out: List[bytes] = [b""] * (max_id + 1)
    for i, b in table.items():
        out[i] = b
    return out


def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _mel_filterbank(
    sr: int,
    n_fft: int,
    n_mels: int,
    fmin: float,
    fmax: float,
) -> np.ndarray:
    if fmax <= 0:
        fmax = sr / 2
    m_min = _hz_to_mel(np.array([fmin], dtype=np.float64))[0]
    m_max = _hz_to_mel(np.array([fmax], dtype=np.float64))[0]
    m_pts = np.linspace(m_min, m_max, n_mels + 2, dtype=np.float64)
    hz_pts = _mel_to_hz(m_pts)

    bin_pts = np.floor((n_fft + 1) * hz_pts / sr).astype(np.int32)
    bin_pts = np.clip(bin_pts, 0, n_fft // 2)

    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(n_mels):
        left = bin_pts[m]
        center = bin_pts[m + 1]
        right = bin_pts[m + 2]

        if center <= left:
            center = left + 1
        if right <= center:
            right = center + 1

        # rising slope
        fb[m, left:center] = (np.arange(left, center, dtype=np.float32) - left) / (center - left)
        # falling slope
        fb[m, center:right] = (right - np.arange(center, right, dtype=np.float32)) / (right - center)
    return fb


def _fbank_80(
    samples: np.ndarray,
    sr: int = 16000,
    n_mels: int = 80,
    n_fft: int = 512,
    win_length: int = 400,  # 25 ms @ 16k
    hop_length: int = 160,  # 10 ms @ 16k
    fmin: float = 0.0,
    fmax: float = 8000.0,
) -> np.ndarray:
    if samples.dtype != np.float32:
        samples = samples.astype(np.float32)

    if samples.ndim != 1:
        samples = samples.reshape(-1)

    if samples.size < win_length:
        pad = win_length - samples.size
        samples = np.pad(samples, (0, pad), mode="constant")

    window = np.hamming(win_length).astype(np.float32)
    fb = _mel_filterbank(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)

    n_frames = 1 + (samples.size - win_length) // hop_length
    feats = np.empty((n_frames, n_mels), dtype=np.float32)

    for i in range(n_frames):
        start = i * hop_length
        frame = samples[start : start + win_length] * window
        spec = np.fft.rfft(frame, n=n_fft)
        power = (spec.real ** 2 + spec.imag ** 2).astype(np.float32)
        mel = fb @ power
        mel = np.maximum(mel, 1e-10)
        feats[i] = np.log(mel)

    return feats


def _cmvn(feats: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mean = feats.mean(axis=0, keepdims=True)
    var = feats.var(axis=0, keepdims=True)
    feats = (feats - mean) / np.sqrt(var + eps)
    return feats


def _lfr_stack(feats: np.ndarray, lfr_m: int = 7, lfr_n: int = 6) -> np.ndarray:
    """
    Low Frame Rate stacking. Common in FunASR.
    Produces shape: [T', 80*lfr_m]
    """
    t, d = feats.shape
    if t == 0:
        return np.zeros((0, d * lfr_m), dtype=np.float32)

    out = []
    i = 0
    while i < t:
        chunk = feats[i : i + lfr_m]
        if chunk.shape[0] < lfr_m:
            pad = np.repeat(chunk[-1:], repeats=lfr_m - chunk.shape[0], axis=0)
            chunk = np.concatenate([chunk, pad], axis=0)
        out.append(chunk.reshape(-1))
        i += lfr_n
    return np.stack(out, axis=0).astype(np.float32)


def _ctc_greedy(ids: np.ndarray, blank_id: int) -> List[int]:
    out: List[int] = []
    prev: Optional[int] = None
    for x in ids.tolist():
        if x == blank_id:
            prev = x
            continue
        if prev is None or x != prev:
            out.append(x)
        prev = x
    return out


def _decode_tiktoken(token_bytes: List[bytes], ids: List[int]) -> str:
    bs = bytearray()
    for i in ids:
        if 0 <= i < len(token_bytes):
            bs.extend(token_bytes[i])
    return bs.decode("utf-8", errors="ignore")


@dataclass
class FunASRCTCOnnxConfig:
    encoder_path: str
    decoder_path: str
    tiktoken_path: str
    provider: str = "cpu"
    num_threads: int = 4
    blank_id: Optional[int] = None


class FunASRCTCOnnxRecognizer:
    """
    Run a 2-stage ONNX CTC model:
      encoder(speech, speech_lengths) -> encoder_out
      decoder(encoder_out, encoder_out_lens) -> ctc_logits

    This is used when the model export does not include sherpa-onnx metadata.
    """

    def __init__(self, cfg: FunASRCTCOnnxConfig):
        try:
            import onnxruntime as ort  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "缺少 onnxruntime 依赖，无法加载 encoder/decoder.onnx。\n"
                "请在虚拟环境中安装：pip install onnxruntime"
            ) from e

        self._ort = ort
        self._token_bytes = _read_tiktoken_table(cfg.tiktoken_path)

        so = ort.SessionOptions()
        so.intra_op_num_threads = max(1, int(cfg.num_threads))

        providers = ["CPUExecutionProvider"]
        if cfg.provider and cfg.provider.lower() != "cpu":
            # keep it explicit; on mac you may try CoreML later
            providers = [cfg.provider]

        self._enc = ort.InferenceSession(str(cfg.encoder_path), sess_options=so, providers=providers)
        self._dec = ort.InferenceSession(str(cfg.decoder_path), sess_options=so, providers=providers)

        env_blank = os.environ.get("LEGAL_ASR_CTC_BLANK_ID")
        blank_id: Optional[int] = cfg.blank_id
        if env_blank:
            try:
                blank_id = int(env_blank)
            except Exception:
                blank_id = blank_id

        self._blank_id = blank_id if blank_id is not None else self._auto_detect_blank_id()

    def _auto_detect_blank_id(self) -> int:
        # Heuristic: run all-zero features and take the most frequent argmax id as blank.
        speech = np.zeros((1, 20, 560), dtype=np.float32)
        speech_lengths = np.array([speech.shape[1]], dtype=np.int64)
        enc_out, enc_lens = self._enc.run(None, {"speech": speech, "speech_lengths": speech_lengths})
        logits, out_lens = self._dec.run(None, {"encoder_out": enc_out, "encoder_out_lens": enc_lens})
        t = int(out_lens[0])
        ids = np.argmax(logits[0, :t, :], axis=-1)
        values, counts = np.unique(ids, return_counts=True)
        return int(values[np.argmax(counts)])

    def _transcribe_samples(self, samples: np.ndarray) -> str:
        feats80 = _fbank_80(samples, sr=16000)
        if os.environ.get("LEGAL_ASR_CMVN", "1") == "1":
            feats80 = _cmvn(feats80)

        if not np.isfinite(feats80).all():
            feats80 = np.nan_to_num(feats80, nan=0.0, posinf=0.0, neginf=0.0)

        lfr = _lfr_stack(feats80, lfr_m=7, lfr_n=6)

        speech = lfr.reshape(1, lfr.shape[0], lfr.shape[1]).astype(np.float32)
        speech_lengths = np.array([speech.shape[1]], dtype=np.int64)

        enc_out, enc_lens = self._enc.run(None, {"speech": speech, "speech_lengths": speech_lengths})
        logits, out_lens = self._dec.run(None, {"encoder_out": enc_out, "encoder_out_lens": enc_lens})
        t = int(out_lens[0])
        ids = np.argmax(logits[0, :t, :], axis=-1)

        kept = _ctc_greedy(ids, blank_id=self._blank_id)
        text = _decode_tiktoken(self._token_bytes, kept)
        return text.strip()

    def transcribe_wav(self, wav_path: str) -> str:
        import wave

        with wave.open(wav_path, "rb") as f:
            if f.getframerate() != 16000:
                raise ValueError("采样率不匹配：必须为 16000Hz（请先转换为 16k 单声道 WAV）")
            if f.getnchannels() != 1:
                raise ValueError("声道不匹配：必须为单声道（请先转换为 mono）")
            chunk_sec = float(os.environ.get("LEGAL_ASR_CHUNK_SEC", "0"))
            if chunk_sec <= 0:
                frames = f.readframes(f.getnframes())
                samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                return self._transcribe_samples(samples)

            frames_per_chunk = max(16000, int(16000 * chunk_sec))
            overlap_sec = float(os.environ.get("LEGAL_ASR_CHUNK_OVERLAP", "0"))
            overlap_frames = max(0, int(16000 * overlap_sec))
            prev_tail = np.empty((0,), dtype=np.float32)
            texts: List[str] = []
            while True:
                frames = f.readframes(frames_per_chunk)
                if not frames:
                    break
                samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                if prev_tail.size > 0:
                    samples = np.concatenate([prev_tail, samples])
                text = self._transcribe_samples(samples)
                if text:
                    texts.append(text)
                if overlap_frames > 0:
                    if samples.size >= overlap_frames:
                        prev_tail = samples[-overlap_frames:]
                    else:
                        prev_tail = samples
            joiner = os.environ.get("LEGAL_ASR_CHUNK_JOIN", "")
            return joiner.join(texts)
