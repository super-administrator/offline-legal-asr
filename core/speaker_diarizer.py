import os
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import onnxruntime as ort
import sherpa_onnx
from numpy.lib.stride_tricks import as_strided


@dataclass
class SpeakerSegment:
    start: float
    end: float
    speaker: int


def _resolve_threads(env_key: str, default: int) -> int:
    env_val = os.environ.get(env_key)
    if env_val:
        try:
            return max(1, int(env_val))
        except Exception:
            return max(1, default)
    return max(1, default)


class _OnnxSegmentationModel:
    def __init__(self, filename: str):
        num_threads = _resolve_threads("LEGAL_ASR_SPK_THREADS", os.cpu_count() or 4)
        session_opts = ort.SessionOptions()
        session_opts.inter_op_num_threads = max(1, num_threads // 2)
        session_opts.intra_op_num_threads = num_threads
        self.model = ort.InferenceSession(
            filename,
            sess_options=session_opts,
            providers=["CPUExecutionProvider"],
        )
        meta = self.model.get_modelmeta().custom_metadata_map
        self.window_size = int(meta["window_size"])
        self.sample_rate = int(meta["sample_rate"])
        self.window_shift = int(0.1 * self.window_size)
        self.receptive_field_size = int(meta["receptive_field_size"])
        self.receptive_field_shift = int(meta["receptive_field_shift"])
        self.num_speakers = int(meta["num_speakers"])
        self.powerset_max_classes = int(meta["powerset_max_classes"])
        self.num_classes = int(meta["num_classes"])

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.expand_dims(x, axis=1)
        (y,) = self.model.run(
            [self.model.get_outputs()[0].name], {self.model.get_inputs()[0].name: x}
        )
        return y


def _load_wav_16k_mono(filename: str) -> np.ndarray:
    with wave.open(filename, "rb") as f:
        sample_rate = f.getframerate()
        num_channels = f.getnchannels()
        samples = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16).astype(
            np.float32
        ) / 32768.0
    if num_channels > 1:
        samples = samples.reshape(-1, num_channels)[:, 0]
    if sample_rate != 16000:
        raise ValueError(
            f"说话人分离要求 16k WAV；当前采样率为 {sample_rate}。请先转为 16k。"
        )
    return samples


def _get_powerset_mapping(
    num_classes: int, num_speakers: int, powerset_max_classes: int
) -> np.ndarray:
    mapping = np.zeros((num_classes, num_speakers))
    k = 1
    for i in range(1, powerset_max_classes + 1):
        if i == 1:
            for j in range(0, num_speakers):
                mapping[k, j] = 1
                k += 1
        elif i == 2:
            for j in range(0, num_speakers):
                for m in range(j + 1, num_speakers):
                    mapping[k, j] = 1
                    mapping[k, m] = 1
                    k += 1
    return mapping


def _to_multi_label(y: np.ndarray, mapping: np.ndarray) -> np.ndarray:
    y = np.argmax(y, axis=-1)
    labels = mapping[y.reshape(-1)].reshape(y.shape[0], y.shape[1], -1)
    return labels


def _speaker_count(labels: np.ndarray, seg_m: _OnnxSegmentationModel) -> np.ndarray:
    labels = labels.sum(axis=-1)
    num_frames = (
        int(
            (seg_m.window_size + (labels.shape[0] - 1) * seg_m.window_shift)
            / seg_m.receptive_field_shift
        )
        + 1
    )
    ans = np.zeros((num_frames,))
    count = np.zeros((num_frames,))
    for i in range(labels.shape[0]):
        this_chunk = labels[i]
        start = int(i * seg_m.window_shift / seg_m.receptive_field_shift + 0.5)
        end = start + this_chunk.shape[0]
        ans[start:end] += this_chunk
        count[start:end] += 1
    ans /= np.maximum(count, 1e-12)
    return (ans + 0.5).astype(np.int8)


def _load_speaker_extractor(filename: str):
    num_threads = _resolve_threads("LEGAL_ASR_SPK_THREADS", os.cpu_count() or 4)
    config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
        model=filename,
        num_threads=num_threads,
        debug=0,
    )
    if not config.validate():
        raise ValueError(f"Invalid speaker embedding config: {config}")
    return sherpa_onnx.SpeakerEmbeddingExtractor(config)


def _get_embeddings(
    embedding_filename: str,
    audio: np.ndarray,
    labels: np.ndarray,
    seg_m: _OnnxSegmentationModel,
) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    extractor = _load_speaker_extractor(embedding_filename)
    buffer = np.empty(seg_m.window_size)
    num_chunks, num_frames, num_speakers = labels.shape
    chunk_speaker_pair: List[Tuple[int, int]] = []
    embeddings = []

    for i in range(num_chunks):
        labels_t = labels[i].T
        sample_offset = i * seg_m.window_shift
        for j in range(num_speakers):
            frames = labels_t[j]
            if frames.sum() < 10:
                continue
            start = None
            idx = 0
            for k in range(num_frames):
                if frames[k] != 0:
                    if start is None:
                        start = k
                elif start is not None:
                    start_samples = int(start / num_frames * seg_m.window_size) + sample_offset
                    end_samples = int(k / num_frames * seg_m.window_size) + sample_offset
                    num_samples = end_samples - start_samples
                    buffer[idx : idx + num_samples] = audio[start_samples:end_samples]
                    idx += num_samples
                    start = None
            if start is not None:
                start_samples = int(start / num_frames * seg_m.window_size) + sample_offset
                end_samples = int(k / num_frames * seg_m.window_size) + sample_offset
                num_samples = end_samples - start_samples
                buffer[idx : idx + num_samples] = audio[start_samples:end_samples]
                idx += num_samples
            stream = extractor.create_stream()
            stream.accept_waveform(sample_rate=seg_m.sample_rate, waveform=buffer[:idx])
            stream.input_finished()
            if not extractor.is_ready(stream):
                continue
            embedding = np.array(extractor.compute(stream))
            chunk_speaker_pair.append((i, j))
            embeddings.append(embedding)

    if not embeddings:
        return [], np.empty((0, 0), dtype=np.float32)
    return chunk_speaker_pair, np.array(embeddings)


class SpeakerDiarizer:
    def __init__(
        self,
        seg_model: str,
        embedding_model: str,
        num_clusters: int = 2,
        enable_kmeans_fallback: bool = True,
    ):
        self.seg_model = _OnnxSegmentationModel(seg_model)
        self.embedding_model = embedding_model
        self.num_clusters = num_clusters
        self.enable_kmeans_fallback = enable_kmeans_fallback
        self.merge_gap_sec = float(os.environ.get("LEGAL_ASR_SPK_MERGE_GAP", "0.30"))
        self.short_seg_sec = float(os.environ.get("LEGAL_ASR_SPK_SHORT_SEG", "0.60"))
        self.enable_short_smoothing = os.environ.get("LEGAL_ASR_SPK_SHORT_SMOOTH", "1") == "1"
        self.neighbor_min_sec = float(os.environ.get("LEGAL_ASR_SPK_NEIGHBOR_MIN", "0.80"))

    @staticmethod
    def from_default_paths(base_dir: str, num_clusters: int = 2) -> Optional["SpeakerDiarizer"]:
        base = Path(base_dir).resolve()
        seg_candidates = [
            base / "models" / "diarization" / "sherpa-onnx-pyannote-segmentation-3-0" / "model.int8.onnx",
            base / "models" / "diarization" / "sherpa-onnx-pyannote-segmentation-3-0" / "model.onnx",
        ]
        emb_candidates = [
            base / "models" / "diarization" / "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx",
        ]

        env_seg = os.environ.get("LEGAL_ASR_SEG_MODEL")
        env_emb = os.environ.get("LEGAL_ASR_SPK_EMB_MODEL")
        if env_seg:
            seg_candidates.insert(0, Path(env_seg).expanduser())
        if env_emb:
            emb_candidates.insert(0, Path(env_emb).expanduser())

        seg_model = next((str(p.resolve()) for p in seg_candidates if p.exists()), None)
        emb_model = next((str(p.resolve()) for p in emb_candidates if p.exists()), None)
        if not seg_model or not emb_model:
            return None
        enable_kmeans_fallback = os.environ.get("LEGAL_ASR_SPK_KMEANS_FALLBACK", "1") == "1"
        return SpeakerDiarizer(
            seg_model=seg_model,
            embedding_model=emb_model,
            num_clusters=num_clusters,
            enable_kmeans_fallback=enable_kmeans_fallback,
        )

    def diarize(self, wav_path: str) -> List[SpeakerSegment]:
        audio = _load_wav_16k_mono(wav_path)
        seg_m = self.seg_model

        if audio.shape[0] < seg_m.window_size:
            pad_size = seg_m.window_size - audio.shape[0]
            audio = np.pad(audio, (0, pad_size))

        num = max(1, (audio.shape[0] - seg_m.window_size) // seg_m.window_shift + 1)
        samples = as_strided(
            audio,
            shape=(num, seg_m.window_size),
            strides=(seg_m.window_shift * audio.strides[0], audio.strides[0]),
        )

        output = []
        batch_size = 32
        for i in range(0, samples.shape[0], batch_size):
            y = seg_m(samples[i : i + batch_size])
            output.append(y)
        y = np.vstack(output)

        mapping = _get_powerset_mapping(
            num_classes=seg_m.num_classes,
            num_speakers=seg_m.num_speakers,
            powerset_max_classes=seg_m.powerset_max_classes,
        )
        labels = _to_multi_label(y, mapping=mapping)
        speakers_per_frame = _speaker_count(labels=labels, seg_m=seg_m)
        if speakers_per_frame.max() == 0:
            return []

        chunk_speaker_pair, embeddings = _get_embeddings(
            self.embedding_model, audio=audio, labels=labels, seg_m=seg_m
        )
        if embeddings.size == 0:
            return []

        clustering = sherpa_onnx.FastClustering(
            sherpa_onnx.FastClusteringConfig(num_clusters=self.num_clusters)
        )
        cluster_labels = np.array(clustering(embeddings), dtype=np.int32)
        if (
            self.enable_kmeans_fallback
            and
            self.num_clusters >= 2
            and embeddings.shape[0] >= self.num_clusters
            and len(np.unique(cluster_labels)) < self.num_clusters
        ):
            # Fallback: some calls collapse to one speaker; run local k-means on embeddings.
            cluster_labels = self._kmeans_cluster(embeddings, self.num_clusters)

        chunk_speaker_to_cluster = {}
        for (chunk_idx, speaker_idx), cluster_idx in zip(chunk_speaker_pair, cluster_labels):
            chunk_speaker_to_cluster[(chunk_idx, speaker_idx)] = int(cluster_idx)

        num_speakers = int(max(cluster_labels) + 1)
        relabels = np.zeros((labels.shape[0], labels.shape[1], num_speakers))
        for i in range(labels.shape[0]):
            for j in range(labels.shape[1]):
                for k in range(labels.shape[2]):
                    if (i, k) not in chunk_speaker_to_cluster:
                        continue
                    t = chunk_speaker_to_cluster[(i, k)]
                    if labels[i, j, k] == 1:
                        relabels[i, j, t] = 1

        num_frames = (
            int(
                (seg_m.window_size + (relabels.shape[0] - 1) * seg_m.window_shift)
                / seg_m.receptive_field_shift
            )
            + 1
        )
        count = np.zeros((num_frames, relabels.shape[-1]))
        for i in range(relabels.shape[0]):
            this_chunk = relabels[i]
            start = int(i * seg_m.window_shift / seg_m.receptive_field_shift + 0.5)
            end = start + this_chunk.shape[0]
            count[start:end] += this_chunk

        sorted_count = np.argsort(-count, axis=-1)
        final = np.zeros((count.shape[0], count.shape[1]))
        for i, (c, sc) in enumerate(zip(speakers_per_frame, sorted_count)):
            for k in range(c):
                final[i, sc[k]] = 1

        final = final.T
        segments: List[SpeakerSegment] = []
        onset = 0.5
        offset = 0.5
        scale = seg_m.receptive_field_shift / seg_m.sample_rate
        scale_offset = seg_m.receptive_field_size / seg_m.sample_rate * 0.5

        for speaker_idx in range(final.shape[0]):
            frames = final[speaker_idx]
            is_active = frames[0] > onset
            start = 0 if is_active else None
            for i in range(1, len(frames)):
                if is_active and frames[i] < offset:
                    segments.append(
                        SpeakerSegment(
                            start=float(start * scale + scale_offset),
                            end=float(i * scale + scale_offset),
                            speaker=speaker_idx,
                        )
                    )
                    is_active = False
                elif (not is_active) and frames[i] > onset:
                    start = i
                    is_active = True
            if is_active and start is not None:
                segments.append(
                    SpeakerSegment(
                        start=float(start * scale + scale_offset),
                        end=float((len(frames) - 1) * scale + scale_offset),
                        speaker=speaker_idx,
                    )
                )

        segments.sort(key=lambda x: x.start)
        segments = self._post_process_segments(segments)

        # Fallback: 某些双人电话会被上游分段模型塌缩成单说话人
        # 当目标簇数>=2且当前结果<2时，尝试按等长语音块做 embedding 聚类兜底。
        if self.num_clusters >= 2:
            uniq = {s.speaker for s in segments}
            if len(uniq) < 2:
                alt = self._chunk_level_fallback(audio, seg_m.sample_rate)
                if len({s.speaker for s in alt}) >= 2:
                    return alt

        return segments

    def _post_process_segments(self, segments: List[SpeakerSegment]) -> List[SpeakerSegment]:
        if not segments:
            return segments

        # 1) 先按同说话人近邻时间间隔合并
        merged: List[SpeakerSegment] = []
        for seg in sorted(segments, key=lambda x: x.start):
            if not merged:
                merged.append(seg)
                continue
            prev = merged[-1]
            if seg.speaker == prev.speaker and seg.start - prev.end <= self.merge_gap_sec:
                prev.end = max(prev.end, seg.end)
            else:
                merged.append(seg)

        # 2) 对极短段做邻居平滑（更保守）
        if self.enable_short_smoothing and self.short_seg_sec > 0:
            for i in range(1, len(merged) - 1):
                cur = merged[i]
                prev = merged[i - 1]
                nxt = merged[i + 1]
                dur = cur.end - cur.start
                prev_dur = prev.end - prev.start
                next_dur = nxt.end - nxt.start
                if (
                    dur <= self.short_seg_sec
                    and prev.speaker == nxt.speaker
                    and cur.speaker != prev.speaker
                    and prev_dur >= self.neighbor_min_sec
                    and next_dur >= self.neighbor_min_sec
                    and cur.start - prev.end <= self.merge_gap_sec
                    and nxt.start - cur.end <= self.merge_gap_sec
                ):
                    cur.speaker = prev.speaker

        # 3) 再合并一次，输出更连贯段
        final_segments: List[SpeakerSegment] = []
        for seg in merged:
            if not final_segments:
                final_segments.append(SpeakerSegment(seg.start, seg.end, seg.speaker))
                continue
            prev = final_segments[-1]
            if seg.speaker == prev.speaker and seg.start - prev.end <= self.merge_gap_sec:
                prev.end = max(prev.end, seg.end)
            else:
                final_segments.append(SpeakerSegment(seg.start, seg.end, seg.speaker))
        return final_segments

    def _chunk_level_fallback(self, audio: np.ndarray, sample_rate: int) -> List[SpeakerSegment]:
        window_sec = float(os.environ.get("LEGAL_ASR_SPK_FALLBACK_WIN", "1.60"))
        hop_sec = float(os.environ.get("LEGAL_ASR_SPK_FALLBACK_HOP", "0.80"))
        min_energy = float(os.environ.get("LEGAL_ASR_SPK_FALLBACK_RMS", "0.008"))
        min_seg_sec = float(os.environ.get("LEGAL_ASR_SPK_MIN_SEG", "0.35"))

        win = max(1, int(window_sec * sample_rate))
        hop = max(1, int(hop_sec * sample_rate))
        if len(audio) < win:
            return []

        extractor = _load_speaker_extractor(self.embedding_model)
        ranges = []
        embs = []
        for start in range(0, len(audio) - win + 1, hop):
            end = start + win
            chunk = audio[start:end]
            rms = float(np.sqrt(np.mean(chunk * chunk) + 1e-12))
            if rms < min_energy:
                continue
            stream = extractor.create_stream()
            stream.accept_waveform(sample_rate=sample_rate, waveform=chunk)
            stream.input_finished()
            if not extractor.is_ready(stream):
                continue
            emb = np.array(extractor.compute(stream), dtype=np.float32)
            ranges.append((start, end))
            embs.append(emb)

        if len(embs) < self.num_clusters:
            return []

        labels = self._kmeans_cluster(np.stack(embs, axis=0), self.num_clusters)
        segs: List[SpeakerSegment] = []
        cur_label = int(labels[0])
        cur_start = ranges[0][0]
        cur_end = ranges[0][1]
        for i in range(1, len(ranges)):
            st, ed = ranges[i]
            lb = int(labels[i])
            contiguous = st - cur_end <= int(0.25 * sample_rate)
            if lb == cur_label and contiguous:
                cur_end = max(cur_end, ed)
            else:
                s = cur_start / sample_rate
                e = cur_end / sample_rate
                if e - s >= min_seg_sec:
                    segs.append(SpeakerSegment(start=float(s), end=float(e), speaker=cur_label))
                cur_label = lb
                cur_start, cur_end = st, ed
        s = cur_start / sample_rate
        e = cur_end / sample_rate
        if e - s >= min_seg_sec:
            segs.append(SpeakerSegment(start=float(s), end=float(e), speaker=cur_label))

        if not segs:
            return []
        segs.sort(key=lambda x: x.start)
        return self._post_process_segments(segs)

    @staticmethod
    def _kmeans_cluster(embeddings: np.ndarray, k: int, iters: int = 30) -> np.ndarray:
        x = embeddings.astype(np.float32)
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        x = x / np.maximum(norms, 1e-12)

        # Deterministic init: farthest-point style
        centers = [x[0]]
        for _ in range(1, k):
            dist2 = np.min(
                np.stack([np.sum((x - c) ** 2, axis=1) for c in centers], axis=1),
                axis=1,
            )
            centers.append(x[int(np.argmax(dist2))])
        centers = np.stack(centers, axis=0)

        labels = np.zeros((x.shape[0],), dtype=np.int32)
        for _ in range(iters):
            d = np.stack([np.sum((x - c) ** 2, axis=1) for c in centers], axis=1)
            new_labels = np.argmin(d, axis=1).astype(np.int32)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for i in range(k):
                idx = np.where(labels == i)[0]
                if idx.size == 0:
                    continue
                centers[i] = np.mean(x[idx], axis=0)
                n = np.linalg.norm(centers[i])
                if n > 0:
                    centers[i] /= n
        return labels
