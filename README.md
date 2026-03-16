# offline-legal-asr
离线政话录音转写工具（麒麟V10 arm64 适配）Offline call transcription tool (Kylin V10 arm64)

# 录音转文字

离线语音转写终端，面向通话录音转写场景。GUI 使用 **PySide6**，ASR 使用 **sherpa‑onnx** 加载本地 ONNX 模型。  
项目在 **银河麒麟 V10 (arm64) 虚拟机**中开发，在**银河麒麟 V10 (arm64) 物理机**机验证，**未使用虚拟环境**。

> 本项目在实现与打包过程中使用了 **OpenAI Codex CLI** 协助。

---

## 功能

- 离线转写（无公网依赖）
- 支持 mp3/m4a/mp4/wav（依赖 ffmpeg）
- 简单中文断句/标点（启发式）
- 说话人区分（可选）
- 热词纠错（可选）

---

## 系统要求

- OS：银河麒麟 V10 / Linux arm64
- Python：3.8+（建议与目标机一致）
- 依赖：PySide6、sherpa‑onnx、onnxruntime、numpy、pydub
- ffmpeg：用于 m4a/mp3 等格式转码

---

## 快速运行

```bash
python3 main.py
```

---

## 模型目录

支持以下结构（择一）：

**A. SenseVoice 标准包**
```
model.onnx
tokens.txt
```

**B. CTC 分体包**
```
encoder.onnx
decoder.onnx
multilingual.tiktoken
```

**C. FunASR‑nano**
```
encoder_adaptor.onnx
llm.onnx
embedding.onnx
tokenizer/
```

将模型放入：
```
models/<模型目录>
```
或设置：
```
LEGAL_ASR_MODEL_DIR=/opt/legal-asr/models/<模型目录>
```

---

## 热词纠错

编辑 `hotwords.txt`（UTF‑8），每行一条规则：
```
错误=>正确
皇上法院=>黄山法院
撤述=>撤诉
```

支持分隔符：`=>` / `=` / `,`  
默认启用；可通过环境变量关闭：
```
LEGAL_ASR_HOTWORDS=0
```

---

## 说话人区分（可选）

默认寻找：
- `models/diarization/sherpa-onnx-pyannote-segmentation-3-0/model.onnx`（或 `model.int8.onnx`）
- `models/diarization/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx`

也可显式指定：
```
LEGAL_ASR_SEG_MODEL=/path/to/seg/model.onnx
LEGAL_ASR_SPK_EMB_MODEL=/path/to/embedding.onnx
```

---

## 环境变量（常用）

- `FFMPEG_PATH`：ffmpeg 路径
- `LEGAL_ASR_MODEL_DIR`：模型目录
- `LEGAL_ASR_PUNCT=1`：启用标点恢复
- `LEGAL_ASR_HOTWORDS=1`：启用热词纠错
- `LEGAL_ASR_HOTWORDS_FILE`：热词文件路径
- `LEGAL_ASR_NUM_THREADS`：ASR 线程数（默认使用全部 CPU 核心）
- `LEGAL_ASR_SPK_THREADS`：说话人分离线程数

---

## 打包（deb）

**单文件程序（不含模型）**
```bash
APP_VERSION=1.0.0 \
APP_TITLE="语音转写" \
ICON_SRC="$PWD/legal-asr.png" \
bash scripts/pkg/build_app_deb_onefile.sh
```

**源码最小包（仅 main.py + core/）**
```bash
APP_VERSION=1.0.0 \
APP_TITLE="语音转写" \
ICON_SRC="$PWD/legal-asr.png" \
bash scripts/pkg/build_app_deb_source_only.sh
```

安装后模型放入：
```
/opt/legal-asr/models/
```

---

## 许可

本项目采用 **Apache License 2.0**，详见 `LICENSE`。
