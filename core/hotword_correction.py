import os
from pathlib import Path
from typing import Dict, List, Tuple


def load_hotwords(path: str) -> List[Tuple[str, str]]:
    """
    Hotword file format (UTF-8):
      wrong=>correct
      wrong=correct
      wrong,correct
    Lines starting with # are comments.
    """
    p = Path(path).expanduser()
    if not p.exists():
        return []
    pairs: List[Tuple[str, str]] = []
    for raw in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=>" in line:
            a, b = line.split("=>", 1)
        elif "=" in line:
            a, b = line.split("=", 1)
        elif "," in line:
            a, b = line.split(",", 1)
        else:
            continue
        a = a.strip()
        b = b.strip()
        if a and b:
            pairs.append((a, b))
    # sort by longer wrong first to reduce partial override
    pairs.sort(key=lambda x: len(x[0]), reverse=True)
    return pairs


def apply_hotwords(text: str, pairs: List[Tuple[str, str]]) -> str:
    if not text or not pairs:
        return text
    for wrong, correct in pairs:
        text = text.replace(wrong, correct)
    return text


def resolve_hotword_file(base_dir: str) -> str:
    env = os.environ.get("LEGAL_ASR_HOTWORDS_FILE")
    if env:
        return str(Path(env).expanduser())
    return str(Path(base_dir).resolve() / "hotwords.txt")
