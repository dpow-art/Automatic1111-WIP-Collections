import re
from typing import Dict, List, Tuple


LORA_RE = re.compile(r"<lora:([^:>]+):([0-9]*\.?[0-9]+)>")


def extract_loras(prompt: str) -> List[Tuple[str, float]]:
    matches = []
    for name, weight in LORA_RE.findall(prompt or ""):
        try:
            matches.append((name.strip(), float(weight)))
        except ValueError:
            continue
    return matches


def detect_platform(metadata: Dict) -> str:
    text = " ".join([str(v) for v in metadata.values()]).lower()
    if "comfy" in text:
        return "ComfyUI"
    if "forge" in text:
        return "Forge"
    if "automatic1111" in text or "a1111" in text:
        return "Automatic1111"
    return "Unknown"
