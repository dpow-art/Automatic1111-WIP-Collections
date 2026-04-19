from pathlib import Path
from typing import Dict, List


class LoraMatcher:
    def __init__(self, lora_dirs: List[Path] | None = None):
        self.lora_dirs = [Path(p) for p in (lora_dirs or [])]
        self._inventory = self._scan()

    def _scan(self) -> Dict[str, str]:
        inventory: Dict[str, str] = {}
        for root in self.lora_dirs:
            if not root.exists():
                continue
            for path in root.rglob("*.safetensors"):
                inventory[path.stem.lower()] = str(path)
        return inventory

    def annotate_loras(self, loras: List[dict]) -> List[dict]:
        annotated = []
        for lora in loras:
            name = (lora.get("name") or "").lower()
            match = self._inventory.get(name)
            lora["local_status"] = "green" if match else lora.get("local_status", "red")
            lora["local_filename"] = match or lora.get("local_filename")
            annotated.append(lora)
        return annotated
