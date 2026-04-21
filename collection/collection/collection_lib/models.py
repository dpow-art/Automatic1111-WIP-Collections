from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class LoraReference:
    name: str
    weight: float = 1.0
    model_id: Optional[int] = None
    version_id: Optional[int] = None
    local_status: str = "red"
    local_filename: Optional[str] = None


@dataclass
class ItemRecord:
    title: str
    image_url: str
    creator_name: str
    post_url: str
    rating: str
    platform: str
    prompt: str
    negative_prompt: str = ""
    metadata: Dict = field(default_factory=dict)
    loras: List[LoraReference] = field(default_factory=list)
