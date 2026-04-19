from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import json
import requests


SOURCE_MODES = {
    "sfw": "https://civitai.com",
    "full": "https://civitai.red",
}

@dataclass
class CivitaiClient:
    api_key: Optional[str] = None
    source_mode: str = "full"
    timeout: int = 60

    @property
    def base_url(self) -> str:
        return SOURCE_MODES.get(self.source_mode, SOURCE_MODES["full"])

    def _headers(self) -> Dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        response = requests.get(
            f"{self.base_url}{path}",
            params=params or {},
            headers=self._headers(),
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def _trpc_get(self, procedure: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        encoded_input = quote(json.dumps({"json": payload}, separators=(",", ":")))
        response = requests.get(
            f"{self.base_url}/api/trpc/{procedure}?input={encoded_input}",
            headers=self._headers(),
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def get_all_user_collections(self) -> List[Dict[str, Any]]:
        if not self.api_key:
            return []

        payload = self._trpc_get("collection.getAllUser", {"permission": "VIEW"})
        return payload.get("result", {}).get("data", {}).get("json", []) or []

    def get_collection_by_id(self, collection_id: int) -> Dict[str, Any]:
        if not self.api_key:
            return {}

        payload = self._trpc_get("collection.getById", {"id": collection_id})
        return payload.get("result", {}).get("data", {}).get("json", {}) or {}

    def get_collection_images(
        self,
        collection_id: int,
    ) -> List[Dict[str, Any]]:
        if not self.api_key:
            return []

        all_items: List[Dict[str, Any]] = []
        cursor: Optional[int] = None

        while True:
            payload_data: Dict[str, Any] = {
                "collectionId": collection_id,
                "period": "AllTime",
                "sort": "Newest",
                "browsingLevel": 31,
                "include": ["cosmetics"],
                "disablePoi": True,
                "disableMinor": False,
                "authed": True,
            }

            if cursor is not None:
                payload_data["cursor"] = cursor

            payload = self._trpc_get("image.getInfinite", payload_data)
            json_data = payload.get("result", {}).get("data", {}).get("json", {}) or {}

            items = json_data.get("items", []) or []
            next_cursor = json_data.get("nextCursor")

            all_items.extend(items)

            if not next_cursor:
                break

            cursor = next_cursor

        return all_items    
