from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import quote

import json
import requests


SOURCE_MODES = {
    "sfw": "https://civitai.com",
    "full": "https://civitai.red",
}

API_BASE_URL = "https://civitai.com"


@dataclass
class CollectionImagePage:
    page_number: int
    request_cursor: Optional[int]
    next_cursor: Optional[int]
    items: List[Dict[str, Any]]
    first_image_id: Optional[int]
    last_image_id: Optional[int]
    first_created_at: Optional[str]
    last_created_at: Optional[str]


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
            f"{API_BASE_URL}{path}",
            params=params or {},
            headers=self._headers(),
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def _trpc_get(
        self,
        procedure: str,
        payload: Dict[str, Any],
        base_url: Optional[str] = None,
        retries: int = 3,
        log_failures: bool = True,
    ) -> Dict[str, Any]:
        encoded_input = quote(json.dumps({"json": payload}, separators=(",", ":")))
        request_base_url = base_url or self.base_url

        last_exception = None

        for attempt in range(retries):
            try:
                response = requests.get(
                    f"{request_base_url}/api/trpc/{procedure}?input={encoded_input}",
                    headers=self._headers(),
                    timeout=self.timeout,
                )
                response.raise_for_status()
                return response.json()

            except Exception as e:
                last_exception = e
                if log_failures:
                    print(f"[CivitaiClient] Attempt {attempt+1}/{retries} failed for {procedure}: {repr(e)}")

        if log_failures:
            print("[CivitaiClient] All retries failed.")
        raise last_exception

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

    def get_image_by_id(self, image_id: int) -> Dict[str, Any]:
        if not self.api_key:
            return {}

        for base_url in [self.base_url, API_BASE_URL]:
            try:
                payload = self._trpc_get(
                    "image.get",
                    {"id": image_id},
                    base_url=base_url,
                    retries=1,
                    log_failures=False,
                )
                return payload.get("result", {}).get("data", {}).get("json", {}) or {}
            except Exception:
                continue

        return {}

    def get_image_generation_data(self, image_id: int) -> Dict[str, Any]:
        if not self.api_key:
            return {}

        for base_url in [self.base_url, API_BASE_URL]:
            try:
                payload = self._trpc_get(
                    "image.getGenerationData",
                    {"id": image_id},
                    base_url=base_url,
                    retries=1,
                    log_failures=False,
                )
                return payload.get("result", {}).get("data", {}).get("json", {}) or {}
            except Exception:
                continue

        return {}

    def get_model_version(self, version_id: int) -> Dict[str, Any]:
        if not self.api_key:
            return {}

        try:
            return self._get(f"/api/v1/model-versions/{int(version_id)}")
        except Exception:
            return {}

    def get_model_version_by_hash(self, file_hash: str) -> Dict[str, Any]:
        clean_hash = str(file_hash or "").strip()
        if not clean_hash:
            return {}

        try:
            return self._get(f"/api/v1/model-versions/by-hash/{clean_hash}")
        except Exception:
            return {}

    def get_collection_images(
        self,
        collection_id: int,
        on_page: Optional[Callable[[CollectionImagePage], None]] = None,
        start_cursor: Optional[int] = None,
        max_pages: Optional[int] = None,
        max_items: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if not self.api_key:
            return []

        all_items: List[Dict[str, Any]] = []
        cursor: Optional[int] = start_cursor
        page_number = 0

        while True:
            if max_pages is not None and page_number >= max_pages:
                break

            request_cursor = cursor
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

            if request_cursor is not None:
                payload_data["cursor"] = request_cursor

            try:
                payload = self._trpc_get("image.getInfinite", payload_data)
            except Exception as e:
                print(f"[CivitaiClient] PAGE FETCH FAILED at cursor {request_cursor}: {repr(e)}")
                print(f"[CivitaiClient] Stopping pagination early. Returning {len(all_items)} items.")
                break

            json_data = payload.get("result", {}).get("data", {}).get("json", {}) or {}

            items = json_data.get("items", []) or []
            next_cursor = json_data.get("nextCursor")
            page_number += 1

            if max_items is not None:
                remaining = max_items - len(all_items)
                if remaining <= 0:
                    break
                if len(items) > remaining:
                    items = items[:remaining]
                    next_cursor = None

            all_items.extend(items)

            first_image_id = items[0].get("id") if items else None
            last_image_id = items[-1].get("id") if items else None
            first_created_at = items[0].get("createdAt") if items else None
            last_created_at = items[-1].get("createdAt") if items else None

            if on_page is not None:
                on_page(
                    CollectionImagePage(
                        page_number=page_number,
                        request_cursor=request_cursor,
                        next_cursor=next_cursor,
                        items=items,
                        first_image_id=first_image_id,
                        last_image_id=last_image_id,
                        first_created_at=first_created_at,
                        last_created_at=last_created_at,
                    )
                )

            if not next_cursor:
                break

            cursor = next_cursor

        return all_items    

    def iter_collection_image_pages(
        self,
        collection_id: int,
        start_cursor: Optional[int] = None,
        max_pages: Optional[int] = None,
        max_items: Optional[int] = None,
    ) -> List[CollectionImagePage]:
        pages: List[CollectionImagePage] = []

        def _collect_page(page: CollectionImagePage) -> None:
            pages.append(page)

        self.get_collection_images(
            collection_id=collection_id,
            on_page=_collect_page,
            start_cursor=start_cursor,
            max_pages=max_pages,
            max_items=max_items,
        )
        return pages