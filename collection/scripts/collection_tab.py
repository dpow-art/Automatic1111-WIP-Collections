import base64
import hashlib
import html
import json
import re
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
from urllib.parse import quote, urlparse

import requests

import gradio as gr

from modules import script_callbacks, shared

from collection_lib.civitai_api import CivitaiClient
from collection_lib.database import CollectionDatabase


EXTENSION_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = EXTENSION_ROOT / "data"
DEFAULT_DB_PATH = DATA_DIR / "collections.db"
DEFAULT_IMAGE_CACHE_DIR = DATA_DIR / "images"
INITIAL_BATCH_SIZE = 35
NSFW_ON_ICON_PATH = EXTENSION_ROOT / "assets" / "icons" / "nsfw_on.svg"
NSFW_ON_ICON_DATA_URI = "data:image/svg+xml;base64," + base64.b64encode(
    NSFW_ON_ICON_PATH.read_bytes()
).decode("ascii")

_db: Optional[CollectionDatabase] = None
_hide_nsfw: bool = False
_stop_requested: bool = False


def _get_db() -> CollectionDatabase:
    global _db
    if _db is None:
        _db = CollectionDatabase(DEFAULT_DB_PATH)
        _db.initialize()
    return _db


def _settings() -> Dict[str, str]:
    api_key = (getattr(shared.opts, "collection_api_key", "") or "").strip()
    source_mode = (getattr(shared.opts, "collection_source_mode", "full") or "full").strip().lower()
    if source_mode not in {"full", "sfw"}:
        source_mode = "full"

    preview_cache_dir = (
        (getattr(shared.opts, "collection_preview_cache_dir", "") or "").strip()
        or str(DEFAULT_IMAGE_CACHE_DIR / "preview")
    )

    full_download_dir = (
        (getattr(shared.opts, "collection_full_download_dir", "") or "").strip()
        or str(DEFAULT_IMAGE_CACHE_DIR / "full")
    )

    nsfw_filter_mode = (
        getattr(shared.opts, "collection_nsfw_filter_mode", "r_and_above")
        or "r_and_above"
    ).strip().lower()
    if nsfw_filter_mode not in {"r_and_above", "x_and_above"}:
        nsfw_filter_mode = "r_and_above"

    local_preview_cap_enabled = bool(
        getattr(shared.opts, "collection_local_preview_cap_enabled", False)
    )

    local_preview_cap_raw = (
        getattr(shared.opts, "collection_local_preview_cap", "") or ""
    ).strip()
    local_preview_cap_value = 0
    if local_preview_cap_raw:
        try:
            local_preview_cap_value = max(0, int(local_preview_cap_raw))
        except Exception:
            local_preview_cap_value = 0

    local_full_cap_enabled = bool(
        getattr(shared.opts, "collection_local_full_cap_enabled", False)
    )

    local_full_cap_raw = (
        getattr(shared.opts, "collection_local_full_cap", "") or ""
    ).strip()
    local_full_cap_value = 0
    if local_full_cap_raw:
        try:
            local_full_cap_value = max(0, int(local_full_cap_raw))
        except Exception:
            local_full_cap_value = 0

    return {
        "api_key": api_key,
        "source_mode": source_mode,
        "preview_cache_dir": preview_cache_dir,
        "full_download_dir": full_download_dir,
        "nsfw_filter_mode": nsfw_filter_mode,
        "local_preview_cap_enabled": local_preview_cap_enabled,
        "local_preview_cap_value": local_preview_cap_value,
        "local_full_cap_enabled": local_full_cap_enabled,
        "local_full_cap_value": local_full_cap_value,
    }


def _safe_suffix_from_url(url: str) -> str:
    try:
        suffix = Path(urlparse(url).path).suffix.lower()
    except Exception:
        suffix = ""

    if suffix in {".jpg", ".jpeg", ".png", ".webp", ".gif", ".mp4", ".webm", ".mov"}:
        return suffix

    return ""


def _is_video_path(path_or_url: str) -> bool:
    if not path_or_url:
        return False

    try:
        suffix = Path(urlparse(path_or_url).path).suffix.lower()
    except Exception:
        suffix = Path(path_or_url).suffix.lower()

    return suffix in {".mp4", ".webm", ".mov"}


def _item_is_video(item: Dict[str, Any]) -> bool:
    media_type = (item.get("media_type") or "").strip().lower()
    if media_type.startswith("video/"):
        return True
    if media_type == "video":
        return True

    preview_source = item.get("preview_path") or item.get("image_url") or ""
    return _is_video_path(preview_source)

def _to_browser_src(path_or_url: str) -> str:
    if not path_or_url:
        return ""

    parsed = urlparse(path_or_url)
    if parsed.scheme in {"http", "https", "data"}:
        return path_or_url

    local_path = Path(path_or_url).resolve()
    return f"/file={quote(str(local_path))}"

def _slugify_collection_name(name: str) -> str:
    value = (name or "").strip().lower()
    value = re.sub(r"[^\w\s-]", "", value)
    value = re.sub(r"[-\s]+", "_", value).strip("_")
    return value or "collection"

def _get_collection_cache_dir(image_cache_dir: str, collection_name: str, collection_id: int) -> Path:
    base_dir = Path(image_cache_dir)
    folder_name = f"{_slugify_collection_name(collection_name)}_{collection_id}"
    return base_dir / folder_name


def _get_media_root_dirs(settings: Dict[str, Any]) -> List[Path]:
    return [
        Path(settings["preview_cache_dir"]),
        Path(settings["full_download_dir"]),
    ]

def _remove_empty_dirs(root_dir: Path) -> int:
    removed = 0

    if not root_dir.exists():
        return removed

    for directory in sorted(
        [p for p in root_dir.rglob("*") if p.is_dir()],
        key=lambda p: len(p.parts),
        reverse=True,
    ):
        try:
            if not any(directory.iterdir()):
                directory.rmdir()
                removed += 1
        except Exception:
            pass

    return removed


def _download_media_file(media_url: str, cache_dir: Path, media_id: Optional[int]) -> str:
    if not media_url:
        return ""

    cache_dir.mkdir(parents=True, exist_ok=True)

    image_key = str(media_id) if media_id is not None else hashlib.sha1(media_url.encode("utf-8")).hexdigest()
    suffix = _safe_suffix_from_url(media_url)

    if suffix:
        local_path = cache_dir / f"{image_key}{suffix}"
        if local_path.exists():
            return local_path.as_posix()

    retry_delays = [2, 5, 10]
    last_exception: Optional[Exception] = None

    for attempt in range(len(retry_delays) + 1):
        try:
            response = requests.get(media_url, timeout=30)
            response.raise_for_status()

            resolved_suffix = suffix
            if not resolved_suffix:
                content_type = (response.headers.get("Content-Type") or "").lower()
                if "video/mp4" in content_type:
                    resolved_suffix = ".mp4"
                elif "video/webm" in content_type:
                    resolved_suffix = ".webm"
                elif "video/quicktime" in content_type:
                    resolved_suffix = ".mov"
                elif "image/png" in content_type:
                    resolved_suffix = ".png"
                elif "image/webp" in content_type:
                    resolved_suffix = ".webp"
                elif "image/gif" in content_type:
                    resolved_suffix = ".gif"
                else:
                    resolved_suffix = ".jpg"

            local_path = cache_dir / f"{image_key}{resolved_suffix}"

            if not local_path.exists():
                local_path.write_bytes(response.content)

            time.sleep(0.5)
            return local_path.as_posix()

        except Exception as exc:
            last_exception = exc
            if attempt < len(retry_delays):
                delay = retry_delays[attempt]
                print(f"  MEDIA DOWNLOAD RETRY {attempt + 1} for media {media_id}: {exc!r} — waiting {delay}s")
                time.sleep(delay)
            else:
                break

    raise last_exception if last_exception else RuntimeError("Media download failed without exception")


def _render_sfw_indicator() -> str:
    mode = _settings()["source_mode"]
    is_sfw = mode == "sfw"

    color = "#ffffff" if is_sfw else "#2f2f2f"
    opacity = "1.0" if is_sfw else "0.24"
    title = "Safe mode enabled" if is_sfw else "Full mode"

    return f"""
    <div title="{html.escape(title)}" style="
        position:absolute;
        top:10px;
        right:14px;
        font-size:11px;
        font-weight:600;
        letter-spacing:1.2px;
        color:{color};
        opacity:{opacity};
        user-select:none;
        pointer-events:auto;
        z-index:20;
    ">SFW</div>
    """


def _all_collections() -> List[Dict[str, Any]]:
    db = _get_db()
    synced = db.list_collections("synced")
    local = db.list_collections("local")
    return synced + local

def _get_collection_cache_status(collection: Dict[str, Any]) -> Dict[str, Any]:
    db = _get_db()
    collection_id = int(collection["id"])
    collection_type = (collection.get("type") or "").strip().lower()
    items = db.list_items_for_collection(collection_id)

    if not items:
        if collection_type == "synced":
            label = "Synced-Remote"
        else:
            label = "Local"
        return {
            "label": label,
            "cached_count": 0,
            "total_count": 0,
        }

    preview_count = 0
    full_count = 0

    for item in items:
        preview_path = (item.get("preview_path") or "").strip()
        full_path = (item.get("full_path") or "").strip()

        if preview_path and Path(preview_path).exists():
            preview_count += 1
        if full_path and Path(full_path).exists():
            full_count += 1

    total_count = len(items)

    if collection_type == "synced":
        if full_count >= total_count and total_count > 0:
            label = "Synced-Local Full"
            cached_count = full_count
        elif preview_count > 0:
            label = "Synced-Local Prvw"
            cached_count = preview_count
        else:
            label = "Synced-Remote"
            cached_count = 0
    else:
        label = "Local"
        cached_count = preview_count

    return {
        "label": label,
        "cached_count": cached_count,
        "total_count": total_count,
    }


def _refresh_sidebar_payload(selected_collection_id: Optional[int] = None) -> str:
    collections = _all_collections()

    rows: List[str] = []
    for collection in collections:
        cid = int(collection["id"])
        name = html.escape(collection["name"])
        item_count = int(collection.get("item_count", 0))
        active = cid == selected_collection_id

        cache_status = _get_collection_cache_status(collection)
        cache_label_raw = (cache_status["label"] or "").strip()
        cached_count = int(cache_status["cached_count"])
        total_count = int(cache_status["total_count"])

        # Strip prefixes so we only show Prvw / Full / Remote
        display_label = cache_label_raw.replace("Synced-Local ", "")
        display_label = display_label.replace("Synced-", "")

        cache_label = html.escape(display_label)

        bg = "#202020" if active else "transparent"
        border = "#4b4b4b" if active else "transparent"
        text = "#ffffff" if active else "#d4d4d4"

        rows.append(
            f"""
            <button
                type="button"
                class="collection-sidebar-item"
                onclick="(function() {{
                    const root = gradioApp();
                    const selectedEl = root.querySelector('#collection_selected_collection_id textarea, #collection_selected_collection_id input');
                    if (!selectedEl) return;
                    selectedEl.value = '{cid}';
                    selectedEl.dispatchEvent(new Event('input', {{ bubbles: true }}));
                    selectedEl.dispatchEvent(new Event('change', {{ bubbles: true }}));
                }})()"
                style="
                    width:100%;
                    text-align:left;
                    margin:0 0 6px 0;
                    padding:10px 12px;
                    border-radius:10px;
                    border:1px solid {border};
                    background:{bg};
                    color:{text};
                    cursor:pointer;
                "
            >
                <div style="display:flex; align-items:center; justify-content:space-between; gap:8px;">
                    <div style="min-width:0;">
                        <div style="
                            font-size:13px;
                            font-weight:600;
                            white-space:nowrap;
                            overflow:hidden;
                            text-overflow:ellipsis;
                        ">{name}</div>
                        <div style="font-size:11px; color:#8f8f8f; margin-top:2px;">
                            {cache_label} ({cached_count}/{total_count})
                        </div>
                    </div>
                    <div style="font-size:11px; color:#9a9a9a; flex:0 0 auto;">{item_count}</div>
                </div>
            </button>
            """
        )

    if not rows:
        rows.append(
            """
            <div style="padding:10px 4px; color:#8a8a8a; font-size:12px;">
                No collections yet. Click <strong>Sync collections</strong> to pull your Civitai collections.
            </div>
            """
        )

    return f"""
    <div style="position:relative; height:100%; min-height:300px;">
        {_render_sfw_indicator()}
        <div style="
            height:100%;
            overflow:auto;
            padding:10px;
            box-sizing:border-box;
        ">
            {''.join(rows)}
        </div>
    </div>
    """


def _render_controls_bar() -> str:
    global _hide_nsfw

    return f"""
    <div style="
        position:sticky;
        top:0;
        z-index:20;
        background:#111;
        border-bottom:1px solid #202020;
        padding:8px 12px;
        display:flex;
        align-items:center;
        justify-content:flex-end;
        gap:8px;
    ">
        <div style="
            display:flex;
            align-items:center;
            gap:8px;
            color:#666;
            font-size:12px;
        ">
            <div title="Thumbnail scale" style="
                width:18px;
                height:18px;
                border:1px solid #333;
                border-radius:4px;
                position:relative;
                opacity:0.75;
            ">
                <div style="
                    position:absolute;
                    left:3px;
                    top:3px;
                    width:6px;
                    height:10px;
                    border:1px solid #555;
                    border-radius:2px;
                "></div>
            </div>

            <div title="Regular view" style="
                width:18px;
                height:18px;
                border:1px solid #333;
                border-radius:4px;
                display:grid;
                grid-template-columns:1fr 1fr;
                gap:2px;
                padding:2px;
                box-sizing:border-box;
                opacity:0.75;
            ">
                <div style="background:#444;border-radius:1px;"></div>
                <div style="background:#444;border-radius:1px;"></div>
                <div style="background:#444;border-radius:1px;"></div>
                <div style="background:#444;border-radius:1px;"></div>
            </div>

            <div title="Scroll view" style="
                width:18px;
                height:18px;
                border:1px solid #333;
                border-radius:4px;
                display:flex;
                align-items:center;
                justify-content:center;
                opacity:0.75;
                color:#666;
                font-size:12px;
            ">↕</div>

            <div title="Focused view" style="
                width:18px;
                height:18px;
                border:1px solid #333;
                border-radius:4px;
                display:flex;
                align-items:center;
                justify-content:center;
                opacity:0.75;
                color:#666;
                font-size:12px;
            ">▥</div>
        </div>
    </div>
    """


def _get_filtered_items_for_collection(collection_id: int) -> List[Dict[str, Any]]:
    db = _get_db()
    items = db.list_items_for_collection(collection_id)

    global _hide_nsfw
    if _hide_nsfw:
        settings = _settings()
        nsfw_filter_mode = settings["nsfw_filter_mode"]
        cutoff = 4 if nsfw_filter_mode == "r_and_above" else 8

        filtered_items = []
        for item in items:
            try:
                rating_value = int(item.get("rating", 0))
            except Exception:
                rating_value = 0

            if rating_value < cutoff:
                filtered_items.append(item)

        items = filtered_items

    return items


def _render_feed_cards(items: List[Dict[str, Any]]) -> str:
    cards: List[str] = []
    for item in items:
        preview_source = item.get("preview_path") or item.get("image_url") or ""
        browser_src = _to_browser_src(preview_source)
        image_url = html.escape(browser_src)
        title = item.get("title") or "Untitled"
        safe_title = html.escape(title)

        is_mp4 = _item_is_video(item)

        thumb_html = ""
        if image_url and not is_mp4:
            thumb_html = f"""
            <div class="collection-card" style="
                background:#171717;
                border:1px solid #272727;
                border-radius:14px;
                padding:8px;
                box-sizing:border-box;
                width:100%;
            ">
                <div style="
                    width:100%;
                    aspect-ratio:1 / 1.25;
                    overflow:hidden;
                    border-radius:12px;
                    background:#202020;
                ">
                    <img
                        src="{image_url}"
                        alt="{safe_title}"
                        loading="lazy"
                        style="
                            width:100%;
                            height:100%;
                            object-fit:cover;
                            display:block;
                        "
                    />
                </div>
            </div>
            """
        elif is_mp4:
            thumb_html = f"""
            <div class="collection-card" style="
                background:#171717;
                border:1px solid #272727;
                border-radius:14px;
                padding:8px;
                box-sizing:border-box;
                width:100%;
            ">
                <div style="
                    width:100%;
                    aspect-ratio:1 / 1.25;
                    overflow:hidden;
                    border-radius:12px;
                    background:#202020;
            ">
                    <video 
                        class="collection-preview-video"
                        src="{image_url}"
                        autoplay
                        loop
                        muted
                        playsinline
                        preload="metadata"
                        style="
                            width:100%;
                            height:100%;
                            object-fit:cover;
                            display:block;
                            background:#202020;
                        "
                    ></video>
                </div>
            </div>
            """
        else:
            thumb_html = f"""
            <div class="collection-card" style="
                background:#171717;
                border:1px solid #272727;
                border-radius:14px;
                padding:8px;
                box-sizing:border-box;
                width:100%;
            "></div>
            """

        cards.append(thumb_html)

    return "".join(cards)


def _render_feed_batch(collection_id: int, offset: int, limit: int = INITIAL_BATCH_SIZE) -> tuple[str, int, bool]:
    items = _get_filtered_items_for_collection(collection_id)
    batch_items = items[offset:offset + limit]
    cards_html = _render_feed_cards(batch_items)
    next_offset = offset + len(batch_items)
    has_more = next_offset < len(items)
    return cards_html, next_offset, has_more


def _render_feed_html(collection_id: Optional[int]) -> str:
    if not collection_id:
        return """
        <div style="
            height:100%;
            min-height:480px;
            display:flex;
            align-items:center;
            justify-content:center;
            color:#8a8a8a;
            font-size:14px;
            text-align:center;
            padding:24px;
            box-sizing:border-box;
        ">
            Select a collection from the left sidebar.
        </div>
        """

    items = _get_filtered_items_for_collection(collection_id)

    if not items:
        return """
        <div style="
            height:100%;
            min-height:480px;
            display:flex;
            align-items:center;
            justify-content:center;
            color:#8a8a8a;
            font-size:14px;
            text-align:center;
            padding:24px;
            box-sizing:border-box;
        ">
            No items found in this collection yet.
        </div>
        """

    cards_html, next_offset, has_more = _render_feed_batch(
        collection_id=collection_id,
        offset=0,
        limit=INITIAL_BATCH_SIZE,
    )

    sentinel_text = "Loading more..." if has_more else "End of collection"

    return f"""
    <div
        id="collection_feed_root"
        class="collection-feed-root"
        style="
            height:100%;
            min-height:480px;
            overflow:auto;
            padding:12px;
            box-sizing:border-box;
        "
    >
        <div
            id="collection_cards_container"
            class="collection-cards collection-view-grid"
            style="
                display:grid;
                grid-template-columns:repeat(4, minmax(0, 1fr));
                gap:12px;
                align-items:start;
            "
        >
            {cards_html}
        </div>

        <div
            id="collection_feed_sentinel"
            data-next-offset="{next_offset}"
            data-has-more="{str(has_more).lower()}"
            data-loading="0"
            style="
                padding:16px 0 8px 0;
                display:flex;
                justify-content:center;
                color:#8a8a8a;
                font-size:12px;
            "
        >
            {sentinel_text}
        </div>
    </div>
    """


def _load_collection_feed(selected_collection_id_raw: str) -> tuple[str, str]:
    selected_collection_id: Optional[int] = None

    try:
        if selected_collection_id_raw and str(selected_collection_id_raw).strip():
            selected_collection_id = int(str(selected_collection_id_raw).strip())
    except Exception:
        selected_collection_id = None

    return (
        _refresh_sidebar_payload(selected_collection_id),
        _render_feed_html(selected_collection_id),
    )


def _load_more_feed_batch(selected_collection_id_raw: str, batch_offset_raw: str) -> str:
    selected_collection_id: Optional[int] = None
    batch_offset = 0

    try:
        if selected_collection_id_raw and str(selected_collection_id_raw).strip():
            selected_collection_id = int(str(selected_collection_id_raw).strip())
    except Exception:
        selected_collection_id = None

    try:
        if batch_offset_raw and str(batch_offset_raw).strip():
            batch_offset = max(0, int(str(batch_offset_raw).strip()))
    except Exception:
        batch_offset = 0

    if not selected_collection_id:
        payload_json = json.dumps({"html": "", "next_offset": batch_offset, "has_more": False})
        return f'<div class="collection-batch-payload" data-json="{html.escape(payload_json, quote=True)}"></div>'

    cards_html, next_offset, has_more = _render_feed_batch(
        collection_id=selected_collection_id,
        offset=batch_offset,
        limit=INITIAL_BATCH_SIZE,
    )

    payload_json = json.dumps(
        {
            "html": cards_html,
            "next_offset": next_offset,
            "has_more": has_more,
        }
    )
    return f'<div class="collection-batch-payload" data-json="{html.escape(payload_json, quote=True)}"></div>'


def _create_local_collection(name: str) -> str:
    if not name or not name.strip():
        return _refresh_sidebar_payload()

    db = _get_db()
    db.create_collection(name=name.strip(), collection_type="local")
    return _refresh_sidebar_payload()


def _toggle_nsfw_filter(selected_collection_id_raw: str):
    global _hide_nsfw
    _hide_nsfw = not _hide_nsfw

    selected_collection_id: Optional[int] = None
    try:
        if selected_collection_id_raw and str(selected_collection_id_raw).strip():
            selected_collection_id = int(str(selected_collection_id_raw).strip())
    except Exception:
        selected_collection_id = None

    button_variant = "secondary" if _hide_nsfw else "primary"

    return (
        gr.update(variant=button_variant),
        _render_controls_bar(),
        _refresh_sidebar_payload(selected_collection_id),
        _render_feed_html(selected_collection_id),
    )


def _clear_cache() -> str:
    settings = _settings()
    removed_files = 0
    removed_dirs = 0

    for root_dir in _get_media_root_dirs(settings):
        if root_dir.exists():
            for f in root_dir.rglob("*"):
                if f.is_file():
                    try:
                        f.unlink()
                        removed_files += 1
                    except Exception:
                        pass

            removed_dirs += _remove_empty_dirs(root_dir)

    return f"Cache cleared: {removed_files} file(s) removed, {removed_dirs} empty folder(s) removed."


def _reset_extension() -> tuple[str, str]:
    global _db

    db_path = Path(DEFAULT_DB_PATH)
    settings = _settings()

    # Close DB connection
    _db = None

    # Delete database
    if db_path.exists():
        try:
            db_path.unlink()
        except Exception:
            pass

    # Clear media folders
    for root_dir in _get_media_root_dirs(settings):
        if root_dir.exists():
            for f in root_dir.rglob("*"):
                if f.is_file():
                    try:
                        f.unlink()
                    except Exception:
                        pass

            _remove_empty_dirs(root_dir)

    # Recreate fresh DB
    db = _get_db()

    return (
        _refresh_sidebar_payload(),
        "Extension reset complete.",
    )

def _request_stop_jobs() -> str:
    global _stop_requested
    _stop_requested = True
    return "Stopping active job after current item..."


def _sync_collections() -> tuple[str, str]:
    global _stop_requested
    _stop_requested = False

    settings = _settings()
    api_key = settings["api_key"]
    source_mode = settings["source_mode"]
    preview_cache_dir = settings["preview_cache_dir"]
    full_download_dir = settings["full_download_dir"]

    print("SYNC FUNCTION STARTED")
    print("api_key present:", bool(api_key))
    print("source_mode:", source_mode)
    print("preview_cache_dir:", preview_cache_dir)
    print("full_download_dir:", full_download_dir)

    for root_dir in _get_media_root_dirs(settings):
        root_dir.mkdir(parents=True, exist_ok=True)

    db = _get_db()
    client = CivitaiClient(
        api_key=api_key or None,
        source_mode=source_mode,
    )

    print("CLIENT BASE URL:", client.base_url)

    try:
        collections = client.get_all_user_collections()
        print("FETCHED COLLECTION COUNT:", len(collections))

        synced_count = 0

        for collection in collections:
            if _stop_requested:
                print("STOP REQUESTED — stopping sync before next collection")
                break

            collection_id = collection.get("id")
            collection_name = (collection.get("name") or "").strip() or f"Collection {collection_id}"
            collection_type = collection.get("type") or "Image"

            if collection_type != "Image":
                print(f"SKIPPING NON-IMAGE COLLECTION: {collection_name} ({collection_type})")
                continue

            if not collection_id:
                continue

            local_collection_id = db.get_or_create_collection(
                name=collection_name,
                collection_type="synced",
                civitai_id=collection_id,
            )

            images = client.get_collection_images(collection_id=int(collection_id))
            print(f"  IMAGE COUNT: {len(images)}")

            # Only clear AFTER successful fetch
            db.clear_collection_items(local_collection_id)

            for idx, image in enumerate(images):
                meta = image.get("meta") or {}
                user = image.get("user") or {}

                image_id = image.get("id")
                title = image.get("name") or f"Civitai Image {image_id}"

                raw_url = image.get("url") or ""
                filename = image.get("name") or ""
                media_type = (image.get("mimeType") or image.get("type") or "").strip().lower()

                preview_url = (
                    f"https://image.civitai.com/xG1nkqKTMzGDvpLrqFT7WA/{raw_url}/width=512"
                    if raw_url else ""
                )

                if raw_url and filename:
                    if media_type.startswith("video/") or media_type == "video":
                        full_media_url = (
                            f"https://image.civitai.com/xG1nkqKTMzGDvpLrqFT7WA/"
                            f"{raw_url}/transcode=true,original=true,quality=90/{filename}"
                        )
                    else:
                        full_media_url = (
                            f"https://image.civitai.com/xG1nkqKTMzGDvpLrqFT7WA/"
                            f"{raw_url}/original=true,quality=90/{filename}"
                        )
                else:
                    full_media_url = ""

                item_id = db.create_item(
                    civitai_image_id=image_id,
                    civitai_post_id=image.get("postId"),
                    title=title,
                    image_url=preview_url,
                    full_media_url=full_media_url,
                    preview_path="",
                    full_path="",
                    download_status="none",
                    creator_name=user.get("username") or "Unknown",
                    creator_url="",
                    post_url=f"https://civitai.com/images/{image_id}" if image_id else "",
                    rating=str(image.get("nsfwLevel", "Unknown")),
                    platform="Civitai",
                    media_type=media_type,
                    prompt=meta.get("prompt") or "",
                    negative_prompt=meta.get("negativePrompt") or "",
                    metadata_json=json.dumps(image),
                )

                db.add_item_to_collection(local_collection_id, item_id, idx)

            synced_count += 1

        if _stop_requested:
            return (
                _refresh_sidebar_payload(),
                f"Sync stopped by user: {synced_count} collection(s) synced before stopping.",
            )

        return (
            _refresh_sidebar_payload(),
            f"Sync succeeded: {synced_count} collection(s) synced.",
        )

    except Exception as exc:
        print("FETCH ERROR:", repr(exc))
        return (
            _refresh_sidebar_payload(),
            f"Sync failed: {html.escape(str(exc))}",
        )

def _cache_selected_collection(selected_collection_id_raw: str) -> tuple[str, str, str]:
    selected_collection_id: Optional[int] = None

    global _stop_requested
    _stop_requested = False

    try:
        if selected_collection_id_raw and str(selected_collection_id_raw).strip():
            selected_collection_id = int(str(selected_collection_id_raw).strip())
    except Exception:
        selected_collection_id = None

    if not selected_collection_id:
        return (
            _refresh_sidebar_payload(),
            _render_feed_html(None),
            "No collection selected.",
        )

    db = _get_db()
    collection = db.get_collection(selected_collection_id)
    if not collection:
        return (
            _refresh_sidebar_payload(),
            _render_feed_html(None),
            "Selected collection was not found.",
        )

    if collection.get("type") != "synced":
        return (
            _refresh_sidebar_payload(selected_collection_id),
            _render_feed_html(selected_collection_id),
            "Only synced collections can be cached locally.",
        )

    settings = _settings()
    preview_cache_dir = settings["preview_cache_dir"]
    local_preview_cap_enabled = bool(settings["local_preview_cap_enabled"])
    local_preview_cap_value = int(settings["local_preview_cap_value"])

    collection_name = collection.get("name") or f"collection_{selected_collection_id}"
    cache_dir = _get_collection_cache_dir(
        image_cache_dir=preview_cache_dir,
        collection_name=collection_name,
        collection_id=selected_collection_id,
    )

    items = db.list_items_for_collection(selected_collection_id)

    uncached_items: List[Dict[str, Any]] = []
    for item in items:
        preview_path = (item.get("preview_path") or "").strip()
        if preview_path and Path(preview_path).exists():
            continue
        uncached_items.append(item)

    batch_limit = (
        local_preview_cap_value
        if local_preview_cap_enabled and local_preview_cap_value > 0
        else None
    )
    if batch_limit is not None:
        items_to_cache = uncached_items[:batch_limit]
    else:
        items_to_cache = uncached_items

    downloaded = 0
    skipped = len(items) - len(uncached_items)
    failed = 0
    consecutive_failures = 0
    stopped_early = False

    print(f"CACHING COLLECTION: {collection_name} ({selected_collection_id})")
    print(f"CACHE DIRECTORY: {cache_dir.as_posix()}")
    print(f"ITEM COUNT: {len(items)}")
    print(f"UNCACHED ITEM COUNT: {len(uncached_items)}")
    if batch_limit is not None:
        print(f"LOCAL PREVIEW CAP APPLIED: {batch_limit}")
        print(f"BATCH ITEM COUNT: {len(items_to_cache)}")
    for item in items_to_cache:
        if _stop_requested:
            print("STOP REQUESTED — stopping preview download run")
            stopped_early = True
            break

        item_id = int(item["id"])
        image_url = (item.get("image_url") or "").strip()

        if not image_url:
            failed += 1
            consecutive_failures += 1
        else:
            try:
                local_path = _download_media_file(
                    media_url=image_url,
                    cache_dir=cache_dir,
                    media_id=item.get("civitai_image_id") or item_id,
                )
                db.update_item_preview_state(
                    item_id=item_id,
                    preview_path=local_path,
                    download_status="preview",
                )
                downloaded += 1
                consecutive_failures = 0
            except Exception as exc:
                print(f"  PREVIEW DOWNLOAD FAILED for item {item_id}: {exc!r}")
                failed += 1
                consecutive_failures += 1

        if consecutive_failures >= 3:
            cooldown = 15
            print(f"  ADAPTIVE COOLDOWN: {consecutive_failures} consecutive failures — waiting {cooldown}s")
            time.sleep(cooldown)

        if consecutive_failures >= 5:
            print("  STOPPING CACHE RUN EARLY: too many consecutive failures")
            stopped_early = True
            break

    cap_note = ""
    if batch_limit is not None:
        cap_note = f" Batch cap: {batch_limit}."

    if stopped_early:
        status = (
            f"Preview download paused for '{collection_name}': "
            f"{downloaded} downloaded, {skipped} already downloaded, {failed} failed. "
            f"Stopped early.{cap_note}"
        )
    else:
        status = (
            f"Preview download complete for '{collection_name}': "
            f"{downloaded} downloaded, {skipped} already downloaded, {failed} failed.{cap_note}"
        )

    return (
        _refresh_sidebar_payload(selected_collection_id),
        _render_feed_html(selected_collection_id),
        status,
    )


def _full_download_selected_collection(selected_collection_id_raw: str) -> tuple[str, str, str]:
    selected_collection_id: Optional[int] = None

    global _stop_requested
    _stop_requested = False

    try:
        if selected_collection_id_raw and str(selected_collection_id_raw).strip():
            selected_collection_id = int(str(selected_collection_id_raw).strip())
    except Exception:
        selected_collection_id = None

    if not selected_collection_id:
        return (
            _refresh_sidebar_payload(),
            _render_feed_html(None),
            "No collection selected.",
        )

    db = _get_db()
    collection = db.get_collection(selected_collection_id)
    if not collection:
        return (
            _refresh_sidebar_payload(),
            _render_feed_html(None),
            "Selected collection was not found.",
        )

    if collection.get("type") != "synced":
        return (
            _refresh_sidebar_payload(selected_collection_id),
            _render_feed_html(selected_collection_id),
            "Only synced collections can be downloaded locally.",
        )

    settings = _settings()
    full_download_dir = settings["full_download_dir"]
    local_full_cap_enabled = bool(settings["local_full_cap_enabled"])
    local_full_cap_value = int(settings["local_full_cap_value"])

    collection_name = collection.get("name") or f"collection_{selected_collection_id}"
    cache_dir = _get_collection_cache_dir(
        image_cache_dir=full_download_dir,
        collection_name=collection_name,
        collection_id=selected_collection_id,
    )

    items = db.list_items_for_collection(selected_collection_id)

    undownloaded_items: List[Dict[str, Any]] = []
    for item in items:
        full_path = (item.get("full_path") or "").strip()
        if full_path and Path(full_path).exists():
            continue
        undownloaded_items.append(item)

    batch_limit = (
        local_full_cap_value
        if local_full_cap_enabled and local_full_cap_value > 0
        else None
    )
    if batch_limit is not None:
        items_to_download = undownloaded_items[:batch_limit]
    else:
        items_to_download = undownloaded_items

    downloaded = 0
    skipped = len(items) - len(undownloaded_items)
    failed = 0
    consecutive_failures = 0
    stopped_early = False

    print(f"FULL DOWNLOADING COLLECTION: {collection_name} ({selected_collection_id})")
    print(f"FULL DOWNLOAD DIRECTORY: {cache_dir.as_posix()}")
    print(f"ITEM COUNT: {len(items)}")
    print(f"UNDOWNLOADED ITEM COUNT: {len(undownloaded_items)}")
    if batch_limit is not None:
        print(f"LOCAL FULL CAP APPLIED: {batch_limit}")
        print(f"FULL BATCH ITEM COUNT: {len(items_to_download)}")

    for item in items_to_download:
        if _stop_requested:
            print("STOP REQUESTED — stopping full download run")
            stopped_early = True
            break

        item_id = int(item["id"])
        full_media_url = (item.get("full_media_url") or "").strip()

        if not full_media_url:
            failed += 1
            consecutive_failures += 1
        else:
            try:
                local_path = _download_media_file(
                    media_url=full_media_url,
                    cache_dir=cache_dir,
                    media_id=item.get("civitai_image_id") or item_id,
                )
                db.update_item_full_state(
                    item_id=item_id,
                    full_path=local_path,
                    download_status="full",
                )
                downloaded += 1
                consecutive_failures = 0
            except Exception as exc:
                print(f"  FULL DOWNLOAD FAILED for item {item_id}: {exc!r}")
                failed += 1
                consecutive_failures += 1

                status_code = getattr(getattr(exc, "response", None), "status_code", None)
                if status_code == 404:
                    print("  STOPPING FULL DOWNLOAD RUN EARLY: received 404")
                    stopped_early = True
                    break

        if consecutive_failures >= 3:
            cooldown = 15
            print(f"  ADAPTIVE COOLDOWN: {consecutive_failures} consecutive failures — waiting {cooldown}s")
            time.sleep(cooldown)

        if consecutive_failures >= 5:
            print("  STOPPING FULL DOWNLOAD RUN EARLY: too many consecutive failures")
            stopped_early = True
            break

    cap_note = ""
    if batch_limit is not None:
        cap_note = f" Batch cap: {batch_limit}."

    if stopped_early:
        status = (
            f"Full download paused for '{collection_name}': "
            f"{downloaded} downloaded, {skipped} already downloaded, {failed} failed. "
            f"Stopped early.{cap_note}"
        )
    else:
        status = (
            f"Full download complete for '{collection_name}': "
            f"{downloaded} downloaded, {skipped} already downloaded, {failed} failed.{cap_note}"
        )

    return (
        _refresh_sidebar_payload(selected_collection_id),
        _render_feed_html(selected_collection_id),
        status,
    )


def _get_item_detail(item_id: int) -> str:
    db = _get_db()
    detail = db.get_item_detail(item_id)
    if not detail:
        return json.dumps({})

    return json.dumps(detail)


def on_ui_settings() -> None:
    section = ("collection", "Collection")

    shared.opts.add_option(
        "collection_api_key",
        shared.OptionInfo(
            "",
            "Civitai API Key",
            gr.Textbox,
            {"type": "password"},
            section=section,
        ),
    )

    shared.opts.add_option(
        "collection_source_mode",
        shared.OptionInfo(
            "full",
            "Civitai Source Mode",
            gr.Radio,
            {"choices": ["full", "sfw"]},
            section=section,
        ),
    )

    shared.opts.add_option(
        "collection_nsfw_filter_mode",
        shared.OptionInfo(
            "r_and_above",
            "18+ Button Filter Behavior",
            gr.Radio,
            {
                "choices": [
                    ("Hide R, X, XXX", "r_and_above"),
                    ("Hide X, XXX (recommended)", "x_and_above"),
                ]
            },
            section=section,
        ).info("Controls what the 18+ button hides in the collection view"),
    )

    shared.opts.add_option(
        "collection_preview_cache_dir",
        shared.OptionInfo(
            str(DEFAULT_IMAGE_CACHE_DIR / "preview"),
            "Directory For Preview Images",
            gr.Textbox,
            {},
            section=section,
        ),
    )

    shared.opts.add_option(
        "collection_full_download_dir",
        shared.OptionInfo(
            str(DEFAULT_IMAGE_CACHE_DIR / "full"),
            "Directory For Full Image Downloads",
            gr.Textbox,
            {},
            section=section,
        ),
    )

    shared.opts.add_option(
        "collection_local_preview_cap",
        shared.OptionInfo(
            "",
            "Max Files per Collection for Local Preview Images",
            gr.Textbox,
            {"placeholder": "0000"},
            section=section,
        ).info("Leave blank for no cap. Applies only to downloaded preview files."),
    )

    shared.opts.add_option(
        "collection_local_preview_cap_enabled",
        shared.OptionInfo(
            False,
            "Implement Cap On Local Cached Preview Collections",
            gr.Checkbox,
            {},
            section=section,
        ),
    )

    shared.opts.add_option(
        "collection_local_full_cap",
        shared.OptionInfo(
            "",
            "Max Files per Collection for Local Full-Size Images",
            gr.Textbox,
            {"placeholder": "0000"},
            section=section,
        ).info("Leave blank for no cap. Applies only to downloaded full-size files."),
    )

    shared.opts.add_option(
        "collection_local_full_cap_enabled",
        shared.OptionInfo(
            False,
            "Implement Cap On Local Full Download Collections",
            gr.Checkbox,
            {},
            section=section,
        ),
    )


def _refresh_all(selected_collection_id_raw: str) -> tuple[str, str, str]:
    selected_collection_id: Optional[int] = None

    try:
        if selected_collection_id_raw and str(selected_collection_id_raw).strip():
            selected_collection_id = int(str(selected_collection_id_raw).strip())
    except Exception:
        selected_collection_id = None

    return (
        _render_controls_bar(),
        _refresh_sidebar_payload(selected_collection_id),
        _render_feed_html(selected_collection_id),
    )


def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as collection_tab:

        gr.HTML(
            f"""
            <style>
            #collection_toolbar_nsfw_button,
            #collection_toolbar_nsfw_button.lg {{
                min-width: 28px !important;
                width: 28px !important;
                height: 24px !important;
                min-height: 24px !important;
                margin-top: 6px !important;
                margin-left: 4px !important;
                padding: 0 !important;
                position: relative !important;
                overflow: hidden !important;
                border: none !important;
                box-shadow: none !important;
                font-size: 0 !important;
                color: transparent !important;
            }}

            #collection_toolbar_nsfw_button::before,
            #collection_toolbar_nsfw_button.lg::before {{
                content: "" !important;
                position: absolute !important;
                inset: 0 !important;
                background-repeat: no-repeat !important;
                background-position: center !important;
                background-size: 20px 20px !important;
                background-image: url("{NSFW_ON_ICON_DATA_URI}") !important;
                pointer-events: none !important;
                z-index: 2 !important;
            }}

            #collection_toolbar_nsfw_button.primary,
            #collection_toolbar_nsfw_button.lg.primary {{
                background-color: #ff7a1a !important;
            }}

            #collection_toolbar_nsfw_button.secondary,
            #collection_toolbar_nsfw_button.lg.secondary {{
                background-color: #3a4352 !important;
            }}

            #collection_toolbar_nsfw_button.primary:hover,
            #collection_toolbar_nsfw_button.lg.primary:hover {{
                background-color: #ff8c33 !important;
            }}

            #collection_toolbar_nsfw_button.secondary:hover,
            #collection_toolbar_nsfw_button.lg.secondary:hover {{
                background-color: #4a5160 !important;
            }}

            .collection-cards.collection-view-grid {{
                display: grid;
                grid-template-columns: repeat(4, minmax(0, 1fr));
                gap: 12px;
                align-items: start;
            }}

            .collection-card {{
                width: 100%;
            }}
            </style>
            """
        )

        selected_collection_id = gr.Textbox(
            value="",
            visible=False,
            elem_id="collection_selected_collection_id",
        )

        batch_offset = gr.Textbox(
            value="0",
            visible=False,
            elem_id="collection_batch_offset",
        )

        batch_payload_html = gr.HTML(
            value="",
            visible=False,
            elem_id="collection_batch_payload",
        )

        load_more_button = gr.Button(
            "Load More Internal",
            visible=False,
            elem_id="collection_load_more_button",
        )

        with gr.Row(equal_height=True):
            with gr.Column(scale=1, min_width=280):
                gr.HTML(
                    """
                    <div style="
                        padding:10px 12px 8px 12px;
                        color:#e6e6e6;
                        font-size:15px;
                        font-weight:600;
                    ">
                        Collections
                    </div>
                    """
                )

                sidebar_html = gr.HTML(
                    value=_refresh_sidebar_payload(),
                    elem_id="collection_sidebar_html",
                )

                with gr.Row():
                    sync_button = gr.Button("Sync collections", variant="primary")
                    preview_download_button = gr.Button("Preview Download")
                    full_download_button = gr.Button("Full Download")
                    stop_button = gr.Button("Stop", variant="stop")
                    refresh_button = gr.Button("Refresh")
                    clear_cache_button = gr.Button("Clear Cache")
                    reset_button = gr.Button("Reset", variant="stop")

                status_markdown = gr.Markdown("")

            with gr.Column(scale=3):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1):
                        controls_html = gr.HTML(
                            value=_render_controls_bar(),
                            elem_id="collection_controls_html",
                        )

                    with gr.Column(scale=0, min_width=36):
                        nsfw_toggle_button = gr.Button(
                            value="",
                            elem_id="collection_toolbar_nsfw_button",
                            tooltip="18+",
                            variant="primary",
                        )

                feed_html = gr.HTML(
                    value=_render_feed_html(None),
                    elem_id="collection_feed_html",
                )

        sync_button.click(
            fn=_sync_collections,
            inputs=[],
            outputs=[sidebar_html, status_markdown],
        )

        stop_button.click(
            fn=_request_stop_jobs,
            inputs=[],
            outputs=[status_markdown],
        )

        preview_download_button.click(
            fn=_cache_selected_collection,
            inputs=[selected_collection_id],
            outputs=[sidebar_html, feed_html, status_markdown],
        )

        full_download_button.click(
            fn=_full_download_selected_collection,
            inputs=[selected_collection_id],
            outputs=[sidebar_html, feed_html, status_markdown],
        )

        refresh_button.click(
            fn=_refresh_all,
            inputs=[selected_collection_id],
            outputs=[controls_html, sidebar_html, feed_html],
        )

        nsfw_toggle_button.click(
            fn=_toggle_nsfw_filter,
            inputs=[selected_collection_id],
            outputs=[nsfw_toggle_button, controls_html, sidebar_html, feed_html],
       )

        clear_cache_button.click(
            fn=_clear_cache,
            inputs=[],
            outputs=[status_markdown],
        )

        reset_button.click(
            fn=_reset_extension,
            inputs=[],
            outputs=[sidebar_html, status_markdown],
        )

        selected_collection_id.change(
            fn=_load_collection_feed,
            inputs=[selected_collection_id],
            outputs=[sidebar_html, feed_html],
        )

        load_more_button.click(
            fn=_load_more_feed_batch,
            inputs=[selected_collection_id, batch_offset],
            outputs=[batch_payload_html],
        )

    return [(collection_tab, "Collection", "collection_tab")]


script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_ui_tabs(on_ui_tabs)