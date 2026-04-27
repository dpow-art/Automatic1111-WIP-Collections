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

IMAGE_BATCH_SIZE = 35
VIDEO_BATCH_SIZE = 20

NSFW_ON_ICON_PATH = EXTENSION_ROOT / "assets" / "icons" / "nsfw_on.svg"
NSFW_ON_ICON_DATA_URI = "data:image/svg+xml;base64," + base64.b64encode(
    NSFW_ON_ICON_PATH.read_bytes()
).decode("ascii")

MAIN_VIEW_ICON_PATH = EXTENSION_ROOT / "assets" / "icons" / "main_view.svg"
MAIN_VIEW_ICON_DATA_URI = "data:image/svg+xml;base64," + base64.b64encode(
    MAIN_VIEW_ICON_PATH.read_bytes()
).decode("ascii")

SCROLL_VIEW_ICON_PATH = EXTENSION_ROOT / "assets" / "icons" / "scroll_view.svg"
SCROLL_VIEW_ICON_DATA_URI = "data:image/svg+xml;base64," + base64.b64encode(
    SCROLL_VIEW_ICON_PATH.read_bytes()
).decode("ascii")

DETAIL_VIEW_ICON_PATH = EXTENSION_ROOT / "assets" / "icons" / "detail_view.svg"
DETAIL_VIEW_ICON_DATA_URI = "data:image/svg+xml;base64," + base64.b64encode(
    DETAIL_VIEW_ICON_PATH.read_bytes()
).decode("ascii")

PLAY_PAUSE_ICON_PATH = EXTENSION_ROOT / "assets" / "icons" / "Play_pause.svg"
PLAY_PAUSE_ICON_DATA_URI = "data:image/svg+xml;base64," + base64.b64encode(
    PLAY_PAUSE_ICON_PATH.read_bytes()
).decode("ascii")

THUMB_SCALE_ICON_PATH = EXTENSION_ROOT / "assets" / "icons" / "thumb_scale.svg"
THUMB_SCALE_ICON_DATA_URI = "data:image/svg+xml;base64," + base64.b64encode(
    THUMB_SCALE_ICON_PATH.read_bytes()
).decode("ascii")

_db: Optional[CollectionDatabase] = None
_hide_nsfw: bool = False
_stop_requested: bool = False
_current_view: str = "grid"
_active_collection_id: Optional[int] = None
_selected_item_id: Optional[int] = None
_target_local_collection_id: Optional[int] = None
_video_autoplay_enabled: bool = True
_active_cache_filter: str = "sync"
_local_resource_index_cache: Dict[str, Any] = {}
_model_version_hash_cache: Dict[str, Dict[str, str]] = {}
PREVIEW_SIZE_STEPS = [144, 192, 208, 256, 320, 512, 720]
_preview_size: int = 144
PREVIEW_SIZE_MAX = 720

DETAIL_REQUEST_DELAY_SECONDS = 1.25
DETAIL_REQUEST_FAILURE_BACKOFF_SECONDS = 8.0
DETAIL_REQUEST_MAX_CONSECUTIVE_FAILURES = 3


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


def _civitai_web_base_url() -> str:
    source_mode = _settings()["source_mode"]
    if source_mode == "sfw":
        return "https://civitai.com"
    return "https://civitai.red"


def _civitai_page_url(path: str) -> str:
    clean_path = (path or "").strip()
    if not clean_path.startswith("/"):
        clean_path = f"/{clean_path}"
    return f"{_civitai_web_base_url()}{clean_path}"


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
    <div style="
        position:relative;
        height:100%;
        overflow:hidden;
    ">
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


def _refresh_local_collections_payload(
    selected_collection_id: Optional[int] = None,
    target_collection_id: Optional[int] = None,
) -> str:
    global _target_local_collection_id

    if target_collection_id is None:
        target_collection_id = _target_local_collection_id

    db = _get_db()
    collections = db.list_collections("local")

    rows: List[str] = []
    for collection in collections:
        cid = int(collection["id"])
        name = html.escape(collection["name"])
        active = cid == target_collection_id
        items = db.list_items_for_collection(cid)

        image_count = len([item for item in items if not _item_is_video(item)])
        video_count = len([item for item in items if _item_is_video(item)])

        bg = "#202020" if active else "transparent"
        border = "#4b4b4b" if active else "transparent"
        text = "#ffffff" if active else "#d4d4d4"

        rows.append(f"""
        <button
            type="button"
            onclick="(function() {{
                const root = gradioApp();
                const targetEl = root.querySelector('#collection_target_local_collection_id textarea, #collection_target_local_collection_id input');
                const button = root.querySelector('#collection_target_local_collection_button');

                if (!targetEl || !button) return;

                targetEl.value = '{cid}';
                targetEl.dispatchEvent(new Event('input', {{ bubbles: true }}));
                targetEl.dispatchEvent(new Event('change', {{ bubbles: true }}));

                button.click();
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
            <div style="font-size:13px;font-weight:600;">{name}</div>
            <div style="font-size:11px;color:#8f8f8f;margin-top:2px;">
                {image_count} images · {video_count} videos{" · Add Target" if active else ""}
            </div>
        </button>
        """)

    if not rows:
        rows.append("""
        <div style="padding:8px 4px;color:#6f7682;font-size:12px;">
            No local collections yet.
        </div>
        """)

    return f"""
    <div style="
        max-height:180px;
        overflow:auto;
        padding-top:8px;
    ">
        {''.join(rows)}
    </div>
    """


def _render_controls_bar() -> str:
    global _hide_nsfw, _current_view, _preview_size, _video_autoplay_enabled, _active_cache_filter

    view_label = {
        "grid": "Grid",
        "scroll": "Scrolling",
        "detail": "Detailed",
    }.get(_current_view, "Grid")

    video_label = "Videos: Play" if _video_autoplay_enabled else "Videos: Pause"

    filter_label = {
        "sync": "Sync",
        "preview": "Prvw",
        "full": "Full",
    }.get(_active_cache_filter, "Sync")

    return f"""
    <div style="
        padding:4px 12px 8px 12px;
        color:#777;
        font-size:12px;
        display:flex;
        justify-content:flex-end;
        gap:12px;
    ">
        <span>View: {html.escape(view_label)}</span>
        <span>Filter: {html.escape(filter_label)}</span>
        <span>Preview: {int(_preview_size)}px</span>
        <span>{html.escape(video_label)}</span>
    </div>
    """


def _path_exists(path_value: Any) -> bool:
    path_text = str(path_value or "").strip()
    if not path_text:
        return False

    try:
        return Path(path_text).exists()
    except Exception:
        return False


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

    global _active_cache_filter

    if _active_cache_filter == "preview":
        items = [
            item for item in items
            if _path_exists(item.get("preview_path"))
        ]
    elif _active_cache_filter == "full":
        items = [
            item for item in items
            if _path_exists(item.get("full_path"))
        ]

    return items


def _collection_allows_reorder(collection_id: Optional[int]) -> bool:
    global _active_cache_filter

    if not collection_id:
        return False

    if _active_cache_filter != "sync":
        return False

    db = _get_db()
    collection = db.get_collection(int(collection_id))

    return bool(collection and collection.get("type") == "local")


def _render_feed_cards(
    items: List[Dict[str, Any]],
    collection_id: Optional[int] = None,
    allow_reorder: bool = False,
) -> str:
    global _video_autoplay_enabled, _active_cache_filter

    cards: List[str] = []
    video_autoplay_attr = "autoplay" if _video_autoplay_enabled else ""
    show_add_overlay = _active_cache_filter in {"preview", "full"}

    for item in items:
        preview_source = item.get("preview_path") or item.get("image_url") or ""
        browser_src = _to_browser_src(preview_source)
        image_url = html.escape(browser_src)
        title = item.get("title") or "Untitled"
        safe_title = html.escape(title)

        is_mp4 = _item_is_video(item)

        thumb_html = ""
        item_id = int(item["id"])

        reorder_attrs = ""
        if allow_reorder and collection_id:
            reorder_attrs = f"""
                draggable="true"
                data-reorder-item-id="{item_id}"
                ondragstart="(function(event, el) {{
                    event.dataTransfer.setData('text/plain', '{item_id}');
                    el.style.opacity = '0.45';
                }})(event, this)"
                ondragend="this.style.opacity = '1'"
                ondragover="event.preventDefault();"
                ondrop="(function(event, el) {{
                    event.preventDefault();

                    const root = gradioApp();
                    const container = el.closest('#collection_cards_container, .collection-cards');
                    const payloadEl = root.querySelector('#collection_reorder_payload textarea, #collection_reorder_payload input');
                    const button = root.querySelector('#collection_reorder_button');

                    if (!container || !payloadEl || !button) return;

                    const draggedId = event.dataTransfer.getData('text/plain');
                    const cards = Array.from(container.querySelectorAll('[data-reorder-item-id]'));

                    let dragged = null;
                    for (let c of cards) {{
                        if (c.getAttribute('data-reorder-item-id') === draggedId) {{
                            dragged = c;
                            break;
                        }}
                    }}

                    if (!dragged || dragged === el) return;

                    const draggedIndex = cards.indexOf(dragged);
                    const dropIndex = cards.indexOf(el);

                    if (draggedIndex < 0 || dropIndex < 0) return;

                    if (draggedIndex < dropIndex) {{
                        el.after(dragged);
                    }} else {{
                        el.before(dragged);
                    }}

                    const orderedIds = Array.from(container.querySelectorAll('[data-reorder-item-id]'))
                        .map(function(card) {{
                            return card.getAttribute('data-reorder-item-id');
                        }});

                    payloadEl.value = JSON.stringify({{
                        collection_id: {int(collection_id)},
                        item_ids: orderedIds
                    }});

                    payloadEl.dispatchEvent(new Event('input', {{ bubbles: true }}));
                    payloadEl.dispatchEvent(new Event('change', {{ bubbles: true }}));

                    button.click();
                }})(event, this)"
            """

        add_overlay_html = ""
        if show_add_overlay:
            add_overlay_html = f"""
            <button
                type="button"
                class="collection-add-btn"
                title="Add to selected local collection"
                onclick="(function(event) {{
                    event.preventDefault();
                    event.stopPropagation();

                    const root = gradioApp();
                    const itemEl = root.querySelector('#collection_local_add_item_id textarea, #collection_local_add_item_id input');
                    const button = root.querySelector('#collection_local_add_item_button');

                    if (!itemEl || !button) return;

                    itemEl.value = '{item_id}';
                    itemEl.dispatchEvent(new Event('input', {{ bubbles: true }}));
                    itemEl.dispatchEvent(new Event('change', {{ bubbles: true }}));

                    button.click();
                }})(event)"
                style="
                    position:absolute;
                    top:14px;
                    right:14px;
                    width:26px;
                    height:26px;
                    border-radius:999px;
                    background:rgba(0,0,0,0.68);
                    border:1px solid #555;
                    color:#ffffff;
                    font-size:17px;
                    font-weight:800;
                    line-height:1;
                    display:flex;
                    align-items:center;
                    justify-content:center;
                    cursor:pointer;
                    opacity:0;
                    transition:opacity 0.15s ease;
                    z-index:5;
                    padding:0;
                "
            >+</button>
            """

        if image_url and not is_mp4:
            thumb_html = f"""
            <div
                class="collection-card-shell"
                {reorder_attrs}
                style="
                    position:relative;
                    width:100%;
                "
                onmouseover="(function(el) {{
                    const btn = el.querySelector('.collection-add-btn');
                    if (btn) btn.style.opacity = '1';
                }})(this)"
                onmouseout="(function(el) {{
                    const btn = el.querySelector('.collection-add-btn');
                    if (btn) btn.style.opacity = '0';
                }})(this)"
            >
                {add_overlay_html}
                <button
                    id="collection_card_{item_id}"
                    data-item-id="{item_id}"
                    type="button"
                    class="collection-card"
                    onclick="(function() {{
                        const root = gradioApp();
                        const selectedEl = root.querySelector('#collection_selected_item_id textarea, #collection_selected_item_id input');
                        if (!selectedEl) return;
                        selectedEl.value = '{item_id}';
                        selectedEl.dispatchEvent(new Event('input', {{ bubbles: true }}));
                        selectedEl.dispatchEvent(new Event('change', {{ bubbles: true }}));
                    }})()"
                    style="
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
                </button>
            </div>
            """
        elif is_mp4:
            thumb_html = f"""
            <div
                class="collection-card-shell"
                {reorder_attrs}
                style="
                    position:relative;
                    width:100%;
                "
                onmouseover="(function(el) {{
                    const btn = el.querySelector('.collection-add-btn');
                    if (btn) btn.style.opacity = '1';
                }})(this)"
                onmouseout="(function(el) {{
                    const btn = el.querySelector('.collection-add-btn');
                    if (btn) btn.style.opacity = '0';
                }})(this)"
            >
                {add_overlay_html}
                <button
                    id="collection_card_{item_id}"
                    data-item-id="{item_id}"
                    type="button"
                    class="collection-card"
                    onclick="(function() {{
                        const root = gradioApp();
                        const selectedEl = root.querySelector('#collection_selected_item_id textarea, #collection_selected_item_id input');
                        if (!selectedEl) return;
                        selectedEl.value = '{item_id}';
                        selectedEl.dispatchEvent(new Event('input', {{ bubbles: true }}));
                        selectedEl.dispatchEvent(new Event('change', {{ bubbles: true }}));
                    }})()"
                    style="
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
                            {video_autoplay_attr}
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
                </button>
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


def _webui_root() -> Path:
    return EXTENSION_ROOT.parent.parent


import hashlib


def _read_file_sha256(file_path: Path) -> Optional[str]:
    try:
        sha256 = hashlib.sha256()
        with file_path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
    except Exception:
        return None


def _read_local_sidecar_json(model_path: Path) -> Dict[str, Any]:
    sidecar_candidates = [
        model_path.with_suffix(model_path.suffix + ".json"),
        model_path.with_suffix(".json"),
        model_path.with_suffix(".civitai.info"),
        model_path.with_name(model_path.name + ".json"),
        model_path.with_name(model_path.name + ".civitai.info"),
    ]

    for sidecar_path in sidecar_candidates:
        if not sidecar_path.exists() or not sidecar_path.is_file():
            continue

        try:
            return json.loads(sidecar_path.read_text(encoding="utf-8"))
        except Exception:
            continue

    return {}


def _extract_sidecar_hashes(sidecar: Dict[str, Any]) -> Dict[str, str]:
    hashes = sidecar.get("hashes") or {}
    files = sidecar.get("files") or []

    sha256 = ""
    autov2 = ""

    if isinstance(hashes, dict):
        sha256 = str(hashes.get("SHA256") or hashes.get("sha256") or "").strip().lower()
        autov2 = str(hashes.get("AutoV2") or hashes.get("autov2") or "").strip().lower()

    if not sha256:
        sha256 = str(sidecar.get("sha256") or sidecar.get("SHA256") or "").strip().lower()

    if not autov2:
        autov2 = str(sidecar.get("AutoV2") or sidecar.get("autov2") or sidecar.get("modelHash") or "").strip().lower()

    if (not sha256 or not autov2) and isinstance(files, list):
        for file_info in files:
            if not isinstance(file_info, dict):
                continue

            file_hashes = file_info.get("hashes") or {}
            if not isinstance(file_hashes, dict):
                continue

            if not sha256:
                sha256 = str(file_hashes.get("SHA256") or file_hashes.get("sha256") or "").strip().lower()

            if not autov2:
                autov2 = str(file_hashes.get("AutoV2") or file_hashes.get("autov2") or "").strip().lower()

    return {
        "sha256": sha256,
        "autov2": autov2,
    }


def _identity_from_local_resource_row(row: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    model_ids = set()
    version_ids = set()
    hashes = set()

    if not row:
        return {
            "model_ids": model_ids,
            "version_ids": version_ids,
            "hashes": hashes,
        }

    model_id = str(row.get("model_id") or "").strip()
    version_id = str(row.get("version_id") or "").strip()
    hash_sha256 = str(row.get("hash_sha256") or "").strip().lower()
    hash_autov2 = str(row.get("hash_autov2") or "").strip().lower()

    if model_id:
        model_ids.add(model_id)
    if version_id:
        version_ids.add(version_id)
    if hash_sha256:
        hashes.add(hash_sha256)
        hashes.add(hash_sha256[:10])
    if hash_autov2:
        hashes.add(hash_autov2)

    return {
        "model_ids": model_ids,
        "version_ids": version_ids,
        "hashes": hashes,
    }


def _identity_from_hash_cache_row(row: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    model_ids = set()
    version_ids = set()
    hashes = set()

    if not row:
        return {
            "model_ids": model_ids,
            "version_ids": version_ids,
            "hashes": hashes,
        }

    model_id = str(row.get("model_id") or "").strip()
    version_id = str(row.get("version_id") or "").strip()
    hash_sha256 = str(row.get("hash_sha256") or "").strip().lower()
    hash_autov2 = str(row.get("hash_autov2") or "").strip().lower()

    if model_id:
        model_ids.add(model_id)
    if version_id:
        version_ids.add(version_id)
    if hash_sha256:
        hashes.add(hash_sha256)
        hashes.add(hash_sha256[:10])
    if hash_autov2:
        hashes.add(hash_autov2)

    return {
        "model_ids": model_ids,
        "version_ids": version_ids,
        "hashes": hashes,
    }


def _first_set_value(values: set[str]) -> str:
    for value in values:
        if value:
            return str(value)
    return ""


def _extract_civitai_version_identity(payload: Dict[str, Any]) -> Dict[str, Any]:
    model_ids = set()
    version_ids = set()
    hashes = set()

    if not isinstance(payload, dict):
        return {
            "model_ids": model_ids,
            "version_ids": version_ids,
            "hashes": hashes,
        }

    version_id = payload.get("id") or payload.get("modelVersionId")
    model_id = payload.get("modelId")

    if version_id is not None and str(version_id).strip():
        version_ids.add(str(version_id).strip())

    if model_id is not None and str(model_id).strip():
        model_ids.add(str(model_id).strip())

    model_payload = payload.get("model")
    if isinstance(model_payload, dict):
        nested_model_id = model_payload.get("id") or model_payload.get("modelId")
        if nested_model_id is not None and str(nested_model_id).strip():
            model_ids.add(str(nested_model_id).strip())

    for file_info in payload.get("files", []) or []:
        if not isinstance(file_info, dict):
            continue

        file_hashes = file_info.get("hashes") or {}
        if not isinstance(file_hashes, dict):
            continue

        sha256 = str(file_hashes.get("SHA256") or file_hashes.get("sha256") or "").strip().lower()
        autov2 = str(file_hashes.get("AutoV2") or file_hashes.get("autov2") or "").strip().lower()

        if sha256:
            hashes.add(sha256)
            hashes.add(sha256[:10])

        if autov2:
            hashes.add(autov2)

    return {
        "model_ids": model_ids,
        "version_ids": version_ids,
        "hashes": hashes,
    }


def _extract_sidecar_ids(sidecar: Dict[str, Any]) -> tuple[set[str], set[str]]:
    model_ids = set()
    version_ids = set()

    possible_model_keys = [
        "modelId",
        "model_id",
        "id",
    ]

    possible_version_keys = [
        "modelVersionId",
        "model_version_id",
        "versionId",
        "version_id",
    ]

    for key in possible_model_keys:
        value = sidecar.get(key)
        if value is not None and str(value).strip():
            model_ids.add(str(value).strip())

    for key in possible_version_keys:
        value = sidecar.get(key)
        if value is not None and str(value).strip():
            version_ids.add(str(value).strip())

    model = sidecar.get("model")
    if isinstance(model, dict):
        value = model.get("id") or model.get("modelId")
        if value is not None and str(value).strip():
            model_ids.add(str(value).strip())

    version = sidecar.get("modelVersion")
    if isinstance(version, dict):
        value = version.get("id") or version.get("modelVersionId")
        if value is not None and str(value).strip():
            version_ids.add(str(value).strip())

    return model_ids, version_ids


def _resource_search_dirs() -> Dict[str, Path]:
    webui_root = _webui_root()

    lora_dir = Path(
        getattr(shared.cmd_opts, "lora_dir", "") or webui_root / "models" / "Lora"
    )

    embedding_dir = Path(
        getattr(shared.cmd_opts, "embeddings_dir", "") or webui_root / "embeddings"
    )

    checkpoint_dir = Path(
        getattr(shared.cmd_opts, "ckpt_dir", "") or webui_root / "models" / "Stable-diffusion"
    )

    return {
        "lora": lora_dir,
        "embedding": embedding_dir,
        "checkpoint": checkpoint_dir,
    }


def _normalize_resource_lookup_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (value or "").lower())


def _build_local_resource_index(
    include_hashes: bool = False,
    force_refresh: bool = False,
) -> Dict[str, Dict[str, Any]]:
    global _local_resource_index_cache

    now = time.time()
    cached_at = float(_local_resource_index_cache.get("cached_at", 0) or 0)
    cached_index = _local_resource_index_cache.get("index")
    cached_has_hashes = bool(_local_resource_index_cache.get("has_hashes", False))

    if cached_index and not force_refresh:
        if cached_has_hashes:
            return cached_index
        if not include_hashes and now - cached_at < 60:
            return cached_index

    index = {
        "lora": {"model_ids": set(), "version_ids": set(), "hashes": set(), "files": set()},
        "embedding": {"model_ids": set(), "version_ids": set(), "hashes": set(), "files": set()},
        "checkpoint": {"model_ids": set(), "version_ids": set(), "hashes": set(), "files": set()},
    }

    suffixes = {
        "lora": {".safetensors"},
        "embedding": {".pt", ".safetensors"},
        "checkpoint": {".safetensors", ".ckpt"},
    }

    client = None
    if include_hashes:
        settings = _settings()
        client = CivitaiClient(
            api_key=settings["api_key"] or None,
            source_mode=settings["source_mode"],
        )

    by_hash_cache: Dict[str, Dict[str, Any]] = {}

    for resource_type, root_dir in _resource_search_dirs().items():
        if not root_dir.exists():
            continue

        for path in root_dir.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in suffixes.get(resource_type, set()):
                continue

            index[resource_type]["files"].add(path.as_posix())

            sidecar = _read_local_sidecar_json(path)
            if sidecar:
                model_ids, version_ids = _extract_sidecar_ids(sidecar)
                index[resource_type]["model_ids"].update(model_ids)
                index[resource_type]["version_ids"].update(version_ids)

            if include_hashes and path.suffix.lower() in {".safetensors", ".ckpt", ".pt"}:
                db = _get_db()

                try:
                    stat = path.stat()
                    file_size = int(stat.st_size)
                    modified_at = float(stat.st_mtime)
                except Exception:
                    continue

                file_path_key = path.as_posix()
                local_cached_row = db.get_local_resource_file(file_path_key)

                if (
                    local_cached_row
                    and str(local_cached_row.get("resource_type") or "") == resource_type
                    and int(local_cached_row.get("file_size") or 0) == file_size
                    and float(local_cached_row.get("modified_at") or 0.0) == modified_at
                ):
                    identity = _identity_from_local_resource_row(local_cached_row)
                    index[resource_type]["model_ids"].update(identity["model_ids"])
                    index[resource_type]["version_ids"].update(identity["version_ids"])
                    index[resource_type]["hashes"].update(identity["hashes"])
                    continue

                sidecar_hashes = _extract_sidecar_hashes(sidecar) if sidecar else {}

                normalized_hash = sidecar_hashes.get("sha256", "")
                short_hash = sidecar_hashes.get("autov2", "")

                filename_hash_match = re.search(r"\[([0-9a-f]{8,})\]", path.name.lower())
                if not short_hash and filename_hash_match:
                    short_hash = filename_hash_match.group(1).lower()

                if not normalized_hash and not short_hash:
                    file_hash = _read_file_sha256(path)
                    if not file_hash:
                        continue

                    normalized_hash = file_hash.lower()
                    short_hash = normalized_hash[:10]

                if normalized_hash:
                    index[resource_type]["hashes"].add(normalized_hash)
                    index[resource_type]["hashes"].add(normalized_hash[:10])

                if short_hash:
                    index[resource_type]["hashes"].add(short_hash)

                lookup_hash = normalized_hash or short_hash
                cached_row = db.get_civitai_hash_cache(lookup_hash)

                if cached_row:
                    identity = _identity_from_hash_cache_row(cached_row)
                elif normalized_hash or short_hash:
                    # Fast offline fallback: if we know the local hash but have not
                    # enriched it with Civitai identity yet, still cache the local file
                    # and avoid hammering Civitai during normal availability scans.
                    identity = {
                        "model_ids": set(),
                        "version_ids": set(),
                        "hashes": {h for h in {normalized_hash, short_hash} if h},
                    }
                else:
                    if lookup_hash in by_hash_cache:
                        version_payload = by_hash_cache[lookup_hash]
                    elif client is not None:
                        version_payload = client.get_model_version_by_hash(lookup_hash)
                        by_hash_cache[lookup_hash] = version_payload
                        time.sleep(0.2)
                    else:
                        version_payload = {}

                    identity = _extract_civitai_version_identity(version_payload)
                    version_hashes = _extract_resource_hashes(version_payload)

                    if not normalized_hash:
                        normalized_hash = version_hashes.get("sha256", "")
                    if not short_hash:
                        short_hash = version_hashes.get("autov2", "") or (normalized_hash[:10] if normalized_hash else "")

                    db.upsert_civitai_hash_cache(
                        file_hash=lookup_hash,
                        model_id=_first_set_value(identity["model_ids"]),
                        version_id=_first_set_value(identity["version_ids"]),
                        hash_sha256=normalized_hash,
                        hash_autov2=short_hash,
                        raw_json=json.dumps(version_payload),
                    )

                db.upsert_local_resource_file(
                    file_path=file_path_key,
                    resource_type=resource_type,
                    file_size=file_size,
                    modified_at=modified_at,
                    hash_sha256=normalized_hash,
                    hash_autov2=short_hash,
                    model_id=_first_set_value(identity["model_ids"]),
                    version_id=_first_set_value(identity["version_ids"]),
                )

                index[resource_type]["model_ids"].update(identity["model_ids"])
                index[resource_type]["version_ids"].update(identity["version_ids"])
                index[resource_type]["hashes"].update(identity["hashes"])

    _local_resource_index_cache = {
        "cached_at": now,
        "has_hashes": include_hashes,
        "index": index,
    }

    return index


def _render_local_status_dot(
    resource_type: str,
    resource_name: str,
    model_id: Optional[Any] = None,
    version_id: Optional[Any] = None,
    hash_sha256: str = "",
    hash_autov2: str = "",
) -> str:
    index = _build_local_resource_index(
        include_hashes=True,
        force_refresh=False,
    )
    type_index = index.get(resource_type, {})

    model_id_text = str(model_id).strip() if model_id else ""
    version_id_text = str(version_id).strip() if version_id else ""
    hash_sha256_text = str(hash_sha256 or "").strip().lower()
    hash_autov2_text = str(hash_autov2 or "").strip().lower()

    is_local = False
    can_verify = False

    if model_id_text:
        can_verify = True
        if model_id_text in type_index.get("model_ids", set()):
            is_local = True

    if version_id_text:
        can_verify = True
        if version_id_text in type_index.get("version_ids", set()):
            is_local = True

    if hash_sha256_text:
        can_verify = True
        if hash_sha256_text in type_index.get("hashes", set()):
            is_local = True

    if hash_autov2_text:
        can_verify = True
        if hash_autov2_text in type_index.get("hashes", set()):
            is_local = True

    color = "#36d66b" if is_local else "#e04444"
    title = "Verified local file present" if is_local else "Verified missing locally"

    if not can_verify:
        color = "#8a8a8a"
        title = "Unable to verify: no Civitai hash or version ID available"

    return f"""
    <span
        title="{html.escape(title)}"
        style="
            display:inline-block;
            width:8px;
            height:8px;
            border-radius:999px;
            background:{color};
            margin-right:6px;
            vertical-align:middle;
        "
    ></span>
    """


def _find_detail_item(collection_id: int) -> Optional[Dict[str, Any]]:
    global _selected_item_id

    items = _get_filtered_items_for_collection(collection_id)
    if not items:
        return None

    if _selected_item_id is not None:
        for item in items:
            try:
                if int(item["id"]) == int(_selected_item_id):
                    return item
            except Exception:
                pass

    _selected_item_id = int(items[0]["id"])
    return items[0]


def _render_resources_block(item: Dict[str, Any]) -> str:
    db = _get_db()
    detail = db.get_item_detail(int(item["id"])) or {}
    resources = detail.get("resources") or []

    checkpoint_html = ""
    lora_rows = []
    embedding_rows = []

    for r in resources:
        name_raw = r.get("name") or "Unknown"
        name = html.escape(name_raw)
        version = html.escape(r.get("version_name") or "")
        weight = r.get("weight")
        resource_type = (r.get("resource_type") or "").lower()

        model_id = r.get("model_id")
        version_id = r.get("version_id")
        hash_sha256 = r.get("hash_sha256") or ""
        hash_autov2 = r.get("hash_autov2") or ""
        local_dot = _render_local_status_dot(
            resource_type,
            name_raw,
            model_id,
            version_id,
            hash_sha256,
            hash_autov2,
        )

        url = ""
        if model_id:
            url = _civitai_page_url(f"/models/{model_id}")
            if version_id:
                url += f"?modelVersionId={version_id}"

        link_html = f'<a href="{url}" target="_blank" style="color:#9fd4ff;text-decoration:none;">{name}</a>' if url else name

        badge = ""
        if resource_type == "checkpoint":
            badge = '<span style="font-size:10px;background:#3b6ea8;padding:2px 6px;border-radius:6px;margin-left:6px;">CHECKPOINT</span>'
            checkpoint_html = f"""
                <div style="margin-bottom:10px;">
                    {local_dot}{link_html} {badge}
                    <div style="font-size:11px;color:#888;">{version}</div>
                </div>
            """
        elif resource_type == "lora":
            badge = '<span style="font-size:10px;background:#5a3ba8;padding:2px 6px;border-radius:6px;margin-left:6px;">LORA</span>'
            weight_str = f"{float(weight):.2f}" if weight is not None else "1.0"

            lora_rows.append(f"""
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
                    <div>{local_dot}{link_html} {badge}</div>
                    <div style="font-size:11px;color:#aaa;">{weight_str}</div>
                </div>
            """)
        elif resource_type == "embedding":
            badge = '<span style="font-size:10px;background:#3ba86e;padding:2px 6px;border-radius:6px;margin-left:6px;">EMBED</span>'

            embedding_rows.append(f"""
                <div style="margin-bottom:6px;">
                    {local_dot}{link_html} {badge}
                </div>
            """)

    lora_block = "".join(lora_rows)
    embedding_block = "".join(embedding_rows)

    return f"""
    <div style="margin-bottom:16px;">
        <div style="font-size:16px;font-weight:700;margin-bottom:8px;">Resources used</div>

        {checkpoint_html}

        {"<div style='margin-top:10px;'>" + lora_block + "</div>" if lora_block else ""}

        {"<div style='margin-top:10px;'>" + embedding_block + "</div>" if embedding_block else ""}
    </div>
    """


def _render_detail_view(item: Optional[Dict[str, Any]], collection_id: int) -> str:
    if not item:
        return """
        <div style="
            height:72vh;
            display:flex;
            align-items:center;
            justify-content:center;
            color:#888;
            background:#101010;
            border-radius:12px;
        ">
            No item selected.
        </div>
        """

    items = _get_filtered_items_for_collection(collection_id)
    item_ids = [int(row["id"]) for row in items]

    current_item_id = int(item["id"])
    current_index = item_ids.index(current_item_id) if current_item_id in item_ids else 0

    previous_item_id = item_ids[current_index - 1] if current_index > 0 else None
    next_item_id = item_ids[current_index + 1] if current_index < len(item_ids) - 1 else None

    def _nav_button(label: str, target_item_id: Optional[int], side: str) -> str:
        if target_item_id is None:
            return ""

        return f"""
        <button
            type="button"
            onclick="(function(event) {{
                event.stopPropagation();
                const root = gradioApp();
                const selectedEl = root.querySelector('#collection_selected_item_id textarea, #collection_selected_item_id input');
                if (!selectedEl) return;
                selectedEl.value = '{target_item_id}';
                selectedEl.dispatchEvent(new Event('input', {{ bubbles: true }}));
                selectedEl.dispatchEvent(new Event('change', {{ bubbles: true }}));
            }})(event)"
            style="
                position:absolute;
                {side}:14px;
                top:50%;
                transform:translateY(-50%);
                width:42px;
                height:64px;
                border-radius:10px;
                border:1px solid #333;
                background:rgba(0,0,0,0.55);
                color:#f2f2f2;
                font-size:30px;
                line-height:1;
                cursor:pointer;
                z-index:5;
            "
        >{label}</button>
        """

    previous_button = _nav_button("‹", previous_item_id, "left")
    next_button = _nav_button("›", next_item_id, "right")

    media_source = item.get("full_path") or item.get("preview_path") or item.get("image_url") or ""
    media_url = html.escape(_to_browser_src(media_source))
    title = html.escape(item.get("title") or "Untitled")
    creator_raw = item.get("creator_name") or "Unknown"
    creator = html.escape(creator_raw)
    creator_url = html.escape(_civitai_page_url(f"/user/{quote(str(creator_raw))}")) if creator_raw != "Unknown" else ""

    raw_post_url = item.get("post_url") or ""
    if "/images/" in raw_post_url:
        raw_post_url = _civitai_page_url(raw_post_url.split("/images/", 1)[1].join(["/images/", ""]))
    post_url = html.escape(raw_post_url)
    prompt = html.escape(item.get("prompt") or "")
    negative_prompt = html.escape(item.get("negative_prompt") or "")
    sampler = html.escape(str(item.get("sampler") or ""))
    steps = html.escape(str(item.get("steps") or ""))
    cfg_scale = html.escape(str(item.get("cfg_scale") or ""))
    seed = html.escape(str(item.get("seed") or ""))
    checkpoint = html.escape(item.get("checkpoint_name") or "")

    if _item_is_video(item):
        media_html = f"""
        <video
            src="{media_url}"
            controls
            autoplay
            loop
            muted
            playsinline
            onclick="event.stopPropagation();"
            style="
                max-width:100%;
                max-height:72vh;
                width:auto;
                height:auto;
                object-fit:contain;
                display:block;
            "
        ></video>
        """
    else:
        media_html = f"""
        <img
            src="{media_url}"
            alt="{title}"
            style="
                max-width:100%;
                max-height:72vh;
                width:auto;
                height:auto;
                object-fit:contain;
                display:block;
            "
        />
        """

    return f"""
    <div
        tabindex="0"
        onmouseenter="this.focus()"
        onkeydown="(function(event) {{
            if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') return;
            event.preventDefault();

            const targetId = event.key === 'ArrowLeft'
                ? '{previous_item_id or ''}'
                : '{next_item_id or ''}';

            if (!targetId) return;

            const root = gradioApp();
            const selectedEl = root.querySelector('#collection_selected_item_id textarea, #collection_selected_item_id input');
            if (!selectedEl) return;

            selectedEl.value = targetId;
            selectedEl.dispatchEvent(new Event('input', {{ bubbles: true }}));
            selectedEl.dispatchEvent(new Event('change', {{ bubbles: true }}));
        }})(event)"
        style="
            min-height:72vh;
            display:grid;
            grid-template-columns:minmax(0, 1fr) 360px;
            gap:14px;
            background:#101010;
            border-radius:12px;
            padding:12px;
            box-sizing:border-box;
            outline:none;
        "
    >
        <div
            title="Click image to return to scrolling view"
            onclick="(function() {{
                const root = gradioApp();
                const actionEl = root.querySelector('#collection_detail_action textarea, #collection_detail_action input');
                if (!actionEl) return;
                actionEl.value = 'return_scroll:' + Date.now();
                actionEl.dispatchEvent(new Event('input', {{ bubbles: true }}));
                actionEl.dispatchEvent(new Event('change', {{ bubbles: true }}));
            }})()"
            style="
                position:relative;
                min-width:0;
                height:72vh;
                display:flex;
                align-items:center;
                justify-content:center;
                background:#0b0b0b;
                border-radius:10px;
                overflow:hidden;
                cursor:pointer;
            "
        >
            {previous_button}
            {media_html}
            {next_button}
        </div>

        <div style="
            max-height:72vh;
            overflow:auto;
            background:#25262b;
            border-radius:12px;
            padding:16px;
            box-sizing:border-box;
            color:#d8d8d8;
        ">
            <div style="font-size:18px;font-weight:700;margin-bottom:4px;">
                {f'''
                <a
                    href="{post_url}"
                    target="_blank"
                    style="
                        color:#9fd4ff;
                        text-decoration:none;
                    "
                    onmouseover="this.style.textDecoration='underline'"
                    onmouseout="this.style.textDecoration='none'"
                >
                    {title}
                </a>
                ''' if post_url else title}
            </div>

            <div style="font-size:12px;color:#9aa0aa;margin-bottom:16px;">
                Creator:
                {f'''
                <a
                    href="{creator_url}"
                    target="_blank"
                    style="color:#9fd4ff;text-decoration:none;margin-left:4px;"
                >
                    {creator}
                </a>
                ''' if creator_url else creator}
            </div>

            {_render_resources_block(item)}

            <div style="font-size:16px;font-weight:700;margin-bottom:8px;">Generation data</div>

            <div style="font-size:13px;color:#aeb4c0;margin-bottom:10px;">
                <strong>Checkpoint:</strong> {checkpoint or "Unknown"}<br>
                <strong>Sampler:</strong> {sampler or "Unknown"}<br>
                <strong>Steps:</strong> {steps or "Unknown"}<br>
                <strong>CFG:</strong> {cfg_scale or "Unknown"}<br>
                <strong>Seed:</strong> {seed or "Unknown"}
            </div>

            <hr style="border:0;border-top:1px solid #3a3b40;margin:14px 0;">

            <div style="font-size:15px;font-weight:700;margin-bottom:6px;">Prompt</div>
            <div style="font-size:13px;line-height:1.45;color:#b9bec9;white-space:pre-wrap;">{prompt or "No prompt found."}</div>

            <hr style="border:0;border-top:1px solid #3a3b40;margin:14px 0;">

            <div style="font-size:15px;font-weight:700;margin-bottom:6px;">Negative prompt</div>
            <div style="font-size:13px;line-height:1.45;color:#b9bec9;white-space:pre-wrap;">{negative_prompt or "No negative prompt found."}</div>
        </div>

    </div>
    """


def _split_media_items(items: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    image_items: List[Dict[str, Any]] = []
    video_items: List[Dict[str, Any]] = []

    for item in items:
        if _item_is_video(item):
            video_items.append(item)
        else:
            image_items.append(item)

    return image_items, video_items


def _render_feed_batch(
    collection_id: int,
    offset: int,
    limit: int = IMAGE_BATCH_SIZE,
    media_kind: str = "all",
) -> tuple[str, int, bool]:
    items = _get_filtered_items_for_collection(collection_id)

    if media_kind == "image":
        items = [item for item in items if not _item_is_video(item)]
    elif media_kind == "video":
        items = [item for item in items if _item_is_video(item)]

    batch_items = items[offset:offset + limit]
    cards_html = _render_feed_cards(
        batch_items,
        collection_id=collection_id,
        allow_reorder=_collection_allows_reorder(collection_id),
    )
    next_offset = offset + len(batch_items)
    has_more = next_offset < len(items)
    return cards_html, next_offset, has_more


def _render_video_batch_section(
    collection_id: int,
    video_items: List[Dict[str, Any]],
    video_offset: int = 0,
    mixed_collection: bool = False,
) -> str:
    global _preview_size

    total_videos = len(video_items)
    safe_offset = max(0, min(video_offset, max(0, total_videos - 1)))
    safe_offset = (safe_offset // VIDEO_BATCH_SIZE) * VIDEO_BATCH_SIZE

    batch_items = video_items[safe_offset:safe_offset + VIDEO_BATCH_SIZE]
    cards_html = _render_feed_cards(
        batch_items,
        collection_id=collection_id,
        allow_reorder=_collection_allows_reorder(collection_id),
    )

    previous_offset = max(0, safe_offset - VIDEO_BATCH_SIZE)
    next_offset = safe_offset + VIDEO_BATCH_SIZE

    has_previous = safe_offset > 0
    has_next = next_offset < total_videos

    start_label = safe_offset + 1 if total_videos else 0
    end_label = min(safe_offset + len(batch_items), total_videos)

    section_title = "Videos" if mixed_collection else "Video Collection"
    top_margin = "28px" if mixed_collection else "0"

    previous_disabled = "disabled" if not has_previous else ""
    next_disabled = "disabled" if not has_next else ""

    previous_opacity = "0.35" if not has_previous else "1"
    next_opacity = "0.35" if not has_next else "1"

    return f"""
    <div
        id="collection_video_section"
        style="
            margin-top:{top_margin};
            padding-top:4px;
        "
    >
        <div style="
            display:flex;
            align-items:center;
            justify-content:space-between;
            gap:12px;
            margin-bottom:10px;
            color:#d8d8d8;
        ">
            <div>
                <div style="font-size:16px;font-weight:700;">{section_title}</div>
                <div style="font-size:12px;color:#8a8a8a;">
                    Showing videos {start_label}–{end_label} of {total_videos}
                </div>
            </div>

            <div style="display:flex;gap:8px;">
                <button
                    type="button"
                    {previous_disabled}
                    onclick="(function() {{
                        const root = gradioApp();
                        const offsetEl = root.querySelector('#collection_video_batch_offset textarea, #collection_video_batch_offset input');
                        const button = root.querySelector('#collection_video_batch_button');
                        if (!offsetEl || !button) return;
                        offsetEl.value = '{previous_offset}';
                        offsetEl.dispatchEvent(new Event('input', {{ bubbles: true }}));
                        offsetEl.dispatchEvent(new Event('change', {{ bubbles: true }}));
                        button.click();
                    }})()"
                    style="
                        opacity:{previous_opacity};
                        border:1px solid #333;
                        background:#202020;
                        color:#d8d8d8;
                        border-radius:8px;
                        padding:7px 10px;
                        cursor:pointer;
                    "
                >
                    Previous Videos
                </button>

                <button
                    type="button"
                    {next_disabled}
                    onclick="(function() {{
                        const root = gradioApp();
                        const offsetEl = root.querySelector('#collection_video_batch_offset textarea, #collection_video_batch_offset input');
                        const button = root.querySelector('#collection_video_batch_button');
                        if (!offsetEl || !button) return;
                        offsetEl.value = '{next_offset}';
                        offsetEl.dispatchEvent(new Event('input', {{ bubbles: true }}));
                        offsetEl.dispatchEvent(new Event('change', {{ bubbles: true }}));
                        button.click();
                    }})()"
                    style="
                        opacity:{next_opacity};
                        border:1px solid #333;
                        background:#202020;
                        color:#d8d8d8;
                        border-radius:8px;
                        padding:7px 10px;
                        cursor:pointer;
                    "
                >
                    Next Videos
                </button>
            </div>
        </div>

        <div
            class="collection-cards collection-view-grid"
            style="
                display:grid;
                grid-template-columns:repeat(auto-fill, minmax({int(_preview_size)}px, 1fr));
                gap:12px;
                align-items:start;
            "
        >
            {cards_html}
        </div>
    </div>
    """


def _render_feed_html(collection_id: Optional[int], video_offset: int = 0) -> str:
    global _current_view, _preview_size, _selected_item_id

    if _current_view == "detail" and collection_id:
        return _render_detail_view(_find_detail_item(collection_id), collection_id)

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

    image_items, video_items = _split_media_items(items)
    has_images = len(image_items) > 0
    has_videos = len(video_items) > 0
    is_mixed_collection = has_images and has_videos

    initial_offset = 0

    if _current_view == "scroll" and _selected_item_id is not None:
        for index, item in enumerate(image_items if has_images else items):
            try:
                if int(item["id"]) == int(_selected_item_id):
                    initial_offset = max(0, index - 4)
                    break
            except Exception:
                pass

    if has_videos and not has_images:
        video_section_html = _render_video_batch_section(
            collection_id=collection_id,
            video_items=video_items,
            video_offset=video_offset,
            mixed_collection=False,
        )

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
            {video_section_html}
        </div>
        """

    cards_html, next_offset, has_more = _render_feed_batch(
        collection_id=collection_id,
        offset=initial_offset,
        limit=IMAGE_BATCH_SIZE,
        media_kind="image",
    )

    sentinel_text = "Loading more images..." if has_more else "End of images"

    video_section_html = ""
    if is_mixed_collection:
        video_section_html = _render_video_batch_section(
            collection_id=collection_id,
            video_items=video_items,
            video_offset=video_offset,
            mixed_collection=True,
        )

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
        {f'''
        <div style="
            display:flex;
            align-items:center;
            justify-content:space-between;
            gap:12px;
            margin-bottom:10px;
            color:#d8d8d8;
        ">
            <div>
                <div style="font-size:16px;font-weight:700;">Images</div>
                <div style="font-size:12px;color:#8a8a8a;">
                    {len(image_items)} images · {len(video_items)} videos
                </div>
            </div>
        </div>
        ''' if is_mixed_collection else ""}

        <div
            id="collection_cards_container"
            class="collection-cards collection-view-grid"
            style="
                display:grid;
                grid-template-columns:repeat(auto-fill, minmax({int(_preview_size)}px, 1fr));
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

        {video_section_html}
    </div>
    """


def _load_collection_feed(selected_collection_id_raw: str) -> tuple[str, str, str, str]:
    global _active_collection_id

    selected_collection_id: Optional[int] = None

    try:
        if selected_collection_id_raw and str(selected_collection_id_raw).strip():
            selected_collection_id = int(str(selected_collection_id_raw).strip())
    except Exception:
        selected_collection_id = None

    _active_collection_id = selected_collection_id

    return (
        _refresh_sidebar_payload(selected_collection_id),
        _render_feed_html(selected_collection_id),
        _selected_local_collection_name(selected_collection_id),
        _refresh_local_collections_payload(selected_collection_id),
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
        limit=IMAGE_BATCH_SIZE,
        media_kind="image",
    )

    payload_json = json.dumps(
        {
            "html": cards_html,
            "next_offset": next_offset,
            "has_more": has_more,
        }
    )
    return f'<div class="collection-batch-payload" data-json="{html.escape(payload_json, quote=True)}"></div>'


def _set_video_batch(selected_collection_id_raw: str, video_offset_raw: str) -> str:
    selected_collection_id: Optional[int] = None
    video_offset = 0

    try:
        if selected_collection_id_raw and str(selected_collection_id_raw).strip():
            selected_collection_id = int(str(selected_collection_id_raw).strip())
    except Exception:
        selected_collection_id = None

    try:
        if video_offset_raw and str(video_offset_raw).strip():
            video_offset = max(0, int(str(video_offset_raw).strip()))
    except Exception:
        video_offset = 0

    return _render_feed_html(selected_collection_id, video_offset=video_offset)


def _selected_local_collection_name(selected_collection_id: Optional[int]) -> str:
    if not selected_collection_id:
        return ""

    db = _get_db()
    collection = db.get_collection(int(selected_collection_id))
    if not collection or collection.get("type") != "local":
        return ""

    return collection.get("name") or ""


def _unique_local_collection_name(base_name: str = "New Collection") -> str:
    db = _get_db()
    local_names = {
        (collection.get("name") or "").strip().lower()
        for collection in db.list_collections("local")
    }

    clean_base = (base_name or "New Collection").strip() or "New Collection"

    if clean_base.lower() not in local_names:
        return clean_base

    index = 2
    while True:
        candidate = f"{clean_base} {index}"
        if candidate.lower() not in local_names:
            return candidate
        index += 1


def _create_local_collection(name: str):
    global _target_local_collection_id

    db = _get_db()
    collection_name = _unique_local_collection_name(name or "New Collection")
    collection_id = db.create_collection(name=collection_name, collection_type="local")
    _target_local_collection_id = collection_id

    return (
        gr.update(value=str(collection_id)),
        gr.update(value=str(collection_id)),
        _refresh_sidebar_payload(collection_id),
        _render_feed_html(collection_id),
        gr.update(value=collection_name),
        _refresh_local_collections_payload(collection_id, collection_id),
        f"Local collection created: {collection_name}",
    )


def _set_target_local_collection(target_collection_id_raw: str, selected_collection_id_raw: str):
    global _target_local_collection_id, _active_cache_filter, _active_collection_id

    target_collection_id: Optional[int] = None
    selected_collection_id: Optional[int] = None

    try:
        if target_collection_id_raw and str(target_collection_id_raw).strip():
            target_collection_id = int(str(target_collection_id_raw).strip())
    except Exception:
        target_collection_id = None

    try:
        if selected_collection_id_raw and str(selected_collection_id_raw).strip():
            selected_collection_id = int(str(selected_collection_id_raw).strip())
    except Exception:
        selected_collection_id = None

    if not target_collection_id:
        return (
            gr.update(),
            _refresh_sidebar_payload(selected_collection_id),
            _render_feed_html(selected_collection_id),
            _selected_local_collection_name(selected_collection_id),
            _refresh_local_collections_payload(selected_collection_id),
            "No local collection selected as add target.",
        )

    db = _get_db()
    collection = db.get_collection(target_collection_id)

    if not collection or collection.get("type") != "local":
        return (
            gr.update(),
            _refresh_sidebar_payload(selected_collection_id),
            _render_feed_html(selected_collection_id),
            _selected_local_collection_name(selected_collection_id),
            _refresh_local_collections_payload(selected_collection_id),
            "Only local collections can be used as add targets.",
        )

    _target_local_collection_id = target_collection_id

    view_collection_id = selected_collection_id
    selected_update = gr.update()

    if _active_cache_filter == "sync":
        view_collection_id = target_collection_id
        _active_collection_id = target_collection_id
        selected_update = gr.update(value=str(target_collection_id))

    collection_name = collection.get("name") or "Local Collection"

    return (
        selected_update,
        _refresh_sidebar_payload(view_collection_id),
        _render_feed_html(view_collection_id),
        _selected_local_collection_name(view_collection_id),
        _refresh_local_collections_payload(
            view_collection_id,
            target_collection_id,
        ),
        f"Add target selected: {collection_name}",
    )


def _rename_selected_local_collection(selected_collection_id_raw: str, name: str):
    selected_collection_id: Optional[int] = None

    try:
        if selected_collection_id_raw and str(selected_collection_id_raw).strip():
            selected_collection_id = int(str(selected_collection_id_raw).strip())
    except Exception:
        selected_collection_id = None

    if not selected_collection_id:
        return (
            _refresh_sidebar_payload(),
            gr.update(),
            _refresh_local_collections_payload(),
            "Select a local collection before renaming.",
        )

    clean_name = (name or "").strip()
    if not clean_name:
        return (
            _refresh_sidebar_payload(selected_collection_id),
            gr.update(),
            _refresh_local_collections_payload(selected_collection_id),
            "Local collection name cannot be blank.",
        )

    db = _get_db()
    collection = db.get_collection(selected_collection_id)
    if not collection or collection.get("type") != "local":
        return (
            _refresh_sidebar_payload(selected_collection_id),
            gr.update(value=""),
            _refresh_local_collections_payload(selected_collection_id),
            "Only local collections can be renamed.",
        )

    renamed = db.rename_local_collection(selected_collection_id, clean_name)

    if not renamed:
        return (
            _refresh_sidebar_payload(selected_collection_id),
            gr.update(),
            _refresh_local_collections_payload(selected_collection_id),
            "Rename failed.",
        )

    return (
        _refresh_sidebar_payload(selected_collection_id),
        gr.update(value=clean_name),
        _refresh_local_collections_payload(selected_collection_id),
        f"Local collection renamed: {clean_name}",
    )


def _delete_selected_local_collection(selected_collection_id_raw: str, confirm_collection_id_raw: str):
    selected_collection_id: Optional[int] = None
    confirm_collection_id = str(confirm_collection_id_raw or "").strip()

    try:
        if selected_collection_id_raw and str(selected_collection_id_raw).strip():
            selected_collection_id = int(str(selected_collection_id_raw).strip())
    except Exception:
        selected_collection_id = None

    if not selected_collection_id:
        return (
            gr.update(value=""),
            gr.update(value=""),
            _refresh_sidebar_payload(),
            _render_feed_html(None),
            gr.update(value=""),
            _refresh_local_collections_payload(),
            "Select a local collection before deleting.",
        )

    db = _get_db()
    collection = db.get_collection(selected_collection_id)

    if not collection or collection.get("type") != "local":
        return (
            gr.update(value=str(selected_collection_id)),
            gr.update(value=""),
            _refresh_sidebar_payload(selected_collection_id),
            _render_feed_html(selected_collection_id),
            gr.update(value=""),
            _refresh_local_collections_payload(selected_collection_id),
            "Only local collections can be deleted.",
        )

    collection_name = collection.get("name") or "Local Collection"

    if confirm_collection_id != str(selected_collection_id):
        return (
            gr.update(value=str(selected_collection_id)),
            gr.update(value=str(selected_collection_id)),
            _refresh_sidebar_payload(selected_collection_id),
            _render_feed_html(selected_collection_id),
            gr.update(value=collection_name),
            _refresh_local_collections_payload(selected_collection_id),
            f"Press − again to confirm deleting local collection: {collection_name}. Files will not be deleted.",
        )

    deleted = db.delete_local_collection(selected_collection_id)

    if not deleted:
        return (
            gr.update(value=str(selected_collection_id)),
            gr.update(value=""),
            _refresh_sidebar_payload(selected_collection_id),
            _render_feed_html(selected_collection_id),
            gr.update(value=collection_name),
            _refresh_local_collections_payload(selected_collection_id),
            "Delete failed.",
        )

    return (
        gr.update(value=""),
        gr.update(value=""),
        _refresh_sidebar_payload(),
        _render_feed_html(None),
        gr.update(value=""),
        _refresh_local_collections_payload(),
        f"Local collection deleted: {collection_name}. Files were not deleted.",
    )


def _add_item_to_target_local_collection(
    item_id_raw: str,
    target_collection_id_raw: str,
    selected_collection_id_raw: str,
):
    global _active_cache_filter

    if _active_cache_filter not in {"preview", "full"}:
        selected_collection_id: Optional[int] = None
        try:
            if selected_collection_id_raw and str(selected_collection_id_raw).strip():
                selected_collection_id = int(str(selected_collection_id_raw).strip())
        except Exception:
            selected_collection_id = None

        return (
            _refresh_sidebar_payload(selected_collection_id),
            _refresh_local_collections_payload(selected_collection_id),
            _render_feed_html(selected_collection_id),
            "Switch to Prvw or Full filter before adding to a local collection.",
        )

    item_id: Optional[int] = None
    target_collection_id: Optional[int] = None
    selected_collection_id: Optional[int] = None

    try:
        if item_id_raw and str(item_id_raw).strip():
            item_id = int(str(item_id_raw).strip())
    except Exception:
        item_id = None

    try:
        if target_collection_id_raw and str(target_collection_id_raw).strip():
            target_collection_id = int(str(target_collection_id_raw).strip())
    except Exception:
        target_collection_id = None

    try:
        if selected_collection_id_raw and str(selected_collection_id_raw).strip():
            selected_collection_id = int(str(selected_collection_id_raw).strip())
    except Exception:
        selected_collection_id = None

    if not item_id:
        return (
            _refresh_sidebar_payload(selected_collection_id),
            _refresh_local_collections_payload(selected_collection_id),
            _render_feed_html(selected_collection_id),
            "No item selected to add.",
        )

    if not target_collection_id:
        return (
            _refresh_sidebar_payload(selected_collection_id),
            _refresh_local_collections_payload(selected_collection_id),
            _render_feed_html(selected_collection_id),
            "Select a local collection first, then click + on an image or video.",
        )

    db = _get_db()
    target_collection = db.get_collection(target_collection_id)

    if not target_collection or target_collection.get("type") != "local":
        return (
            _refresh_sidebar_payload(selected_collection_id),
            _refresh_local_collections_payload(selected_collection_id),
            _render_feed_html(selected_collection_id),
            "Add target must be a local collection.",
        )

    item = db.get_item_detail(item_id)
    if not item:
        return (
            _refresh_sidebar_payload(selected_collection_id),
            _refresh_local_collections_payload(selected_collection_id),
            _render_feed_html(selected_collection_id),
            "Selected item was not found.",
        )

    if _active_cache_filter == "preview" and not _path_exists(item.get("preview_path")):
        return (
            _refresh_sidebar_payload(selected_collection_id),
            _refresh_local_collections_payload(selected_collection_id),
            _render_feed_html(selected_collection_id),
            "This item does not have a local preview file.",
        )

    if _active_cache_filter == "full" and not _path_exists(item.get("full_path")):
        return (
            _refresh_sidebar_payload(selected_collection_id),
            _refresh_local_collections_payload(selected_collection_id),
            _render_feed_html(selected_collection_id),
            "This item does not have a local full-size file.",
        )

    existing_items = db.list_items_for_collection(target_collection_id)
    order_index = len(existing_items)

    db.add_item_to_collection(
        collection_id=target_collection_id,
        item_id=item_id,
        order_index=order_index,
    )

    target_name = target_collection.get("name") or "Local Collection"

    return (
        _refresh_sidebar_payload(selected_collection_id),
        _refresh_local_collections_payload(selected_collection_id, target_collection_id),
        _render_feed_html(selected_collection_id),
        f"Added reference to local collection: {target_name}",
    )


def _reorder_local_collection_items(reorder_payload_raw: str, selected_collection_id_raw: str):
    selected_collection_id: Optional[int] = None

    try:
        if selected_collection_id_raw and str(selected_collection_id_raw).strip():
            selected_collection_id = int(str(selected_collection_id_raw).strip())
    except Exception:
        selected_collection_id = None

    try:
        payload = json.loads(reorder_payload_raw or "{}")
    except Exception:
        payload = {}

    payload_collection_id = payload.get("collection_id")
    item_ids_raw = payload.get("item_ids") or []

    try:
        payload_collection_id = int(payload_collection_id)
    except Exception:
        payload_collection_id = None

    if not payload_collection_id:
        return (
            _refresh_sidebar_payload(selected_collection_id),
            _render_feed_html(selected_collection_id),
            "No reorder payload.",
        )

    if selected_collection_id != payload_collection_id:
        return (
            _refresh_sidebar_payload(selected_collection_id),
            _render_feed_html(selected_collection_id),
            "Collection changed, reorder ignored.",
        )

    db = _get_db()
    reordered = db.reorder_local_collection_items(payload_collection_id, item_ids_raw)

    if not reordered:
        return (
            _refresh_sidebar_payload(selected_collection_id),
            _render_feed_html(selected_collection_id),
            "Reorder failed.",
        )

    return (
        _refresh_sidebar_payload(selected_collection_id),
        _render_feed_html(selected_collection_id),
        "Order saved.",
    )


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


def _set_collection_view(view_name: str, selected_collection_id_raw: str, selected_item_id_raw: str):
    global _current_view, _selected_item_id, _active_collection_id

    if view_name not in {"grid", "scroll", "detail"}:
        view_name = "grid"

    _current_view = view_name

    selected_collection_id: Optional[int] = None
    try:
        if selected_collection_id_raw and str(selected_collection_id_raw).strip():
            selected_collection_id = int(str(selected_collection_id_raw).strip())
    except Exception:
        selected_collection_id = None

    if selected_collection_id is None:
        selected_collection_id = _active_collection_id
    else:
        _active_collection_id = selected_collection_id

    if _current_view == "detail":
        try:
            if selected_item_id_raw and str(selected_item_id_raw).strip():
                _selected_item_id = int(str(selected_item_id_raw).strip())
        except Exception:
            _selected_item_id = None
    else:
        _selected_item_id = None

    if _current_view == "detail" and selected_collection_id and _selected_item_id is None:
        items = _get_filtered_items_for_collection(selected_collection_id)
        if items:
            _selected_item_id = int(items[0]["id"])

    sidebar_visible = _current_view == "grid"

    return (
        gr.update(variant="primary" if _current_view == "grid" else "secondary"),
        gr.update(variant="primary" if _current_view == "scroll" else "secondary"),
        gr.update(variant="primary" if _current_view == "detail" else "secondary"),
        gr.update(visible=sidebar_visible),
        gr.update(value=str(_selected_item_id or "")),
        _render_controls_bar(),
        _refresh_sidebar_payload(selected_collection_id),
        _render_feed_html(selected_collection_id),
    )

def _open_selected_item_detail(selected_collection_id_raw: str, selected_item_id_raw: str):
    global _current_view, _selected_item_id, _active_collection_id

    if not selected_item_id_raw or not str(selected_item_id_raw).strip():
        return (
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            _render_controls_bar(),
            _refresh_sidebar_payload(_active_collection_id),
            _render_feed_html(_active_collection_id),
        )

    _current_view = "detail"

    selected_collection_id: Optional[int] = None
    try:
        if selected_collection_id_raw and str(selected_collection_id_raw).strip():
            selected_collection_id = int(str(selected_collection_id_raw).strip())
    except Exception:
        selected_collection_id = None

    if selected_collection_id is None:
        selected_collection_id = _active_collection_id
    else:
        _active_collection_id = selected_collection_id

    try:
        if selected_item_id_raw and str(selected_item_id_raw).strip():
            _selected_item_id = int(str(selected_item_id_raw).strip())
    except Exception:
        _selected_item_id = None

    return (
        gr.update(variant="secondary"),
        gr.update(variant="secondary"),
        gr.update(variant="primary"),
        gr.update(visible=False),
        _render_controls_bar(),
        _refresh_sidebar_payload(selected_collection_id),
        _render_feed_html(selected_collection_id),
    )


def _return_detail_to_scroll(selected_collection_id_raw: str, selected_item_id_raw: str, detail_action_raw: str):
    global _current_view, _selected_item_id, _active_collection_id

    # Only act on our specific command (supports timestamped values)
    if not str(detail_action_raw or "").startswith("return_scroll"):
        return (
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
        )

    # Switch back to scrolling view
    _current_view = "scroll"

    # Resolve collection id
    selected_collection_id: Optional[int] = None
    try:
        if selected_collection_id_raw and str(selected_collection_id_raw).strip():
            selected_collection_id = int(str(selected_collection_id_raw).strip())
    except Exception:
        selected_collection_id = None

    # 🔴 Critical fix: fallback to last active collection
    if selected_collection_id is None:
        selected_collection_id = _active_collection_id
    else:
        _active_collection_id = selected_collection_id

    # Preserve selected item (for scroll positioning)
    try:
        if selected_item_id_raw and str(selected_item_id_raw).strip():
            _selected_item_id = int(str(selected_item_id_raw).strip())
    except Exception:
        pass

    return (
        gr.update(variant="secondary"),  # grid
        gr.update(variant="primary"),    # scroll (active)
        gr.update(variant="secondary"),  # detail
        gr.update(visible=False),        # sidebar stays hidden in scroll
        gr.update(value=""),             # reset selected item textbox
        gr.update(value=""),             # reset action trigger
        _render_controls_bar(),
        _refresh_sidebar_payload(selected_collection_id),
        _render_feed_html(selected_collection_id),
    )


def _set_preview_size(preview_step_raw: int, selected_collection_id_raw: str):
    global _preview_size

    try:
        step_index = int(preview_step_raw)
        step_index = max(0, min(len(PREVIEW_SIZE_STEPS) - 1, step_index))
        _preview_size = PREVIEW_SIZE_STEPS[step_index]
    except Exception:
        _preview_size = PREVIEW_SIZE_STEPS[0]

    selected_collection_id: Optional[int] = None
    try:
        if selected_collection_id_raw and str(selected_collection_id_raw).strip():
            selected_collection_id = int(str(selected_collection_id_raw).strip())
    except Exception:
        selected_collection_id = None

    return (
        _render_controls_bar(),
        _render_feed_html(selected_collection_id),
    )


def _scan_local_files_for_availability(selected_collection_id_raw: str):
    selected_collection_id: Optional[int] = None

    try:
        if selected_collection_id_raw and str(selected_collection_id_raw).strip():
            selected_collection_id = int(str(selected_collection_id_raw).strip())
    except Exception:
        selected_collection_id = None

    started_at = time.time()
    index = _build_local_resource_index(
        include_hashes=True,
        force_refresh=True,
    )
    elapsed = max(0.0, time.time() - started_at)

    checkpoint_files = len(index.get("checkpoint", {}).get("files", set()))
    lora_files = len(index.get("lora", {}).get("files", set()))
    embedding_files = len(index.get("embedding", {}).get("files", set()))

    checkpoint_hashes = len(index.get("checkpoint", {}).get("hashes", set()))
    lora_hashes = len(index.get("lora", {}).get("hashes", set()))
    embedding_hashes = len(index.get("embedding", {}).get("hashes", set()))

    status = (
        "Local file availability scan complete.\n\n"
        f"Checkpoint files: {checkpoint_files} | hash entries: {checkpoint_hashes}\n\n"
        f"LoRA files: {lora_files} | hash entries: {lora_hashes}\n\n"
        f"Embedding files: {embedding_files} | hash entries: {embedding_hashes}\n\n"
        f"Scan time: {elapsed:.1f}s."
    )

    return (
        _render_controls_bar(),
        _refresh_sidebar_payload(selected_collection_id),
        _render_feed_html(selected_collection_id),
        status,
    )


def _set_cache_filter(filter_name: str, selected_collection_id_raw: str):
    global _active_cache_filter

    if filter_name not in {"sync", "preview", "full"}:
        filter_name = "sync"

    _active_cache_filter = filter_name

    selected_collection_id: Optional[int] = None
    try:
        if selected_collection_id_raw and str(selected_collection_id_raw).strip():
            selected_collection_id = int(str(selected_collection_id_raw).strip())
    except Exception:
        selected_collection_id = None

    return (
        gr.update(variant="primary" if _active_cache_filter == "sync" else "secondary"),
        gr.update(variant="primary" if _active_cache_filter == "preview" else "secondary"),
        gr.update(variant="primary" if _active_cache_filter == "full" else "secondary"),
        _render_controls_bar(),
        _render_feed_html(selected_collection_id),
    )


def _toggle_video_autoplay(selected_collection_id_raw: str):
    global _video_autoplay_enabled

    _video_autoplay_enabled = not _video_autoplay_enabled

    selected_collection_id: Optional[int] = None
    try:
        if selected_collection_id_raw and str(selected_collection_id_raw).strip():
            selected_collection_id = int(str(selected_collection_id_raw).strip())
    except Exception:
        selected_collection_id = None

    return (
        gr.update(variant="primary" if _video_autoplay_enabled else "secondary"),
        _render_controls_bar(),
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


def _get_image_detail_for_sync(
    client: CivitaiClient,
    image_id: Optional[int],
) -> Dict[str, Any]:
    if not image_id:
        return {}

    return client.get_image_by_id(int(image_id))


def _safe_get(d: Dict[str, Any], *keys, default=None):
    current = d
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
        if current is None:
            return default
    return current


def _to_db_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _to_db_int(value: Any) -> Optional[int]:
    try:
        if value is None or value == "":
            return None
        if isinstance(value, (dict, list, tuple)):
            return None
        return int(float(value))
    except Exception:
        return None


def _to_db_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        if isinstance(value, (dict, list, tuple)):
            return None
        return float(value)
    except Exception:
        return None


def _extract_generation_core(detail: Dict[str, Any]) -> Dict[str, Any]:
    meta = detail.get("meta") or {}
    metadata = detail.get("metadata") or {}
    generation = detail.get("generation") or {}
    gen_meta = generation.get("meta") or {}

    # Prefer generation data first
    steps = gen_meta.get("steps") or meta.get("steps") or metadata.get("steps")
    cfg_scale = gen_meta.get("cfgScale") or meta.get("cfgScale") or metadata.get("cfgScale")
    sampler = gen_meta.get("sampler") or meta.get("sampler") or metadata.get("sampler")
    seed = gen_meta.get("seed") or meta.get("seed") or metadata.get("seed")
    clip_skip = gen_meta.get("clipSkip") or meta.get("clipSkip") or metadata.get("clipSkip")

    width = gen_meta.get("width") or metadata.get("width")
    height = gen_meta.get("height") or metadata.get("height")

    return {
        "steps": _to_db_int(steps),
        "cfg_scale": _to_db_float(cfg_scale),
        "sampler": _to_db_text(sampler),
        "seed": _to_db_text(seed),
        "clip_skip": _to_db_int(clip_skip),
        "width": _to_db_int(width),
        "height": _to_db_int(height),
    }

def _store_generation_params(db: CollectionDatabase, item_id: int, detail: Dict[str, Any]):
    generation = detail.get("generation") or {}
    meta = generation.get("meta") or detail.get("meta") or {}

    db.clear_generation_params_for_item(item_id)

    for key, value in meta.items():
        if value is None:
            continue

        db.add_generation_param(
            item_id=item_id,
            param_key=str(key),
            param_value=_to_db_text(value),
            value_type=type(value).__name__,
            display_group="core",
            source_path=f"generation.meta.{key}" if generation.get("meta") else f"meta.{key}",
        )


def _to_float(value: Any, default: float = 1.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def _normalize_resource_type(value: Any) -> str:
    raw = str(value or "").strip().lower().replace(" ", "").replace("_", "")

    mapping = {
        "checkpoint": "checkpoint",
        "model": "checkpoint",
        "lora": "lora",
        "loramodel": "lora",
        "textualinversion": "embedding",
        "embedding": "embedding",
        "upscaler": "upscaler",
        "workflow": "workflow",
        "workflows": "workflow",
    }

    return mapping.get(raw, raw or "resource")


def _extract_generation_identity(detail: Dict[str, Any]) -> Dict[str, Any]:
    generation = detail.get("generation") or {}
    gen_meta = generation.get("meta") or {}
    resources = generation.get("resources") or []

    generator_name = str(gen_meta.get("Version") or "").strip()
    generator_type = ""
    has_generation_data = 1 if gen_meta else 0
    is_external_generator = 1 if generation.get("external") else 0

    checkpoint_name = str(gen_meta.get("Model") or "").strip()
    checkpoint_version = ""
    vae_name = str(gen_meta.get("VAE") or "").strip()

    for resource in resources:
        model_type = _normalize_resource_type(resource.get("modelType"))
        if model_type == "checkpoint":
            if not checkpoint_name:
                checkpoint_name = str(resource.get("modelName") or "").strip()
            break

    if generator_name:
        generator_type = "internal"

    if is_external_generator:
        generator_type = "external"

    return {
        "generator_name": generator_name,
        "generator_type": generator_type,
        "has_generation_data": has_generation_data,
        "is_external_generator": is_external_generator,
        "checkpoint_name": checkpoint_name,
        "checkpoint_version": checkpoint_version,
        "vae_name": vae_name,
    }


def _extract_resource_hashes(resource: Dict[str, Any]) -> Dict[str, str]:
    hashes = resource.get("hashes") or {}
    files = resource.get("files") or []

    sha256 = ""
    autov2 = ""

    if isinstance(hashes, dict):
        sha256 = str(hashes.get("SHA256") or hashes.get("sha256") or "").strip()
        autov2 = str(hashes.get("AutoV2") or hashes.get("autov2") or "").strip()

    if not sha256 or not autov2:
        for file_info in files if isinstance(files, list) else []:
            file_hashes = file_info.get("hashes") or {}
            if not isinstance(file_hashes, dict):
                continue

            if not sha256:
                sha256 = str(file_hashes.get("SHA256") or file_hashes.get("sha256") or "").strip()

            if not autov2:
                autov2 = str(file_hashes.get("AutoV2") or file_hashes.get("autov2") or "").strip()

    return {
        "sha256": sha256.lower(),
        "autov2": autov2.lower(),
    }


def _get_model_version_hashes(
    client: Optional[CivitaiClient],
    version_id: Optional[Any],
) -> Dict[str, str]:
    global _model_version_hash_cache

    if client is None or version_id is None:
        return {"sha256": "", "autov2": ""}

    version_id_text = str(version_id).strip()
    if not version_id_text:
        return {"sha256": "", "autov2": ""}

    if version_id_text in _model_version_hash_cache:
        return _model_version_hash_cache[version_id_text]

    version_payload = client.get_model_version(int(version_id_text))
    version_hashes = _extract_resource_hashes(version_payload)

    _model_version_hash_cache[version_id_text] = version_hashes

    # Conservative request behavior.
    time.sleep(DETAIL_REQUEST_DELAY_SECONDS)

    return version_hashes


def _fill_missing_resource_hashes(
    client: Optional[CivitaiClient],
    resource_hashes: Dict[str, str],
    version_id: Optional[Any],
) -> Dict[str, str]:
    sha256 = resource_hashes.get("sha256", "")
    autov2 = resource_hashes.get("autov2", "")

    if sha256 and autov2:
        return resource_hashes

    version_hashes = _get_model_version_hashes(client, version_id)

    return {
        "sha256": sha256 or version_hashes.get("sha256", ""),
        "autov2": autov2 or version_hashes.get("autov2", ""),
    }


def _store_resources(
    db: CollectionDatabase,
    item_id: int,
    detail: Dict[str, Any],
    client: Optional[CivitaiClient] = None,
) -> None:
    generation = detail.get("generation") or {}
    generation_resources = generation.get("resources") or []
    gen_meta = generation.get("meta") or {}
    root_meta = detail.get("meta") or {}

    meta_resources = (
        gen_meta.get("resources")
        or root_meta.get("resources")
        or detail.get("resources")
        or []
    )

    db.clear_resources_for_item(item_id)

    for resource in generation_resources:
        resource_type = _normalize_resource_type(resource.get("modelType"))
        name = str(resource.get("modelName") or "").strip()
        version_name = str(resource.get("versionName") or "").strip()
        version_id = resource.get("versionId")
        resource_hashes = _extract_resource_hashes(resource)
        resource_hashes = _fill_missing_resource_hashes(client, resource_hashes, version_id)

        if not name:
            continue

        db.add_resource(
            item_id=item_id,
            resource_type=resource_type,
            name=name,
            version_name=version_name,
            weight=1.0,
            model_id=resource.get("modelId"),
            version_id=version_id,
            hash_sha256=resource_hashes["sha256"],
            hash_autov2=resource_hashes["autov2"],
            local_status="red",
            local_filename=None,
        )

    if generation_resources:
        return

    for resource in meta_resources:
        resource_type = _normalize_resource_type(resource.get("type"))
        name = str(resource.get("name") or "").strip()
        version_name = ""
        weight = _to_float(resource.get("weight"), default=1.0)
        version_id = resource.get("versionId")
        resource_hashes = _extract_resource_hashes(resource)
        resource_hashes = _fill_missing_resource_hashes(client, resource_hashes, version_id)

        if not name:
            continue

        db.add_resource(
            item_id=item_id,
            resource_type=resource_type,
            name=name,
            version_name=version_name,
            weight=weight,
            model_id=resource.get("modelId"),
            version_id=version_id,
            hash_sha256=resource_hashes["sha256"],
            hash_autov2=resource_hashes["autov2"],
            local_status="red",
            local_filename=None,
        )


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
        total_images_found = 0
        reused_from_db = 0
        newly_hydrated = 0
        incomplete_rehydrated = 0
        already_linked_before_sync = 0
        relinked_existing_items = 0
        new_items_created = 0
        orphan_items_removed_before = 0
        orphan_items_removed_after = 0
        partial_sync_detected = False
        partial_sync_reason = ""

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
            image_count = len(images)
            total_images_found += image_count
            print(f"  IMAGE COUNT: {image_count}")

            pre_sync_item_ids = db.get_collection_item_ids(local_collection_id)

            # Only clear AFTER successful fetch
            db.clear_collection_items(local_collection_id)

            consecutive_detail_failures = 0
            collection_hydration_stopped_early = False

            for idx, image in enumerate(images):
                if _stop_requested:
                    print("STOP REQUESTED — stopping sync during image hydration")
                    collection_hydration_stopped_early = True
                    partial_sync_detected = True
                    partial_sync_reason = "stopped by user during image detail hydration"
                    break

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

                existing_item = None

                if image_id:
                    existing_item = db.get_item_by_civitai_image_id(int(image_id))

                    if not existing_item:
                        print(f"UNMATCHED IMAGE ID: {image_id}")
                else:
                    print("IMAGE WITH NO ID:", image)

                if existing_item:
                    existing_item_id = int(existing_item["id"])
                    if existing_item_id in pre_sync_item_ids:
                        already_linked_before_sync += 1
                    else:
                        relinked_existing_items += 1

                if existing_item and db.item_is_hydrated(int(existing_item["id"])):
                    reused_from_db += 1
                    db.add_item_to_collection(
                        local_collection_id,
                        int(existing_item["id"]),
                        idx,
                    )
                    continue

                if existing_item:
                    incomplete_rehydrated += 1
                else:
                    newly_hydrated += 1
                    new_items_created += 1

                detail_payload: Dict[str, Any] = {}
                generation_payload: Dict[str, Any] = {}

                if image_id:
                    try:
                        detail_payload = _get_image_detail_for_sync(client, image_id)

                        try:
                            generation_payload = client.get_image_generation_data(int(image_id))
                            time.sleep(DETAIL_REQUEST_DELAY_SECONDS)
                        except Exception as exc:
                            print(f"  GENERATION DATA FETCH FAILED for image {image_id}: {exc!r}")

                        consecutive_detail_failures = 0
                        time.sleep(DETAIL_REQUEST_DELAY_SECONDS)

                    except Exception as exc:
                        print(f"  DETAIL FETCH FAILED for image {image_id}: {exc!r}")
                        consecutive_detail_failures += 1
                        detail_payload = {}
                        generation_payload = {}

                        if consecutive_detail_failures >= DETAIL_REQUEST_MAX_CONSECUTIVE_FAILURES:
                            print(
                                "  STOPPING DETAIL HYDRATION EARLY: "
                                f"{consecutive_detail_failures} consecutive failures"
                            )
                            collection_hydration_stopped_early = True
                            partial_sync_detected = True
                            partial_sync_reason = (
                                f"detail hydration stopped early after "
                                f"{consecutive_detail_failures} consecutive failures"
                            )
                            break

                        time.sleep(DETAIL_REQUEST_FAILURE_BACKOFF_SECONDS)

                metadata_source = detail_payload or image

                if generation_payload:
                    metadata_source = {
                        **metadata_source,
                        "generation": generation_payload,
                    }

                detail_meta = metadata_source.get("meta") or meta
                generation_meta = (metadata_source.get("generation") or {}).get("meta") or {}
                detail_user = metadata_source.get("user") or user
                raw_metadata_payload = metadata_source

                core = _extract_generation_core(metadata_source)
                identity = _extract_generation_identity(metadata_source)

                if existing_item:
                    item_id = int(existing_item["id"])
                else:
                    item_id = db.create_item(
                        civitai_image_id=image_id,
                        civitai_post_id=image.get("postId"),
                        title=title,
                        image_url=preview_url,
                        full_media_url=full_media_url,
                        preview_path="",
                        full_path="",
                        download_status="none",
                        creator_name=detail_user.get("username") or user.get("username") or "Unknown",
                        creator_url="",
                        post_url=_civitai_page_url(f"/images/{image_id}") if image_id else "",
                        rating=str(image.get("nsfwLevel", "Unknown")),
                        platform="Civitai",
                        media_type=media_type,
                        prompt=generation_meta.get("prompt") or detail_meta.get("prompt") or meta.get("prompt") or "",
                        negative_prompt=generation_meta.get("negativePrompt") or detail_meta.get("negativePrompt") or meta.get("negativePrompt") or "",
                        metadata_json=json.dumps(raw_metadata_payload),

                        generator_name=identity["generator_name"],
                        generator_type=identity["generator_type"],
                        has_generation_data=identity["has_generation_data"],
                        is_external_generator=identity["is_external_generator"],

                        steps=core["steps"],
                        cfg_scale=core["cfg_scale"],
                        sampler=core["sampler"],
                        seed=core["seed"],
                        clip_skip=core["clip_skip"],
                        width=core["width"],
                        height=core["height"],

                        checkpoint_name=identity["checkpoint_name"],
                        checkpoint_version=identity["checkpoint_version"],
                        vae_name=identity["vae_name"],
                    )

                # Link immediately so a partial resource/generation issue cannot leave
                # a newly created item orphaned.
                db.add_item_to_collection(local_collection_id, item_id, idx)

                _store_generation_params(db, item_id, metadata_source)
                _store_resources(db, item_id, metadata_source, client)

            synced_count += 1

        # Do not auto-clean orphans during sync.
        # Some images can be created/relinked during this pass; cleanup should be manual
        # after we confirm stable collection links.
        orphan_items_removed_after = 0

        sync_report = (
            f"Collections synced: {synced_count}\n\n"
            f"Images found: {total_images_found}\n\n"
            f"Already linked before sync: {already_linked_before_sync}\n\n"
            f"Re-linked existing items: {relinked_existing_items}\n\n"
            f"Reused from DB: {reused_from_db}\n\n"
            f"New items created: {new_items_created}\n\n"
            f"Newly hydrated: {newly_hydrated}\n\n"
            f"Incomplete rehydrated: {incomplete_rehydrated}\n\n"
            f"Detail calls skipped: {reused_from_db}\n\n"
            f"Orphan items removed before sync: {orphan_items_removed_before}\n\n"
            f"Orphan items removed after clean sync: {orphan_items_removed_after}"
        )

        if _stop_requested:
            return (
                _refresh_sidebar_payload(),
                f"Sync stopped by user.\n\n{sync_report}",
            )

        if partial_sync_detected:
            return (
                _refresh_sidebar_payload(),
                f"Sync partially completed.\n\nReason: {partial_sync_reason}\n\n{sync_report}",
            )

        return (
            _refresh_sidebar_payload(),
            f"Sync succeeded.\n\n{sync_report}",
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


            #collection_toolbar_grid_button,
            #collection_toolbar_scroll_button,
            #collection_toolbar_detail_button,
            #collection_toolbar_video_button {{
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

            #collection_toolbar_grid_button::before {{
                content: "" !important;
                position: absolute !important;
                inset: 0 !important;
                background-repeat: no-repeat !important;
                background-position: center !important;
                background-size: 20px 20px !important;
                background-image: url("{MAIN_VIEW_ICON_DATA_URI}") !important;
            }}

            #collection_toolbar_scroll_button::before {{
                content: "" !important;
                position: absolute !important;
                inset: 0 !important;
                background-repeat: no-repeat !important;
                background-position: center !important;
                background-size: 20px 20px !important;
                background-image: url("{SCROLL_VIEW_ICON_DATA_URI}") !important;
            }}

            #collection_toolbar_detail_button::before {{
                content: "" !important;
                position: absolute !important;
                inset: 0 !important;
                background-repeat: no-repeat !important;
                background-position: center !important;
                background-size: 20px 20px !important;
                background-image: url("{DETAIL_VIEW_ICON_DATA_URI}") !important;
            }}

            #collection_toolbar_video_button::before {{
                content: "" !important;
                position: absolute !important;
                inset: 0 !important;
                background-repeat: no-repeat !important;
                background-position: center !important;
                background-size: 20px 20px !important;
                background-image: url("{PLAY_PAUSE_ICON_DATA_URI}") !important;
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

            #collection_toolbar_grid_button.primary,
            #collection_toolbar_scroll_button.primary,
            #collection_toolbar_detail_button.primary,
            #collection_toolbar_video_button.primary {{
                background-color: #3a4352 !important;
            }}

            #collection_toolbar_grid_button.secondary,
            #collection_toolbar_scroll_button.secondary,
            #collection_toolbar_detail_button.secondary,
            #collection_toolbar_video_button.secondary {{
                background-color: #ff7a1a !important;
            }}

            #collection_toolbar_grid_button.primary:hover,
            #collection_toolbar_scroll_button.primary:hover,
            #collection_toolbar_detail_button.primary:hover,
            #collection_toolbar_video_button.primary:hover {{
                background-color: #4a5160 !important;
            }}

            #collection_toolbar_grid_button.secondary:hover,
            #collection_toolbar_scroll_button.secondary:hover,
            #collection_toolbar_detail_button.secondary:hover,
            #collection_toolbar_video_button.secondary:hover {{
                background-color: #ff8c33 !important;
            }}

            #collection_toolbar_row {{
                background: #26334f !important;
                border-radius: 10px !important;
                padding: 6px 8px !important;
                margin-bottom: 6px !important;
            }}

            #collection_cache_filter_sync_button,
            #collection_cache_filter_preview_button,
            #collection_cache_filter_full_button {{
                min-width: 46px !important;
                width: 46px !important;
                height: 24px !important;
                min-height: 24px !important;
                padding: 0 6px !important;
                margin-top: 6px !important;
                font-size: 11px !important;
                font-weight: 700 !important;
                border-radius: 7px !important;
            }}

            #collection_preview_size_slider input[type="number"],
            #collection_preview_size_slider .input-wrap {{
                display: none !important;
            }}

            #collection_preview_size_ticks {{
                display: flex !important;
                justify-content: space-between !important;
                padding: 0 5px !important;
                margin-top: -6px !important;
                pointer-events: none !important;
            }}

            #collection_preview_size_ticks span {{
                display: block !important;
                width: 2px !important;
                height: 7px !important;
                border-radius: 999px !important;
                background: #9aa8c7 !important;
                opacity: 0.85 !important;
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

            #collection_list_container {{
                flex: 0 0 auto !important;
                height: 400px !important;
                overflow: hidden !important;
            }}

            #collection_sidebar_html {{
                flex: 0 0 auto !important;
                height: 400px !important;
                overflow: hidden !important;
            }}
            #collection_sidebar_actions {{
                flex: 0 0 auto !important;
                padding: 6px 10px 8px 10px !important;
                border-top: 1px solid #252b35 !important;
            }}
            #collection_sidebar_actions button {{
                min-height: 30px !important;
                height: 30px !important;
                padding: 3px 8px !important;
                border-radius: 8px !important;
                font-size: 14px !important;
                font-weight: 700 !important;
                margin: 0 !important;
            }}

            #collection_sync_button {{
                font-size: 18px !important;
            }}

            #collection_refresh_button {{
                font-size: 18px !important;
            }}

            #collection_sync_row,
            #collection_download_row {{
                display: flex !important;
                flex-wrap: nowrap !important;
                gap: 6px !important;
                width: 100% !important;
            }}

            #collection_sync_row > div:first-child {{
                flex: 1 1 auto !important;
                min-width: 0 !important;
            }}

            #collection_sync_row > div:last-child {{
                flex: 0 0 34px !important;
                min-width: 34px !important;
                max-width: 34px !important;
            }}

            #collection_download_row > div {{
                flex: 1 1 0 !important;
                min-width: 0 !important;
            }}

            #collection_sync_button,
            #collection_preview_download_button,
            #collection_full_download_button,
            #collection_refresh_button {{
                width: 100% !important;
            }}

            #collection_stop_button {{
                min-width: 30px !important;
                width: 30px !important;
                padding: 0 !important;
                font-size: 0 !important;
                color: transparent !important;
                position: relative !important;
            }}

            #collection_stop_button::before {{
                content: "" !important;
                position: absolute !important;
                left: 50% !important;
                top: 50% !important;
                width: 10px !important;
                height: 10px !important;
                transform: translate(-50%, -50%) !important;
                border-radius: 2px !important;
                background: #ff3b3b !important;
            }}

            #collection_status_markdown {{
                font-size: 11px !important;
                color: #9aa0aa !important;
                margin-top: 2px !important;
                margin-bottom: 0px !important;
                padding: 0 !important;
            }}

            #collection_maintenance_tools {{
                margin-top: -8px !important;
            }}

            #collection_maintenance_tools > div {{
                padding-top: 0 !important;
                margin-top: 0 !important;
            }}

            #collection_maintenance_tools button {{
                min-height: 28px !important;
                height: 28px !important;
                padding: 3px 8px !important;
                border-radius: 8px !important;
                font-size: 14px !important;
                font-weight: 700 !important;
                margin: 0 !important;
            }}

            #collection_local_collections_panel {{
                flex: 0 0 auto !important;
                padding: 8px 10px 10px 10px !important;
                border-top: 1px solid #252b35 !important;
            }}

            #collection_local_collections_panel button {{
                min-height: 28px !important;
                height: 28px !important;
                padding: 2px 8px !important;
                border-radius: 8px !important;
                font-size: 13px !important;
                font-weight: 700 !important;
                margin: 0 !important;
            }}

            #collection_local_name_input textarea,
            #collection_local_name_input input {{
                min-height: 30px !important;
                height: 30px !important;
                font-size: 12px !important;
                padding: 4px 8px !important;
            }}
            </style>
            """
        )

        selected_collection_id = gr.Textbox(
            value="",
            visible=False,
            elem_id="collection_selected_collection_id",
        )

        selected_item_id = gr.Textbox(
            value="",
            visible=False,
            elem_id="collection_selected_item_id",
        )

        detail_action = gr.Textbox(
            value="",
            visible=False,
            elem_id="collection_detail_action",
        )

        batch_offset = gr.Textbox(
            value="0",
            visible=False,
            elem_id="collection_batch_offset",
        )

        video_batch_offset = gr.Textbox(
            value="0",
            visible=False,
            elem_id="collection_video_batch_offset",
        )

        local_delete_confirm_id = gr.Textbox(
            value="",
            visible=False,
            elem_id="collection_local_delete_confirm_id",
        )

        target_local_collection_id = gr.Textbox(
            value="",
            visible=False,
            elem_id="collection_target_local_collection_id",
        )

        local_add_item_id = gr.Textbox(
            value="",
            visible=False,
            elem_id="collection_local_add_item_id",
        )

        reorder_payload = gr.Textbox(
            value="",
            visible=False,
            elem_id="collection_reorder_payload",
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

        video_batch_button = gr.Button(
            "Video Batch Internal",
            visible=False,
            elem_id="collection_video_batch_button",
        )

        local_add_item_button = gr.Button(
            "Local Add Internal",
            visible=False,
            elem_id="collection_local_add_item_button",
        )

        target_local_collection_button = gr.Button(
            "Target Local Collection Internal",
            visible=False,
            elem_id="collection_target_local_collection_button",
        )

        reorder_button = gr.Button(
            "Reorder Internal",
            visible=False,
            elem_id="collection_reorder_button",
        )

        with gr.Row(equal_height=True):
            with gr.Column(scale=1, min_width=280, visible=True) as sidebar_column:
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

                with gr.Column(elem_id="collection_list_container"):
                    sidebar_html = gr.HTML(
                        value=_refresh_sidebar_payload(),
                        elem_id="collection_sidebar_html",
                    )

                with gr.Column(elem_id="collection_sidebar_actions"):
                    gr.HTML(
                        """
                        <div style="
                            color:#9aa0aa;
                            font-size:11px;
                            font-weight:700;
                            letter-spacing:0.08em;
                            text-transform:uppercase;
                            margin-bottom:6px;
                        ">
                            Actions
                        </div>
                        """
                    )

                    with gr.Row(equal_height=True, elem_id="collection_sync_row"):
                        with gr.Column(scale=1, min_width=0):
                            sync_button = gr.Button(
                                "Sync",
                                variant="primary",
                                elem_id="collection_sync_button",
                            )

                        with gr.Column(scale=0, min_width=34):
                            stop_button = gr.Button(
                                "Stop",
                                variant="stop",
                                elem_id="collection_stop_button",
                            )

                    with gr.Row(equal_height=True, elem_id="collection_download_row"):
                        with gr.Column(scale=1, min_width=0):
                            preview_download_button = gr.Button(
                                "Prvw Download",
                                elem_id="collection_preview_download_button",
                            )

                        with gr.Column(scale=1, min_width=0):
                            full_download_button = gr.Button(
                                "Full Download",
                                elem_id="collection_full_download_button",
                            )

                    refresh_button = gr.Button(
                        "Refresh",
                        elem_id="collection_refresh_button",
                    )

                    status_markdown = gr.Markdown(
                        "",
                        elem_id="collection_status_markdown",
                    )

                    with gr.Accordion(
                        "Maintenance",
                        open=False,
                        elem_id="collection_maintenance_tools",
                    ):
                        scan_local_files_button = gr.Button("Scan Local Files")
                        clear_cache_button = gr.Button("Clear Cache")
                        reset_button = gr.Button("Reset", variant="stop")

                with gr.Column(elem_id="collection_local_collections_panel"):
                    gr.HTML(
                        """
                        <div style="
                            color:#9aa0aa;
                            font-size:11px;
                            font-weight:700;
                            letter-spacing:0.08em;
                            text-transform:uppercase;
                            margin-bottom:6px;
                        ">
                            Local Collections
                        </div>
                        """
                    )

                    local_collection_name = gr.Textbox(
                        value="",
                        placeholder="New Collection",
                        show_label=False,
                        elem_id="collection_local_name_input",
                    )

                    with gr.Row(equal_height=True):
                        local_create_button = gr.Button(
                            "+",
                            elem_id="collection_local_create_button",
                            tooltip="Create a new local collection",
                        )
                        local_delete_button = gr.Button(
                            "−",
                            variant="stop",
                            elem_id="collection_local_delete_button",
                            tooltip="Delete selected local collection",
                        )
                        local_rename_button = gr.Button(
                            "Rename",
                            elem_id="collection_local_rename_button",
                            tooltip="Rename selected local collection",
                        )

                    local_collections_html = gr.HTML(
                        value=_refresh_local_collections_payload(),
                        elem_id="collection_local_collections_html",
                    )

                    gr.HTML(
                        """
                        <div style="
                            color:#6f7682;
                            font-size:11px;
                            line-height:1.35;
                            margin-top:6px;
                        ">
                            Local collections are reference-only. They do not copy or delete physical files.
                        </div>
                        """
                    )

            with gr.Column(scale=3):
                with gr.Row(equal_height=False, elem_id="collection_toolbar_row"):
                    with gr.Column(scale=0, min_width=36):
                        gr.HTML(
                            value=f"""
                            <div style="
                                width:28px;
                                height:24px;
                                margin-top:6px;
                                margin-left:4px;
                                background-image:url('{THUMB_SCALE_ICON_DATA_URI}');
                                background-repeat:no-repeat;
                                background-position:center;
                                background-size:28px 28px;
                            "></div>
                            """
                        )

                    with gr.Column(scale=1, min_width=90):
                        preview_size_slider = gr.Slider(
                            minimum=0,
                            maximum=len(PREVIEW_SIZE_STEPS) - 1,
                            step=1,
                            value=PREVIEW_SIZE_STEPS.index(_preview_size),
                            show_label=False,
                            elem_id="collection_preview_size_slider",
                        )
                        gr.HTML(
                            """
                            <div id="collection_preview_size_ticks">
                                <span></span>
                                <span></span>
                                <span></span>
                                <span></span>
                                <span></span>
                                <span></span>
                                <span></span>
                            </div>
                            """
                        )

                    with gr.Column(scale=0, min_width=50):
                        cache_filter_sync_button = gr.Button(
                            value="Sync",
                            elem_id="collection_cache_filter_sync_button",
                            tooltip="Show all synced items",
                            variant="primary",
                        )

                    with gr.Column(scale=0, min_width=50):
                        cache_filter_preview_button = gr.Button(
                            value="Prvw",
                            elem_id="collection_cache_filter_preview_button",
                            tooltip="Show items with preview files",
                            variant="secondary",
                        )

                    with gr.Column(scale=0, min_width=50):
                        cache_filter_full_button = gr.Button(
                            value="Full",
                            elem_id="collection_cache_filter_full_button",
                            tooltip="Show items with full downloads",
                            variant="secondary",
                        )

                    with gr.Column(scale=0, min_width=36):
                        grid_button = gr.Button(
                            value="",
                            elem_id="collection_toolbar_grid_button",
                            tooltip="Grid view",
                            variant="primary",
                        )

                    with gr.Column(scale=0, min_width=36):
                        scroll_button = gr.Button(
                            value="",
                            elem_id="collection_toolbar_scroll_button",
                            tooltip="Scrolling view",
                            variant="secondary",
                        )

                    with gr.Column(scale=0, min_width=36):
                        detail_button = gr.Button(
                            value="",
                            elem_id="collection_toolbar_detail_button",
                            tooltip="Detailed view",
                            variant="secondary",
                        )

                    with gr.Column(scale=0, min_width=36):
                        video_button = gr.Button(
                            value="",
                            elem_id="collection_toolbar_video_button",
                            tooltip="Play / Pause videos",
                            variant="primary",
                        )

                    with gr.Column(scale=0, min_width=36):
                        nsfw_toggle_button = gr.Button(
                            value="",
                            elem_id="collection_toolbar_nsfw_button",
                            tooltip="18+",
                            variant="primary",
                        )

                controls_html = gr.HTML(
                    value=_render_controls_bar(),
                    elem_id="collection_controls_html",
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

        local_create_button.click(
            fn=_create_local_collection,
            inputs=[local_collection_name],
            outputs=[
                selected_collection_id,
                target_local_collection_id,
                sidebar_html,
                feed_html,
                local_collection_name,
                local_collections_html,
                status_markdown,
            ],
        )

        local_rename_button.click(
            fn=_rename_selected_local_collection,
            inputs=[selected_collection_id, local_collection_name],
            outputs=[
                sidebar_html,
                local_collection_name,
                local_collections_html,
                status_markdown,
            ],
        )

        local_delete_button.click(
            fn=_delete_selected_local_collection,
            inputs=[selected_collection_id, local_delete_confirm_id],
            outputs=[
                selected_collection_id,
                local_delete_confirm_id,
                sidebar_html,
                feed_html,
                local_collection_name,
                local_collections_html,
                status_markdown,
            ],
        )

        local_add_item_button.click(
            fn=_add_item_to_target_local_collection,
            inputs=[
                local_add_item_id,
                target_local_collection_id,
                selected_collection_id,
            ],
            outputs=[
                sidebar_html,
                local_collections_html,
                feed_html,
                status_markdown,
            ],
        )

        target_local_collection_button.click(
            fn=_set_target_local_collection,
            inputs=[
                target_local_collection_id,
                selected_collection_id,
            ],
            outputs=[
                selected_collection_id,
                sidebar_html,
                feed_html,
                local_collection_name,
                local_collections_html,
                status_markdown,
            ],
        )

        reorder_button.click(
            fn=_reorder_local_collection_items,
            inputs=[
                reorder_payload,
                selected_collection_id,
            ],
            outputs=[
                sidebar_html,
                feed_html,
                status_markdown,
            ],
        )

        nsfw_toggle_button.click(
            fn=_toggle_nsfw_filter,
            inputs=[selected_collection_id],
            outputs=[nsfw_toggle_button, controls_html, sidebar_html, feed_html],
       )

        grid_button.click(
            fn=lambda selected_collection_id_raw, selected_item_id_raw: _set_collection_view(
                "grid",
                selected_collection_id_raw,
                selected_item_id_raw,
            ),
            inputs=[selected_collection_id, selected_item_id],
            outputs=[
                grid_button,
                scroll_button,
                detail_button,
                sidebar_column,
                selected_item_id,
                controls_html,
                sidebar_html,
                feed_html,
            ],
        )

        scroll_button.click(
            fn=lambda selected_collection_id_raw, selected_item_id_raw: _set_collection_view(
                "scroll",
                selected_collection_id_raw,
                selected_item_id_raw,
            ),
            inputs=[selected_collection_id, selected_item_id],
            outputs=[
                grid_button,
                scroll_button,
                detail_button,
                sidebar_column,
                selected_item_id,
                controls_html,
                sidebar_html,
                feed_html,
            ],
        )

        detail_button.click(
            fn=lambda selected_collection_id_raw, selected_item_id_raw: _set_collection_view(
                "detail",
                selected_collection_id_raw,
                selected_item_id_raw,
            ),
            inputs=[selected_collection_id, selected_item_id],
            outputs=[
                grid_button,
                scroll_button,
                detail_button,
                sidebar_column,
                selected_item_id,
                controls_html,
                sidebar_html,
                feed_html,
            ],
        )

        preview_size_slider.change(
            fn=_set_preview_size,
            inputs=[preview_size_slider, selected_collection_id],
            outputs=[controls_html, feed_html],
        )

        video_button.click(
            fn=_toggle_video_autoplay,
            inputs=[selected_collection_id],
            outputs=[video_button, controls_html, feed_html],
        )

        cache_filter_sync_button.click(
            fn=lambda selected_collection_id_raw: _set_cache_filter(
                "sync",
                selected_collection_id_raw,
            ),
            inputs=[selected_collection_id],
            outputs=[
                cache_filter_sync_button,
                cache_filter_preview_button,
                cache_filter_full_button,
                controls_html,
                feed_html,
            ],
        )

        cache_filter_preview_button.click(
            fn=lambda selected_collection_id_raw: _set_cache_filter(
                "preview",
                selected_collection_id_raw,
            ),
            inputs=[selected_collection_id],
            outputs=[
                cache_filter_sync_button,
                cache_filter_preview_button,
                cache_filter_full_button,
                controls_html,
                feed_html,
            ],
        )

        cache_filter_full_button.click(
            fn=lambda selected_collection_id_raw: _set_cache_filter(
                "full",
                selected_collection_id_raw,
            ),
            inputs=[selected_collection_id],
            outputs=[
                cache_filter_sync_button,
                cache_filter_preview_button,
                cache_filter_full_button,
                controls_html,
                feed_html,
            ],
        )

        clear_cache_button.click(
            fn=_clear_cache,
            inputs=[],
            outputs=[status_markdown],
        )

        scan_local_files_button.click(
            fn=_scan_local_files_for_availability,
            inputs=[selected_collection_id],
            outputs=[controls_html, sidebar_html, feed_html, status_markdown],
        )

        reset_button.click(
            fn=_reset_extension,
            inputs=[],
            outputs=[sidebar_html, status_markdown],
        )

        selected_collection_id.change(
            fn=_load_collection_feed,
            inputs=[selected_collection_id],
            outputs=[
                sidebar_html,
                feed_html,
                local_collection_name,
                local_collections_html,
            ],
        ).then(
            fn=lambda: gr.update(value=""),
            inputs=[],
            outputs=[local_delete_confirm_id],
        )

        selected_item_id.change(
            fn=_open_selected_item_detail,
            inputs=[selected_collection_id, selected_item_id],
            outputs=[
                grid_button,
                scroll_button,
                detail_button,
                sidebar_column,
                controls_html,
                sidebar_html,
                feed_html,
            ],
        )

        detail_action.change(
            fn=_return_detail_to_scroll,
            inputs=[selected_collection_id, selected_item_id, detail_action],
            outputs=[
                grid_button,
                scroll_button,
                detail_button,
                sidebar_column,
                selected_item_id,
                detail_action,
                controls_html,
                sidebar_html,
                feed_html,
            ],
        )

        load_more_button.click(
            fn=_load_more_feed_batch,
            inputs=[selected_collection_id, batch_offset],
            outputs=[batch_payload_html],
        )

        video_batch_button.click(
            fn=_set_video_batch,
            inputs=[selected_collection_id, video_batch_offset],
            outputs=[feed_html],
        )

    return [(collection_tab, "Collection", "collection_tab")]


script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_ui_tabs(on_ui_tabs)