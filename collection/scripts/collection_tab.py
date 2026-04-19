import html
import json
from pathlib import Path
from typing import Optional, List, Dict, Any

import gradio as gr

from modules import script_callbacks, shared

from collection_lib.civitai_api import CivitaiClient
from collection_lib.database import CollectionDatabase


EXTENSION_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = EXTENSION_ROOT / "data"
DEFAULT_DB_PATH = DATA_DIR / "collections.db"
DEFAULT_IMAGE_CACHE_DIR = DATA_DIR / "images"

_db: Optional[CollectionDatabase] = None
_hide_nsfw: bool = False


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

    image_cache_dir = (
        (getattr(shared.opts, "collection_cache_dir", "") or "").strip()
        or str(DEFAULT_IMAGE_CACHE_DIR)
    )

    return {
        "api_key": api_key,
        "source_mode": source_mode,
        "image_cache_dir": image_cache_dir,
    }


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


def _refresh_sidebar_payload(selected_collection_id: Optional[int] = None) -> str:
    collections = _all_collections()

    rows: List[str] = []
    for collection in collections:
        cid = int(collection["id"])
        name = html.escape(collection["name"])
        ctype = html.escape(collection["type"].title())
        item_count = int(collection.get("item_count", 0))
        active = cid == selected_collection_id

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
                    const el = root.querySelector('#collection_selected_collection_id textarea, #collection_selected_collection_id input');
                    if (!el) return;
                    el.value = '{cid}';
                    el.dispatchEvent(new Event('input', {{ bubbles: true }}));
                    el.dispatchEvent(new Event('change', {{ bubbles: true }}));
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
                        <div style="font-size:11px; color:#8f8f8f; margin-top:2px;">{ctype}</div>
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
    nsfw_color = "#f0f0f0" if _hide_nsfw else "#8a8a8a"

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

    db = _get_db()
    items = db.list_items_for_collection(collection_id)

    global _hide_nsfw
    if _hide_nsfw:
        filtered_items = []
        for item in items:
            try:
                rating_value = int(item.get("rating", 0))
            except Exception:
                rating_value = 0

            if rating_value < 4:
                filtered_items.append(item)

        items = filtered_items

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

    cards: List[str] = []
    for item in items:
        image_url = html.escape(item.get("image_url") or "")
        title = item.get("title") or "Untitled"
        safe_title = html.escape(title)

        is_mp4 = title.lower().endswith(".mp4")

        thumb_html = ""
        if image_url and not is_mp4:
            thumb_html = f"""
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
            """
        elif is_mp4:
            thumb_html = f"""
            <div style="
                width:100%;
                aspect-ratio:1 / 1.25;
                overflow:hidden;
                border-radius:12px;
                background:#202020;
            ">
                <video
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
                    "
                ></video>
            </div>
            """

        cards.append(
            f"""
            <div style="
                break-inside:avoid;
                margin-bottom:12px;
                background:#171717;
                border:1px solid #272727;
                border-radius:14px;
                padding:8px;
                box-sizing:border-box;
            ">
                {thumb_html}
            </div>
            """
        )

    return f"""
    <div style="
        height:100%;
        min-height:480px;
        overflow:auto;
        padding:12px;
        box-sizing:border-box;
    ">
        <div style="
            column-count:4;
            column-gap:12px;
        ">
            {''.join(cards)}
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


def _create_local_collection(name: str) -> str:
    if not name or not name.strip():
        return _refresh_sidebar_payload()

    db = _get_db()
    db.create_collection(name=name.strip(), collection_type="local")
    return _refresh_sidebar_payload()


def _toggle_nsfw_filter(selected_collection_id_raw: str) -> tuple[str, str, str]:
    global _hide_nsfw
    _hide_nsfw = not _hide_nsfw

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


def _clear_cache() -> str:
    cache_dir = Path(DEFAULT_IMAGE_CACHE_DIR)
    removed = 0

    if cache_dir.exists():
        for f in cache_dir.iterdir():
            if f.is_file():
                try:
                    f.unlink()
                    removed += 1
                except Exception:
                    pass

    return f"Cache cleared: {removed} file(s) removed."


def _reset_extension() -> tuple[str, str]:
    global _db

    db_path = Path(DEFAULT_DB_PATH)
    cache_dir = Path(DEFAULT_IMAGE_CACHE_DIR)

    # Close DB connection
    _db = None

    # Delete database
    if db_path.exists():
        try:
            db_path.unlink()
        except Exception:
            pass

    # Clear cache folder
    if cache_dir.exists():
        for f in cache_dir.iterdir():
            if f.is_file():
                try:
                    f.unlink()
                except Exception:
                    pass

    # Recreate fresh DB
    db = _get_db()

    return (
        _refresh_sidebar_payload(),
        "Extension reset complete.",
    )


def _sync_collections() -> tuple[str, str]:
    settings = _settings()
    api_key = settings["api_key"]
    source_mode = settings["source_mode"]
    image_cache_dir = settings["image_cache_dir"]

    print("SYNC FUNCTION STARTED")
    print("api_key present:", bool(api_key))
    print("source_mode:", source_mode)
    print("image_cache_dir:", image_cache_dir)

    Path(image_cache_dir).mkdir(parents=True, exist_ok=True)

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
            collection_id = collection.get("id")
            collection_name = (collection.get("name") or "").strip() or f"Collection {collection_id}"
            collection_type = collection.get("type") or "Image"

            if collection_type != "Image":
                print(f"SKIPPING NON-IMAGE COLLECTION: {collection_name} ({collection_type})")
                continue

            if not collection_id:
                continue

            print(f"SYNCING COLLECTION: {collection_name} ({collection_id})")

            local_collection_id = db.get_or_create_collection(
                name=collection_name,
                collection_type="synced",
                civitai_id=collection_id,
            )

            db.clear_collection_items(local_collection_id)

            images = client.get_collection_images(collection_id=int(collection_id))
            print(f"  IMAGE COUNT: {len(images)}")

            for idx, image in enumerate(images):
                meta = image.get("meta") or {}
                user = image.get("user") or {}

                image_id = image.get("id")
                title = image.get("name") or f"Civitai Image {image_id}"

                raw_url = image.get("url") or ""
                image_url = (
                    f"https://image.civitai.com/xG1nkqKTMzGDvpLrqFT7WA/{raw_url}/width=512"
                    if raw_url else ""
                )

                item_id = db.create_item(
                    civitai_post_id=image.get("postId"),
                    title=title,
                    image_url=image_url,
                    local_path=image_url,
                    creator_name=user.get("username") or "Unknown",
                    creator_url="",
                    post_url=f"https://civitai.com/images/{image_id}" if image_id else "",
                    rating=str(image.get("nsfwLevel", "Unknown")),
                    platform="Civitai",
                    prompt=meta.get("prompt") or "",
                    negative_prompt=meta.get("negativePrompt") or "",
                    metadata_json=json.dumps(image),
                )

                db.add_item_to_collection(local_collection_id, item_id, idx)

            synced_count += 1

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
        "collection_cache_dir",
        shared.OptionInfo(
            str(DEFAULT_IMAGE_CACHE_DIR),
            "Image Cache Directory",
            gr.Textbox,
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
            """
            <style>
            #collection_toolbar_nsfw_button,
            #collection_toolbar_nsfw_button.lg {
                min-width: 28px !important;
                width: 28px !important;
                height: 24px !important;
                min-height: 24px !important;
                padding: 0 4px !important;
                font-size: 10px !important;
                line-height: 1 !important;
                border-radius: 6px !important;
                margin-top: 6px !important;
                margin-left: 4px !important;
            }
            </style>
            <script>
            setTimeout(() => {
                const btn = gradioApp().querySelector('#collection_toolbar_nsfw_button');
                if (btn) {
                    btn.setAttribute('title', '18+');
                }
            }, 0);
            </script>
            """
        )

        selected_collection_id = gr.Textbox(
            value="",
            visible=False,
            elem_id="collection_selected_collection_id",
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
                            "🚫",
                            elem_id="collection_toolbar_nsfw_button",
                            tooltip="18+",
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

        refresh_button.click(
            fn=_refresh_all,
            inputs=[selected_collection_id],
            outputs=[controls_html, sidebar_html, feed_html],
        )

        nsfw_toggle_button.click(
            fn=_toggle_nsfw_filter,
            inputs=[selected_collection_id],
            outputs=[controls_html, sidebar_html, feed_html],
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

    return [(collection_tab, "Collection", "collection_tab")]


script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_ui_tabs(on_ui_tabs)