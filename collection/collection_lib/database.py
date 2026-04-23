import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional


class CollectionDatabase:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def initialize(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS collections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    type TEXT NOT NULL,
                    civitai_id INTEGER,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    civitai_image_id INTEGER,
                    civitai_post_id INTEGER,
                    title TEXT,
                    image_url TEXT,
                    full_media_url TEXT,
                    preview_path TEXT,
                    full_path TEXT,
                    download_status TEXT DEFAULT 'none',
                    creator_name TEXT,
                    creator_url TEXT,
                    post_url TEXT,
                    rating TEXT,
                    platform TEXT,
                    media_type TEXT,
                    prompt TEXT,
                    negative_prompt TEXT,
                    metadata_json TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS collection_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    collection_id INTEGER NOT NULL,
                    item_id INTEGER NOT NULL,
                    order_index INTEGER NOT NULL DEFAULT 0,
                    added_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(collection_id, item_id)
                );

                CREATE TABLE IF NOT EXISTS resources (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    item_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    weight REAL DEFAULT 1.0,
                    model_id INTEGER,
                    version_id INTEGER,
                    local_status TEXT DEFAULT 'red',
                    local_filename TEXT
                );

                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
                """
            )

            columns = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(items)").fetchall()
            }
            if "civitai_image_id" not in columns:
                conn.execute("ALTER TABLE items ADD COLUMN civitai_image_id INTEGER")
            if "media_type" not in columns:
                conn.execute("ALTER TABLE items ADD COLUMN media_type TEXT")
            if "full_media_url" not in columns:
                conn.execute("ALTER TABLE items ADD COLUMN full_media_url TEXT")            

    def create_collection(self, name: str, collection_type: str, civitai_id: Optional[int] = None) -> int:
        with self.connect() as conn:
            cur = conn.execute(
                "INSERT INTO collections (name, type, civitai_id) VALUES (?, ?, ?)",
                (name, collection_type, civitai_id),
            )
            return int(cur.lastrowid)

    def list_collections(self, collection_type: str) -> List[Dict[str, Any]]:
        query = """
            SELECT c.id, c.name, c.type, c.updated_at,
                   COUNT(ci.item_id) AS item_count
            FROM collections c
            LEFT JOIN collection_items ci ON ci.collection_id = c.id
            WHERE c.type = ?
            GROUP BY c.id, c.name, c.type, c.updated_at
            ORDER BY c.name COLLATE NOCASE ASC
        """
        with self.connect() as conn:
            rows = conn.execute(query, (collection_type,)).fetchall()
        return [dict(row) for row in rows]

    def get_collection(self, collection_id: int) -> Optional[Dict[str, Any]]:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM collections WHERE id = ?",
                (collection_id,),
            ).fetchone()
        return dict(row) if row else None

    def list_items_for_collection(self, collection_id: int) -> List[Dict[str, Any]]:
        query = """
            SELECT i.*, ci.order_index
            FROM collection_items ci
            JOIN items i ON i.id = ci.item_id
            WHERE ci.collection_id = ?
            ORDER BY ci.order_index ASC, i.id ASC
        """
        with self.connect() as conn:
            rows = conn.execute(query, (collection_id,)).fetchall()
        return [dict(row) for row in rows]

    def get_item_detail(self, item_id: int) -> Optional[Dict[str, Any]]:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM items WHERE id = ?", (item_id,)).fetchone()
            if not row:
                return None
            detail = dict(row)
            detail["metadata_json"] = json.loads(detail.get("metadata_json") or "{}")
            detail["loras"] = [
                dict(r)
                for r in conn.execute(
                    "SELECT * FROM resources WHERE item_id = ? ORDER BY id ASC", (item_id,)
                ).fetchall()
            ]
            return detail

    def set_setting(self, key: str, value: str) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO settings (key, value, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (key, value),
            )

    def get_setting(self, key: str, default: str = "") -> str:
        with self.connect() as conn:
            row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else default

    def get_settings(self) -> Dict[str, str]:
        keys = {
            "api_key": "",
            "source_mode": "full",
            "image_cache_dir": "",
        }
        with self.connect() as conn:
            rows = conn.execute("SELECT key, value FROM settings").fetchall()
        settings = dict(keys)
        for row in rows:
            settings[row["key"]] = row["value"]
        return settings

    def get_or_create_collection(
        self,
        name: str,
        collection_type: str,
        civitai_id: Optional[int] = None,
    ) -> int:
        with self.connect() as conn:
            if civitai_id is not None:
                row = conn.execute(
                    "SELECT id FROM collections WHERE type = ? AND civitai_id = ?",
                    (collection_type, civitai_id),
                ).fetchone()
                if row:
                    conn.execute(
                        """
                        UPDATE collections
                        SET name = ?, updated_at = CURRENT_TIMESTAMP
                        WHERE id = ?
                        """,
                        (name, row["id"]),
                    )
                    return int(row["id"])

            row = conn.execute(
                "SELECT id FROM collections WHERE type = ? AND name = ?",
                (collection_type, name),
            ).fetchone()
            if row:
                conn.execute(
                    """
                    UPDATE collections
                    SET updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                    """,
                    (row["id"],),
                )
                return int(row["id"])

            cur = conn.execute(
                "INSERT INTO collections (name, type, civitai_id) VALUES (?, ?, ?)",
                (name, collection_type, civitai_id),
            )
            return int(cur.lastrowid)

    def create_item(
        self,
        *,
        civitai_image_id: Optional[int],
        civitai_post_id: Optional[int],
        title: str,
        image_url: str,
        full_media_url: str,
        preview_path: str,
        full_path: str = "",
        download_status: str = "none",
        creator_name: str,
        creator_url: str,
        post_url: str,
        rating: str,
        platform: str,
        media_type: str,
        prompt: str,
        negative_prompt: str,
        metadata_json: str,
    ) -> int:
        with self.connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO items (
                    civitai_image_id, civitai_post_id, title, image_url, full_media_url, preview_path, full_path, download_status, creator_name, creator_url,
                    post_url, rating, platform, media_type, prompt, negative_prompt, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    civitai_image_id, civitai_post_id, title, image_url, full_media_url, preview_path, full_path, download_status, creator_name, creator_url,
                    post_url, rating, platform, media_type, prompt, negative_prompt, metadata_json,
                ),
            )
            return int(cur.lastrowid)

    def update_item_preview_state(
        self,
        item_id: int,
        preview_path: str,
        download_status: str,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE items
                SET preview_path = ?, download_status = ?
                WHERE id = ?
                """,
                (preview_path, download_status, item_id),
            )

    def update_item_full_state(
        self,
        item_id: int,
        full_path: str,
        download_status: str,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE items
                SET full_path = ?, download_status = ?
                WHERE id = ?
                """,
                (full_path, download_status, item_id),
            )

    def add_item_to_collection(self, collection_id: int, item_id: int, order_index: int = 0) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO collection_items (collection_id, item_id, order_index)
                VALUES (?, ?, ?)
                """,
                (collection_id, item_id, order_index),
            )

    def clear_collection_items(self, collection_id: int) -> None:
        with self.connect() as conn:
            conn.execute(
                "DELETE FROM collection_items WHERE collection_id = ?",
                (collection_id,),
            )

    def add_resource(
        self,
        item_id: int,
        name: str,
        weight: float = 1.0,
        model_id: Optional[int] = None,
        version_id: Optional[int] = None,
        local_status: str = "red",
        local_filename: Optional[str] = None,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO resources (
                    item_id, name, weight, model_id, version_id, local_status, local_filename
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (item_id, name, weight, model_id, version_id, local_status, local_filename),
            )

    def clear_resources_for_item(self, item_id: int) -> None:
        with self.connect() as conn:
            conn.execute(
                "DELETE FROM resources WHERE item_id = ?",
                (item_id,),
            )


    def seed_demo_data_if_empty(self) -> None:
        with self.connect() as conn:
            existing = conn.execute("SELECT COUNT(*) AS n FROM collections").fetchone()["n"]
            if existing:
                return

            synced_id = conn.execute(
                "INSERT INTO collections (name, type, civitai_id) VALUES (?, ?, ?)",
                ("Demo Synced", "synced", 1001),
            ).lastrowid
            local_id = conn.execute(
                "INSERT INTO collections (name, type) VALUES (?, ?)",
                ("Favorites", "local"),
            ).lastrowid

            demo_items = [
                {
                    "title": "Painterly character study",
                    "image_url": "https://placehold.co/480x720/png",
                    "preview_path": "https://placehold.co/480x720/png",
                    "full_path": "",
                    "download_status": "preview",
                    "creator_name": "Demo Creator",
                    "creator_url": "https://example.com/creator/demo",                    "post_url": "https://example.com/post/1",
                    "rating": "PG",
                    "platform": "Automatic1111",
                    "prompt": "masterpiece, painterly portrait, dramatic lighting",
                    "negative_prompt": "blurry, lowres",
                    "metadata_json": json.dumps({"sampler": "DPM++ 2M", "steps": 28, "cfg": 7, "size": "832x1216"}),
                    "loras": [("PainterlyStyleXL", 0.8), ("HandsFixer", 0.4)],
                },
                {
                    "title": "Graphic poster concept",
                    "image_url": "https://placehold.co/720x960/png",
                    "preview_path": "https://placehold.co/720x960/png",
                    "full_path": "",
                    "download_status": "preview",
                    "creator_name": "Demo Creator",
                    "creator_url": "https://example.com/creator/demo",
                    "post_url": "https://example.com/post/2",
                    "rating": "R",
                    "platform": "Forge",
                    "prompt": "graphic poster, high contrast, editorial design",
                    "negative_prompt": "muddy colors",
                    "metadata_json": json.dumps({"sampler": "Euler a", "steps": 24, "cfg": 6.5, "size": "1024x1365"}),
                    "loras": [("PosterTone", 1.0)],
                },
            ]

            for idx, item in enumerate(demo_items):
                item_id = conn.execute(
                    """
                    INSERT INTO items (
                        title, image_url, preview_path, full_path, download_status, creator_name, creator_url,
                        post_url, rating, platform, prompt, negative_prompt, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        item["title"], item["image_url"], item["preview_path"], item["full_path"], item["download_status"], item["creator_name"],
                        item["creator_url"], item["post_url"], item["rating"], item["platform"],
                        item["prompt"], item["negative_prompt"], item["metadata_json"],
                    ),
                ).lastrowid

                conn.execute(
                    "INSERT INTO collection_items (collection_id, item_id, order_index) VALUES (?, ?, ?)",
                    (synced_id, item_id, idx),
                )
                conn.execute(
                    "INSERT INTO collection_items (collection_id, item_id, order_index) VALUES (?, ?, ?)",
                    (local_id, item_id, idx),
                )

                for name, weight in item["loras"]:
                    conn.execute(
                        "INSERT INTO resources (item_id, name, weight) VALUES (?, ?, ?)",
                        (item_id, name, weight),
                    )
