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

                    generator_name TEXT,
                    generator_type TEXT,
                    has_generation_data INTEGER DEFAULT 0,
                    is_external_generator INTEGER DEFAULT 0,

                    steps INTEGER,
                    cfg_scale REAL,
                    sampler TEXT,
                    scheduler TEXT,
                    seed TEXT,
                    clip_skip INTEGER,
                    width INTEGER,
                    height INTEGER,

                    denoise_strength REAL,
                    hires_enabled INTEGER DEFAULT 0,
                    hires_upscaler TEXT,
                    hires_steps INTEGER,
                    hires_denoise_strength REAL,
                    resize_mode TEXT,

                    checkpoint_name TEXT,
                    checkpoint_version TEXT,
                    vae_name TEXT,

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
                    resource_type TEXT,
                    name TEXT NOT NULL,
                    version_name TEXT,
                    weight REAL DEFAULT 1.0,
                    model_id INTEGER,
                    version_id INTEGER,
                    hash_sha256 TEXT,
                    hash_autov2 TEXT,
                    local_status TEXT DEFAULT 'red',
                    local_filename TEXT
                );

                CREATE TABLE IF NOT EXISTS generation_params (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    item_id INTEGER NOT NULL,
                    param_key TEXT NOT NULL,
                    param_value TEXT,
                    value_type TEXT,
                    display_group TEXT,
                    source_path TEXT,
                    sort_order INTEGER DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS civitai_hash_cache (
                    file_hash TEXT PRIMARY KEY,
                    model_id TEXT,
                    version_id TEXT,
                    hash_sha256 TEXT,
                    hash_autov2 TEXT,
                    raw_json TEXT,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS local_resource_files (
                    file_path TEXT PRIMARY KEY,
                    resource_type TEXT,
                    file_size INTEGER,
                    modified_at REAL,
                    hash_sha256 TEXT,
                    hash_autov2 TEXT,
                    model_id TEXT,
                    version_id TEXT,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
                """
            )

            columns = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(items)").fetchall()
            }
            item_column_defs = {
                "civitai_image_id": "INTEGER",
                "media_type": "TEXT",
                "full_media_url": "TEXT",
                "creator_url": "TEXT",
                "generator_name": "TEXT",
                "generator_type": "TEXT",
                "has_generation_data": "INTEGER DEFAULT 0",
                "is_external_generator": "INTEGER DEFAULT 0",
                "steps": "INTEGER",
                "cfg_scale": "REAL",
                "sampler": "TEXT",
                "scheduler": "TEXT",
                "seed": "TEXT",
                "clip_skip": "INTEGER",
                "width": "INTEGER",
                "height": "INTEGER",
                "denoise_strength": "REAL",
                "hires_enabled": "INTEGER DEFAULT 0",
                "hires_upscaler": "TEXT",
                "hires_steps": "INTEGER",
                "hires_denoise_strength": "REAL",
                "resize_mode": "TEXT",
                "checkpoint_name": "TEXT",
                "checkpoint_version": "TEXT",
                "vae_name": "TEXT",
            }
            for column_name, column_def in item_column_defs.items():
                if column_name not in columns:
                    conn.execute(f"ALTER TABLE items ADD COLUMN {column_name} {column_def}")

            resource_columns = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(resources)").fetchall()
            }
            resource_column_defs = {
                "resource_type": "TEXT",
                "version_name": "TEXT",
                "hash_sha256": "TEXT",
                "hash_autov2": "TEXT",
            }
            for column_name, column_def in resource_column_defs.items():
                if column_name not in resource_columns:
                    conn.execute(f"ALTER TABLE resources ADD COLUMN {column_name} {column_def}")

            local_resource_columns = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(local_resource_files)").fetchall()
            }
            local_resource_column_defs = {
                "resource_type": "TEXT",
                "file_size": "INTEGER",
                "modified_at": "REAL",
                "hash_sha256": "TEXT",
                "hash_autov2": "TEXT",
                "model_id": "TEXT",
                "version_id": "TEXT",
            }
            for column_name, column_def in local_resource_column_defs.items():
                if column_name not in local_resource_columns:
                    conn.execute(f"ALTER TABLE local_resource_files ADD COLUMN {column_name} {column_def}")

        self.cleanup_duplicate_items()


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

    def get_collection_item_ids(self, collection_id: int) -> set[int]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT item_id FROM collection_items WHERE collection_id = ?",
                (collection_id,),
            ).fetchall()

        return {int(row["item_id"]) for row in rows}

    def cleanup_orphan_items(self) -> int:
        with self.connect() as conn:
            orphan_rows = conn.execute(
                """
                SELECT i.id
                FROM items i
                LEFT JOIN collection_items ci ON ci.item_id = i.id
                WHERE ci.item_id IS NULL
                """
            ).fetchall()

            orphan_ids = [int(row["id"]) for row in orphan_rows]

            for item_id in orphan_ids:
                conn.execute("DELETE FROM resources WHERE item_id = ?", (item_id,))
                conn.execute("DELETE FROM generation_params WHERE item_id = ?", (item_id,))
                conn.execute("DELETE FROM items WHERE id = ?", (item_id,))

        return len(orphan_ids)

    def get_item_detail(self, item_id: int) -> Optional[Dict[str, Any]]:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM items WHERE id = ?", (item_id,)).fetchone()
            if not row:
                return None
            detail = dict(row)
            detail["metadata_json"] = json.loads(detail.get("metadata_json") or "{}")
            detail["resources"] = [
                dict(r)
                for r in conn.execute(
                    "SELECT * FROM resources WHERE item_id = ? ORDER BY id ASC", (item_id,)
                ).fetchall()
            ]
            return detail

    def get_item_by_civitai_image_id(self, image_id: int) -> Optional[Dict[str, Any]]:
        if image_id is None:
            return None

        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM items
                WHERE civitai_image_id = ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (image_id,),
            ).fetchone()

        return dict(row) if row else None

    def item_has_resources(self, item_id: int) -> bool:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM resources WHERE item_id = ?",
                (item_id,),
            ).fetchone()

        return bool(row and int(row["n"]) > 0)

    def item_has_generation_params(self, item_id: int) -> bool:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM generation_params WHERE item_id = ?",
                (item_id,),
            ).fetchone()

        return bool(row and int(row["n"]) > 0)

    def item_is_hydrated(self, item_id: int) -> bool:
        # For incremental sync, "hydrated" must mean resource data exists.
        # Generation params alone are not enough because the detail panel depends
        # on stored checkpoint / LoRA / embedding resources for local availability dots.
        return self.item_has_resources(item_id)

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

    def get_civitai_hash_cache(self, file_hash: str) -> Optional[Dict[str, Any]]:
        clean_hash = str(file_hash or "").strip().lower()
        if not clean_hash:
            return None

        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM civitai_hash_cache
                WHERE file_hash = ?
                   OR hash_sha256 = ?
                   OR hash_autov2 = ?
                LIMIT 1
                """,
                (clean_hash, clean_hash, clean_hash),
            ).fetchone()

        return dict(row) if row else None

    def upsert_civitai_hash_cache(
        self,
        file_hash: str,
        model_id: str = "",
        version_id: str = "",
        hash_sha256: str = "",
        hash_autov2: str = "",
        raw_json: str = "",
    ) -> None:
        clean_hash = str(file_hash or "").strip().lower()
        if not clean_hash:
            return

        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO civitai_hash_cache (
                    file_hash, model_id, version_id, hash_sha256, hash_autov2, raw_json, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(file_hash) DO UPDATE SET
                    model_id = excluded.model_id,
                    version_id = excluded.version_id,
                    hash_sha256 = excluded.hash_sha256,
                    hash_autov2 = excluded.hash_autov2,
                    raw_json = excluded.raw_json,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    clean_hash,
                    str(model_id or ""),
                    str(version_id or ""),
                    str(hash_sha256 or "").lower(),
                    str(hash_autov2 or "").lower(),
                    raw_json or "",
                ),
            )

    def get_local_resource_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        clean_path = str(file_path or "").strip()
        if not clean_path:
            return None

        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM local_resource_files WHERE file_path = ?",
                (clean_path,),
            ).fetchone()

        return dict(row) if row else None

    def upsert_local_resource_file(
        self,
        file_path: str,
        resource_type: str,
        file_size: int,
        modified_at: float,
        hash_sha256: str = "",
        hash_autov2: str = "",
        model_id: str = "",
        version_id: str = "",
    ) -> None:
        clean_path = str(file_path or "").strip()
        if not clean_path:
            return

        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO local_resource_files (
                    file_path, resource_type, file_size, modified_at,
                    hash_sha256, hash_autov2, model_id, version_id, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(file_path) DO UPDATE SET
                    resource_type = excluded.resource_type,
                    file_size = excluded.file_size,
                    modified_at = excluded.modified_at,
                    hash_sha256 = excluded.hash_sha256,
                    hash_autov2 = excluded.hash_autov2,
                    model_id = excluded.model_id,
                    version_id = excluded.version_id,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    clean_path,
                    str(resource_type or ""),
                    int(file_size or 0),
                    float(modified_at or 0.0),
                    str(hash_sha256 or "").lower(),
                    str(hash_autov2 or "").lower(),
                    str(model_id or ""),
                    str(version_id or ""),
                ),
            )

    def cleanup_duplicate_items(self) -> None:
        with self.connect() as conn:
            duplicate_rows = conn.execute(
                """
                SELECT civitai_image_id
                FROM items
                WHERE civitai_image_id IS NOT NULL
                GROUP BY civitai_image_id
                HAVING COUNT(*) > 1
                """
            ).fetchall()

            for row in duplicate_rows:
                image_id = row["civitai_image_id"]

                ranked_rows = conn.execute(
                    """
                    SELECT
                        i.id,
                        (SELECT COUNT(*) FROM resources r WHERE r.item_id = i.id) AS resource_count,
                        (SELECT COUNT(*) FROM generation_params gp WHERE gp.item_id = i.id) AS param_count,
                        (SELECT COUNT(*) FROM collection_items ci WHERE ci.item_id = i.id) AS collection_count
                    FROM items i
                    WHERE i.civitai_image_id = ?
                    ORDER BY resource_count DESC, param_count DESC, collection_count DESC, id ASC
                    """,
                    (image_id,),
                ).fetchall()

                if not ranked_rows:
                    continue

                keep_id = int(ranked_rows[0]["id"])
                duplicate_ids = [int(r["id"]) for r in ranked_rows[1:]]

                for duplicate_id in duplicate_ids:
                    duplicate_links = conn.execute(
                        """
                        SELECT collection_id, order_index
                        FROM collection_items
                        WHERE item_id = ?
                        """,
                        (duplicate_id,),
                    ).fetchall()

                    for link in duplicate_links:
                        conn.execute(
                            """
                            INSERT OR IGNORE INTO collection_items (collection_id, item_id, order_index)
                            VALUES (?, ?, ?)
                            """,
                            (link["collection_id"], keep_id, link["order_index"]),
                        )

                    conn.execute("DELETE FROM collection_items WHERE item_id = ?", (duplicate_id,))
                    conn.execute("DELETE FROM resources WHERE item_id = ?", (duplicate_id,))
                    conn.execute("DELETE FROM generation_params WHERE item_id = ?", (duplicate_id,))
                    conn.execute("DELETE FROM items WHERE id = ?", (duplicate_id,))

            conn.execute(
                """
                DELETE FROM resources
                WHERE item_id NOT IN (SELECT id FROM items)
                """
            )

            conn.execute(
                """
                DELETE FROM generation_params
                WHERE item_id NOT IN (SELECT id FROM items)
                """
            )

            conn.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_items_civitai_image_id_unique
                ON items(civitai_image_id)
                WHERE civitai_image_id IS NOT NULL
                """
            )

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
        generator_name: str = "",
        generator_type: str = "",
        has_generation_data: int = 0,
        is_external_generator: int = 0,
        steps: Optional[int] = None,
        cfg_scale: Optional[float] = None,
        sampler: str = "",
        scheduler: str = "",
        seed: str = "",
        clip_skip: Optional[int] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        denoise_strength: Optional[float] = None,
        hires_enabled: int = 0,
        hires_upscaler: str = "",
        hires_steps: Optional[int] = None,
        hires_denoise_strength: Optional[float] = None,
        resize_mode: str = "",
        checkpoint_name: str = "",
        checkpoint_version: str = "",
        vae_name: str = "",
    ) -> int:
        with self.connect() as conn:

            # 🔴 Prevent duplicate inserts at source
            if civitai_image_id is not None:
                existing = conn.execute(
                    """
                    SELECT id
                    FROM items
                    WHERE civitai_image_id = ?
                    LIMIT 1
                    """,
                    (civitai_image_id,),
                ).fetchone()

                if existing:
                    return int(existing["id"])

            cur = conn.execute(
                """
                INSERT INTO items (
                    civitai_image_id, civitai_post_id, title, image_url, full_media_url, preview_path, full_path, download_status,
                    creator_name, creator_url, post_url, rating, platform, media_type, prompt, negative_prompt, metadata_json,
                    generator_name, generator_type, has_generation_data, is_external_generator,
                    steps, cfg_scale, sampler, scheduler, seed, clip_skip, width, height,
                    denoise_strength, hires_enabled, hires_upscaler, hires_steps, hires_denoise_strength, resize_mode,
                    checkpoint_name, checkpoint_version, vae_name
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    civitai_image_id, civitai_post_id, title, image_url, full_media_url, preview_path, full_path, download_status,
                    creator_name, creator_url, post_url, rating, platform, media_type, prompt, negative_prompt, metadata_json,
                    generator_name, generator_type, has_generation_data, is_external_generator,
                    steps, cfg_scale, sampler, scheduler, seed, clip_skip, width, height,
                    denoise_strength, hires_enabled, hires_upscaler, hires_steps, hires_denoise_strength, resize_mode,
                    checkpoint_name, checkpoint_version, vae_name,
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
        resource_type: str = "",
        version_name: str = "",
        weight: float = 1.0,
        model_id: Optional[int] = None,
        version_id: Optional[int] = None,
        hash_sha256: str = "",
        hash_autov2: str = "",
        local_status: str = "red",
        local_filename: Optional[str] = None,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO resources (
                    item_id, resource_type, name, version_name, weight, model_id, version_id,
                    hash_sha256, hash_autov2, local_status, local_filename
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item_id,
                    resource_type,
                    name,
                    version_name,
                    weight,
                    model_id,
                    version_id,
                    hash_sha256,
                    hash_autov2,
                    local_status,
                    local_filename,
                ),
            )

    def clear_resources_for_item(self, item_id: int) -> None:
        with self.connect() as conn:
            conn.execute(
                "DELETE FROM resources WHERE item_id = ?",
                (item_id,),
            )

    def clear_generation_params_for_item(self, item_id: int) -> None:
        with self.connect() as conn:
            conn.execute(
                "DELETE FROM generation_params WHERE item_id = ?",
                (item_id,),
            )

    def add_generation_param(
        self,
        item_id: int,
        param_key: str,
        param_value: str,
        value_type: str = "string",
        display_group: str = "",
        source_path: str = "",
        sort_order: int = 0,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO generation_params (
                    item_id, param_key, param_value, value_type, display_group, source_path, sort_order
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item_id,
                    param_key,
                    param_value,
                    value_type,
                    display_group,
                    source_path,
                    sort_order,
                ),
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
                    "creator_url": "https://example.com/creator/demo",
                    "post_url": "https://example.com/post/1",
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
