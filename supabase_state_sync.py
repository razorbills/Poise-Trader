import asyncio
import base64
import logging
import os
from datetime import datetime, timezone
from typing import List, Optional

import requests


class SupabaseStateSync:
    def __init__(
        self,
        supabase_url: str,
        supabase_key: str,
        table: str = "bot_state",
        keys: Optional[List[str]] = None,
        interval_seconds: int = 60,
        timeout_seconds: int = 20,
    ):
        self.logger = logging.getLogger("SupabaseStateSync")
        self.supabase_url = (supabase_url or "").rstrip("/")
        self.supabase_key = supabase_key
        self.table = table
        self.keys = keys or []
        self.interval_seconds = max(10, int(interval_seconds))
        self.timeout_seconds = max(5, int(timeout_seconds))

        self._stop_event: Optional[asyncio.Event] = None
        self._task: Optional[asyncio.Task] = None

    @classmethod
    def from_env(cls) -> Optional["SupabaseStateSync"]:
        backend = str(os.getenv("STATE_BACKEND", "") or "").strip().lower()
        if backend and backend != "supabase":
            return None

        supabase_url = os.getenv("SUPABASE_URL")
        if not supabase_url:
            project_ref = str(os.getenv("SUPABASE_PROJECT_REF", "") or "").strip()
            if project_ref:
                supabase_url = f"https://{project_ref}.supabase.co"

        supabase_key = (
            os.getenv("SUPABASE_SERVICE_ROLE_KEY")
            or os.getenv("SUPABASE_KEY")
            or os.getenv("SUPABASE_ANON_KEY")
        )

        if not supabase_url or not supabase_key:
            return None

        keys_env = str(
            os.getenv(
                "STATE_SYNC_KEYS",
                "learned_knowledge.json,trading_state.json,strategy_learning_data.pkl",
            )
            or ""
        )
        keys = [k.strip() for k in keys_env.split(",") if k.strip()]

        interval_seconds = int(str(os.getenv("STATE_SYNC_INTERVAL_SECONDS", "60") or "60").strip() or 60)
        table = str(os.getenv("SUPABASE_STATE_TABLE", "bot_state") or "bot_state").strip() or "bot_state"

        return cls(
            supabase_url=supabase_url,
            supabase_key=supabase_key,
            table=table,
            keys=keys,
            interval_seconds=interval_seconds,
        )

    def _headers(self) -> dict:
        return {
            "apikey": self.supabase_key,
            "Authorization": f"Bearer {self.supabase_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _rest_url(self, path: str) -> str:
        return f"{self.supabase_url}/rest/v1/{self.table}{path}" 

    def _local_path_for_key(self, key: str) -> str:
        if "/" in key or "\\" in key:
            return key
        return os.path.join("data", key)

    def _read_local_b64(self, key: str) -> Optional[str]:
        path = self._local_path_for_key(key)
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            raw = f.read()
        return base64.b64encode(raw).decode("ascii")

    def _write_local_b64(self, key: str, content_base64: str) -> None:
        path = self._local_path_for_key(key)
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        os.makedirs("data", exist_ok=True)
        raw = base64.b64decode(content_base64.encode("ascii"))
        with open(path, "wb") as f:
            f.write(raw)

    def download_key(self, key: str) -> bool:
        try:
            url = self._rest_url(f"?file_key=eq.{key}&select=content_base64")
            r = requests.get(url, headers=self._headers(), timeout=self.timeout_seconds)
            if r.status_code == 404:
                return False
            r.raise_for_status()
            data = r.json()
            if not isinstance(data, list) or not data:
                return False
            row = data[0]
            content_b64 = row.get("content_base64")
            if not content_b64:
                return False
            self._write_local_b64(key, content_b64)
            try:
                path = self._local_path_for_key(key)
                size = os.path.getsize(path) if os.path.exists(path) else None
                if size is not None:
                    self.logger.info(f"✅ Restored {key} from Supabase ({size} bytes)")
                else:
                    self.logger.info(f"✅ Restored {key} from Supabase")
            except Exception:
                self.logger.info(f"✅ Restored {key} from Supabase")
            return True
        except Exception as e:
            self.logger.warning(f"State download failed for {key}: {e}")
            return False

    def upload_key(self, key: str) -> bool:
        try:
            content_b64 = self._read_local_b64(key)
            if not content_b64:
                return False

            url = self._rest_url("?on_conflict=file_key")
            payload = [
                {
                    "file_key": key,
                    "content_base64": content_b64,
                }
            ]
            headers = dict(self._headers())
            headers["Prefer"] = "resolution=merge-duplicates"

            r = requests.post(url, headers=headers, json=payload, timeout=self.timeout_seconds)
            r.raise_for_status()
            return True
        except Exception as e:
            self.logger.warning(f"State upload failed for {key}: {e}")
            return False

    async def restore_on_startup(self) -> None:
        if not self.keys:
            return
        restored = 0
        missing = 0
        for key in self.keys:
            ok = await asyncio.to_thread(self.download_key, key)
            if ok:
                restored += 1
            else:
                missing += 1
                self.logger.info(f"No state found in Supabase for {key} (skipping)")

        self.logger.info(
            f"☁️ Supabase restore complete: restored={restored}, missing={missing}, total={len(self.keys)}"
        )

    async def sync_once(self) -> None:
        if not self.keys:
            return
        for key in self.keys:
            await asyncio.to_thread(self.upload_key, key)

    def start_background_sync(self) -> Optional[asyncio.Task]:
        if self._task:
            return self._task
        self._stop_event = asyncio.Event()
        self.logger.info(
            f"☁️ Supabase background sync started (interval={self.interval_seconds}s, keys={len(self.keys)})"
        )
        self._task = asyncio.create_task(self._run_loop())
        return self._task

    async def stop(self) -> None:
        if self._stop_event:
            self._stop_event.set()
        if self._task:
            try:
                await self._task
            except Exception:
                pass
        self._task = None
        self._stop_event = None

    async def _run_loop(self) -> None:
        while True:
            if self._stop_event and self._stop_event.is_set():
                return
            await self.sync_once()
            await asyncio.sleep(self.interval_seconds)
