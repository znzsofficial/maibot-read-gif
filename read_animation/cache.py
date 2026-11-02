"""Single durable description/attempt store; corrupt or unknown data fails closed."""

from __future__ import annotations

import asyncio
import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any

CACHE_VERSION = 2
MAX_STORE_BYTES = 16 * 1024 * 1024


async def finish_io(function, *args):
    """Do not release a writer lock while its cancelled filesystem thread runs."""
    task = asyncio.create_task(asyncio.to_thread(function, *args))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        await task
        raise


class DescriptionCache:
    def __init__(
        self, path: Path | None, *, max_entries: int, retention_days: int, logger
    ) -> None:
        self._path = path
        self._max_entries = max_entries
        self._retention_days = retention_days
        self._logger = logger
        self._entries: dict[str, dict[str, Any]] = {}
        self._attempts: dict[str, str] = {}
        self._lock = asyncio.Lock()
        self._dirty = False
        self._healthy = True

    async def load(self) -> None:
        async with self._lock:
            try:
                payload = await finish_io(self._read_cache_file)
                if payload is None:
                    return
                if not isinstance(payload, dict) or payload.get("version") not in (
                    1,
                    CACHE_VERSION,
                ):
                    raise ValueError("unsupported cache version")
                entries = payload.get("entries")
                attempts = payload.get("attempts", {})
                if (
                    not isinstance(entries, dict)
                    or not isinstance(attempts, dict)
                    or len(attempts) > 8
                ):
                    raise ValueError("invalid cache shape")
                for key, entry in entries.items():
                    if (
                        not isinstance(key, str)
                        or len(key) > 256
                        or not isinstance(entry, dict)
                        or any(
                            not isinstance(entry.get(k), str) or len(entry[k]) > 4096
                            for k in ("header", "description", "kind")
                        )
                        or any(
                            not isinstance(entry.get(k), (int, float))
                            or not math.isfinite(entry[k])
                            for k in ("created_at", "last_access")
                        )
                    ):
                        raise ValueError("invalid cache entry")
                if any(
                    not isinstance(k, str)
                    or len(k) > 256
                    or v not in ("started", "unknown")
                    for k, v in attempts.items()
                ):
                    raise ValueError("invalid attempt record")
                self._entries = entries
                # A previous process's started request has an unknown remote outcome.
                self._attempts = {key: "unknown" for key in attempts}
                self._prune()
                self._dirty = False
            except (OSError, ValueError, TypeError):
                self._healthy = False
                self._logger.warning(
                    "动图缓存/请求记录损坏或不可读，保留原文件并停止新识别"
                )

    def _read_cache_file(self):
        if self._path is None or not self._path.exists():
            return None
        if self._path.is_symlink() or self._path.stat().st_size > MAX_STORE_BYTES:
            raise ValueError("unsafe cache file")
        return json.loads(self._path.read_text(encoding="utf-8"))

    def _prune(self):
        now = time.time()
        if self._retention_days > 0:
            self._entries = {
                key: value
                for key, value in self._entries.items()
                if 0 <= now - value["created_at"] < self._retention_days * 86400
            }
        if self._max_entries > 0 and len(self._entries) > self._max_entries:
            ranked = sorted(
                self._entries, key=lambda key: self._entries[key]["last_access"]
            )
            for key in ranked[: len(ranked) - self._max_entries]:
                del self._entries[key]

    async def configure(self, *, max_entries, retention_days):
        async with self._lock:
            self._max_entries, self._retention_days = max_entries, retention_days
            self._prune()
            self._dirty = True
            await self._save_locked()

    async def get(self, key):
        async with self._lock:
            if not self._healthy:
                return None
            entry = self._entries.get(key)
            if entry is None:
                return None
            age = time.time() - entry["created_at"]
            if age < 0 or (
                self._retention_days > 0 and age >= self._retention_days * 86400
            ):
                del self._entries[key]
                self._dirty = True
                return None
            entry["last_access"] = time.time()
            self._dirty = True
            return dict(entry)

    async def put(self, key, *, header, description, kind):
        async with self._lock:
            if not self._healthy:
                return
            now = time.time()
            self._entries[key] = dict(
                header=header,
                description=description,
                kind=kind,
                created_at=now,
                last_access=now,
            )
            self._prune()
            self._dirty = True
            await self._save_locked()

    async def begin(self, key, limit):
        """Persist before dispatch. Unknown attempts occupy slots until reviewed."""
        async with self._lock:
            if (
                not self._healthy
                or key in self._attempts
                or len(self._attempts) >= limit
            ):
                return False
            self._attempts[key] = "started"
            self._dirty = True
            await self._save_locked()
            return self._healthy

    async def end(self, key, *, known):
        async with self._lock:
            if key not in self._attempts:
                return
            if known:
                del self._attempts[key]
            else:
                self._attempts[key] = "unknown"
            self._dirty = True
            await self._save_locked()

    async def save(self):
        async with self._lock:
            await self._save_locked()

    async def _save_locked(self):
        if not self._healthy or not self._dirty:
            return
        if self._path is None:
            self._dirty = False
            return
        payload = json.dumps(
            dict(version=CACHE_VERSION, entries=self._entries, attempts=self._attempts),
            ensure_ascii=False,
        ).encode("utf-8")
        if len(payload) > MAX_STORE_BYTES:
            self._healthy = False
            self._logger.warning("动图记录超过持久化预算，停止新识别并保留已有文件")
            return
        try:
            await finish_io(self._write_cache_file, payload)
            self._dirty = False
        except OSError:
            self._healthy = False
            self._logger.warning("动图记录保存失败，停止新识别；不覆盖损坏记录")

    def _write_cache_file(self, payload):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if self._path.is_symlink():
            raise OSError("unsafe cache destination")
        fd, temporary = tempfile.mkstemp(
            prefix=self._path.name + ".", suffix=".tmp", dir=self._path.parent
        )
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self._path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    async def close(self):
        await self.save()
