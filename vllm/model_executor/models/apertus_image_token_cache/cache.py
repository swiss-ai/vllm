# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sqlite3
import time
from dataclasses import dataclass
from typing import Any

from vllm.logger import init_logger

from .config import ApertusImageTokenCacheConfig
from .guard_serde import (
    deserialize_collision_guard_record,
    serialize_collision_guard_record,
)
from .lru import ThreadSafeLRUCache
from .records import CollisionGuardCacheRecord, CollisionGuardVerificationData
from .sqlite_backend import SQLiteImagePromptBackend

logger = init_logger(__name__)


@dataclass
class _CacheStats:
    memory_hits: int = 0
    sqlite_hits: int = 0
    misses: int = 0
    sqlite_write_success: int = 0
    sqlite_write_failure: int = 0
    sqlite_busy_timeouts: int = 0
    collision_guard_validation_success: int = 0
    collision_guard_validation_failure: int = 0
    sqlite_open_time_ms: int = 0
    sqlite_read_time_ms: int = 0
    sqlite_write_time_ms: int = 0
    preload_rows_loaded: int = 0
    preload_bytes_loaded: int = 0
    preload_time_ms: int = 0
    preload_failures: int = 0


class ApertusImageTokenizationCache:
    def __init__(self, config: ApertusImageTokenCacheConfig) -> None:
        self._config = config
        self._memory: ThreadSafeLRUCache[str, str | CollisionGuardCacheRecord] = (
            ThreadSafeLRUCache(0)
        )
        self._disk_enabled = config.enabled
        self._disk_backend: SQLiteImagePromptBackend | None = None
        self._disk_init_failed = False
        self._preload_attempted = False
        self._stats = _CacheStats()
        self._cache_mode = "guard" if config.collision_guard else "normal"

        if not config.enabled:
            logger.debug(
                "Apertus image token cache disabled: %s",
                config.disabled_reason,
            )
        else:
            logger.info(
                "Apertus image token cache enabled at %s "
                "(sqlite_db=%s, mode=%s, memory_cache_size=%d, "
                "sqlite_busy_timeout_ms=%d, sqlite_mmap_size=%d).",
                config.cache_dir,
                config.sqlite_db_path,
                self._cache_mode,
                config.memory_cache_size,
                config.sqlite_busy_timeout_ms,
                config.sqlite_mmap_size,
            )
            logger.info(
                "Apertus image token cache in-memory cache is unbounded for this "
                "backend; %s is accepted but ignored for eviction.",
                config.MEMORY_SIZE_ENV,
            )
            # Initialize the SQLite backend eagerly so preload happens before
            # the first cache lookup.
            self._ensure_disk_backend()

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    @property
    def collision_guard_enabled(self) -> bool:
        return self._config.collision_guard

    def _debug(self, message: str, *args: Any) -> None:
        if self._config.debug_logging:
            logger.info(message, *args)

    def _increment_stat(self, field_name: str) -> None:
        setattr(self._stats, field_name, getattr(self._stats, field_name) + 1)

    @staticmethod
    def _elapsed_ms(start: float) -> int:
        return int((time.perf_counter() - start) * 1000)

    @staticmethod
    def _is_busy_timeout_error(exc: BaseException) -> bool:
        if not isinstance(exc, sqlite3.OperationalError):
            return False
        lowered = str(exc).lower()
        return "database is locked" in lowered or "database is busy" in lowered

    def _ensure_disk_backend(self) -> SQLiteImagePromptBackend | None:
        if not self._disk_enabled or self._disk_init_failed:
            return None
        if self._disk_backend is not None:
            return self._disk_backend

        sqlite_db_path = self._config.sqlite_db_path
        if sqlite_db_path is None:
            return None

        start = time.perf_counter()
        had_existing_db = sqlite_db_path.exists()
        try:
            sqlite_db_path.parent.mkdir(parents=True, exist_ok=True)
            self._disk_backend = SQLiteImagePromptBackend(
                db_path=sqlite_db_path,
                busy_timeout_ms=self._config.sqlite_busy_timeout_ms,
                mmap_size=self._config.sqlite_mmap_size,
            )
            # Force open + pragma setup once so open timing can be observed.
            self._disk_backend.get(mode=self._cache_mode, key="__warmup__")
            self._stats.sqlite_open_time_ms += self._elapsed_ms(start)
            self._debug(
                "Apertus image token cache SQLite backend opened path=%s open_ms=%d",
                sqlite_db_path,
                self._elapsed_ms(start),
            )
            if had_existing_db:
                self._preload_memory_cache(self._disk_backend)
            else:
                self._preload_attempted = True
                logger.info(
                    "Apertus image token cache preload skipped mode=%s "
                    "(database did not exist before initialization).",
                    self._cache_mode,
                )
            return self._disk_backend
        except Exception as exc:
            self._stats.sqlite_open_time_ms += self._elapsed_ms(start)
            self._disk_init_failed = True
            logger.warning(
                "Failed to initialize Apertus image token SQLite cache at %s: %s. "
                "Falling back to uncached tokenization.",
                sqlite_db_path,
                exc,
            )
            return None

    def _encode_disk_value(
        self,
        prompt: str,
        guard_data: CollisionGuardVerificationData | None,
    ) -> bytes:
        if not self._config.collision_guard:
            return prompt.encode("utf-8")

        if guard_data is None:
            raise ValueError("Collision guard data is required when guard is enabled.")

        record = CollisionGuardCacheRecord(
            prompt=prompt,
            fingerprint_payload=guard_data.fingerprint_payload,
            raw_image_mode=guard_data.raw_image_mode,
            raw_image_size=guard_data.raw_image_size,
            raw_image_bytes=guard_data.raw_image_bytes,
            raw_image_sha256=guard_data.raw_image_sha256,
            resized_image_mode=guard_data.resized_image_mode,
            resized_image_size=guard_data.resized_image_size,
            resized_image_sha256=guard_data.resized_image_sha256,
        )
        return serialize_collision_guard_record(record)

    def _validate_guard_record(
        self,
        record: CollisionGuardCacheRecord,
        guard_data: CollisionGuardVerificationData,
        cache_key: str,
        *,
        source: str,
    ) -> str | None:
        if record.fingerprint_payload != guard_data.fingerprint_payload:
            return "fingerprint-payload"
        if record.raw_image_mode != guard_data.raw_image_mode:
            return "raw-image-mode"
        if record.raw_image_size != guard_data.raw_image_size:
            return "raw-image-size"
        if record.raw_image_sha256 != guard_data.raw_image_sha256:
            return "raw-image-sha256"
        if record.raw_image_bytes != guard_data.raw_image_bytes:
            return "raw-image-bytes"
        if record.resized_image_mode != guard_data.resized_image_mode:
            return "resized-image-mode"
        if record.resized_image_size != guard_data.resized_image_size:
            return "resized-image-size"
        if record.resized_image_sha256 != guard_data.resized_image_sha256:
            return "resized-image-sha256"

        self._increment_stat("collision_guard_validation_success")
        self._debug(
            "Apertus image token cache collision-guard validation success "
            "source=%s key=%s",
            source,
            cache_key,
        )
        return None

    def _log_collision_guard_mismatch(
        self,
        *,
        cache_key: str,
        reason: str,
        source: str,
        expected: CollisionGuardCacheRecord,
        current: CollisionGuardVerificationData,
    ) -> None:
        key_prefix = cache_key[:12]
        logger.warning(
            "Apertus image token cache collision-guard mismatch "
            "(source=%s, key_prefix=%s, reason=%s, raw_sha_expected=%s, "
            "raw_sha_current=%s, resized_sha_expected=%s, resized_sha_current=%s, "
            "raw_mode_expected=%s, raw_mode_current=%s, raw_size_expected=%s, "
            "raw_size_current=%s, resized_mode_expected=%s, "
            "resized_mode_current=%s, resized_size_expected=%s, "
            "resized_size_current=%s). Treating as miss.",
            source,
            key_prefix,
            reason,
            expected.raw_image_sha256,
            current.raw_image_sha256,
            expected.resized_image_sha256,
            current.resized_image_sha256,
            expected.raw_image_mode,
            current.raw_image_mode,
            expected.raw_image_size,
            current.raw_image_size,
            expected.resized_image_mode,
            current.resized_image_mode,
            expected.resized_image_size,
            current.resized_image_size,
        )
        self._increment_stat("collision_guard_validation_failure")

    def _decode_disk_value(
        self,
        value: bytes,
        guard_data: CollisionGuardVerificationData | None,
        cache_key: str,
    ) -> str | None:
        if not self._config.collision_guard:
            return value.decode("utf-8")

        if guard_data is None:
            return None

        record = deserialize_collision_guard_record(value)
        mismatch_reason = self._validate_guard_record(
            record,
            guard_data,
            cache_key,
            source="sqlite",
        )
        if mismatch_reason is not None:
            self._log_collision_guard_mismatch(
                cache_key=cache_key,
                reason=mismatch_reason,
                source="sqlite",
                expected=record,
                current=guard_data,
            )
            return None
        return record.prompt

    def _memory_lookup(
        self,
        cache_key: str,
        guard_data: CollisionGuardVerificationData | None,
    ) -> str | None:
        record = self._memory.get(cache_key)
        if record is None:
            return None

        if not self._config.collision_guard:
            if isinstance(record, str):
                self._increment_stat("memory_hits")
                self._debug("Apertus image token cache memory hit key=%s", cache_key)
                return record
            self._memory.remove(cache_key)
            return None

        if not isinstance(record, CollisionGuardCacheRecord):
            self._memory.remove(cache_key)
            return None
        if guard_data is None:
            self._memory.remove(cache_key)
            return None

        mismatch_reason = self._validate_guard_record(
            record,
            guard_data,
            cache_key,
            source="memory",
        )
        if mismatch_reason is not None:
            self._log_collision_guard_mismatch(
                cache_key=cache_key,
                reason=mismatch_reason,
                source="memory",
                expected=record,
                current=guard_data,
            )
            self._memory.remove(cache_key)
            return None

        self._increment_stat("memory_hits")
        self._debug("Apertus image token cache memory hit key=%s", cache_key)
        return record.prompt

    def _preload_memory_cache(self, backend: SQLiteImagePromptBackend) -> None:
        if self._preload_attempted:
            return
        self._preload_attempted = True
        logger.info(
            "Apertus image token cache preload started mode=%s",
            self._cache_mode,
        )
        start = time.perf_counter()

        loaded = 0
        loaded_bytes = 0
        try:
            rows = backend.preload_rows(mode=self._cache_mode, max_entries=None)
            for cache_key, value in rows:
                loaded_bytes += len(value)
                if self._config.collision_guard:
                    self._memory.put(
                        cache_key,
                        deserialize_collision_guard_record(value),
                    )
                else:
                    self._memory.put(cache_key, value.decode("utf-8"))
                loaded += 1
        except Exception as exc:
            self._increment_stat("preload_failures")
            logger.warning(
                "Apertus image token cache preload failed mode=%s: %s. "
                "Continuing without preload.",
                self._cache_mode,
                exc,
            )
        finally:
            elapsed_ms = self._elapsed_ms(start)
            self._stats.preload_time_ms += elapsed_ms
            self._stats.preload_rows_loaded += loaded
            self._stats.preload_bytes_loaded += loaded_bytes
            logger.info(
                "Apertus image token cache preload complete mode=%s rows=%d bytes=%d "
                "elapsed_ms=%d",
                self._cache_mode,
                loaded,
                loaded_bytes,
                elapsed_ms,
            )

    def get(
        self,
        cache_key: str,
        *,
        guard_data: CollisionGuardVerificationData | None = None,
    ) -> str | None:
        memory_hit = self._memory_lookup(cache_key, guard_data)
        if memory_hit is not None:
            return memory_hit

        backend = self._ensure_disk_backend()
        if backend is None:
            self._increment_stat("misses")
            self._debug(
                "Apertus image token cache full miss key=%s (no SQLite backend)",
                cache_key,
            )
            return None

        read_start = time.perf_counter()
        try:
            disk_value = backend.get(mode=self._cache_mode, key=cache_key)
        except Exception as exc:
            self._stats.sqlite_read_time_ms += self._elapsed_ms(read_start)
            if self._is_busy_timeout_error(exc):
                self._increment_stat("sqlite_busy_timeouts")
            logger.warning(
                "Failed reading Apertus image token SQLite key %s: %s",
                cache_key,
                exc,
            )
            self._increment_stat("misses")
            return None
        self._stats.sqlite_read_time_ms += self._elapsed_ms(read_start)

        if disk_value is None:
            self._increment_stat("misses")
            self._debug("Apertus image token cache full miss key=%s", cache_key)
            return None

        try:
            prompt = self._decode_disk_value(disk_value, guard_data, cache_key)
        except Exception as exc:
            logger.warning(
                "Failed parsing Apertus image token SQLite value for key %s: %s",
                cache_key,
                exc,
            )
            self._increment_stat("misses")
            return None

        if prompt is None:
            self._increment_stat("misses")
            self._debug(
                "Apertus image token cache full miss key=%s (invalid disk record)",
                cache_key,
            )
            return None

        if self._config.collision_guard:
            if guard_data is None:
                return None
            self._memory.put(cache_key, deserialize_collision_guard_record(disk_value))
        else:
            self._memory.put(cache_key, prompt)

        self._increment_stat("sqlite_hits")
        self._debug("Apertus image token cache SQLite hit key=%s", cache_key)
        return prompt

    def put(
        self,
        cache_key: str,
        prompt: str,
        *,
        guard_data: CollisionGuardVerificationData | None = None,
    ) -> None:
        if self._config.collision_guard:
            if guard_data is None:
                return
            self._memory.put(
                cache_key,
                CollisionGuardCacheRecord(
                    prompt=prompt,
                    fingerprint_payload=guard_data.fingerprint_payload,
                    raw_image_mode=guard_data.raw_image_mode,
                    raw_image_size=guard_data.raw_image_size,
                    raw_image_bytes=guard_data.raw_image_bytes,
                    raw_image_sha256=guard_data.raw_image_sha256,
                    resized_image_mode=guard_data.resized_image_mode,
                    resized_image_size=guard_data.resized_image_size,
                    resized_image_sha256=guard_data.resized_image_sha256,
                ),
            )
        else:
            self._memory.put(cache_key, prompt)

        backend = self._ensure_disk_backend()
        if backend is None:
            return

        write_start = time.perf_counter()
        try:
            inserted = backend.put_if_absent(
                mode=self._cache_mode,
                key=cache_key,
                value=self._encode_disk_value(prompt, guard_data),
            )
            self._stats.sqlite_write_time_ms += self._elapsed_ms(write_start)
            if inserted:
                self._increment_stat("sqlite_write_success")
                self._debug(
                    "Apertus image token cache SQLite write success key=%s",
                    cache_key,
                )
            else:
                self._debug(
                    "Apertus image token cache SQLite key already existed key=%s",
                    cache_key,
                )
        except Exception as exc:
            self._stats.sqlite_write_time_ms += self._elapsed_ms(write_start)
            self._increment_stat("sqlite_write_failure")
            if self._is_busy_timeout_error(exc):
                self._increment_stat("sqlite_busy_timeouts")
            logger.warning(
                "Failed writing Apertus image token SQLite key %s: %s",
                cache_key,
                exc,
            )

    def get_stats(self) -> dict[str, int]:
        return {
            "memory_hits": self._stats.memory_hits,
            "sqlite_hits": self._stats.sqlite_hits,
            "misses": self._stats.misses,
            "sqlite_write_success": self._stats.sqlite_write_success,
            "sqlite_write_failure": self._stats.sqlite_write_failure,
            "sqlite_busy_timeouts": self._stats.sqlite_busy_timeouts,
            "collision_guard_validation_success": (
                self._stats.collision_guard_validation_success
            ),
            "collision_guard_validation_failure": (
                self._stats.collision_guard_validation_failure
            ),
            "sqlite_open_time_ms": self._stats.sqlite_open_time_ms,
            "sqlite_read_time_ms": self._stats.sqlite_read_time_ms,
            "sqlite_write_time_ms": self._stats.sqlite_write_time_ms,
            "preload_rows_loaded": self._stats.preload_rows_loaded,
            "preload_bytes_loaded": self._stats.preload_bytes_loaded,
            "preload_time_ms": self._stats.preload_time_ms,
            "preload_failures": self._stats.preload_failures,
        }

    def close(self) -> None:
        if self._disk_backend is not None:
            self._disk_backend.close()
