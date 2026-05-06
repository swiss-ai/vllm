# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import base64
import json
import multiprocessing as mp
import sqlite3
import threading
import time
from pathlib import Path

import pytest
import torch
from PIL import Image

from vllm.model_executor.models.apertus_image_token_cache.cache import (
    ApertusImageTokenizationCache,
)
from vllm.model_executor.models.apertus_image_token_cache.config import (
    ApertusImageTokenCacheConfig,
)
from vllm.model_executor.models.apertus_image_token_cache.guard_serde import (
    deserialize_collision_guard_record,
)
from vllm.model_executor.models.apertus_image_token_cache.hashing import stable_cache_key
from vllm.model_executor.models.apertus_image_token_cache.records import (
    CollisionGuardVerificationData,
)
from vllm.model_executor.models.apertus_utils import ApertusImageTokenizer

pytestmark = pytest.mark.cpu_test


class _PromptTokenizer:
    boi_token = "<|img_start|>"
    img_token = "<|img_token_start|>"
    eol_token = "<|img_end_of_row|>"
    eoi_token = "<|img_end|>"


class _FakeVisionTokenizer:
    def __init__(self) -> None:
        self._param = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))
        self.encode_calls = 0

    def parameters(self):
        return iter((self._param,))

    def encode(self, *_args, **_kwargs):
        self.encode_calls += 1
        return torch.tensor([[5]], dtype=torch.int64)


def _mm_kwargs() -> dict[str, object]:
    return {
        "apertus_vision_tokenizer_device": "cpu",
        "apertus_vision_tokenizer_dtype": "float32",
        "apertus_min_pixels": 16 * 16,
        "apertus_max_pixels": 16 * 16,
        "apertus_vq_hub": "BAAI/Emu3.5-VisionTokenizer",
        "apertus_vq_type": "ibq",
    }


def _make_image() -> Image.Image:
    return Image.new("RGB", (16, 16), color=(31, 63, 95))


def _sqlite_db_path(cache_dir: Path) -> Path:
    return cache_dir / "image_tokens" / "apertus_image_token_cache.sqlite3"


def _read_single_sqlite_entry(cache_dir: Path, *, mode: str) -> tuple[str, bytes]:
    conn = sqlite3.connect(str(_sqlite_db_path(cache_dir)))
    try:
        row = conn.execute(
            "SELECT cache_key, value FROM image_token_cache WHERE mode=?;",
            (mode,),
        ).fetchall()
        assert len(row) == 1
        return str(row[0][0]), bytes(row[0][1])
    finally:
        conn.close()


def _make_guard_data(image: Image.Image) -> CollisionGuardVerificationData:
    raw_bytes = image.tobytes()
    payload = "payload"
    return CollisionGuardVerificationData(
        fingerprint_payload=payload,
        raw_image_mode=image.mode,
        raw_image_size=image.size,
        raw_image_bytes=raw_bytes,
        raw_image_sha256="raw-sha",
        resized_image_mode=image.mode,
        resized_image_size=image.size,
        resized_image_sha256="resized-sha",
    )


def _mp_worker(cache_dir: str, worker_id: int, queue: mp.Queue) -> None:
    try:
        cache = ApertusImageTokenizationCache(
            ApertusImageTokenCacheConfig(
                cache_dir=Path(cache_dir),
                collision_guard=False,
                memory_cache_size=128,
                sqlite_busy_timeout_ms=3000,
                sqlite_mmap_size=1024 * 1024,
                debug_logging=False,
                disabled_reason=None,
            )
        )
        for idx in range(200):
            key = f"shared-key-{idx % 20}"
            prompt = f"prompt-{idx % 20}"
            cache.put(key, prompt)
            resolved = cache.get(key)
            if resolved != prompt:
                queue.put(f"worker {worker_id} mismatch for {key}: {resolved!r}")
                cache.close()
                return
        cache.close()
        queue.put("ok")
    except Exception as exc:  # pragma: no cover - defensive in subprocess
        queue.put(f"worker {worker_id} exception: {exc}")


def test_key_payload_is_deterministic_and_time_independent():
    logical_key = (
        "apertus-image-tokenization-v1",
        "resized-image-hash",
        (112, 128),
        65536,
        1960000,
        16,
        "BAAI/Emu3.5-VisionTokenizer",
        "ibq",
        "cuda",
        "torch.bfloat16",
        "<|img_start|>",
        "<|img_token_start|>",
        "<|img_end_of_row|>",
        "<|img_end|>",
        "<|visual token {token_id}|>",
    )
    key1, payload1 = stable_cache_key(logical_key)
    key2, payload2 = stable_cache_key(logical_key)

    assert key1 == key2
    assert payload1 == payload2
    assert "slurm" not in payload1.lower()
    assert "pid" not in payload1.lower()
    assert "hostname" not in payload1.lower()
    assert "uuid" not in payload1.lower()


def test_normal_and_guard_coexist_in_same_sqlite_table(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True)

    normal_cache = ApertusImageTokenizationCache(
        ApertusImageTokenCacheConfig(
            cache_dir=cache_dir,
            collision_guard=False,
            memory_cache_size=16,
            sqlite_busy_timeout_ms=5000,
            sqlite_mmap_size=1024 * 1024,
            debug_logging=False,
            disabled_reason=None,
        )
    )
    guard_cache = ApertusImageTokenizationCache(
        ApertusImageTokenCacheConfig(
            cache_dir=cache_dir,
            collision_guard=True,
            memory_cache_size=16,
            sqlite_busy_timeout_ms=5000,
            sqlite_mmap_size=1024 * 1024,
            debug_logging=False,
            disabled_reason=None,
        )
    )

    image = _make_image()
    key = "same-key"
    normal_cache.put(key, "normal-prompt")
    guard_cache.put(key, "guard-prompt", guard_data=_make_guard_data(image))
    normal_cache.close()
    guard_cache.close()

    conn = sqlite3.connect(str(_sqlite_db_path(cache_dir)))
    try:
        rows = conn.execute(
            "SELECT mode, cache_key, value FROM image_token_cache WHERE cache_key=? "
            "ORDER BY mode;",
            (key,),
        ).fetchall()
        assert len(rows) == 2
        assert rows[0][0] == "guard"
        assert rows[1][0] == "normal"
        assert bytes(rows[1][2]) == b"normal-prompt"
        record = deserialize_collision_guard_record(bytes(rows[0][2]))
        assert record.prompt == "guard-prompt"
    finally:
        conn.close()


def test_cache_disabled_env_unset_uses_normal_tokenization(monkeypatch):
    monkeypatch.delenv("VLLM_APERTUS_IMAGE_TOKEN_CACHE_DIR", raising=False)
    monkeypatch.delenv(
        "VLLM_APERTUS_IMAGE_TOKEN_CACHE_COLLISION_GUARD",
        raising=False,
    )

    tokenizer = ApertusImageTokenizer()
    fake_vision = _FakeVisionTokenizer()
    monkeypatch.setattr(tokenizer, "load_vision_tokenizer", lambda _kwargs: fake_vision)

    outputs = tokenizer.encode_images(
        [_make_image(), _make_image()],
        tokenizer=_PromptTokenizer(),
        mm_processor_kwargs=_mm_kwargs(),
    )

    assert len(outputs) == 2
    assert fake_vision.encode_calls == 2


def test_invalid_cache_path_disables_cache_without_creating_directory(
    monkeypatch,
    tmp_path,
):
    invalid_dir = tmp_path / "does-not-exist"
    monkeypatch.setenv("VLLM_APERTUS_IMAGE_TOKEN_CACHE_DIR", str(invalid_dir))
    monkeypatch.delenv(
        "VLLM_APERTUS_IMAGE_TOKEN_CACHE_COLLISION_GUARD",
        raising=False,
    )

    tokenizer = ApertusImageTokenizer()
    fake_vision = _FakeVisionTokenizer()
    monkeypatch.setattr(tokenizer, "load_vision_tokenizer", lambda _kwargs: fake_vision)

    _ = tokenizer.encode_images(
        [_make_image()],
        tokenizer=_PromptTokenizer(),
        mm_processor_kwargs=_mm_kwargs(),
    )

    assert fake_vision.encode_calls == 1
    assert not invalid_dir.exists()


def test_normal_mode_sqlite_value_is_raw_prompt_bytes(monkeypatch, tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True)
    monkeypatch.setenv("VLLM_APERTUS_IMAGE_TOKEN_CACHE_DIR", str(cache_dir))
    monkeypatch.delenv(
        "VLLM_APERTUS_IMAGE_TOKEN_CACHE_COLLISION_GUARD",
        raising=False,
    )

    tokenizer = ApertusImageTokenizer()
    fake_vision = _FakeVisionTokenizer()
    monkeypatch.setattr(tokenizer, "load_vision_tokenizer", lambda _kwargs: fake_vision)

    prompt = tokenizer.encode_images(
        [_make_image()],
        tokenizer=_PromptTokenizer(),
        mm_processor_kwargs=_mm_kwargs(),
    )[0]
    if tokenizer._image_prompt_cache is not None:
        tokenizer._image_prompt_cache.close()
    _key, value = _read_single_sqlite_entry(cache_dir, mode="normal")

    assert value == prompt.encode("utf-8")


def test_non_guard_mode_does_not_use_guard_serializer(monkeypatch, tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True)
    monkeypatch.setenv("VLLM_APERTUS_IMAGE_TOKEN_CACHE_DIR", str(cache_dir))
    monkeypatch.delenv(
        "VLLM_APERTUS_IMAGE_TOKEN_CACHE_COLLISION_GUARD",
        raising=False,
    )

    def _boom(_record):
        raise RuntimeError("guard serializer should not be used in non-guard mode")

    monkeypatch.setattr(
        "vllm.model_executor.models.apertus_image_token_cache.cache.serialize_collision_guard_record",
        _boom,
    )

    tokenizer = ApertusImageTokenizer()
    fake_vision = _FakeVisionTokenizer()
    monkeypatch.setattr(tokenizer, "load_vision_tokenizer", lambda _kwargs: fake_vision)

    outputs = tokenizer.encode_images(
        [_make_image(), _make_image()],
        tokenizer=_PromptTokenizer(),
        mm_processor_kwargs=_mm_kwargs(),
    )
    assert len(outputs) == 2
    assert fake_vision.encode_calls == 1


def test_cache_enabled_persists_across_tokenizer_instances(monkeypatch, tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True)
    monkeypatch.setenv("VLLM_APERTUS_IMAGE_TOKEN_CACHE_DIR", str(cache_dir))
    monkeypatch.delenv(
        "VLLM_APERTUS_IMAGE_TOKEN_CACHE_COLLISION_GUARD",
        raising=False,
    )

    first_tokenizer = ApertusImageTokenizer()
    first_fake_vision = _FakeVisionTokenizer()
    monkeypatch.setattr(
        first_tokenizer,
        "load_vision_tokenizer",
        lambda _kwargs: first_fake_vision,
    )
    first_out = first_tokenizer.encode_images(
        [_make_image()],
        tokenizer=_PromptTokenizer(),
        mm_processor_kwargs=_mm_kwargs(),
    )
    assert first_fake_vision.encode_calls == 1

    second_out_same_process = first_tokenizer.encode_images(
        [_make_image()],
        tokenizer=_PromptTokenizer(),
        mm_processor_kwargs=_mm_kwargs(),
    )
    assert second_out_same_process == first_out
    assert first_fake_vision.encode_calls == 1

    second_tokenizer = ApertusImageTokenizer()
    second_fake_vision = _FakeVisionTokenizer()
    monkeypatch.setattr(
        second_tokenizer,
        "load_vision_tokenizer",
        lambda _kwargs: second_fake_vision,
    )
    second_out = second_tokenizer.encode_images(
        [_make_image()],
        tokenizer=_PromptTokenizer(),
        mm_processor_kwargs=_mm_kwargs(),
    )
    assert second_out == first_out
    assert second_fake_vision.encode_calls == 0
    assert _sqlite_db_path(cache_dir).exists()


def test_collision_guard_stores_raw_normalized_bytes(monkeypatch, tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True)
    monkeypatch.setenv("VLLM_APERTUS_IMAGE_TOKEN_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("VLLM_APERTUS_IMAGE_TOKEN_CACHE_COLLISION_GUARD", "1")

    tokenizer = ApertusImageTokenizer()
    fake_vision = _FakeVisionTokenizer()
    monkeypatch.setattr(tokenizer, "load_vision_tokenizer", lambda _kwargs: fake_vision)

    image = _make_image()
    _ = tokenizer.encode_images(
        [image],
        tokenizer=_PromptTokenizer(),
        mm_processor_kwargs=_mm_kwargs(),
    )
    if tokenizer._image_prompt_cache is not None:
        tokenizer._image_prompt_cache.close()
    _key, value = _read_single_sqlite_entry(cache_dir, mode="guard")
    payload = json.loads(value.decode("utf-8"))

    normalized = tokenizer.coerce_pil_image(image)
    expected_raw_bytes = normalized.tobytes()
    raw_image_payload = payload["raw_image"]

    assert payload["prompt"].startswith("<|img_start|>")
    assert tuple(raw_image_payload["size"]) == normalized.size
    assert raw_image_payload["mode"] == normalized.mode
    assert base64.b64decode(raw_image_payload["bytes_b64"].encode("ascii")) == (
        expected_raw_bytes
    )


def test_collision_guard_mismatch_treated_as_miss(monkeypatch, tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True)
    monkeypatch.setenv("VLLM_APERTUS_IMAGE_TOKEN_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("VLLM_APERTUS_IMAGE_TOKEN_CACHE_COLLISION_GUARD", "1")

    tokenizer = ApertusImageTokenizer()
    fake_vision = _FakeVisionTokenizer()
    monkeypatch.setattr(tokenizer, "load_vision_tokenizer", lambda _kwargs: fake_vision)
    _ = tokenizer.encode_images(
        [_make_image()],
        tokenizer=_PromptTokenizer(),
        mm_processor_kwargs=_mm_kwargs(),
    )
    assert fake_vision.encode_calls == 1
    if tokenizer._image_prompt_cache is not None:
        tokenizer._image_prompt_cache.close()

    conn = sqlite3.connect(str(_sqlite_db_path(cache_dir)))
    try:
        row = conn.execute(
            "SELECT cache_key, value FROM image_token_cache WHERE mode='guard';"
        ).fetchone()
        assert row is not None
        key = str(row[0])
        value = json.loads(bytes(row[1]).decode("utf-8"))
        value["raw_image"]["bytes_b64"] = base64.b64encode(b"corrupt-bytes").decode(
            "ascii"
        )
        conn.execute(
            "UPDATE image_token_cache SET value=? WHERE mode='guard' AND cache_key=?;",
            (
                json.dumps(
                    value,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                ).encode("utf-8"),
                key,
            ),
        )
        conn.commit()
    finally:
        conn.close()

    second_tokenizer = ApertusImageTokenizer()
    second_fake_vision = _FakeVisionTokenizer()
    monkeypatch.setattr(
        second_tokenizer,
        "load_vision_tokenizer",
        lambda _kwargs: second_fake_vision,
    )
    _ = second_tokenizer.encode_images(
        [_make_image()],
        tokenizer=_PromptTokenizer(),
        mm_processor_kwargs=_mm_kwargs(),
    )
    assert second_fake_vision.encode_calls == 1


def test_sqlite_failure_falls_back_to_tokenization(monkeypatch, tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True)
    monkeypatch.setenv("VLLM_APERTUS_IMAGE_TOKEN_CACHE_DIR", str(cache_dir))
    monkeypatch.delenv(
        "VLLM_APERTUS_IMAGE_TOKEN_CACHE_COLLISION_GUARD",
        raising=False,
    )

    tokenizer = ApertusImageTokenizer()
    fake_vision = _FakeVisionTokenizer()
    monkeypatch.setattr(tokenizer, "load_vision_tokenizer", lambda _kwargs: fake_vision)

    _ = tokenizer.encode_images(
        [_make_image()],
        tokenizer=_PromptTokenizer(),
        mm_processor_kwargs=_mm_kwargs(),
    )
    cache = tokenizer._image_prompt_cache
    assert cache is not None
    cache._disk_backend = None  # type: ignore[attr-defined]

    class _FailingBackend:
        def get(self, *, mode: str, key: str):
            del mode, key
            raise RuntimeError("read failure")

        def put_if_absent(self, *, mode: str, key: str, value: bytes):
            del mode, key, value
            raise RuntimeError("write failure")

    monkeypatch.setattr(cache, "_ensure_disk_backend", lambda: _FailingBackend())
    tokenizer2 = ApertusImageTokenizer()
    monkeypatch.setattr(tokenizer2, "load_vision_tokenizer", lambda _kwargs: fake_vision)
    tokenizer2._image_prompt_cache = cache

    outputs = tokenizer2.encode_images(
        [Image.new("RGB", (16, 16), color=(1, 2, 3))],
        tokenizer=_PromptTokenizer(),
        mm_processor_kwargs=_mm_kwargs(),
    )
    assert len(outputs) == 1
    assert outputs[0].startswith("<|img_start|>")


def test_prompt_equivalence_uncached_and_cached(monkeypatch, tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True)
    mm_kwargs = _mm_kwargs()
    image = _make_image()

    monkeypatch.delenv("VLLM_APERTUS_IMAGE_TOKEN_CACHE_DIR", raising=False)
    uncached_tokenizer = ApertusImageTokenizer()
    uncached_fake_vision = _FakeVisionTokenizer()
    monkeypatch.setattr(
        uncached_tokenizer,
        "load_vision_tokenizer",
        lambda _kwargs: uncached_fake_vision,
    )
    uncached_prompt = uncached_tokenizer.encode_images(
        [image],
        tokenizer=_PromptTokenizer(),
        mm_processor_kwargs=mm_kwargs,
    )[0]

    monkeypatch.setenv("VLLM_APERTUS_IMAGE_TOKEN_CACHE_DIR", str(cache_dir))
    cached_tokenizer_first = ApertusImageTokenizer()
    cached_fake_vision_first = _FakeVisionTokenizer()
    monkeypatch.setattr(
        cached_tokenizer_first,
        "load_vision_tokenizer",
        lambda _kwargs: cached_fake_vision_first,
    )
    cached_prompt_first = cached_tokenizer_first.encode_images(
        [image],
        tokenizer=_PromptTokenizer(),
        mm_processor_kwargs=mm_kwargs,
    )[0]

    cached_tokenizer_second = ApertusImageTokenizer()
    cached_fake_vision_second = _FakeVisionTokenizer()
    monkeypatch.setattr(
        cached_tokenizer_second,
        "load_vision_tokenizer",
        lambda _kwargs: cached_fake_vision_second,
    )
    cached_prompt_second = cached_tokenizer_second.encode_images(
        [image],
        tokenizer=_PromptTokenizer(),
        mm_processor_kwargs=mm_kwargs,
    )[0]

    assert uncached_prompt == cached_prompt_first == cached_prompt_second
    assert cached_fake_vision_second.encode_calls == 0


def test_cache_stats_expose_hits_misses_and_guard_validation(monkeypatch, tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True)
    monkeypatch.setenv("VLLM_APERTUS_IMAGE_TOKEN_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("VLLM_APERTUS_IMAGE_TOKEN_CACHE_COLLISION_GUARD", "1")

    tokenizer = ApertusImageTokenizer()
    fake_vision = _FakeVisionTokenizer()
    monkeypatch.setattr(tokenizer, "load_vision_tokenizer", lambda _kwargs: fake_vision)

    _ = tokenizer.encode_images(
        [_make_image(), _make_image()],
        tokenizer=_PromptTokenizer(),
        mm_processor_kwargs=_mm_kwargs(),
    )
    cache = tokenizer._image_prompt_cache
    assert isinstance(cache, ApertusImageTokenizationCache)
    stats = cache.get_stats()
    assert stats["memory_hits"] >= 1
    assert stats["misses"] >= 1
    assert stats["collision_guard_validation_success"] >= 1
    assert "sqlite_hits" in stats
    assert "sqlite_read_time_ms" in stats


def test_insert_or_ignore_preserves_first_value(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True)

    cache = ApertusImageTokenizationCache(
        ApertusImageTokenCacheConfig(
            cache_dir=cache_dir,
            collision_guard=False,
            memory_cache_size=8,
            sqlite_busy_timeout_ms=5000,
            sqlite_mmap_size=1024 * 1024,
            debug_logging=False,
            disabled_reason=None,
        )
    )
    cache.put("k", "first")
    cache.put("k", "second")
    cache.close()

    conn = sqlite3.connect(str(_sqlite_db_path(cache_dir)))
    try:
        value = conn.execute(
            "SELECT value FROM image_token_cache WHERE mode='normal' AND cache_key='k';"
        ).fetchone()
        assert value is not None
        assert bytes(value[0]) == b"first"
    finally:
        conn.close()


def test_preload_loads_active_mode_only(monkeypatch, tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True)

    monkeypatch.setenv("VLLM_APERTUS_IMAGE_TOKEN_CACHE_DIR", str(cache_dir))
    monkeypatch.delenv(
        "VLLM_APERTUS_IMAGE_TOKEN_CACHE_COLLISION_GUARD",
        raising=False,
    )
    tokenizer = ApertusImageTokenizer()
    fake_vision = _FakeVisionTokenizer()
    monkeypatch.setattr(tokenizer, "load_vision_tokenizer", lambda _kwargs: fake_vision)
    _ = tokenizer.encode_images(
        [_make_image()],
        tokenizer=_PromptTokenizer(),
        mm_processor_kwargs=_mm_kwargs(),
    )
    assert tokenizer._image_prompt_cache is not None
    tokenizer._image_prompt_cache.close()

    monkeypatch.setenv("VLLM_APERTUS_IMAGE_TOKEN_CACHE_COLLISION_GUARD", "1")
    tokenizer_guard = ApertusImageTokenizer()
    fake_vision_guard = _FakeVisionTokenizer()
    monkeypatch.setattr(
        tokenizer_guard,
        "load_vision_tokenizer",
        lambda _kwargs: fake_vision_guard,
    )
    _ = tokenizer_guard.encode_images(
        [_make_image()],
        tokenizer=_PromptTokenizer(),
        mm_processor_kwargs=_mm_kwargs(),
    )
    assert tokenizer_guard._image_prompt_cache is not None
    tokenizer_guard._image_prompt_cache.close()

    monkeypatch.delenv(
        "VLLM_APERTUS_IMAGE_TOKEN_CACHE_COLLISION_GUARD",
        raising=False,
    )
    tokenized = ApertusImageTokenizer()
    fake_vision2 = _FakeVisionTokenizer()
    monkeypatch.setattr(tokenized, "load_vision_tokenizer", lambda _kwargs: fake_vision2)
    outputs = tokenized.encode_images(
        [_make_image()],
        tokenizer=_PromptTokenizer(),
        mm_processor_kwargs=_mm_kwargs(),
    )
    assert len(outputs) == 1
    assert fake_vision2.encode_calls == 0
    cache = tokenized._image_prompt_cache
    assert cache is not None
    stats = cache.get_stats()
    assert stats["preload_rows_loaded"] >= 1
    assert stats["memory_hits"] >= 1
    assert stats["sqlite_hits"] == 0
    assert stats["preload_failures"] == 0


def test_preload_ignores_memory_cap_and_loads_all_rows(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True)

    writer = ApertusImageTokenizationCache(
        ApertusImageTokenCacheConfig(
            cache_dir=cache_dir,
            collision_guard=False,
            memory_cache_size=1,
            sqlite_busy_timeout_ms=5000,
            sqlite_mmap_size=1024 * 1024,
            debug_logging=False,
            disabled_reason=None,
        )
    )
    for idx in range(20):
        writer.put(f"k-{idx}", f"v-{idx}")
    writer.close()

    reader = ApertusImageTokenizationCache(
        ApertusImageTokenCacheConfig(
            cache_dir=cache_dir,
            collision_guard=False,
            memory_cache_size=1,
            sqlite_busy_timeout_ms=5000,
            sqlite_mmap_size=1024 * 1024,
            debug_logging=False,
            disabled_reason=None,
        )
    )
    stats = reader.get_stats()
    assert stats["preload_rows_loaded"] >= 20
    for idx in range(20):
        assert reader.get(f"k-{idx}") == f"v-{idx}"
    stats_after = reader.get_stats()
    assert stats_after["memory_hits"] >= 20
    assert stats_after["sqlite_hits"] == 0
    reader.close()


def test_multi_process_shared_sqlite_read_write_stress(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True)

    queue: mp.Queue = mp.Queue()
    procs = [
        mp.Process(target=_mp_worker, args=(str(cache_dir), idx, queue))
        for idx in range(4)
    ]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join(timeout=30)
        assert proc.exitcode == 0

    results = [queue.get(timeout=5) for _ in procs]
    assert results.count("ok") == len(procs)

    conn = sqlite3.connect(str(_sqlite_db_path(cache_dir)))
    try:
        count = conn.execute(
            "SELECT COUNT(*) FROM image_token_cache WHERE mode='normal';"
        ).fetchone()
        assert count is not None
        assert int(count[0]) <= 20
        assert int(count[0]) > 0
    finally:
        conn.close()


def test_busy_timeout_error_falls_back_without_hanging(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True)

    config = ApertusImageTokenCacheConfig(
        cache_dir=cache_dir,
        collision_guard=False,
        memory_cache_size=8,
        sqlite_busy_timeout_ms=200,
        sqlite_mmap_size=0,
        debug_logging=False,
        disabled_reason=None,
    )
    cache = ApertusImageTokenizationCache(config)
    cache.put("seed", "value")

    hold_conn = sqlite3.connect(
        str(_sqlite_db_path(cache_dir)),
        timeout=1.0,
        check_same_thread=False,
    )
    hold_conn.execute("PRAGMA journal_mode=WAL;").fetchone()
    hold_conn.execute("BEGIN EXCLUSIVE;")
    hold_conn.execute(
        "INSERT OR REPLACE INTO image_token_cache (mode, cache_key, value) "
        "VALUES ('normal', '__lock__', '1');"
    )

    lock_released = threading.Event()

    def _release_lock():
        time.sleep(0.8)
        hold_conn.commit()
        hold_conn.close()
        lock_released.set()

    releaser = threading.Thread(target=_release_lock, daemon=True)
    releaser.start()

    start = time.perf_counter()
    cache.put("key-under-lock", "value-under-lock")
    elapsed = time.perf_counter() - start
    stats = cache.get_stats()
    cache.close()

    lock_released.wait(timeout=5)
    releaser.join(timeout=5)

    assert elapsed < 1.0
    assert stats["sqlite_write_failure"] >= 1
    assert stats["sqlite_busy_timeouts"] >= 1
