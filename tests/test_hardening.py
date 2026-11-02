"""Offline regression for the concrete concurrency, lifecycle and provenance bugs."""

import asyncio
import copy
import logging
import threading
import time
import unittest
from unittest.mock import patch

from read_animation.cache import DescriptionCache
from read_animation.config import ReadAnimationConfig
from read_animation.context import DerivedContext, display_id
from read_animation.frames import AnimationReadError, AnimationReader
from read_animation.pipeline import AnimationProcessor
from tests.fakes import FakeLlm, EMPTY_RESPONSE_RESULT
from tests.image_fixtures import make_animated_gif, make_image_component, make_message
from tests.temp_utils import cache_file, reset_cache_file


def item(session="s", message="m"):
    return {
        "item_type": "UserMessageItem",
        "meta": {
            "item_id": "source",
            "logical_turn_id": None,
            "timestamp": "2026-11-02T12:00:00",
        },
        "parts": [
            {
                "type": "text",
                "text": f'<message msg_id="{message}" session_id="{session}">\n',
            }
        ],
    }


class CacheTests(unittest.IsolatedAsyncioTestCase):
    def store(self):
        return DescriptionCache(
            cache_file("hardening.json"),
            max_entries=1000,
            retention_days=1,
            logger=logging.getLogger("test"),
        )

    async def asyncSetUp(self):
        reset_cache_file(cache_file("hardening.json"))

    async def test_ttl_checked_on_hit(self):
        store = self.store()
        await store.load()
        with patch("read_animation.cache.time.time", return_value=1_000_000):
            await store.put("k", header="GIF", description="desc", kind="gif")
        with patch("read_animation.cache.time.time", return_value=2_000_000):
            self.assertIsNone(await store.get("k"))

    async def test_unknown_survives_reload_and_holds_capacity(self):
        store = self.store()
        await store.load()
        self.assertTrue(await store.begin("k", 1))
        other = self.store()
        await other.load()
        self.assertFalse(await other.begin("k", 1))
        self.assertFalse(await other.begin("different", 1))
        self.assertEqual(other._attempts, {"k": "unknown"})

    async def test_completed_request_frees_capacity(self):
        store = self.store()
        await store.load()
        self.assertTrue(await store.begin("k", 1))
        await store.end("k", known=True)
        self.assertTrue(await store.begin("next", 1))

    async def test_corrupt_file_is_not_overwritten(self):
        path = cache_file("hardening.json")
        path.write_bytes(b"{corrupt")
        store = self.store()
        await store.load()
        self.assertFalse(await store.begin("k", 1))
        await store.put("k", header="GIF", description="desc", kind="gif")
        await store.close()
        self.assertEqual(path.read_bytes(), b"{corrupt")

    async def test_put_persists_without_unload(self):
        store = self.store()
        await store.load()
        await store.put("k", header="GIF", description="desc", kind="gif")
        other = self.store()
        await other.load()
        self.assertEqual((await other.get("k"))["description"], "desc")


class WorkTests(unittest.IsolatedAsyncioTestCase):
    async def test_simultaneous_distinct_jobs_reserve_decode_slots_atomically(self):
        cfg = ReadAnimationConfig()
        cfg.perf.max_concurrent = 2
        processor = await self.build(cfg=cfg)
        release = threading.Event()

        def blocked(*args, **kwargs):
            release.wait(5)
            return None

        with patch("read_animation.pipeline.analyze_animation", side_effect=blocked):
            tasks = [
                asyncio.create_task(
                    processor.process_message(
                        make_message(
                            [make_image_component(make_animated_gif(frame_count=n))]
                        )
                    )
                )
                for n in (3, 4, 5, 6)
            ]
            try:
                await asyncio.sleep(0.05)
                self.assertEqual(len(processor._decodes), 2)
            finally:
                release.set()
                await asyncio.gather(*tasks)

    async def test_timeout_keeps_actual_request_and_late_success_is_cached(self):
        entered, release = asyncio.Event(), asyncio.Event()
        calls = 0

        async def fake(**kwargs):
            nonlocal calls
            calls += 1
            entered.set()
            await release.wait()
            return {"success": True, "response": "late"}

        cfg = ReadAnimationConfig()
        cfg.vlm.timeout_ms = (
            1000  # Smallest validated production value, no config bypass.
        )
        processor = await self.build(fake, cfg)
        first = await processor.process_message(self.message())
        self.assertIsNotNone(first)
        self.assertTrue(entered.is_set())
        self.assertEqual(len(processor._requests), 1)
        self.assertEqual(list(processor._cache._attempts.values()), ["started"])
        await processor.process_message(self.message())
        self.assertEqual(calls, 1)
        release.set()
        await asyncio.gather(*list(processor._requests))
        result = await processor.process_message(self.message())
        self.assertIn("late", result["raw_message"][-1]["data"])
        self.assertEqual(calls, 1)
        self.assertFalse(processor._cache._attempts)

    async def test_real_sdk_llm_capability_shapes(self):
        from maibot_sdk.context import PluginContext

        calls = []

        async def rpc(method, plugin_id, payload, **kwargs):
            calls.append((method, plugin_id, payload))
            self.assertEqual(method, "cap.call")
            self.assertEqual(payload["capability"], "llm.generate")
            self.assertEqual(payload["args"]["task_name"], "vlm")
            return {"success": True, "response": "actual SDK shape"}

        ctx = PluginContext("github.fabiocsa.read-gif", rpc_call=rpc)
        processor = await self.build(ctx.llm.generate)
        result = await processor.process_message(self.message())
        self.assertIn("actual SDK shape", result["raw_message"][-1]["data"])
        self.assertEqual(len(calls), 1)

    async def test_pending_bound_and_no_post_close_admission(self):
        cfg = ReadAnimationConfig()
        cfg.perf.max_pending = 1
        processor = await self.build(cfg=cfg)
        processor._messages = 1
        self.assertIsNone(await processor.process_message(self.message()))
        processor._messages = 0
        await processor.close()
        self.assertIsNone(await processor.process_message(self.message()))

    async def build(self, fake=None, cfg=None):
        reset_cache_file(cache_file("work.json"))
        processor = AnimationProcessor(
            config=cfg or ReadAnimationConfig(),
            llm_generate=fake or FakeLlm(),
            logger=logging.getLogger("test"),
            cache_path=cache_file("work.json"),
        )
        await processor.initialize()
        self.addAsyncCleanup(processor.close)
        return processor

    def message(self):
        return make_message([make_image_component(make_animated_gif())])

    async def test_concurrent_same_image_is_single_flight(self):
        entered, release = asyncio.Event(), asyncio.Event()
        calls = []

        async def fake(**kwargs):
            calls.append(kwargs)
            entered.set()
            await release.wait()
            return {"success": True, "response": "synthetic"}

        processor = await self.build(fake)
        first = asyncio.create_task(processor.process_message(self.message()))
        await entered.wait()
        second = asyncio.create_task(processor.process_message(self.message()))
        await asyncio.sleep(0.02)
        release.set()
        self.assertTrue(all(await asyncio.gather(first, second)))
        self.assertEqual(len(calls), 1)

    async def test_no_default_empty_retry(self):
        fake = FakeLlm(responses=[EMPTY_RESPONSE_RESULT])
        processor = await self.build(fake)
        result = await processor.process_message(self.message())
        self.assertIsNotNone(result)
        self.assertEqual(len(fake.calls), 1)
        self.assertEqual(fake.calls[0]["max_tokens"], 4096)

    async def test_cancelled_waiter_does_not_cancel_shared_work(self):
        entered, release = asyncio.Event(), asyncio.Event()
        calls = 0

        async def fake(**kwargs):
            nonlocal calls
            calls += 1
            entered.set()
            await release.wait()
            return {"success": True, "response": "synthetic"}

        processor = await self.build(fake)
        first = asyncio.create_task(processor.process_message(self.message()))
        await entered.wait()
        second = asyncio.create_task(processor.process_message(self.message()))
        await asyncio.sleep(0.02)
        first.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await first
        release.set()
        self.assertIsNotNone(await second)
        self.assertEqual(calls, 1)

    async def test_config_change_cannot_commit_old_result(self):
        entered, release = asyncio.Event(), asyncio.Event()

        async def fake(**kwargs):
            entered.set()
            await release.wait()
            return {"success": True, "response": "stale"}

        processor = await self.build(fake)
        first = asyncio.create_task(processor.process_message(self.message()))
        await entered.wait()
        cfg = ReadAnimationConfig()
        cfg.plugin.enabled = False
        await processor.update_config(cfg)
        self.assertIsNone(await first)
        release.set()
        await asyncio.gather(*list(processor._requests))
        self.assertFalse(processor._cache._entries)
        self.assertIsNone(await processor.process_message(self.message()))

    async def test_unload_marks_sent_request_unknown(self):
        entered = asyncio.Event()

        async def fake(**kwargs):
            entered.set()
            await asyncio.Event().wait()

        processor = await self.build(fake)
        first = asyncio.create_task(processor.process_message(self.message()))
        await entered.wait()
        await processor.close()
        self.assertIsNone(await first)
        self.assertEqual(list(processor._cache._attempts.values()), ["unknown"])

    async def test_decode_thread_keeps_capacity_until_finished(self):
        entered, release = threading.Event(), threading.Event()
        cfg = ReadAnimationConfig()
        cfg.perf.max_concurrent = 1
        processor = await self.build(cfg=cfg)

        def decode(*args, **kwargs):
            entered.set()
            release.wait(5)
            return None

        with patch("read_animation.pipeline.analyze_animation", side_effect=decode):
            first = asyncio.create_task(processor.process_message(self.message()))
            await asyncio.to_thread(entered.wait, 2)
            await processor.update_config(cfg)
            self.assertIsNone(await first)
            self.assertEqual(len(processor._decodes), 1)
            self.assertIsNone(await processor.process_message(self.message()))
            release.set()
            await asyncio.gather(*list(processor._decodes))

    async def test_encoded_size_guard_before_base64_allocation(self):
        processor = await self.build()
        message = self.message()
        message["raw_message"][0]["binary_data_base64"] = "A" * (12 * 1024 * 1024)
        with patch(
            "read_animation.pipeline.base64.b64decode",
            side_effect=AssertionError("must not decode"),
        ):
            self.assertIsNone(await processor.process_message(message))

    async def test_adapter_at_and_raw_at_are_detected_before_host_flags(self):
        cfg = ReadAnimationConfig()
        cfg.trigger.trigger_mode = "mentioned"
        processor = await self.build(cfg=cfg)
        message = self.message()
        message["account_id"] = "bot"
        message["raw_message"].insert(
            0, {"type": "at", "data": {"target_user_id": "bot"}}
        )
        self.assertTrue(processor._should_process_message(message))
        message["raw_message"][0]["data"]["target_user_id"] = "other"
        self.assertFalse(processor._should_process_message(message))
        message["message_info"]["additional_config"]["at_bot"] = True
        self.assertTrue(processor._should_process_message(message))


class ContextTests(unittest.TestCase):
    def setUp(self):
        self.context = DerivedContext()
        self.context.record({"session_id": "s", "message_id": "m"}, ["derived"])

    def test_inject_preserves_original_items_and_is_idempotent(self):
        items = [item()]
        original = copy.deepcopy(items)
        result = self.context.inject(items, session_id="s", item_schema_version=1)
        self.assertEqual(items, original)
        self.assertEqual(len(result), 2)
        self.assertIn("非用户原话", result[-1]["parts"][0]["text"])
        self.assertIsNone(
            self.context.inject(result, session_id="s", item_schema_version=1)
        )

    def test_cross_scope_and_body_spoof_are_rejected(self):
        self.assertIsNone(
            self.context.inject(
                [item("other")], session_id="other", item_schema_version=1
            )
        )
        value = item()
        value["parts"][0]["text"] = "user says\n" + value["parts"][0]["text"]
        self.assertIsNone(
            self.context.inject([value], session_id="s", item_schema_version=1)
        )

    def test_focus_requires_authoritative_reply_schema(self):
        definitions = [
            {
                "type": "function",
                "function": {
                    "name": "reply",
                    "parameters": {
                        "required": ["session_id"],
                        "properties": {"session_id": {"type": "string", "enum": ["s"]}},
                    },
                },
            }
        ]
        self.assertIsNone(self.context.inject([item()], item_schema_version=1))
        self.assertIsNotNone(
            self.context.inject(
                [item()], item_schema_version=1, tool_definitions=definitions
            )
        )
        self.assertIsNone(
            self.context.inject(
                [item("other")], item_schema_version=1, tool_definitions=definitions
            )
        )

    def test_replyer_target_uses_raw_host_id(self):
        self.assertIsNotNone(
            self.context.inject(
                [item(message="not-target")],
                session_id="s",
                item_schema_version=1,
                reply_message_id="m",
            )
        )
        self.assertIsNone(
            self.context.inject(
                [item(message="not-target")],
                session_id="other",
                item_schema_version=1,
                reply_message_id="m",
            )
        )

    def test_long_id_alias_and_expiry(self):
        raw = "long-platform-message-id"
        self.context.record({"session_id": "s", "message_id": raw}, ["long"])
        self.assertIsNotNone(
            self.context.inject(
                [item(message=display_id(raw))], session_id="s", item_schema_version=1
            )
        )
        with patch(
            "read_animation.context.time.monotonic",
            return_value=time.monotonic() + 1000,
        ):
            self.assertIsNone(
                self.context.inject([item()], session_id="s", item_schema_version=1)
            )


class DecodeTests(unittest.TestCase):
    def test_apng_poster_is_not_an_animation_frame(self):
        from io import BytesIO
        from PIL import Image

        stream = BytesIO()
        poster = Image.new("RGBA", (32, 32), "red")
        poster.save(
            stream,
            format="PNG",
            save_all=True,
            default_image=True,
            append_images=[
                Image.new("RGBA", (32, 32), "blue"),
                Image.new("RGBA", (32, 32), "green"),
            ],
            duration=[100, 200],
            loop=0,
        )
        reader = AnimationReader(stream.getvalue(), "apng", max_scan_frames=3000)
        try:
            self.assertEqual(reader.meta.frame_count, 2)
            self.assertEqual(reader.meta.total_duration_ms, 300)
            self.assertEqual(
                reader.render_frame(0, max_edge=32, background="white").getpixel(
                    (0, 0)
                ),
                (0, 0, 255),
            )
        finally:
            reader.close()

    def test_actual_sheet_budget_is_enforced(self):
        from read_animation.sheet import build_frame_sheet

        reader = AnimationReader(make_animated_gif(), "gif", max_scan_frames=3000)
        try:
            with self.assertRaises(ValueError):
                build_frame_sheet(
                    reader,
                    reader.sample(8),
                    max_edge=512,
                    columns=4,
                    background="white",
                    max_sheet_pixels=10,
                )
        finally:
            reader.close()

    def test_pixel_and_deadline_budget(self):
        with self.assertRaises(AnimationReadError):
            AnimationReader(
                make_animated_gif(), "gif", max_scan_frames=3000, max_decode_pixels=100
            )
        with self.assertRaises(AnimationReadError):
            AnimationReader(
                make_animated_gif(), "gif", max_scan_frames=3000, deadline=0
            )

    def test_time_weighted_sampling(self):
        reader = AnimationReader(make_animated_gif(), "gif", max_scan_frames=3000)
        try:
            reader._durations = [1000] + [10] * 7
            reader._prefix_ms = [0, 1000, 1010, 1020, 1030, 1040, 1050, 1060, 1070]
            self.assertEqual([sample.index for sample in reader.sample(4)], [0, 7])
        finally:
            reader.close()
