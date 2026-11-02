"""Read GIF: preserve raw IM, deliver derived descriptions only to request context."""

from __future__ import annotations

import asyncio
from typing import Any

from maibot_sdk import CONFIG_RELOAD_SCOPE_SELF, HookHandler, MaiBotPlugin
from maibot_sdk.types import ErrorPolicy, HookMode, HookOrder

from .read_animation.config import ReadAnimationConfig
from .read_animation.context import DerivedContext
from .read_animation.pipeline import CACHE_FILE_NAME, AnimationProcessor

HOOK_TIMEOUT_MS = 45000


class ReadGifPlugin(MaiBotPlugin):
    config_model = ReadAnimationConfig

    def __init__(self):
        super().__init__()
        self._processor = None
        self._lifecycle = asyncio.Lock()
        self._generation = 0
        self._derived = DerivedContext()

    async def on_load(self):
        async with self._lifecycle:
            self._generation += 1
            self._derived.clear()
            if self._processor is not None:
                await self._processor.close()
            self._processor = AnimationProcessor(
                config=self.config,
                llm_generate=self.ctx.llm.generate,
                logger=self.ctx.logger,
                cache_path=self.ctx.paths.data_dir / CACHE_FILE_NAME,
            )
            await self._processor.initialize()
        self.ctx.logger.info("GIF 动图识别插件已加载（维护版0.2.1，门卫准入后识别）")

    async def on_unload(self):
        async with self._lifecycle:
            self._generation += 1
            self._derived.clear()
            processor, self._processor = self._processor, None
            if processor is not None:
                await processor.close()
        self.ctx.logger.info("GIF 动图识别插件已卸载")

    async def on_config_update(
        self, scope: str, config_data: dict[str, object], version: str
    ):
        del config_data, version
        async with self._lifecycle:
            if scope != CONFIG_RELOAD_SCOPE_SELF or self._processor is None:
                return
            self._generation += 1
            self._derived.clear()
            await self._processor.update_config(self.config)
        self.ctx.logger.info("GIF 动图识别配置已更新")

    @HookHandler(
        "chat.receive.before_process",
        name="animation_frame_sheet",
        description="分析入站动图并保存短期派生描述，不改写用户原消息。",
        mode=HookMode.BLOCKING,
        order=HookOrder.NORMAL,
        timeout_ms=HOOK_TIMEOUT_MS,
        error_policy=ErrorPolicy.SKIP,
    )
    async def handle_before_process(
        self, message: dict[str, Any] | None = None, **kwargs
    ):
        processor, generation = self._processor, self._generation
        if processor is None or not isinstance(message, dict):
            return None
        try:
            notes = []
            result = await processor.process_message(message, annotations=notes)
        except Exception:
            self.ctx.logger.warning("处理动图失败，保持原消息；错误正文已省略")
            return None
        if (
            result is not None
            and generation == self._generation
            and processor is self._processor
        ):
            self._derived.record(message, notes)
        return None  # No generated text is returned to the raw IM/storage/learning pipeline.

    def _request_context(self, **kwargs):
        if (
            self._processor is None
            or not self.config.plugin.enabled
            or self.config.trigger.trigger_mode == "off"
        ):
            return None
        items = self._derived.inject(
            kwargs.get("items"),
            session_id=kwargs.get("session_id", ""),
            tool_definitions=kwargs.get("tool_definitions"),
            item_schema_version=kwargs.get("item_schema_version"),
            reply_message_id=kwargs.get("reply_message_id", ""),
        )
        if items is not None:
            return {"action": "continue", "modified_kwargs": {**kwargs, "items": items}}
        return None

    @HookHandler(
        "maisaka.planner.before_request",
        name="animation_derived_planner",
        mode=HookMode.BLOCKING,
        order=HookOrder.LATE,
        timeout_ms=1000,
        error_policy=ErrorPolicy.SKIP,
    )
    async def planner_context(self, **kwargs):
        return self._request_context(**kwargs)

    @HookHandler(
        "maisaka.replyer.before_model_request",
        name="animation_derived_replyer",
        mode=HookMode.BLOCKING,
        order=HookOrder.LATE,
        timeout_ms=1000,
        error_policy=ErrorPolicy.SKIP,
    )
    async def replyer_context(self, **kwargs):
        return self._request_context(**kwargs)


def create_plugin() -> MaiBotPlugin:
    return ReadGifPlugin()
