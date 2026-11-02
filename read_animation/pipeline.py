"""动图识别管线：检测、抽帧、视觉描述与消息注入。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import asyncio
import base64
import hashlib
import logging
import time
import json
from threading import Event

from .cache import DescriptionCache
from .config import FramesSection, ReadAnimationConfig
from .frames import AnimationMeta, AnimationReadError, AnimationReader
from .sheet import build_frame_sheet, fit_frame_edge
from .sniff import AnimationKind, detect_animation_kind
from .vlm import (
    PROMPT_VERSION,
    LlmGenerate,
    VlmEmptyResponseError,
    describe_frame_sheet,
)

HANDLED_COMPONENT_TYPES = frozenset({"image", "emoji"})
MAX_FORWARD_DEPTH = 3
MESSAGE_BUDGET_SECONDS = 40.0
CACHE_FILE_NAME = "description_cache.json"
CAVEAT_TEXT = "（基于等间隔采样帧生成，帧间细节可能有遗漏）"
EMPTY_RESPONSE_RETRY_MIN_TOKENS = 2048
EMPTY_RESPONSE_RETRY_MAX_TOKENS = 8192
SKIP_REASON_TEXT = {
    "too_many_frames": "帧数超出解析上限，未做视觉识别",
    "too_long": "动图时长超出上限，未做视觉识别",
}


class NotAnimatedError(AnimationReadError):
    """图片只有一帧，不属于动图。"""


@dataclass(frozen=True)
class AnimationAnalysis:
    """同步分析阶段的产物。"""

    meta: AnimationMeta
    sheet_bytes: bytes
    sheet_media_type: str
    sampled_count: int
    frame_edge: int = 0
    skip_reason: str = ""


def analyze_animation(
    data: bytes,
    kind: AnimationKind,
    frames_config: FramesSection,
    *,
    build_sheet: bool = True,
    deadline: float = float("inf"),
    cancelled: Event | None = None,
) -> AnimationAnalysis:
    """解码动图并按需生成帧板（同步函数，应在线程池中调用）。

    Args:
        data: 动图文件字节。
        kind: 嗅探得到的动图格式。
        frames_config: 抽帧与帧板配置。
        build_sheet: 是否生成帧板；``False`` 时只收集元信息。

    Returns:
        AnimationAnalysis: 分析结果；体量超限时 ``skip_reason`` 非空。

    Raises:
        NotAnimatedError: 图片只有一帧时抛出。
        AnimationReadError: 动图解码失败时抛出。
    """

    reader = AnimationReader(
        data,
        kind,
        max_scan_frames=int(frames_config.max_total_frames),
        max_decode_pixels=frames_config.max_decode_pixels,
        deadline=deadline,
        cancelled=cancelled,
    )
    try:
        meta = reader.meta
        if meta.frame_count <= 1:
            raise NotAnimatedError("图片只有一帧")

        if meta.scan_truncated or meta.frame_count > int(
            frames_config.max_total_frames
        ):
            return AnimationAnalysis(
                meta=meta,
                sheet_bytes=b"",
                sheet_media_type="",
                sampled_count=0,
                skip_reason="too_many_frames",
            )

        if meta.total_duration_ms > int(frames_config.max_duration_ms):
            return AnimationAnalysis(
                meta=meta,
                sheet_bytes=b"",
                sheet_media_type="",
                sampled_count=0,
                skip_reason="too_long",
            )

        if not build_sheet:
            return AnimationAnalysis(
                meta=meta, sheet_bytes=b"", sheet_media_type="", sampled_count=0
            )

        samples = reader.sample(int(frames_config.max_frames))
        if not samples:
            raise NotAnimatedError("未能抽取到有效帧")

        frame_edge = fit_frame_edge(
            width=meta.width,
            height=meta.height,
            frame_count=len(samples),
            columns=int(frames_config.columns),
            max_edge=int(frames_config.max_edge),
            max_sheet_pixels=int(frames_config.max_sheet_pixels),
        )
        sheet_bytes, sheet_media_type = build_frame_sheet(
            reader,
            samples,
            max_edge=frame_edge,
            columns=int(frames_config.columns),
            background=str(frames_config.background),
            max_sheet_pixels=frames_config.max_sheet_pixels,
        )
        return AnimationAnalysis(
            meta=meta,
            sheet_bytes=sheet_bytes,
            sheet_media_type=sheet_media_type,
            sampled_count=len(samples),
            frame_edge=frame_edge,
        )
    finally:
        reader.close()


class AnimationProcessor:
    """把一条入站消息里的动图替换成"原组件 + 描述文本"的处理器。"""

    def __init__(
        self,
        *,
        config: ReadAnimationConfig,
        llm_generate: LlmGenerate,
        logger: logging.Logger,
        cache_path: Path | None,
    ) -> None:
        """初始化处理器。

        Args:
            config: 插件配置。
            llm_generate: 宿主提供的 ``ctx.llm.generate``。
            logger: 插件日志器。
            cache_path: 缓存文件路径；``None`` 表示不启用缓存。
        """

        self._config = config.model_copy(deep=True)
        self._llm_generate = llm_generate
        self._logger = logger
        self._cache_path = cache_path
        self._cache = self._build_cache(cache_path, config)
        self._epoch = 0
        self._ready = False
        self._closed = False
        self._messages = 0
        self._jobs: dict[str, asyncio.Task] = {}
        self._waiters: dict[str, int] = {}
        self._decodes: set[asyncio.Task] = set()
        self._requests: set[asyncio.Task] = set()
        self._lifecycle = asyncio.Lock()

    async def initialize(self) -> None:
        """加载缓存。"""

        async with self._lifecycle:
            if self._closed:
                return
            await self._cache.load()
            self._ready = True

    async def close(self) -> None:
        """保存缓存并释放资源。"""

        async with self._lifecycle:
            self._closed = True
            self._ready = False
            self._epoch += 1
            await self._cancel_jobs()
            for task in list(self._requests):
                task.cancel()
            if self._requests:
                await asyncio.gather(*list(self._requests), return_exceptions=True)
            await self._cache.close()

    async def update_config(self, config: ReadAnimationConfig) -> None:
        """配置热更新后重建缓存与并发控制。

        Args:
            config: 新的插件配置。
        """

        async with self._lifecycle:
            if self._closed:
                return
            self._ready = False
            self._epoch += 1
            await self._cancel_jobs()
            self._config = config.model_copy(deep=True)
            # Keep the same store/attempts and live decode/request accounting.
            await self._cache.configure(
                max_entries=config.cache.max_entries,
                retention_days=config.cache.retention_days,
            )
            self._ready = True

    async def _cancel_jobs(self):
        tasks = list(self._jobs.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def _valid(self, epoch):
        return self._ready and not self._closed and epoch == self._epoch

    @staticmethod
    def _track(task, collection):
        collection.add(task)

        def completed(done):
            collection.discard(done)
            if not done.cancelled():
                done.exception()  # retrieve errors even after all waiters have left

        task.add_done_callback(completed)
        return task

    def _build_cache(
        self, cache_path: Path | None, config: ReadAnimationConfig
    ) -> DescriptionCache | None:
        """按配置构造缓存对象。"""

        return DescriptionCache(
            cache_path,
            max_entries=int(config.cache.max_entries),
            retention_days=int(config.cache.retention_days),
            logger=self._logger,
        )

    async def process_message(
        self, message: dict[str, Any], *, annotations: list[str] | None = None
    ) -> dict[str, Any] | None:
        """处理一条入站消息。

        Args:
            message: 插件 Hook 收到的消息字典。

        Returns:
            Optional[Dict[str, Any]]: 注入描述后的消息副本；无需修改时返回 ``None``。
        """

        cfg = self._config
        epoch = self._epoch
        if (
            not self._valid(epoch)
            or not cfg.plugin.enabled
            or cfg.trigger.trigger_mode == "off"
        ):
            return None
        if not self._should_process_message(message):
            return None

        raw_message = message.get("raw_message")
        if not isinstance(raw_message, list) or not raw_message:
            return None

        if self._messages >= cfg.perf.max_pending:
            return None
        self._messages += 1
        deadline = time.monotonic() + MESSAGE_BUDGET_SECONDS
        budget = {
            "remaining": max(0, int(cfg.trigger.max_images_per_message)),
            "components": 128,
        }
        try:
            new_raw_message = await self._process_component_list(
                raw_message,
                budget,
                deadline,
                depth=0,
                cfg=cfg,
                epoch=epoch,
                annotations=annotations,
            )
        finally:
            self._messages -= 1
        if not self._valid(epoch):
            return None
        if new_raw_message is None:
            return None

        mutated = dict(message)
        mutated["raw_message"] = new_raw_message
        return mutated

    def _should_process_message(self, message: Mapping[str, Any]) -> bool:
        """按触发范围判断是否处理该消息。"""

        trigger_mode = self._config.trigger.trigger_mode
        if trigger_mode == "all":
            return True
        if bool(message.get("is_mentioned")) or bool(message.get("is_at")):
            return True

        message_info = message.get("message_info")
        group_info = (
            message_info.get("group_info")
            if isinstance(message_info, Mapping)
            else None
        )
        additional = (
            message_info.get("additional_config", {})
            if isinstance(message_info, Mapping)
            else {}
        )
        # Adapter-owned flags exist before Host computes the top-level markers.
        if isinstance(additional, Mapping) and (
            additional.get("at_bot") or additional.get("is_mentioned")
        ):
            return True
        account = message.get("account_id")
        for component in message.get("raw_message", [])[:128]:
            if not isinstance(component, Mapping):
                continue
            if component.get("type") == "at" and isinstance(
                component.get("data"), Mapping
            ):
                if account and component["data"].get("target_user_id") == account:
                    return True
            if (
                component.get("type") == "reply"
                and account
                and component.get("target_message_sender_id") == account
            ):
                return True
        return not group_info

    async def _process_component_list(
        self,
        components: list[Any],
        budget: dict[str, int],
        deadline: float,
        *,
        depth: int,
        cfg: ReadAnimationConfig,
        epoch: int,
        annotations: list[str] | None,
    ) -> list[Any] | None:
        """递归处理消息组件列表，返回注入后的新列表或 ``None``。"""

        changed = False
        output: list[Any] = []
        for index, component in enumerate(components):
            if (
                budget["components"] <= 0
                or not self._valid(epoch)
                or time.monotonic() >= deadline
            ):
                # Preserve untouched source components without additional decode/model work.
                output.extend(components[index:])
                break
            budget["components"] -= 1
            if not isinstance(component, Mapping):
                output.append(component)
                continue

            component_type = str(component.get("type") or "")
            if component_type == "forward" and depth < MAX_FORWARD_DEPTH:
                rewritten_forward = await self._rewrite_forward(
                    component,
                    budget,
                    deadline,
                    depth + 1,
                    cfg=cfg,
                    epoch=epoch,
                    annotations=annotations,
                )
                if rewritten_forward is not None:
                    output.append(rewritten_forward)
                    changed = True
                    continue

            if component_type in HANDLED_COMPONENT_TYPES:
                injected_text = await self._build_injection(
                    component, budget, deadline, cfg=cfg, epoch=epoch
                )
                if injected_text:
                    output.append(component)
                    output.append({"type": "text", "data": injected_text})
                    if annotations is not None:
                        annotations.append(injected_text)
                    changed = True
                    continue

            output.append(component)

        return output if changed else None

    async def _rewrite_forward(
        self,
        component: Mapping[str, Any],
        budget: dict[str, int],
        deadline: float,
        depth: int,
        *,
        cfg: ReadAnimationConfig,
        epoch: int,
        annotations: list[str] | None,
    ) -> Mapping[str, Any] | None:
        """递归处理合并转发节点里的动图。"""

        nodes = component.get("data")
        if not isinstance(nodes, list):
            return None

        changed = False
        rewritten_nodes: list[Any] = []
        for node in nodes:
            if not isinstance(node, Mapping):
                rewritten_nodes.append(node)
                continue
            content = node.get("content")
            if not isinstance(content, list):
                rewritten_nodes.append(node)
                continue

            new_content = await self._process_component_list(
                content,
                budget,
                deadline,
                depth=depth,
                cfg=cfg,
                epoch=epoch,
                annotations=annotations,
            )
            if new_content is None:
                rewritten_nodes.append(node)
                continue

            rewritten_node = dict(node)
            rewritten_node["content"] = new_content
            rewritten_nodes.append(rewritten_node)
            changed = True

        if not changed:
            return None
        rewritten_component = dict(component)
        rewritten_component["data"] = rewritten_nodes
        return rewritten_component

    async def _build_injection(
        self,
        component: Mapping[str, Any],
        budget: dict[str, int],
        deadline: float,
        *,
        cfg: ReadAnimationConfig,
        epoch: int,
    ) -> str | None:
        """Bound admission before decode; coalesce simultaneous identical images."""
        component_type = str(component.get("type") or "")
        if component_type == "image" and not cfg.trigger.handle_images:
            return None
        if component_type == "emoji" and not cfg.trigger.handle_emojis:
            return None

        raw_base64 = component.get("binary_data_base64")
        if not isinstance(raw_base64, str) or not raw_base64:
            return None
        if (
            len(raw_base64) > ((cfg.frames.max_input_bytes + 2) // 3) * 4
            or not raw_base64.isascii()
        ):
            return None
        # The encoded content identity allows same-image admission before allocating decoded data.
        digest = hashlib.sha256(raw_base64.encode("ascii")).hexdigest()
        fingerprint = json.dumps(
            [PROMPT_VERSION, cfg.frames.model_dump(), cfg.vlm.model_dump()],
            sort_keys=True,
        )
        key = digest + ":" + hashlib.sha256(fingerprint.encode()).hexdigest()
        cached = await self._cache.get(key) if cfg.cache.enabled else None
        if cached is not None:
            return self._format_text(cached["header"], cached["description"], cfg=cfg)
        if not self._valid(epoch):
            return None
        job = self._jobs.get(key)
        if job is None:
            if (
                len(self._jobs) >= cfg.perf.max_pending
                or len(self._decodes) >= cfg.perf.max_concurrent
            ):
                return None
            can_describe = budget["remaining"] > 0
            # Reserve the per-message unique-image allowance before creating its shared job.
            if can_describe:
                budget["remaining"] -= 1
            job = asyncio.create_task(
                self._analyze_unique(
                    raw_base64, digest, key, cfg, epoch, deadline, can_describe
                )
            )
            self._jobs[key] = job

            def done(task):
                if self._jobs.get(key) is task:
                    self._jobs.pop(key, None)
                if not task.cancelled():
                    task.exception()

            job.add_done_callback(done)
        self._waiters[key] = self._waiters.get(key, 0) + 1
        try:
            result = await asyncio.wait_for(
                asyncio.shield(job), max(0.001, deadline - time.monotonic())
            )
            return result if self._valid(epoch) else None
        except asyncio.TimeoutError:
            return None
        except asyncio.CancelledError:
            if not self._valid(epoch):
                return None
            raise
        finally:
            count = self._waiters.get(key, 1) - 1
            if count:
                self._waiters[key] = count
            else:
                self._waiters.pop(key, None)
                if not job.done():
                    job.cancel()

    async def _analyze_unique(
        self, raw_base64, digest, key, cfg, epoch, deadline, can_describe
    ):
        # Several messages can pass the earlier admission check before their jobs
        # are scheduled. Recheck here, with no await between this and reservation.
        if not self._valid(epoch) or len(self._decodes) >= cfg.perf.max_concurrent:
            return None
        cancelled = Event()

        def decode():
            data = base64.b64decode(raw_base64, validate=True)
            if not data or len(data) > cfg.frames.max_input_bytes:
                return None
            if base64.b64encode(data).decode("ascii") != raw_base64:
                return None  # Do not bypass durable same-content identity with alternate padding.
            kind = detect_animation_kind(data)
            if kind is None:
                return None
            return analyze_animation(
                data,
                kind,
                cfg.frames,
                build_sheet=can_describe,
                deadline=deadline,
                cancelled=cancelled,
            )

        decode_task = self._track(
            asyncio.create_task(asyncio.to_thread(decode)), self._decodes
        )
        try:
            analysis = await asyncio.wait_for(
                asyncio.shield(decode_task), max(0.001, deadline - time.monotonic())
            )
        except (ValueError, AnimationReadError, asyncio.TimeoutError):
            return None
        finally:
            cancelled.set()
        if analysis is None or not self._valid(epoch):
            return None
        header = self._build_header(analysis.meta)
        if analysis.skip_reason:
            skip_text = SKIP_REASON_TEXT.get(analysis.skip_reason, "未做视觉识别")
            return self._format_text(header, "", note=skip_text, cfg=cfg)
        if not can_describe:
            return self._format_text(header, "", note="已达每条消息识别上限", cfg=cfg)
        attempt_key = (
            digest
            + ":"
            + hashlib.sha256(
                json.dumps([cfg.vlm.task_name, cfg.vlm.model_name]).encode()
            ).hexdigest()
        )
        max_tokens = cfg.vlm.max_tokens
        for attempt in range(2 if cfg.vlm.retry_empty_response else 1):
            if not self._valid(epoch) or time.monotonic() >= deadline:
                return None
            if not await self._cache.begin(attempt_key, cfg.perf.max_concurrent):
                return self._format_text(
                    header, "", note="识别繁忙或先前请求结果未知", cfg=cfg
                )

            async def call(tokens=max_tokens):
                known = False
                try:
                    if not self._valid(epoch) or time.monotonic() >= deadline:
                        known = True  # No capability was dispatched, so it is safe to release.
                        return ""
                    result = await describe_frame_sheet(
                        self._llm_generate,
                        sheet_bytes=analysis.sheet_bytes,
                        sheet_media_type=analysis.sheet_media_type,
                        header=header,
                        sampled_count=analysis.sampled_count,
                        total_frames=analysis.meta.frame_count,
                        task_name=cfg.vlm.task_name.strip(),
                        model_name=cfg.vlm.model_name.strip(),
                        temperature=cfg.vlm.temperature,
                        max_tokens=tokens,
                        max_chars=cfg.vlm.max_chars,
                    )
                    known = True
                    if self._valid(epoch) and cfg.cache.enabled:
                        await self._cache.put(
                            key,
                            header=header,
                            description=result,
                            kind=analysis.meta.kind,
                        )
                    return result
                except VlmEmptyResponseError:
                    known = True  # A completed empty response is not an unknown RPC attempt.
                    raise
                finally:
                    await self._cache.end(attempt_key, known=known)

            request = self._track(asyncio.create_task(call()), self._requests)
            try:
                description = await asyncio.wait_for(
                    asyncio.shield(request),
                    max(
                        0.001,
                        min(cfg.vlm.timeout_ms / 1000, deadline - time.monotonic()),
                    ),
                )
                return self._format_text(header, description, cfg=cfg)
            except VlmEmptyResponseError:
                retry_tokens = min(
                    max(max_tokens * 2, EMPTY_RESPONSE_RETRY_MIN_TOKENS),
                    EMPTY_RESPONSE_RETRY_MAX_TOKENS,
                )
                if retry_tokens <= max_tokens:
                    break
                max_tokens = retry_tokens
            except asyncio.TimeoutError:
                self._logger.warning(
                    "动图识别等待超时；保留在途占用，不重发，远端费用/取消状态未知"
                )
                break
            except Exception:
                self._logger.warning("动图识别失败；错误正文已省略，保留未知请求记录")
                break
        return self._format_text(header, "", note="视觉描述不可用", cfg=cfg)

    def _format_text(
        self, header: str, description: str, *, note: str = "", cfg=None
    ) -> str:
        """构造最终注入到聊天中的文本。"""

        cfg = cfg or self._config
        prefix = cfg.inject.prefix.strip()
        if cfg.inject.include_metadata and header:
            leading = f"{prefix} {header}".strip() if prefix else header
        else:
            leading = prefix or "[动图]"

        segments = [leading]
        if description:
            segments.append(f"画面：{description}")
            if cfg.inject.include_caveat:
                segments.append(CAVEAT_TEXT)
        elif note:
            segments.append(f"（{note}）")
        else:
            segments.append("（视觉描述不可用）")
        return "；".join(segment for segment in segments if segment)

    @staticmethod
    def _build_header(meta: AnimationMeta) -> str:
        """构造注入文本里的动图元信息头。"""

        if meta.loop == 0:
            loop_text = "循环"
        elif meta.loop is None:
            loop_text = "播放一次"
        else:
            loop_text = f"重复{meta.loop}次"
        fps_text = f"{meta.fps:.0f}" if meta.fps > 0 else "?"
        return (
            f"{meta.kind.upper()} {meta.width}x{meta.height} · {meta.frame_count}帧 · "
            f"{meta.duration_seconds:.1f}s · 约{fps_text}fps · {loop_text}"
        )
