"""Explicit isolated real Host/Runner integration; only the bottom RPC is synthetic."""

from pathlib import Path
from tempfile import TemporaryDirectory
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import AsyncMock
import argparse
import asyncio
import base64
import importlib.metadata
import io
import json
import os
import shutil
import sys


async def verify(args):
    root = Path(__file__).resolve().parents[1]
    host = args.host_source.resolve()
    with ExitStack() as cleanup:
        directory = cleanup.enter_context(
            TemporaryDirectory(prefix="gif-runner-", dir=args.temp_dir)
        )
        instance = Path(directory).resolve()
        old_cwd = Path.cwd()
        os.chdir(instance)
        cleanup.callback(os.chdir, old_cwd)
        os.environ.update(
            MAIBOT_INSTANCE_DIR=str(instance),
            MAIBOT_LOCALE="zh-CN",
            MAIBOT_HOST_VERSION="1.3.19",
        )
        sys.path.insert(0, str(host))
        sys.path.insert(0, str(host / "scripts"))
        import neko_pytest_guard  # noqa: F401
        from src.plugin_runtime.runner.runner_main import (
            PluginRunner,
            PluginActivationStatus,
        )
        from src.plugin_runtime.runner import runner_main
        from src.plugin_runtime.protocol.envelope import Envelope
        from src.plugin_runtime.hook_payloads import (
            serialize_prompt_items,
            deserialize_prompt_items,
        )
        from src.llm_models.payload_content.context_item import (
            UserMessageItem,
            ContextItemMeta,
            ContextTextPart,
        )
        from src.llm_models.payload_content.context_protocol import ContextProtocolMode
        from src.common.database.database import engine
        from src.common.logger import close_handlers
        from PIL import Image

        cleanup.callback(engine.dispose)
        cleanup.callback(close_handlers)
        assert runner_main._PROJECT_ROOT.resolve() == instance, (
            "Refuse production instance paths"
        )
        assert Path(runner_main.__file__).resolve().is_relative_to(host)
        target = instance / "plugins/github_fabiocsa_read-gif"
        target.mkdir(parents=True)
        for name in ("plugin.py", "_manifest.json"):
            shutil.copyfile(root / name, target / name)
        shutil.copytree(
            root / "read_animation",
            target / "read_animation",
            ignore=shutil.ignore_patterns("__pycache__"),
        )
        (target / "config.toml").write_text(
            '[plugin]\nenabled=true\nconfig_version="1.3.0"\n', encoding="utf-8"
        )
        runner = PluginRunner(
            host_address="unix:///unused-gif.sock",
            session_token="synthetic",
            plugin_dirs=[str(target.parent)],
        )
        registrations, capabilities = [], []

        async def rpc(method, **kwargs):
            if method == "cap.call":
                payload = kwargs["payload"]
                assert payload["capability"] == "llm.generate", (
                    "No real send/API capability allowed"
                )
                assert payload["args"]["task_name"] == "vlm"
                capabilities.append(payload["capability"])
                return SimpleNamespace(
                    error=None,
                    payload={
                        "result": {"success": True, "response": "synthetic animation"}
                    },
                )
            assert method in (
                "plugin.bootstrap",
                "plugin.register_components",
                "plugin.unregister",
            ), method
            if method == "plugin.register_components":
                registrations.append(kwargs["payload"])
            return SimpleNamespace(error=None, payload={"accepted": True})

        runner._rpc_client.send_request = AsyncMock(side_effect=rpc)
        metas = runner._loader.discover_and_load([str(target.parent)])
        assert not runner._loader.failed_plugins and len(metas) == 1
        meta = metas[0]
        assert runner._prepare_plugin_activation(meta) is None and not meta.config_error
        assert await runner._activate_plugin(meta) == PluginActivationStatus.LOADED
        try:
            assert len(registrations[0]["components"]) == 3
            frames = [
                Image.new("RGB", (32, 32), color) for color in ("red", "blue", "green")
            ]
            stream = io.BytesIO()
            frames[0].save(
                stream,
                format="GIF",
                save_all=True,
                append_images=frames[1:],
                duration=100,
                loop=0,
            )
            message = {
                "message_id": "1001",
                "session_id": "s",
                "account_id": "bot",
                "platform": "qq",
                "message_info": {
                    "group_info": {"group_id": "1", "group_name": "synthetic"},
                    "user_info": {"user_id": "2", "user_nickname": "synthetic"},
                    "additional_config": {},
                },
                "raw_message": [
                    {
                        "type": "image",
                        "data": "",
                        "hash": "synthetic",
                        "binary_data_base64": base64.b64encode(
                            stream.getvalue()
                        ).decode(),
                    }
                ],
            }

            async def invoke(name, **arguments):
                result = await runner._handle_hook_invoke(
                    Envelope(
                        request_id=1,
                        message_type="request",
                        method="plugin.invoke_hook",
                        plugin_id="github.fabiocsa.read-gif",
                        payload={"component_name": name, "args": arguments},
                    )
                )
                assert result.error is None and result.payload["success"], result
                return result.payload

            original = json.dumps(message, sort_keys=True)
            received = await invoke("animation_frame_sheet", message=message)
            assert "modified_kwargs" not in received and original == json.dumps(
                message, sort_keys=True
            )
            source = UserMessageItem(
                meta=ContextItemMeta.create(),
                parts=(ContextTextPart('<message msg_id="1001">\n'),),
            )
            raw = serialize_prompt_items([source])
            planned = await invoke(
                "animation_derived_planner",
                items=raw,
                item_schema_version=1,
                session_id="s",
                tool_definitions=[],
            )
            normalized = deserialize_prompt_items(
                planned["modified_kwargs"]["items"],
                item_schema_version=planned["modified_kwargs"]["item_schema_version"],
                mode=ContextProtocolMode.REQUEST_CONTEXT,
                original_items=[source],
            )
            assert len(normalized) == 2 and normalized[0] is source
            assert "synthetic animation" in str(planned["modified_kwargs"]["items"][-1])
            replied = await invoke(
                "animation_derived_replyer",
                items=raw,
                item_schema_version=1,
                session_id="s",
                reply_message_id="1001",
            )
            assert len(replied["modified_kwargs"]["items"]) == 2
            denied = await invoke(
                "animation_derived_planner",
                items=raw,
                item_schema_version=1,
                session_id="other",
            )
            assert denied.get("modified_kwargs") is None
            for enabled in (False, True):
                result = await runner._handle_config_updated(
                    Envelope(
                        request_id=2,
                        message_type="request",
                        method="plugin.config_updated",
                        plugin_id="github.fabiocsa.read-gif",
                        payload={
                            "plugin_id": "github.fabiocsa.read-gif",
                            "config_scope": "self",
                            "config_version": "synthetic",
                            "config_data": {
                                "plugin": {
                                    "enabled": enabled,
                                    "config_version": "1.3.0",
                                }
                            },
                        },
                    )
                )
                assert result.error is None
                assert meta.instance._processor._config.plugin.enabled is enabled
                assert not meta.instance._derived._records
            assert len(capabilities) == 1
        finally:
            await runner._invoke_plugin_on_unload(meta)
            assert meta.instance._processor is None
        result = {
            "isolated": True,
            "real_runner_activation": True,
            "hook_count": 3,
            "real_host_item_roundtrip": True,
            "raw_message_unchanged": True,
            "cross_scope_denied": True,
            "config_reload_and_unload": True,
            "sdk": importlib.metadata.version("maibot-plugin-sdk"),
            "synthetic_llm_calls": len(capabilities),
            "real_model_calls": 0,
            "QQ_sends": 0,
        }
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host-source", type=Path, required=True)
    parser.add_argument("--temp-dir", type=Path)
    asyncio.run(verify(parser.parse_args()))
