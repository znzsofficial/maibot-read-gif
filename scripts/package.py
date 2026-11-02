"""Allowlisted local-only source package and isolated actual SDK discovery."""

from pathlib import Path
from tempfile import TemporaryDirectory
import hashlib
import json
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
FILES = (
    "plugin.py",
    "_manifest.json",
    "LICENSE",
    "README.md",
    "NOTICE.md",
    "CHANGELOG.md",
    *(
        f"read_animation/{name}.py"
        for name in (
            "__init__",
            "cache",
            "config",
            "context",
            "frames",
            "pipeline",
            "sheet",
            "sniff",
            "vlm",
        )
    ),
)


def verify():
    # No real config/cache/logs/keys/venv/tests can enter the runtime source artifact.
    with TemporaryDirectory(prefix="gif-source-") as directory:
        artifact = Path(directory) / "gif-source.zip"
        hashes = {}
        with zipfile.ZipFile(artifact, "w", zipfile.ZIP_DEFLATED) as archive:
            for name in FILES:
                path = ROOT / name
                if path.is_symlink() or not path.is_file():
                    raise ValueError("Missing/unsafe source member: " + name)
                data = path.read_bytes()
                hashes[name] = hashlib.sha256(data).hexdigest()
                archive.writestr("read-gif/" + name, data)
        extracted = Path(directory) / "unpacked"
        with zipfile.ZipFile(artifact) as archive:
            if set(archive.namelist()) != {"read-gif/" + name for name in FILES}:
                raise ValueError("Unexpected artifact member")
            archive.extractall(extracted)
        root = extracted / "read-gif"
        for name, digest in hashes.items():
            if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
                raise ValueError("Extracted source mismatch")
        program = """import importlib.util,json,pathlib,sys
root=pathlib.Path(sys.argv[1]).resolve()
spec=importlib.util.spec_from_file_location('isolated_gif',root/'plugin.py',submodule_search_locations=[str(root)])
module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
from maibot_sdk import MaiBotPlugin
plugin=module.create_plugin()
assert isinstance(plugin,MaiBotPlugin)
components=plugin.get_components()
assert len(components)==3 and all(c['type']=='HOOK_HANDLER' for c in components)
assert plugin.build_default_config()['vlm']['max_tokens']==4096
assert plugin.build_default_config()['vlm']['retry_empty_response'] is False
assert plugin.build_default_config()['plugin']['config_version']=='1.3.0'
assert pathlib.Path(module.__file__).resolve().is_relative_to(root)
assert json.loads((root/'_manifest.json').read_text(encoding='utf-8'))['id']=='github.fabiocsa.read-gif'
print('Isolated package / real SDK / three Hook discovery passed; no on_load/model/QQ')
"""
        subprocess.run(
            [sys.executable, "-I", "-X", "utf8", "-c", program, str(root)],
            cwd=directory,
            check=True,
        )
        print(
            json.dumps(
                {
                    "members": len(FILES),
                    "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                    "public_release": False,
                }
            )
        )


if __name__ == "__main__":
    verify()
