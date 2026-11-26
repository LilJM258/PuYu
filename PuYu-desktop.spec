# PyInstaller desktop build. Model weights remain beside the executable.
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

project = Path(SPECPATH)
datas = []
for package in ("gradio", "gradio_client", "safehttpx", "huggingface_hub", "tokenizers", "peft"):
    datas += collect_data_files(package)
for package in ("gradio", "gradio_client", "huggingface-hub", "pydantic", "pydantic-core", "transformers", "peft", "accelerate", "torch", "pywebview"):
    datas += copy_metadata(package)
for name in ("web/style.css", "data/mock/examples.json", "evaluation/human/rubric.md", "configs/web_model.json", "desktop/icon.ico"):
    path = project / name
    datas.append((str(path), str(path.parent.relative_to(project))))

hidden = ["webview.platforms.winforms", "webview.platforms.edgechromium", "pythonnet", "clr",
          "transformers.models.qwen2.configuration_qwen2", "transformers.models.qwen2.modeling_qwen2",
          "transformers.models.qwen2.tokenization_qwen2", "transformers.models.gpt2.tokenization_gpt2",
          "transformers.models.auto.configuration_auto", "transformers.models.auto.modeling_auto",
          "transformers.models.auto.tokenization_auto", "uvicorn.logging", "uvicorn.loops.asyncio",
          "uvicorn.protocols.http.h11_impl", "uvicorn.protocols.websockets.websockets_impl",
          "uvicorn.lifespan.on"]
hidden += collect_submodules("gradio", filter=lambda name: not name.startswith("gradio.cli"))
hidden += collect_submodules("peft.tuners.lora")

a = Analysis([str(project / "run_desktop.py")], pathex=[str(project)], binaries=[], datas=datas,
             hiddenimports=hidden, hookspath=[], hooksconfig={}, runtime_hooks=[],
             excludes=["playwright", "pytest", "IPython", "notebook", "matplotlib", "tensorflow",
                       "jax", "PyQt5", "PyQt6", "PySide2", "PySide6", "tkinter"],
             noarchive=False, optimize=0)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="PuYu", debug=False,
          bootloader_ignore_signals=False, strip=False, upx=False, console=False,
          disable_windowed_traceback=False, icon=str(project / "desktop/icon.ico"))
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="PuYu")
