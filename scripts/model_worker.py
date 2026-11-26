"""One owned local GPU job; JSON file protocol avoids stdout contamination."""
import argparse
import ctypes
import json
import os
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def watch_parent(pid):
    if os.name != "nt":
        return
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x00100000, False, pid)
    if not handle:
        os._exit(3)
    try:
        kernel.WaitForSingleObject(handle, 0xFFFFFFFF)
        os._exit(3)
    finally:
        kernel.CloseHandle(handle)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--parent-pid", type=int, required=True)
    args = parser.parse_args()
    # Windows venv python.exe can be a launcher that owns the real interpreter.
    # Observe both the UI service and this launcher, including timeout kills.
    for parent_pid in {args.parent_pid, os.getppid()}:
        threading.Thread(target=watch_parent, args=(parent_pid,), daemon=True).start()
    awake_requested = False
    if os.name == 'nt':
        # Thread-scoped request only; it is also released automatically on exit.
        awake_requested = bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
    request = json.loads(args.request.read_text(encoding="utf-8"))
    config = args.request.with_name("config.json")
    config.write_text(json.dumps(request["config"]), encoding="utf-8")
    os.environ["PUYU_MODEL_CONFIG"] = str(config)
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import torch
    torch.set_num_threads(4)
    from inference.model_pipeline import generate_review
    report = generate_review(request["essay"], request["tasks"])
    report["model_info"].update({"torch_version": torch.__version__,
                               "cuda_version": torch.version.cuda,
                               "gpu_name": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                               "prevent_idle_sleep_requested": awake_requested})
    args.result.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if awake_requested:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == "__main__":
    main()
