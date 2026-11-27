"""PuYu Windows desktop entry point and owned local-service lifecycle."""

import argparse
import json
import multiprocessing
import os
import socket
import sys
import threading
import time
import traceback
from pathlib import Path

from app_paths import resource_root, writable_root


SPLASH = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<style>body{margin:0;background:#f5f9fc;color:#123f3a;font:17px/1.8 'Microsoft YaHei',sans-serif;display:grid;place-items:center;height:100vh}main{max-width:620px;padding:40px}h1{font-size:34px}p{color:#536b72}.dot{height:5px;background:#1ab8a7;animation:pulse 1.5s infinite;border-radius:8px}@keyframes pulse{50%{opacity:.2}}</style>
<main><h1>英语作文批改工作台</h1><p>正在准备批改工作台，请稍候。</p><div class="dot"></div></main></html>'''


def configure_runtime():
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    log_dir = writable_root() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    # Windowed frozen programs have no stdout/stderr. Libraries still use them.
    if sys.stdout is None:
        sys.stdout = (log_dir / "desktop.log").open("a", encoding="utf-8", buffering=1)
    if sys.stderr is None:
        sys.stderr = (log_dir / "desktop.log").open("a", encoding="utf-8", buffering=1)
    return log_dir


def serve_ui(connection):
    configure_runtime()
    app = None
    try:
        from web.app import build_app, launch_options
        for attempt in range(3):
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", 0))
                port = probe.getsockname()[1]
            try:
                app = build_app().queue(max_size=16)
                app.launch(server_name="127.0.0.1", server_port=port, share=False,
                           inbrowser=False, prevent_thread_lock=True, run_history=False,
                           allowed_paths=[str(writable_root() / "results/web_week2"),
                                          str(writable_root() / "results/human_week34")],
                           quiet=True, **launch_options())
                break
            except OSError:
                if app is not None:
                    app.close()
                if attempt == 2:
                    raise
        connection.send({"ready": True, "port": port, "pid": os.getpid()})
        # Only the owning desktop process holds the other end of this pipe.
        connection.recv()
    except EOFError:
        pass
    except Exception:
        error = traceback.format_exc()
        print(error, file=sys.stderr, flush=True)
        try:
            connection.send({"ready": False, "error": error})
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        if app is not None:
            app.close()
        connection.close()


class DesktopApp:
    def __init__(self, diagnostics=None, self_test=False, test_model=False):
        self.diagnostics = diagnostics
        self.self_test = self_test
        self.test_model = test_model
        self.closed = threading.Event()
        self.cleanup_lock = threading.Lock()
        self.cleaned_up = False
        self.worker = None
        self.connection = None
        self.window = None
        self.result = {"desktop_started": False, "self_test_passed": False}

    def record(self, name, value):
        self.result[name] = value
        if self.diagnostics:
            self.diagnostics.mkdir(parents=True, exist_ok=True)
            (self.diagnostics / "desktop-checks.json").write_text(
                json.dumps(self.result, ensure_ascii=False, indent=2), encoding="utf-8")

    def close(self):
        with self.cleanup_lock:
            if self.cleaned_up:
                return
            self.closed.set()
            if self.connection is not None:
                try:
                    self.connection.send("close")
                except (BrokenPipeError, EOFError, OSError):
                    pass
            if self.worker is not None:
                self.worker.join(timeout=3)
                if self.worker.is_alive():
                    self.worker.terminate()
                    self.worker.join(timeout=3)
                self.record("service_stopped", not self.worker.is_alive())
            if self.connection is not None:
                self.connection.close()
            self.cleaned_up = True

    def wait_for_dom(self, expression, timeout=45):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and not self.closed.is_set():
            try:
                if self.window.evaluate_js(expression):
                    return
            except Exception:
                pass
            self.closed.wait(0.25)
        raise RuntimeError("桌面页面未在限定时间内完成加载或操作。")

    def native_smoke_test(self):
        self.wait_for_dom("document.querySelector('#review-button button, button#review-button') !== null", timeout=60)
        title = self.window.evaluate_js("document.querySelector('h1')?.textContent")
        self.record("page_title", title)
        self.record("native_window", str(type(self.window.native)))
        self.record("window_title", str(self.window.native.Text))
        if self.result["window_title"] != "英语作文批改工作台":
            raise RuntimeError("桌面窗口标题与工作台名称不一致。")
        self.record("desktop_started", title == "英语作文批改工作台")
        self.window.evaluate_js("Array.from(document.querySelectorAll('button')).find(b=>b.textContent.trim()==='载入样例').click()")
        self.wait_for_dom("document.querySelector('#essay-input textarea')?.value.startsWith('Many students prefers')")
        self.window.evaluate_js("Array.from(document.querySelectorAll('button')).find(b=>b.textContent.trim()==='生成批改报告').click()")
        self.wait_for_dom("document.querySelectorAll('#scores .score-card').length === 6", timeout=60)
        self.record("six_scores_rendered", True)
        labels = self.window.evaluate_js(
            "Array.from(document.querySelectorAll('#scores .score-card > span')).map(e=>e.textContent.trim())")
        self.record("score_labels", labels)
        if labels != ["连贯性", "句法", "词汇", "措辞", "语法", "规范"]:
            raise RuntimeError("六维评分标签与团队约定不一致。")
        self.wait_for_dom("document.querySelector('#generation-timing').textContent.includes('总处理')")
        self.record("timing_visible", bool(self.window.evaluate_js("document.querySelector('#generation-timing').textContent.includes('总处理')")))
        self.record("native_report_status", self.window.evaluate_js("document.querySelector('#report-status textarea').value"))
        exported = self.read_exported_report()
        self.record("report_downloadable", exported.get("source") == "inference.mock_api.mock_generate_review")
        self.capture_window("desktop-window.png")
        self.window.evaluate_js("Array.from(document.querySelectorAll('[role=tab]')).find(b=>b.textContent.trim()==='人工反馈评价').click()")
        self.wait_for_dom("document.querySelector('#review-essay textarea')?.value.startsWith('Many students prefers')")
        self.record('human_tab_mock_report_synced', True)
        self.window.evaluate_js("Array.from(document.querySelectorAll('[role=tab]')).find(b=>b.textContent.trim()==='批改工作台').click()")
        self.window.evaluate_js("""(()=>{const t=document.querySelector('#essay-input textarea');
          Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype,'value').set.call(t,'word '.repeat(301));
          t.dispatchEvent(new Event('input',{bubbles:true}));})()""")
        self.wait_for_dom("document.querySelector('#input-limit').textContent.includes('301 / 300')")
        self.window.evaluate_js("Array.from(document.querySelectorAll('button')).find(b=>b.textContent.trim()==='生成批改报告').click()")
        self.wait_for_dom("document.querySelector('#report-status textarea').value.includes('超过 300')")
        self.record("word_limit_preserves_input", self.window.evaluate_js("document.querySelector('#essay-input textarea').value === 'word '.repeat(301)"))
        if self.test_model:
            self.test_local_model()
        self.record("self_test_passed", True)

    def read_exported_report(self):
        from urllib.request import ProxyHandler, build_opener
        self.window.evaluate_js("Array.from(document.querySelectorAll('button')).find(b=>b.textContent.trim()==='导出本次报告').click()")
        self.wait_for_dom("document.querySelector('a[href*=\"review-\"][href*=\".json\"]') !== null")
        url = self.window.evaluate_js("document.querySelector('a[href*=\"review-\"][href*=\".json\"]').href")
        with build_opener(ProxyHandler({})).open(url, timeout=15) as response:
            return json.loads(response.read())

    def capture_window(self, filename, retry=True):
        if not self.diagnostics:
            return
        try:
            import base64
            from System import Action, Func, Object, String
            from System.Threading.Tasks import Task
            engine = self.window.native.browser
            view = engine.webview
            captured, completed = {}, threading.Event()

            def capture_ready(task):
                try:
                    captured.update(json.loads(str(task.Result)))
                except Exception as exc:
                    captured['error'] = str(exc)
                finally:
                    completed.set()

            view.Invoke(Func[Object](lambda: view.CoreWebView2.CallDevToolsProtocolMethodAsync(
                "Page.captureScreenshot", '{"format":"png","captureBeyondViewport":false}').ContinueWith(
                    Action[Task[String]](capture_ready), engine.syncContextTaskScheduler)))
            if not completed.wait(20) or 'data' not in captured:
                raise RuntimeError(captured.get('error', '桌面截图等待超时'))
            (self.diagnostics / filename).write_bytes(base64.b64decode(captured['data']))
        except Exception as exc:
            if retry:
                self.closed.wait(1)
                return self.capture_window(filename, retry=False)
            self.record("screenshot_error", str(exc))

    def test_local_model(self):
        from inference.context import load_model_config
        config = load_model_config()
        self.window.evaluate_js("Array.from(document.querySelectorAll('#output-mode input')).find(e=>e.value==='本地模型（Qwen）').click()")
        self.window.evaluate_js("Array.from(document.querySelectorAll('#model-tasks input')).filter(e=>!e.checked).forEach(e=>e.click())")
        self.window.evaluate_js("""(()=>{const t=document.querySelector('#essay-input textarea');
          Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype,'value').set.call(t,'Many students prefers online learning because it saves travel time. Schools should combine online lessons with classroom meetings.');
          t.dispatchEvent(new Event('input',{bubbles:true}));})()""")
        self.wait_for_dom(f"document.querySelector('#input-limit').textContent.includes('18 / {config['max_essay_words']}')")
        self.window.evaluate_js("Array.from(document.querySelectorAll('button')).find(b=>b.textContent.trim()==='生成批改报告').click()")
        self.wait_for_dom("document.querySelector('#essay-input textarea').disabled", timeout=15)
        self.record('input_locked_during_model_generation', True)
        self.wait_for_dom(f"document.querySelector('#report-status textarea').value.includes('实际模型组别 {config['model_group']}')", timeout=600)
        self.wait_for_dom("document.querySelector('#generation-timing').textContent.includes('本地模型实际生成耗时')")
        report = self.read_exported_report()
        if self.diagnostics:
            (self.diagnostics / 'model-report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        seconds = report.get('timing', {}).get('generation_seconds', {}).get('feedback')
        if not seconds or not report.get('raw_outputs', {}).get('feedback'):
            raise RuntimeError('打包模型未产生实际输出，请检查模型依赖和权重。')
        self.record('packaged_model_called', True)
        self.record('packaged_model_generation_seconds', seconds)
        self.record('packaged_model_errors', report.get('errors', {}))
        self.record('model_generation_seconds', report['timing']['generation_seconds'])
        self.record('model_valid_tasks', list(report['valid_outputs']))
        self.record('model_identity', report['model_report']['model_info'])
        if set(report['raw_outputs']) != {'scoring', 'feedback', 'correction'}:
            raise RuntimeError('实际模型三任务未全部产生输出。')
        self.window.evaluate_js("document.querySelector('#report-column').scrollIntoView({block:'start'})")
        self.closed.wait(0.5)
        self.capture_window('model-b-report.png')
        self.window.evaluate_js("document.querySelector('#correction-output').scrollIntoView({block:'start'})")
        self.closed.wait(0.5)
        self.record('correction_change_highlights', self.window.evaluate_js(
            "document.querySelectorAll('#correction-output mark.revision-changed').length"))
        self.capture_window('model-b-correction.png')
        self.window.evaluate_js("Array.from(document.querySelectorAll('[role=tab]')).find(b=>b.textContent.trim()==='人工反馈评价').click()")
        self.wait_for_dom("document.querySelector('[role=tab][aria-selected=true]')?.textContent.trim()==='人工反馈评价'")
        self.wait_for_dom("document.querySelector('#review-essay textarea')?.value.startsWith('Many students prefers')")
        self.record('human_review_has_actual_essay', True)
        self.record('human_rating_fields_blank', self.window.evaluate_js(
            "Array.from(document.querySelectorAll('input[type=number]')).every(e=>e.value==='')"))
        self.window.evaluate_js("document.querySelector('#review-essay').scrollIntoView({block:'start'})")
        self.closed.wait(0.5)
        self.capture_window('human-review.png')
        self.window.evaluate_js("Array.from(document.querySelectorAll('button')).find(b=>b.textContent.trim()==='批改工作台').click()")
        excess = config['max_essay_words'] + 1
        self.window.evaluate_js(f"""(()=>{{const t=document.querySelector('#essay-input textarea');
          Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype,'value').set.call(t,'word '.repeat({excess}));
          t.dispatchEvent(new Event('input',{{bubbles:true}}));}})()""")
        self.wait_for_dom(f"document.querySelector('#input-limit').textContent.includes('{excess} / {config['max_essay_words']}')")
        self.window.evaluate_js("Array.from(document.querySelectorAll('button')).find(b=>b.textContent.trim()==='生成批改报告').click()")
        self.wait_for_dom(f"document.querySelector('#report-status textarea').value.includes('最多接收 {config['max_essay_words']} 词')")
        self.record('B_word_limit_preserves_input', self.window.evaluate_js(
            f"document.querySelector('#essay-input textarea').value === 'word '.repeat({excess})"))

    def start(self):
        try:
            context = multiprocessing.get_context("spawn")
            with self.cleanup_lock:
                if self.closed.is_set():
                    return
                self.connection, child_connection = context.Pipe()
                self.worker = context.Process(target=serve_ui, args=(child_connection,), daemon=True)
                self.worker.start()
                child_connection.close()
            self.record("service_pid", self.worker.pid)
            deadline = time.monotonic() + 180
            while not self.connection.poll(0.2):
                if self.closed.is_set():
                    return
                if not self.worker.is_alive():
                    raise RuntimeError("批改服务启动失败，请查看运行日志。")
                if time.monotonic() > deadline:
                    raise RuntimeError("启动超时，请查看运行日志后重新打开。")
            response = self.connection.recv()
            if not response.get("ready"):
                raise RuntimeError(response.get("error", "启动失败"))
            url = f"http://127.0.0.1:{response['port']}/?__theme=light"
            self.record("service_url", url)
            if self.closed.is_set():
                return
            self.window.load_url(url)
            if self.self_test:
                self.native_smoke_test()
                self.closed.wait(2)
                self.window.destroy()
        except Exception:
            error = traceback.format_exc()
            print(error, file=sys.stderr, flush=True)
            self.record("error", error)
            if self.self_test:
                try:
                    self.record('ui_debug', self.window.evaluate_js("""({
                      selectedTabs:Array.from(document.querySelectorAll('[role=tab][aria-selected=true]')).map(e=>e.textContent.trim()),
                      reviewEssay:document.querySelector('#review-essay textarea')?.value,
                      reviewStatus:document.querySelector('#review-status textarea')?.value,
                      workStatus:document.querySelector('#report-status textarea')?.value,
                      reviewElements:document.querySelectorAll('#review-essay').length
                    })"""))
                except Exception:
                    pass
            if self.self_test:
                self.window.destroy()
            elif not self.closed.is_set():
                from html import escape
                self.window.load_html(SPLASH.replace("正在准备批改工作台，请稍候。", "启动遇到问题，请打开日志目录查看详情。")
                                      .replace('<div class="dot"></div>', '<p>' + escape(str(writable_root() / "logs")) + '</p>'))


def main():
    parser = argparse.ArgumentParser(description="PuYu desktop application")
    parser.add_argument("--self-test", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--self-test-model", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--diagnostics-dir", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.self_test:
        os.environ["PUYU_SELF_TEST"] = "1"
    configure_runtime()
    import webview
    webview.settings["ALLOW_DOWNLOADS"] = True
    app = DesktopApp(args.diagnostics_dir, args.self_test, args.self_test_model)
    app.window = webview.create_window("英语作文批改工作台", html=SPLASH, width=1280, height=900,
                                       min_size=(900, 680), background_color="#f5f9fc")
    app.window.events.closed += app.close
    try:
        storage = writable_root() / "desktop-profile"
        storage.mkdir(parents=True, exist_ok=True)
        webview.start(app.start, gui="edgechromium", private_mode=False, storage_path=str(storage),
                      icon=str(resource_root() / "desktop/icon.ico"))
    except Exception:
        app.close()
        error = traceback.format_exc()
        print(error, file=sys.stderr, flush=True)
        try:
            import ctypes
            ctypes.windll.user32.MessageBoxW(None,
                "桌面窗口无法启动。请确认 Microsoft Edge WebView2 Runtime 已安装。\n运行日志位于 " + str(writable_root() / "logs"),
                "英语作文批改工作台", 0x10)
        except Exception:
            pass
        return 1
    if args.self_test:
        app.close()
        return 0 if app.result.get("self_test_passed") and app.result.get("service_stopped") else 1
    app.close()
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
