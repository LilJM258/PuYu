import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from inference.context import load_model_config, task_prompt
from inference import model_pipeline
from web.app import MODEL_EXAMPLE_LABEL, LOCAL_MODE, input_summary, load_example


class Week34Tests(unittest.TestCase):
    def test_qwen3_prompt_disables_thinking(self):
        tokenizer = Mock(chat_template='template')
        tokenizer.apply_chat_template.return_value = 'rendered'
        self.assertEqual(task_prompt('Essay.', 'feedback', tokenizer), 'rendered')
        self.assertFalse(tokenizer.apply_chat_template.call_args.kwargs['enable_thinking'])

    def test_real_example_has_no_preloaded_answers(self):
        result = load_example(MODEL_EXAMPLE_LABEL)
        self.assertEqual(result[1:4], ('', '', ''))
        self.assertEqual(result[4], LOCAL_MODE)
        self.assertIsNone(result[-1])

    def test_profile_word_limit_applies_before_model_load(self):
        config = load_model_config()
        config['max_essay_words'] = 150
        with patch('inference.model_pipeline.load_model_config', return_value=config), \
                patch('inference.model_pipeline._model') as model:
            with self.assertRaisesRegex(ValueError, '150'):
                model_pipeline.generate_review('word ' * 151)
        model.assert_not_called()
        with patch('web.app.model_profile', return_value=(config, 'B')):
            self.assertIn('151 / 150', input_summary('word ' * 151, LOCAL_MODE))

    def test_worker_failure_does_not_fall_back_to_mock(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime, worker = root/'python.exe', root/'scripts/worker.py'
            worker.parent.mkdir()
            runtime.touch(); worker.touch()
            config = dict(load_model_config(), runtime_python=str(runtime), worker_path=str(worker))
            with patch('app_paths.writable_root', return_value=root), \
                    patch('subprocess.run', return_value=SimpleNamespace(returncode=1)):
                with self.assertRaisesRegex(ValueError, '推理进程启动失败'):
                    model_pipeline._external_review('Essay.', ['feedback'], config)

    def test_worker_config_cannot_recursively_spawn_itself(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime, worker = root/'python.exe', root/'scripts/worker.py'
            worker.parent.mkdir()
            runtime.touch(); worker.touch()
            config = dict(load_model_config(), runtime_python=str(runtime), worker_path=str(worker))
            expected = {'model_info': {'model_group': 'B'}, 'timings': {}, 'raw_outputs': {'feedback': 'bad-json'}}

            def subprocess_run(command, **kwargs):
                request = json.loads(Path(command[command.index('--request')+1]).read_text(encoding='utf-8'))
                self.assertNotIn('runtime_python', request['config'])
                self.assertNotIn('worker_path', request['config'])
                self.assertEqual(request['essay'], 'Essay.')
                Path(command[command.index('--result')+1]).write_text(json.dumps(expected), encoding='utf-8')
                return SimpleNamespace(returncode=0)

            with patch('app_paths.writable_root', return_value=root), patch('subprocess.run', side_effect=subprocess_run):
                result = model_pipeline._external_review('Essay.', ['feedback'], config)
            self.assertEqual(result['raw_outputs']['feedback'], 'bad-json')
            self.assertGreaterEqual(result['timings']['backend_process_seconds'], 0)


if __name__ == '__main__':
    unittest.main()
