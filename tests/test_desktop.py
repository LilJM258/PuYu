import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import app_paths
from inference.context import load_model_config
from run_desktop import DesktopApp


class DesktopTests(unittest.TestCase):
    def test_frozen_paths_separate_writable_data_from_installation(self):
        with patch.object(app_paths.sys, 'frozen', True, create=True), \
                patch.object(app_paths.sys, 'executable', 'C:/Apps/PuYu/PuYu.exe'), \
                patch.dict(app_paths.os.environ, {'LOCALAPPDATA': 'C:/Users/Test/AppData/Local'}):
            self.assertEqual(app_paths.install_root(), Path('C:/Apps/PuYu'))
            self.assertEqual(app_paths.writable_root(), Path('C:/Users/Test/AppData/Local/PuYu'))

    def test_packaged_model_is_resolved_beside_exe_and_cache_is_writable(self):
        config = load_model_config()
        config['model_path'] = ''
        config['cache_dir'] = '.cache/huggingface'
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            assets, install, user = folder/'assets', folder/'app', folder/'user'
            (assets/'configs').mkdir(parents=True)
            (assets/'configs/web_model.json').write_text(json.dumps(config), encoding='utf-8')
            model = install/'models/Qwen3-1.7B'
            model.mkdir(parents=True)
            (model/'config.json').write_text('{}', encoding='utf-8')
            with patch('inference.context.ROOT', assets), \
                    patch('inference.context.install_root', return_value=install), \
                    patch('inference.context.writable_root', return_value=user), \
                    patch('inference.context.is_packaged', return_value=True), \
                    patch.dict('os.environ', {}, clear=True):
                actual = load_model_config()
            self.assertEqual(actual['model_path'], str(model))
            self.assertEqual(actual['cache_dir'], str(user/'.cache/huggingface'))

    def test_close_is_idempotent_and_stops_owned_service(self):
        app = DesktopApp()
        app.connection = Mock()
        app.worker = Mock()
        alive = [True]
        app.worker.is_alive.side_effect = lambda: alive[0]
        app.worker.terminate.side_effect = lambda: alive.__setitem__(0, False)
        app.close()
        app.close()
        app.connection.send.assert_called_once_with('close')
        app.connection.close.assert_called_once()
        app.worker.terminate.assert_called_once()
        self.assertTrue(app.result['service_stopped'])


if __name__ == '__main__':
    unittest.main()
