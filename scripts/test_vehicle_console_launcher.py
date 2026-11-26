"""Check repository imports in the backend environment created by the launcher."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import vehicle_console_launcher as launcher


class BackendImportTests(unittest.TestCase):
    def test_backend_can_import_saved_run_contracts(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            (directory/'preflight.json').write_text(json.dumps({'manufacturer':{'available':True}}))
            with patch.dict(os.environ, {'T1_CONSOLE_DB1':'console-only', 'PYTHONPATH':str(directory)}), \
                    patch.object(launcher.signal, 'signal'), \
                    patch.object(launcher, 'listening', return_value=False), \
                    patch.object(launcher.subprocess, 'run', return_value=Mock(returncode=0)), \
                    patch.object(launcher.subprocess, 'Popen', side_effect=RuntimeError('capture spawn')) as spawn:
                with self.assertRaisesRegex(RuntimeError, 'capture spawn'):
                    launcher.start_services(directory, argparse.Namespace(check=False, console_only=True))
                environment = spawn.call_args.kwargs['env']
            self.assertIn(str(directory), environment['PYTHONPATH'].split(os.pathsep))
            result = subprocess.run([sys.executable, '-B', '-c',
                'import sys; sys.path[0] = sys.argv[1]; '
                'import vehicle.settings; import common.recognition_transport',
                str(launcher.ROOT/'vehicle')], cwd=directory, env=environment,
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
