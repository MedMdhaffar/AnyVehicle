"""Recognition assets and imports stay independent of the former Django apps."""
import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]


class RecognitionLayoutTests(unittest.TestCase):
    def test_console_imports_without_former_app_packages(self):
        code = '''
import importlib.abc
import sys

class RemovedApps(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        prefixes = (
            'vehicle.license_plate_service',
            'vehicle.vehicle_classification_manufacturer_service',
            'vehicle.vehicle_color_service',
            'license_plate_service',
            'vehicle_classification_manufacturer_service',
            'vehicle_color_service',
        )
        if any(fullname == p or fullname.startswith(p + '.') for p in prefixes):
            raise AssertionError('Former Django app imported: ' + fullname)

sys.meta_path.insert(0, RemovedApps())
from unittest.mock import patch
import psycopg
with patch.object(psycopg, 'connect', side_effect=AssertionError('No database access')), \
     patch('subprocess.Popen', side_effect=AssertionError('No worker launch')):
    import vehicle.common.real_video.recognition_engine
    import vehicle.common.real_video.plate_engine
    import vehicle.common.recognition_db1
    from vehicle.common.manufacturer.test_database import ManufacturerTestDatabase
    from vehicle.common.real_video.test_database import fixture_module
    fixture_module()
    import django
    django.setup()
    from django.urls import get_resolver
    get_resolver().url_patterns
'''
        environment = {**os.environ, 'DJANGO_SETTINGS_MODULE': 'vehicle.settings',
                       'PYTHONPATH': os.pathsep.join((str(ROOT/'vehicle'), str(ROOT))),
                       'PYTHONDONTWRITEBYTECODE': '1'}
        result = subprocess.run([sys.executable, '-B', '-c', code], cwd=ROOT/'vehicle',
                                env=environment, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_default_asset_paths_are_package_relative_from_another_directory(self):
        code = '''
import json
from pathlib import Path
from vehicle.recognition.config import paths
from vehicle.recognition.attributes.manufacturer.config import RecognitionConfig
from vehicle.recognition.plate.license_plate_plugin.config import load_default_config
from vehicle.recognition.plate.license_plate_plugin.reference_index import ReferenceIndex
from vehicle.recognition.plate.license_plate_plugin.reference_index import compatibility_record
from vehicle.recognition.plate.license_plate_plugin.dinov3 import repository_revision, CHECKPOINT_SHA256
from vehicle.recognition.plate.license_plate_plugin.utils import sha256_file
from types import SimpleNamespace

assets = paths()
attributes = RecognitionConfig.from_env()
category = load_default_config()
all_paths = list(assets.values()) + [Path(attributes.model_path), Path(attributes.projector_path),
    category.dinov3_repo, category.dinov3_checkpoint, category.paddle_model_dir,
    category.reference_manifest, category.reference_index]
assert all(p.exists() for p in all_paths), all_paths
assert all(p.is_relative_to(Path(__import__('vehicle.recognition', fromlist=['']).__file__).parent)
           for p in all_paths), all_paths
encoder = SimpleNamespace(checkpoint=category.dinov3_checkpoint,
    checkpoint_sha256=sha256_file(category.dinov3_checkpoint),
    repo_revision=repository_revision(category.dinov3_repo))
assert encoder.checkpoint_sha256 == CHECKPOINT_SHA256
ReferenceIndex.load(category.reference_index, compatibility_record(category, encoder))
print(json.dumps([str(p) for p in all_paths]))
'''
        environment = {k: v for k, v in os.environ.items()
                       if not k.startswith(('T1_', 'MANUFACTURER_', 'PLATE_'))}
        environment.update(PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE='1')
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, '-B', '-c', code], cwd=directory,
                                    env=environment, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(json.loads(result.stdout))

    def test_local_audit_artifacts_preserve_locked_bytes(self):
        for name in ('mohamed_lpr_assets.json', 'mohamed_lpr_assets.v2.json'):
            lock = json.loads((ROOT/'vehicle/common'/name).read_text())
            for label, item in lock['files'].items():
                if item['base'] != 'target':continue
                with self.subTest(lock=name, artifact=label):
                    path = ROOT/item['path']
                    self.assertTrue(path.is_file(), str(path))
                    self.assertEqual(path.stat().st_size, item['bytes'])
                    digest = hashlib.sha256()
                    with path.open('rb') as stream:
                        for block in iter(lambda:stream.read(1024*1024), b''):
                            digest.update(block)
                    self.assertEqual(digest.hexdigest(), item['sha256'])
