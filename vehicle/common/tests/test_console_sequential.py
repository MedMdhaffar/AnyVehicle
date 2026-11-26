"""Console worker acceptance: >16 sequential lifetimes, async manufacturer, stop/EOF.

Real worker process, real MP4 decoder, run-owned Redis, real Redis metadata
workers and V2 publication; injected detectors/plate executors and an
in-process persistent manufacturer fixture (no GPU, no PostgreSQL).
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[3]
SERVER = os.environ.get('T1_REDIS_SERVER', '/home/intern/part-time/aziz/redis-native-validation/redis-8.2.11/src/redis-server')
FFMPEG = shutil.which('ffmpeg') or '/usr/bin/ffmpeg'


@unittest.skipUnless(Path(SERVER).is_file() and Path(FFMPEG).is_file(), 'requires redis-server and ffmpeg')
class ConsoleSequentialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix='waycon-sequential-')
        cls.video = Path(cls.tmp.name)/'sequential.mp4'
        # Sharp synthetic frames so the quality gate admits vehicle crops.
        subprocess.run([FFMPEG, '-v', 'error', '-f', 'lavfi', '-i', 'testsrc2=size=1280x720:rate=25', '-frames:v', '800',
                        '-c:v', 'libx264', '-preset', 'ultrafast', '-pix_fmt', 'yuv420p', str(cls.video)],
                       check=True, timeout=120, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    @classmethod
    def long_video(cls):
        path = Path(cls.tmp.name)/'sequential-long.mp4'
        if not path.exists():
            subprocess.run([FFMPEG, '-v', 'error', '-f', 'lavfi', '-i', 'testsrc2=size=1280x720:rate=25', '-frames:v', '1240',
                            '-c:v', 'libx264', '-preset', 'ultrafast', '-pix_fmt', 'yuv420p', str(path)],
                           check=True, timeout=180, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return path

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def start(self, *, vehicles=40, manufacturer=True, extra=None, video=None):
        from vehicle.common.console_runs import initial_status, atomic_json
        directory = Path(tempfile.mkdtemp(prefix='run-', dir=self.tmp.name))
        os.chmod(directory, 0o700)
        run_id = directory.name.replace('run-', '')[:36]
        atomic_json(directory/'status.json', initial_status(run_id, 'mp4', manufacturer, 1, None))
        atomic_json(directory/'config.json', dict(run_id=run_id, source_kind='mp4', source=str(video or self.video),
                                                  rtsp_transport='tcp', manufacturer=manufacturer, sample_every=1, max_seconds=None))
        env = {k: v for k, v in os.environ.items() if not k.startswith(('T1_', 'WAYCON_'))}
        env.update(WAYCON_CONFIG=str(directory/'absent-config.json'), WAYCON_CONSOLE_TEST_PROFILE='sequential',
                   WAYCON_CONSOLE_TEST_VEHICLES=str(vehicles), T1_REDIS_SERVER=SERVER, CUDA_VISIBLE_DEVICES='',
                   PYTHONDONTWRITEBYTECODE='1', **(extra or {}))
        child = subprocess.Popen([sys.executable, '-B', str(ROOT/'scripts/vehicle_console_worker.py'), str(directory)],
                                 cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        self.addCleanup(lambda: child.poll() is None and child.kill())
        return directory, child

    def finish(self, directory, child, timeout=600):
        _, err = child.communicate(timeout=timeout)
        status = json.loads((directory/'status.json').read_text())
        failure = (directory/'failure.json').read_text() if (directory/'failure.json').exists() else None
        return status, failure, err.decode()[-3000:]

    def test_forty_sequential_lifetimes_quality_selection_and_async_manufacturer(self):
        directory, child = self.start(vehicles=40)
        status, failure, err = self.finish(directory, child)
        self.assertIsNone(failure, failure or err)
        self.assertEqual((status['status'], status['end_reason'], status['full_video_complete']), ('completed', 'eof', True))
        counters = status['counters']
        self.assertEqual((counters['tracks'], counters['canonical_redis_deliveries'], counters['released_lifetimes']), (40, 40, 40))
        self.assertEqual(counters['manufacturer_calls'], 40)            # at-most-once per lifetime
        self.assertEqual(counters['manufacturer_review_results'], 40)
        self.assertEqual(counters['vehicle_detector_calls'], 800)        # one detector, one decode
        self.assertTrue(all(status['cleanup'].values()), status['cleanup'])
        runtime = status['manufacturer_runtime']
        self.assertEqual((runtime['mode'], runtime['selection'], runtime['model_loads']), ('persistent', 'quality_lifetime_v1', 1))
        makes = {row['manufacturer']['make'] for row in status['rows']}
        self.assertEqual(makes, {'toyota'})
        self.assertEqual({row['state'] for row in status['rows']}, {'settled'})
        self.assertEqual(status['delivery']['review_ack'], 'acknowledged')
        # DB1 manufacturer persistence stays disabled: the canonical bundle has
        # no make fact and an explicit unsupported terminal omission.
        bundles = [json.loads(p.read_text()) for p in directory.glob('delivery-*.json')]
        self.assertEqual(len(bundles), 40)
        for bundle in bundles:
            self.assertNotIn('fact_vehicle_make', {f['row_kind'] for f in bundle['facts']})
            self.assertIn(dict(domain='manufacturer', reason='unsupported_capability'), bundle['omissions'])

    def test_legacy_first_eligible_policy_remains_selectable(self):
        directory, child = self.start(vehicles=20, extra=dict(T1_MANUFACTURER_SELECTION='first_eligible_v1'))
        status, failure, err = self.finish(directory, child)
        self.assertIsNone(failure, failure or err)
        self.assertEqual((status['status'], status['counters']['tracks'], status['counters']['manufacturer_calls']), ('completed', 20, 20))
        self.assertEqual(status['manufacturer_runtime']['selection'], 'first_eligible_v1')

    def test_manufacturer_disabled_more_than_16_lifetimes(self):
        directory, child = self.start(vehicles=30, manufacturer=False)
        status, failure, err = self.finish(directory, child)
        self.assertIsNone(failure, failure or err)
        self.assertEqual((status['status'], status['counters']['tracks'], status['counters']['canonical_redis_deliveries']), ('completed', 30, 30))
        self.assertEqual(status['counters']['manufacturer_calls'], 0)

    def test_tentative_plate_sightings_deliver_and_unconfirmed_plates_block_only_their_lifetime(self):
        directory, child = self.start(vehicles=20, manufacturer=False, extra=dict(WAYCON_CONSOLE_TEST_PLATES='tentative'))
        status, failure, err = self.finish(directory, child)
        self.assertIsNone(failure, failure or err)
        counters = status['counters']
        self.assertEqual((status['status'], counters['tracks']), ('completed', 20))
        self.assertEqual(counters['blocked_lifetimes'], {'UNCONFIRMED_PLATE_TRACK': 4})   # vehicles 5, 10, 15, 20
        self.assertEqual(counters['canonical_redis_deliveries'], 16)
        rows = {row['track_id']: row for row in status['rows']}
        blocked = sorted(sid for sid, row in rows.items() if row['state'] == 'blocked')
        self.assertEqual(len(blocked), 4)
        self.assertTrue(all(rows[sid]['plate']['status'] == 'blocked' for sid in blocked))
        settled = [row for row in rows.values() if row['state'] == 'settled']
        self.assertEqual(len(settled), 16)
        self.assertTrue(all(row['plate']['status'] != 'no_plate' for row in settled))      # OCR ran on tracked plates
        self.assertEqual(len(list(directory.glob('delivery-*.json'))), 16)
        self.assertTrue(all(status['cleanup'].values()), status['cleanup'])

    ROTATION = dict(WAYCON_CONSOLE_TEST_REDIS_PROFILE='small', WAYCON_CONSOLE_TEST_PLATES='tentative')

    def test_redis_epoch_rotation_exceeds_previous_per_run_capacity(self):
        directory, child = self.start(vehicles=60, extra=self.ROTATION, video=self.long_video())
        status, failure, err = self.finish(directory, child, timeout=1500)
        self.assertIsNone(failure, failure or err)
        c = status['counters']
        self.assertEqual((status['status'], status['end_reason'], c['tracks']), ('completed', 'eof', 60))
        self.assertEqual(c['blocked_lifetimes'], {'UNCONFIRMED_PLATE_TRACK': 12})
        self.assertEqual(c['canonical_redis_deliveries'], 48)
        self.assertEqual((c['manufacturer_calls'], c['manufacturer_review_results']), (60, 60))   # at most once each
        self.assertGreaterEqual(c['redis_epochs'], 3)
        retired = c['retired_epochs']
        self.assertEqual([e['index'] for e in retired], list(range(len(retired))))
        self.assertTrue(all(e['deleted_keys'] > 0 and e['usage'] >= 0.5 for e in retired))
        self.assertTrue(0 < sum(e['deliveries'] for e in retired) <= 48)   # earlier epochs delivered and were retired
        self.assertEqual(status['delivery']['review_ack'], 'acknowledged')
        bundles = sorted(int(p.stem.split('-')[1]) for p in directory.glob('delivery-*.json'))
        self.assertEqual(len(bundles), 48);self.assertEqual(len(set(bundles)), 48)
        print('ROTATION_EVIDENCE ' + json.dumps(dict(epochs=c['redis_epochs'], retired=retired, tracks=c['tracks'],
              deliveries=c['canonical_redis_deliveries'], blocked=c['blocked_lifetimes'])), file=sys.stderr)
        self.assertTrue(all(status['cleanup'].values()), status['cleanup'])

    def test_same_load_without_rotation_fails_closed_at_the_old_capacity(self):
        directory, child = self.start(vehicles=60, extra=dict(self.ROTATION, WAYCON_CONSOLE_TEST_NO_ROTATION='1'),
                                      video=self.long_video())
        status, failure, err = self.finish(directory, child, timeout=1500)
        self.assertEqual(status['status'], 'failed')
        self.assertEqual(json.loads(failure)['pipeline_code'], 'CAPACITY_EXCEEDED')
        print('NO_ROTATION_EVIDENCE ' + json.dumps(dict(status=status['status'], analyzed=status['analyzed_frames'],
              error_code=status.get('error_code'), failure=json.loads(failure)['type'] if failure else None)), file=sys.stderr)
        self.assertLess(status['analyzed_frames'], 1240)
        self.assertTrue(all(status['cleanup'].values()), status['cleanup'])

    @unittest.skipUnless(os.environ.get('DB1_P4_EXISTING_SERVER') == '1' and os.environ.get('DB1_P4_ADMIN_DSN'),
                         'requires the explicit disposable P4 PostgreSQL fixture')
    def test_epoch_rotation_delivers_every_epoch_to_db1_before_retirement(self):
        from psycopg.conninfo import conninfo_to_dict
        port = conninfo_to_dict(os.environ['DB1_P4_ADMIN_DSN']).get('port', '5432')
        directory, child = self.start(vehicles=30, manufacturer=False, video=self.long_video(), extra=dict(
            self.ROTATION, T1_CONSOLE_DB1='fixture', WAYCON_CONSOLE_TEST_DB1_FIXTURE='1',
            WAYCON_POSTGRES_MODE='existing', WAYCON_PG_PORT=str(port)))
        status, failure, err = self.finish(directory, child, timeout=2400)
        self.assertIsNone(failure, failure or err)
        c = status['counters']
        self.assertEqual((status['status'], c['tracks'], c['canonical_redis_deliveries']), ('completed', 30, 24))
        self.assertGreaterEqual(c['redis_epochs'], 2)
        self.assertEqual((c['db1_committed_tracks'], c['db1_acknowledged_tracks']), (24, 24))
        self.assertTrue(list(directory.glob('db1-progress-e*.json')))
        print('DB1_ROTATION_EVIDENCE ' + json.dumps(dict(epochs=c['redis_epochs'], retired=c['retired_epochs'],
              committed=c['db1_committed_tracks'], acknowledged=c['db1_acknowledged_tracks'],
              progress_files=sorted(p.name for p in directory.glob('db1-progress*.json')))), file=sys.stderr)
        self.assertEqual((status['delivery']['postgres_commit'], status['delivery']['redis_ack']), ('committed', 'acknowledged'))
        self.assertTrue(all(status['cleanup'].values()), status['cleanup'])

    def test_controlled_stop_mid_run_cleans_up_and_marks_incomplete(self):
        directory, child = self.start(vehicles=40)
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline and child.poll() is None:
            try:
                if json.loads((directory/'status.json').read_text()).get('analyzed_frames', 0) >= 300:break
            except (ValueError, OSError):
                pass
            time.sleep(.05)
        (directory/'stop').write_text('stop')
        status, failure, err = self.finish(directory, child)
        self.assertEqual((status['status'], status['end_reason']), ('stopped', 'controlled_stop'), err)
        self.assertTrue(all(status['cleanup'].values()), status['cleanup'])
        self.assertTrue(status['stop_requested'])
        self.assertLess(status['analyzed_frames'], 800)


if __name__ == '__main__':
    unittest.main()
