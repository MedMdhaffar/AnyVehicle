"""Short real decoder/Redis worker checks with explicit empty model fixtures."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from uuid import uuid4

from vehicle.common.console_runs import atomic_json, initial_status

ROOT=Path(__file__).resolve().parents[3]


@unittest.skipUnless(shutil.which('ffmpeg'),'ffmpeg required')
class VideoWorkerTests(unittest.TestCase):
    def test_stop_during_decoding_reaps_resources_without_eof_claim(self):
        import time
        server='/home/intern/part-time/aziz/redis-native-validation/redis-8.2.11/src/redis-server'
        if not Path(server).is_file():self.skipTest('configured Redis server required')
        with tempfile.TemporaryDirectory(prefix='waycon-midstop-') as name:
            root=Path(name);video=root/'stop.mov'
            subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','testsrc2=size=64x48:rate=25',
                '-frames:v','40','-c:v','mpeg4',str(video)],check=True,timeout=15)
            run_id=str(uuid4());directory=root/run_id;directory.mkdir()
            atomic_json(directory/'config.json',dict(run_id=run_id,source_kind='mp4',source=str(video),manufacturer=False,sample_every=1,max_seconds=None))
            atomic_json(directory/'status.json',initial_status(run_id,'mp4',False,1,None))
            child=subprocess.Popen([sys.executable,'-B',str(ROOT/'scripts/vehicle_console_worker.py'),str(directory)],
                env={**os.environ,'WAYCON_CONSOLE_TEST_PROFILE':'empty','WAYCON_CONSOLE_TEST_DELAY':'.1',
                    'T1_CONSOLE_DB1':'console-only','T1_REDIS_SERVER':server,'WAYCON_GPU_LOCK':str(root/'gpu.lock')},
                stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            try:
                deadline=time.monotonic()+30
                while time.monotonic()<deadline:
                    status=json.loads((directory/'status.json').read_text())
                    if status.get('analyzed_frames',0)>=2:break
                    self.assertIsNone(child.poll());time.sleep(.05)
                self.assertGreaterEqual(status.get('analyzed_frames',0),2)
                (directory/'stop').touch(mode=0o600)
                self.assertEqual(child.wait(timeout=15),0)
                status=json.loads((directory/'status.json').read_text())
                self.assertEqual(status['status'],'stopped')
                self.assertEqual(status['end_reason'],'controlled_stop')
                self.assertFalse(status['full_video_complete'])
                self.assertLess(status['analyzed_frames'],40)
                self.assertTrue(all(status['cleanup'].values()))
            finally:
                if child.poll() is None:child.terminate();child.wait(timeout=15)

    def test_sequential_full_eof_and_stop_keep_separate_history(self):
        server='/home/intern/part-time/aziz/redis-native-validation/redis-8.2.11/src/redis-server'
        if not Path(server).is_file():self.skipTest('configured Redis server required')
        with tempfile.TemporaryDirectory(prefix='waycon-full-eof-') as name:
            root=Path(name)
            video=root/'tiny.mkv'
            subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','testsrc2=size=64x48:rate=25',
                '-frames:v','8','-c:v','mpeg4',str(video)],check=True,timeout=15)
            statuses=[]
            for stop in (False,False,True):
                run_id=str(uuid4());directory=root/run_id;directory.mkdir()
                atomic_json(directory/'config.json',dict(run_id=run_id,source_kind='mp4',source=str(video),manufacturer=False,sample_every=1,max_seconds=None))
                atomic_json(directory/'status.json',initial_status(run_id,'mp4',False,1,None))
                if stop:(directory/'stop').touch()
                result=subprocess.run([sys.executable,'-B',str(ROOT/'scripts/vehicle_console_worker.py'),str(directory)],
                    env={**os.environ,'WAYCON_CONSOLE_TEST_PROFILE':'empty','T1_CONSOLE_DB1':'console-only',
                        'T1_REDIS_SERVER':server,'WAYCON_GPU_LOCK':str(root/'gpu.lock')},capture_output=True,timeout=40)
                self.assertEqual(result.returncode,0, result.stderr[-1000:])
                status=json.loads((directory/'status.json').read_text());statuses.append(status)
                self.assertTrue(all(status['cleanup'].values()))
                if not stop:
                    self.assertEqual(status['analyzed_frames'],8)
                    self.assertEqual(status['end_reason'],'eof')
                    self.assertTrue(status['full_video_complete'])
                else:
                    self.assertEqual(status['status'],'stopped')
                    self.assertFalse(status['full_video_complete'])
            self.assertEqual(len({s['run_id'] for s in statuses}),3)
            self.assertTrue(all((root/s['run_id']/'status.json').exists() for s in statuses))
