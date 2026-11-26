"""scripts/run_waycon_rtsp.sh: configuration, secrets, outage/reconnect, Ctrl+C, cleanup.

End-to-end cases use the real launcher, RunStore, console worker, latest-frame RTSP
decoder, run-owned Redis and an owned loopback VLC RTSP server. Only the vehicle,
plate and manufacturer models are the explicit sequential fixtures (no GPU).
"""
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[3]
LAUNCHER = ROOT/'scripts/run_waycon_rtsp.sh'
WORKER = os.fsencode(str(ROOT/'scripts/vehicle_console_worker.py'))
SERVER = os.environ.get('T1_REDIS_SERVER') or os.environ.get('VEHICLE_REDIS_SERVER') or ''
FFMPEG, VLC = '/usr/bin/ffmpeg', '/usr/bin/cvlc'
SECRET = 's3cret-Token-9f2c'
# The only local RTSP server (VLC's RTP/RTSP output) answers TCP-interleaved SETUP with
# 461 Unsupported transport, so live-stream cases use UDP; TCP is the launcher default.
LIVE = ['--transport', 'udp']


def private_camera_dirs():
    return {p for p in Path(tempfile.gettempdir()).glob('waycon-camera-*') if p.stat().st_uid == os.getuid()}


def free_port():
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        return probe.getsockname()[1]


def workers():
    found = []
    for entry in Path('/proc').iterdir():
        if entry.name.isdigit():
            try:
                if WORKER in (entry/'cmdline').read_bytes().split(b'\0'):found.append(int(entry.name))
            except OSError:
                pass
    return found


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='waycon-rtsp-test-')
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        os.chmod(self.dir, 0o700)

    def env(self, **extra):
        env = {k: v for k, v in os.environ.items() if not k.startswith(('T1_', 'WAYCON_', 'DB1_'))}
        env.update(WAYCON_CONFIG=str(self.dir/'absent.json'), WAYCON_RTSP_ROOT=str(self.dir/'state'),
                   WAYCON_LAUNCHER_ROOT=str(self.dir/'web'), T1_CONSOLE_PYTHON=sys.executable,
                   T1_REDIS_SERVER=SERVER, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
        env.update(extra)
        return env

    def launch(self, args, env, timeout=60):
        result = subprocess.run([str(LAUNCHER), *args], env=env, capture_output=True, text=True, timeout=timeout)
        return result.returncode, result.stdout+result.stderr

    def assert_no_secret(self, *texts):
        for text in texts:
            self.assertNotIn(SECRET, text)
        for path in self.dir.rglob('*'):
            if path.is_file() and path.suffix != '.uri':
                self.assertNotIn(SECRET.encode(), path.read_bytes(), str(path.relative_to(self.dir)))


class ConfigurationTests(Base):
    def test_missing_credentials_fail_before_any_service(self):
        code, out = self.launch(['--console-only'], self.env())
        self.assertEqual(code, 2)
        self.assertIn('Missing camera configuration', out)
        self.assertFalse((self.dir/'state').exists())

    def test_both_sources_are_refused(self):
        uri = self.dir/'camera.uri'; uri.write_text('rtsp://127.0.0.1:1/x'); uri.chmod(0o600)
        code, out = self.launch([], self.env(WAYCON_RTSP_URI_FILE=str(uri), WAYCON_RTSP_URI='rtsp://127.0.0.1:1/x'))
        self.assertEqual((code, 'Set only one' in out), (2, True))

    def test_invalid_urls_are_rejected_without_echo(self):
        for bad in (f'http://user:{SECRET}@cam/x', f'rtsp://user:{SECRET}@/x', f'rtsp://user:{SECRET}@cam/a\nb', ''):
            code, out = self.launch(['--console-only'], self.env(WAYCON_RTSP_URI=bad))
            self.assertEqual(code, 2, out)
            self.assertTrue('Invalid RTSP URL' in out or 'Missing camera configuration' in out, out)
            self.assertNotIn(SECRET, out)

    def test_uri_file_must_be_private(self):
        uri = self.dir/'camera.uri'; uri.write_text(f'rtsp://u:{SECRET}@127.0.0.1:1/x'); uri.chmod(0o644)
        code, out = self.launch(['--console-only'], self.env(WAYCON_RTSP_URI_FILE=str(uri)))
        self.assertEqual(code, 2)
        self.assertIn('mode 0600', out)
        self.assertNotIn(SECRET, out)

    def test_runtime_pins_refuse_legacy_or_v1(self):
        for name, value in (('T1_MANUFACTURER_RUNTIME', 'legacy'), ('T1_MANUFACTURER_QUALITY', 'v1'),
                            ('T1_MANUFACTURER_SELECTION', 'first_eligible_v1')):
            code, out = self.launch(['--console-only'], self.env(WAYCON_RTSP_URI='rtsp://127.0.0.1:1/x', **{name: value}))
            self.assertEqual(code, 2, out)
            self.assertIn('requires '+name, out)

    def test_argument_validation(self):
        for args in (['--sample-every', '0'], ['--transport', 'http'], ['--max-seconds', '-1'],
                     ['--reconnect-attempts', '101'], ['--unknown']):
            code, _ = self.launch(args, self.env(WAYCON_RTSP_URI='rtsp://127.0.0.1:1/x'))
            self.assertEqual(code, 2, args)

    def test_missing_python_or_redis_fail_clearly(self):
        code, out = self.launch([], self.env(T1_CONSOLE_PYTHON=str(self.dir/'missing-python')))
        self.assertEqual(code, 2)
        self.assertIn('Python environment not found', out)
        code, out = self.launch(['--console-only'], self.env(WAYCON_RTSP_URI='rtsp://127.0.0.1:1/x',
                                                              T1_REDIS_SERVER=str(self.dir/'none')))
        self.assertEqual(code, 2)
        self.assertIn('T1_REDIS_SERVER', out)

    def test_refuses_second_inference_pipeline(self):
        # A running web console launcher (its lock is held) ...
        web = self.dir/'web'; web.mkdir(mode=0o700)
        with (web/'launcher.lock').open('a') as held:
            fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
            code, out = self.launch(['--console-only'], self.env(WAYCON_RTSP_URI='rtsp://127.0.0.1:1/x'))
        self.assertEqual(code, 2)
        self.assertIn('web console launcher is running', out)
        # ... or any other console worker process.
        if not workers():
            other = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(60)', str(ROOT/'scripts/vehicle_console_worker.py')])
            self.addCleanup(lambda: (other.kill(), other.wait()))
        code, out = self.launch(['--console-only'], self.env(WAYCON_RTSP_URI='rtsp://127.0.0.1:1/x'))
        self.assertEqual(code, 2)
        self.assertIn('another console worker is running', out)


@unittest.skipUnless(Path(SERVER).is_file() and Path(FFMPEG).is_file() and Path(VLC).is_file(),
                     'requires redis-server, ffmpeg and cvlc')
class EndToEndTests(Base):
    @classmethod
    def setUpClass(cls):
        cls.media = tempfile.TemporaryDirectory(prefix='waycon-rtsp-media-')
        cls.clip = Path(cls.media.name)/'loop.mp4'
        # Sharp synthetic 25 fps H.264 with repeated headers (late joiners decode at once).
        subprocess.run([FFMPEG, '-v', 'error', '-f', 'lavfi', '-i', 'testsrc2=size=1280x720:rate=25', '-t', '600',
                        '-c:v', 'libx264', '-preset', 'ultrafast', '-tune', 'zerolatency', '-g', '25',
                        '-x264-params', 'repeat-headers=1', '-pix_fmt', 'yuv420p', str(cls.clip)],
                       check=True, timeout=300, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    @classmethod
    def tearDownClass(cls):
        cls.media.cleanup()

    def setUp(self):
        super().setUp()
        if workers():
            self.skipTest('another console worker is running on this host (the launcher would refuse)')
        self.port = free_port()
        self.vlc = None
        uri = self.dir/'camera.uri'
        uri.write_text(f'rtsp://operator:{SECRET}@127.0.0.1:{self.port}/owned'); uri.chmod(0o600)
        self.uri = uri
        self.addCleanup(self.stop_camera)
        self.camera_dirs_before = private_camera_dirs()

    def start_camera(self):
        config = self.dir/'vlc.conf'
        config.write_text(f'sout=#rtp{{sdp=rtsp://127.0.0.1:{self.port}/owned}}\nsout-keep=1\nrtsp-host=127.0.0.1\n')
        self.vlc = subprocess.Popen([VLC, '--config', str(config), '--intf', 'dummy', '--no-audio', '--loop',
                                     '--no-media-library', str(self.clip)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                    env={**os.environ, 'XDG_CONFIG_HOME': str(self.dir/'vlc-config'),
                                         'XDG_CACHE_HOME': str(self.dir/'vlc-cache')})
        deadline = time.monotonic()+15
        while time.monotonic() < deadline:
            with socket.socket() as probe:
                if probe.connect_ex(('127.0.0.1', self.port)) == 0:return
            time.sleep(.1)
        self.fail('loopback RTSP source did not listen')

    def stop_camera(self):
        if self.vlc is not None and self.vlc.poll() is None:
            self.vlc.kill();self.vlc.wait(10)
        self.vlc = None

    def fixture_env(self, vehicles=200, **extra):
        return self.env(WAYCON_RTSP_URI_FILE=str(self.uri), WAYCON_CONSOLE_TEST_PROFILE='sequential',
                        WAYCON_CONSOLE_TEST_VEHICLES=str(vehicles), WAYCON_RTSP_TEST_SKIP_PREFLIGHT='1', **extra)

    def runs(self):
        root = self.dir/'state/runs'
        found = [json.loads((d/'status.json').read_text()) for d in root.iterdir() if (d/'status.json').exists()] if root.exists() else []
        return sorted(found, key=lambda s: s['started_at'])

    def popen(self, args, env):
        log = (self.dir/'launcher.log').open('w')
        child = subprocess.Popen([str(LAUNCHER), *args], env=env, stdout=log, stderr=subprocess.STDOUT)
        self.addCleanup(lambda: child.poll() is None and (child.kill(), child.wait()))
        return child

    def wait_for(self, predicate, timeout, child=None, every=.5):
        deadline, peak = time.monotonic()+timeout, 0
        while time.monotonic() < deadline:
            peak = max(peak, len(workers()))
            if predicate():return peak
            if child is not None and child.poll() is not None and not predicate():
                self.fail('launcher exited early: '+(self.dir/'launcher.log').read_text()[-3000:])
            time.sleep(every)
        self.fail('timed out; log tail: '+(self.dir/'launcher.log').read_text()[-3000:])

    def assert_clean(self, status):
        self.assertEqual(status['cleanup'], dict(decoder_closed=True, models_closed=True, redis_removed=True,
                                                 private_camera_removed=True, native_children_exited=True), status)

    def test_camera_connection_failure_is_bounded_and_clean(self):
        code, out = self.launch(['--console-only', '--reconnect-attempts', '1', '--reconnect-delay', '1'],
                                self.fixture_env(), timeout=240)
        self.assertEqual(code, 3, out)
        self.assertIn('Camera unavailable: 2 consecutive connection failures', out)
        runs = self.runs()
        self.assertEqual(len(runs), 2)
        for status in runs:
            self.assertEqual((status['status'], status['error_code']), ('failed', 'input_unavailable'))
            self.assert_clean(status)
        self.assertEqual(workers(), [])
        self.assertEqual(private_camera_dirs()-self.camera_dirs_before, set())
        self.assert_no_secret(out)

    def test_bounded_run_delivers_every_lifetime_through_redis(self):
        self.start_camera()
        code, out = self.launch(['--console-only', '--max-seconds', '40', *LIVE], self.fixture_env(), timeout=300)
        self.assertEqual(code, 0, out)
        (status,) = self.runs()
        counters = status['counters']
        self.assertEqual((status['status'], status['end_reason']), ('partial', 'time_limit'))
        self.assertGreater(status['analyzed_frames'], 20)
        self.assertGreater(counters['tracks'], 0)
        self.assertEqual(counters['canonical_redis_deliveries'], counters['tracks']-sum(counters['blocked_lifetimes'].values()))
        self.assertEqual(status['manufacturer_runtime']['quality_version'], 'manufacturer-crop-quality-v2')
        self.assertEqual(status['manufacturer_runtime']['selection'], 'quality_lifetime_v1')
        # Live capture is independent of inference: decoded >= analyzed, excess counted, never queued.
        self.assertGreaterEqual(status['decoded_frames'], status['analyzed_frames'])
        self.assertEqual(status['decoded_frames'], status['analyzed_frames']+status['sampling_skips']+status['overload_drops']
                         + (status.get('selected_pending') or 0), status)
        self.assert_clean(status)
        self.assertEqual(workers(), [])
        self.assert_no_secret(out)

    def test_outage_settles_then_reconnects_with_a_new_run_and_ctrl_c_is_clean(self):
        self.start_camera()
        child = self.popen(['--console-only', '--reconnect-attempts', '6', '--reconnect-delay', '1',
                            '--progress-seconds', '2', *LIVE], self.fixture_env())
        peak = self.wait_for(lambda: any((s.get('analyzed_frames') or 0) >= 60 for s in self.runs()), 240, child)
        self.stop_camera()  # camera outage mid-run
        peak = max(peak, self.wait_for(lambda: self.runs() and self.runs()[0]['status'] == 'partial'
                                       and not self.runs()[0].get('active_slot'), 120, child))
        first = self.runs()[0]
        self.assertEqual(first['end_reason'], 'camera_outage', first)
        self.assertEqual(first['phase'], 'Partial run: camera outage')
        counters = first['counters']
        self.assertGreater(counters['tracks'], 0)
        # Lifetimes observed before the outage are closed, settled and delivered.
        self.assertEqual(counters['canonical_redis_deliveries'], counters['tracks']-sum(counters['blocked_lifetimes'].values()))
        self.assert_clean(first)
        time.sleep(2)
        self.start_camera()  # camera back: the launcher reconnects with a NEW run
        peak = max(peak, self.wait_for(lambda: any(s['run_id'] != first['run_id'] and (s.get('analyzed_frames') or 0) >= 10
                                                   for s in self.runs()), 240, child))
        child.send_signal(signal.SIGINT)  # Ctrl+C
        self.assertEqual(child.wait(180), 0, (self.dir/'launcher.log').read_text()[-3000:])
        runs = self.runs()
        last = runs[-1]
        self.assertEqual((last['status'], last['end_reason'], last['stop_requested']), ('stopped', 'controlled_stop', True))
        for status in runs:
            self.assert_clean(status)
        self.assertEqual(len({s['run_id'] for s in runs}), len(runs))
        self.assertLessEqual(peak, 1, 'more than one inference worker ran at once')
        self.assertEqual(workers(), [])
        self.assertEqual(private_camera_dirs()-self.camera_dirs_before, set())
        log = (self.dir/'launcher.log').read_text()
        self.assertIn('reconnecting with a new run', log)
        self.assertIn('Stopped. Owned worker', log)
        self.assert_no_secret(log)

    @unittest.skipUnless(os.environ.get('DB1_P4_ADMIN_DSN') and os.environ.get('WAYCON_PG_PORT'),
                         'requires the disposable PostgreSQL 16 fixture (existing-server mode)')
    def test_db1_fixture_delivery_commits_before_ack(self):
        self.start_camera()
        env = self.fixture_env(vehicles=8, WAYCON_CONSOLE_TEST_DB1_FIXTURE='1', WAYCON_POSTGRES_MODE='existing',
                               WAYCON_PG_PORT=os.environ['WAYCON_PG_PORT'], DB1_P4_ADMIN_DSN=os.environ['DB1_P4_ADMIN_DSN'])
        code, out = self.launch(['--max-seconds', '45', *LIVE], env, timeout=400)
        self.assertEqual(code, 0, out)
        (status,) = self.runs()
        counters = status['counters']
        delivered = counters['canonical_redis_deliveries']
        self.assertGreater(delivered, 0)
        self.assertEqual((counters['db1_committed_tracks'], counters['db1_acknowledged_tracks']), (delivered, delivered), counters)
        self.assertEqual(status['delivery']['postgres_commit'], 'committed')
        self.assert_clean(status)
        self.assert_no_secret(out)
        self.assertNotIn('postgresql://', out)


if __name__ == '__main__':
    unittest.main()
