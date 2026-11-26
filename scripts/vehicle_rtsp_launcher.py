"""Foreground supervisor for ONE RTSP camera through the existing console run lifecycle.

It does not add a pipeline: it calls `RunStore.start(source_kind='rtsp')`, exactly as
the web console's Start does. That gives the same private camera staging (mode-0600
`camera.json`, endpoint never in argv/logs/status), the same `vehicle_console_worker.py`
(shared YOLO26 + ByteTrack, HSV colour, plate OCR, persistent manufacturer worker,
crop-quality v2, run-owned Redis with epoch rotation, DB1 commit-before-ACK), and the
same single-slot latest-frame decoder (no frame queue). No web server is started.

Reconnection is bounded and follows the architecture: an RTSP Input never reopens,
so a lost camera ends the run (`camera_outage`, lifetimes closed/settled/delivered)
and a NEW run (new Input, clock, Engine and tracker) is started after a backoff.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
EXIT_OK, EXIT_FAILED, EXIT_CONFIG, EXIT_CAMERA = 0, 1, 2, 3
STOP_DRAIN_SECONDS = 330  # the console launcher's run-slot drain bound
REQUIRED = dict(T1_MANUFACTURER_RUNTIME='persistent', T1_MANUFACTURER_QUALITY='v2',
                T1_MANUFACTURER_SELECTION='quality_lifetime_v1')


class LaunchError(RuntimeError):
    """Message is operator-facing and never contains the endpoint."""
    def __init__(self, message, code=EXIT_FAILED):
        super().__init__(message)
        self.code = code


def tell(message):
    print('[waycon-rtsp '+time.strftime('%H:%M:%S')+'] '+message, file=sys.stderr, flush=True)


def camera_uri():
    """Exactly one source: an owned mode-0600 file (recommended) or an env variable.

    Both variables are removed from this process's environment before any child
    starts, so the endpoint reaches the worker only through the private snapshot.
    """
    from vehicle.common.real_video.rtsp import uri_file, InputFault
    path, direct = os.environ.pop('WAYCON_RTSP_URI_FILE', None), os.environ.pop('WAYCON_RTSP_URI', None)
    if path and direct:
        raise LaunchError('Set only one of WAYCON_RTSP_URI_FILE or WAYCON_RTSP_URI.', EXIT_CONFIG)
    if not path and not direct:
        raise LaunchError('Missing camera configuration: set WAYCON_RTSP_URI_FILE to an owned mode-0600 file '
                          'containing the rtsp:// URL (recommended), or WAYCON_RTSP_URI.', EXIT_CONFIG)
    if path:
        try:
            uri = uri_file(path)
        except InputFault:
            raise LaunchError('WAYCON_RTSP_URI_FILE must be an existing regular file owned by you with mode 0600 '
                              '(chmod 600) and at most 16 KiB.', EXIT_CONFIG) from None
    else:
        uri = direct.strip()
    validate_uri(uri)
    return uri


def validate_uri(uri):
    """Run the worker's own validator on a throw-away private snapshot."""
    from vehicle.common.real_video.rtsp import stage_camera, InputFault
    scratch = Path(tempfile.mkdtemp(prefix='waycon-rtsp-validate-'))
    try:
        os.chmod(scratch, 0o700)
        stage_camera(uri, scratch, camera_id='validation')
    except InputFault:
        raise LaunchError('Invalid RTSP URL (value not shown): expected rtsp://[user:password@]host[:port]/path '
                          'without spaces or control characters.', EXIT_CONFIG) from None
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def pin_runtime():
    for name, value in REQUIRED.items():
        current = os.environ.get(name)
        if current not in (None, '', value):
            raise LaunchError(f'This launcher requires {name}={value}; unset {name} '
                              '(rollbacks remain available through the web console).', EXIT_CONFIG)
        os.environ[name] = value


def other_inference_owner(launcher_root):
    """Refuse a second GPU pipeline: the web console or any other console worker."""
    web = Path(os.environ.get('WAYCON_LAUNCHER_ROOT', ROOT/'vehicle/media/waycon-launcher'))/'launcher.lock'
    if web.exists() and web.resolve() != (launcher_root/'launcher.lock').resolve():
        with web.open('a') as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(handle, fcntl.LOCK_UN)
            except BlockingIOError:
                return 'the WAYCON web console launcher is running (stop it with scripts/run_vehicle_console.sh --stop)'
    worker = os.fsencode(str(ROOT/'scripts/vehicle_console_worker.py'))
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():continue
        try:
            if entry.stat().st_uid != os.getuid():continue
            if worker in (entry/'cmdline').read_bytes().split(b'\0'):
                return f'another console worker is running (pid {entry.name})'
        except OSError:
            continue
    return None


def test_fixture_mode():
    """Explicit fixture-model tests only; both variables are required."""
    return (os.environ.get('WAYCON_CONSOLE_TEST_PROFILE') in ('empty', 'sequential')
            and os.environ.get('WAYCON_RTSP_TEST_SKIP_PREFLIGHT') == '1')


def preflight(directory, mode):
    """Reuse the console launcher's preflight: CUDA, artifacts, owned Redis, DB1 mode."""
    if test_fixture_mode():
        tell('TEST PROFILE: native preflight skipped explicitly (fixture models only).')
        return dict(manufacturer=dict(available=True))
    tell('Checking CUDA, local model artifacts, run-owned Redis and the DB1 mode (no camera is opened)…')
    with (directory/'preflight.log').open('wb') as log:
        result = subprocess.run([sys.executable, '-B', str(ROOT/'scripts/vehicle_console_launcher.py'),
                                 '--preflight', str(directory), mode], stdout=log, stderr=log,
                                stdin=subprocess.DEVNULL, timeout=600, start_new_session=True)
    report_path = directory/'preflight.json'
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    for failure in report.get('failures', []):tell(failure)
    if result.returncode or report.get('status') != 'ready':
        raise LaunchError('Preflight failed; see '+str(directory/'preflight.log')+'. No camera was opened.', EXIT_CONFIG)
    if not report.get('manufacturer', {}).get('available'):
        raise LaunchError('Manufacturer runtime unavailable (verified local lock and assets required); '
                          'see '+str(directory/'prerequisites.json')+'.', EXIT_CONFIG)
    return report


def remove_private_camera(directory):
    """Safety net: the worker deletes the endpoint snapshot; ensure it after a crash too."""
    try:
        source = Path(json.loads((directory/'config.json').read_text())['source'])
    except (OSError, ValueError, KeyError):
        return True
    private = source.parent
    if private.name.startswith('waycon-camera-') and private.exists():
        info = private.lstat()
        if stat.S_ISDIR(info.st_mode) and info.st_uid == os.getuid():
            shutil.rmtree(private, ignore_errors=True)
    return not private.exists()


class Supervisor:
    def __init__(self, args, root):
        from vehicle.common.console_runs import RunStore
        self.args, self.root = args, root
        self.store = RunStore(root/'runs')
        self.stop = threading.Event()
        self.interrupts = 0

    def interrupt(self, *_):
        self.interrupts += 1
        self.stop.set()

    def wait(self, seconds):
        return self.stop.wait(seconds)

    def summary(self, status):
        counters = status.get('counters') or {}
        return dict(run_id=status['run_id'], status=status['status'], end_reason=status.get('end_reason'),
                    error_code=status.get('error_code'), input_code=status.get('input_code'),
                    decoded_frames=status.get('decoded_frames'), analyzed_frames=status.get('analyzed_frames'),
                    sampling_skips=status.get('sampling_skips'), overload_drops=status.get('overload_drops'),
                    tracks=counters.get('tracks'), canonical_redis_deliveries=counters.get('canonical_redis_deliveries'),
                    blocked_lifetimes=counters.get('blocked_lifetimes'), redis_epochs=counters.get('redis_epochs'),
                    db1_committed_tracks=counters.get('db1_committed_tracks'),
                    db1_acknowledged_tracks=counters.get('db1_acknowledged_tracks'),
                    manufacturer_calls=counters.get('manufacturer_calls'),
                    manufacturer_model_loads=(status.get('manufacturer_runtime') or {}).get('model_loads'),
                    delivery=status.get('delivery'), cleanup=status.get('cleanup'))

    def run_once(self, attempt):
        from vehicle.common.console_runs import TERMINAL
        status = self.store.start(source_kind='rtsp', rtsp_uri=self.uri, rtsp_transport=self.args.transport,
                                  manufacturer=True, sample_every=self.args.sample_every,
                                  max_seconds=self.args.max_seconds)
        run_id = status['run_id']
        directory = self.store.directory(run_id)
        tell(f'Run {run_id} started (connection attempt {attempt}); worker owns the camera.')
        last_print, previous = 0.0, None
        stop_sent = escalated = None
        while True:
            status = self.store.status(run_id)
            if status['status'] in TERMINAL and not status['active_slot']:
                break
            now = time.monotonic()
            if self.stop.is_set() and stop_sent is None:
                tell('Ctrl+C: stopping the camera input and draining model work (press Ctrl+C again to '
                     'escalate after the drain bound)…')
                self.store.stop(run_id)
                stop_sent = now
            if stop_sent is not None and escalated is None and (
                    now-stop_sent > STOP_DRAIN_SECONDS or self.interrupts >= 3):
                # Same graceful handler as Stop (the worker maps SIGTERM to Stop).
                child = self._child(directory)
                if child is not None and child.poll() is None:
                    tell('Drain bound reached; sending SIGTERM to the owned worker.')
                    child.terminate()
                escalated = now
            if escalated is not None and now-escalated > 30:
                child = self._child(directory)
                if child is not None and child.poll() is None:
                    tell('Worker did not exit; killing its owned process group. Cleanup cannot be confirmed.')
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait(10)
            if now-last_print >= self.args.progress_seconds:
                current = (now, status.get('decoded_frames') or 0, status.get('analyzed_frames') or 0)
                rate = ''
                if previous and current[0] > previous[0]:
                    span = current[0]-previous[0]
                    rate = (f' | decode {(current[1]-previous[1])/span:.2f} fps, '
                            f'analyze {(current[2]-previous[2])/span:.2f} fps')
                tell(f"{status['status']}: {status.get('phase')} | decoded {current[1]}, analyzed {current[2]}, "
                     f"sampling skips {status.get('sampling_skips') or 0}, overload drops "
                     f"{status.get('overload_drops') or 0}, tracks {len(status.get('rows') or [])}{rate}")
                previous, last_print = current, now
            time.sleep(.5)
        removed = remove_private_camera(directory)
        status = self.store.status(run_id)
        try:  # fixed, non-sensitive codes recorded by the worker
            failure = json.loads((directory/'failure.json').read_text())
            status.setdefault('input_code', failure.get('input_code'))
            status['pipeline_code'] = failure.get('pipeline_code')
        except (OSError, ValueError):
            pass
        result = self.summary(status)
        result['pipeline_code'] = status.get('pipeline_code')
        result['private_camera_removed_after_exit'] = removed
        (self.root/'last-run.json').write_text(json.dumps(result, indent=2)+'\n')
        tell('Run ended: '+json.dumps({k: result[k] for k in ('status', 'end_reason', 'error_code', 'input_code',
             'analyzed_frames', 'tracks', 'canonical_redis_deliveries', 'db1_committed_tracks', 'cleanup')}))
        return status

    def _child(self, directory):
        from vehicle.common import console_runs
        return console_runs._CHILDREN.get(str(directory))

    def serve(self):
        failures, runs = 0, 0
        while True:
            if self.stop.is_set():
                return EXIT_OK
            runs += 1
            status = self.run_once(failures+1)
            reason, state = status.get('end_reason'), status['status']
            if status.get('stop_requested') or reason == 'controlled_stop' or self.stop.is_set():
                tell('Stopped. Owned worker, decoder, Redis and private camera snapshot retired.')
                return EXIT_OK if state in ('stopped', 'completed', 'partial') else EXIT_FAILED
            if reason == 'time_limit':
                tell('Configured --max-seconds reached; not reconnecting.')
                return EXIT_OK
            if reason != 'camera_outage':
                tell('Run failed for a non-camera reason ('+str(status.get('error_code'))+'); not reconnecting. '
                     'See '+str(self.store.directory(status['run_id'])/'failure.json'))
                return EXIT_FAILED
            # A run that analyzed frames proves the connection worked; restart the count.
            failures = 1 if (status.get('analyzed_frames') or 0) > 0 else failures+1
            if failures > self.args.reconnect_attempts:
                tell(f'Camera unavailable: {failures} consecutive connection failures '
                     f"(last input code: {status.get('input_code') or status.get('error_code')}). Check the camera, "
                     'network and URL. Exiting.')
                return EXIT_CAMERA
            if self.args.max_runs and runs >= self.args.max_runs:
                tell('Camera outage and --max-runs reached; exiting.')
                return EXIT_CAMERA
            delay = min(self.args.reconnect_max_delay, self.args.reconnect_delay*2**(failures-1))
            tell(f'Camera outage ({status.get("input_code") or status.get("error_code")}); reconnecting with a new '
                 f'run in {delay:.0f}s (attempt {failures} of {self.args.reconnect_attempts}).')
            if self.wait(delay):
                tell('Stopped during reconnect backoff.')
                return EXIT_OK


def parse(argv):
    parser = argparse.ArgumentParser(prog='run_waycon_rtsp.sh', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--check', action='store_true', help='validate configuration and run the native preflight; never opens the camera')
    parser.add_argument('--console-only', action='store_true', help='explicitly run without DB1 (PostgreSQL) delivery')
    parser.add_argument('--transport', choices=('tcp', 'udp'), default=os.environ.get('WAYCON_RTSP_TRANSPORT', 'tcp'))
    parser.add_argument('--sample-every', type=int, default=None,
                        help='analyze every Nth decoded frame (console RTSP default 5); ByteTrack is aged in source frames')
    parser.add_argument('--max-seconds', type=float, default=None, help='bounded run (smoke tests); default unbounded')
    parser.add_argument('--reconnect-attempts', type=int, default=int(os.environ.get('WAYCON_RTSP_RECONNECT_ATTEMPTS', '5')))
    parser.add_argument('--reconnect-delay', type=float, default=float(os.environ.get('WAYCON_RTSP_RECONNECT_DELAY', '2')))
    parser.add_argument('--reconnect-max-delay', type=float, default=60.0)
    parser.add_argument('--max-runs', type=int, default=0, help='total runs including reconnects (0 = no limit)')
    parser.add_argument('--progress-seconds', type=float, default=10.0)
    args = parser.parse_args(argv)
    if args.sample_every is not None and args.sample_every < 1:parser.error('--sample-every must be >= 1')
    if args.max_seconds is not None and not args.max_seconds > 0:parser.error('--max-seconds must be > 0')
    if not 0 <= args.reconnect_attempts <= 100:parser.error('--reconnect-attempts must be 0..100')
    if not (0 < args.reconnect_delay <= args.reconnect_max_delay <= 600):parser.error('invalid reconnect delays')
    if args.max_runs < 0 or args.progress_seconds <= 0:parser.error('invalid --max-runs/--progress-seconds')
    return args


def main(argv=None):
    os.umask(0o077)
    args = parse(argv)
    from vehicle.common.console_stack import configure
    configure(load_dsn=False)
    uri = camera_uri()  # before anything else: fail fast on configuration
    pin_runtime()
    mode = 'console-only' if args.console_only else 'fixture'
    os.environ.update(T1_CONSOLE_DB1=mode, PYTHONDONTWRITEBYTECODE='1', YOLO_OFFLINE='true', YOLO_AUTOINSTALL='false',
                      COLOR_AGENT_ENABLED='false', PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK='True', OMP_NUM_THREADS='1',
                      OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    postgres_mode = os.environ.get('WAYCON_POSTGRES_MODE', 'owned')
    if postgres_mode == 'existing':
        os.environ['DB1_P4_EXISTING_SERVER'] = '1'
        os.environ.pop('DB1_P4_ISOLATED', None)
    elif postgres_mode == 'owned':
        for name in ('DB1_P4_ADMIN_DSN', 'DB1_P4_EXISTING_SERVER', 'WAYCON_DSN_FROM_FILE'):
            os.environ.pop(name, None)
    else:
        raise LaunchError('WAYCON_POSTGRES_MODE must be owned or existing.', EXIT_CONFIG)
    redis = os.environ.get('T1_REDIS_SERVER')
    if not redis or not os.access(redis, os.X_OK):
        raise LaunchError('T1_REDIS_SERVER must name an executable redis-server (set it or redis_server in '
                          '~/.config/waycon/console.json).', EXIT_CONFIG)
    root = Path(os.environ.get('WAYCON_RTSP_ROOT', ROOT/'vehicle/media/waycon-rtsp')).resolve()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    root.chmod(0o700)
    lock = (root/'launcher.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise LaunchError('Another RTSP launcher already uses '+str(root)+'.', EXIT_CONFIG) from None
    busy = other_inference_owner(root)
    if busy:
        raise LaunchError('Refusing to start a second inference pipeline: '+busy+'.', EXIT_CONFIG)
    supervisor = Supervisor(args, root)
    supervisor.uri = uri
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, supervisor.interrupt)
    from vehicle.common.console_prerequisites import inventory
    from vehicle.common.console_runs import atomic_json
    from contextlib import ExitStack
    prerequisites = inventory(mode)
    atomic_json(root/'prerequisites.json', prerequisites)
    for warning in prerequisites['warnings']:tell(warning)
    if prerequisites['missing'] and not test_fixture_mode():
        for missing in prerequisites['missing']:tell(missing)
        raise LaunchError('Project prerequisites are missing; nothing was installed or started. See '
                          +str(root/'prerequisites.json'), EXIT_CONFIG)
    with ExitStack() as resources:
        if mode == 'fixture' and postgres_mode == 'owned':
            from vehicle.common.console_postgres import OwnedPostgres
            tell('Starting a launcher-owned local PostgreSQL 16 test instance…')
            resources.enter_context(OwnedPostgres(root))
        preflight(root, mode)
        if supervisor.stop.is_set():
            tell('Stopped during preflight; no camera was opened.')
            return EXIT_OK
        if args.check:
            tell('Configuration, camera URL syntax and preflight passed. No camera was opened.')
            return EXIT_OK
        if supervisor.stop.is_set():
            return EXIT_OK
        tell(f'Starting one RTSP camera: transport {args.transport}, DB1 {mode}, manufacturer persistent, '
             'crop quality v2. Ctrl+C stops cleanly.')
        return supervisor.serve()


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except LaunchError as error:
        tell(str(error))
        raise SystemExit(error.code)
    except KeyboardInterrupt:
        tell('Interrupted before a run started; nothing is running.')
        raise SystemExit(EXIT_OK)
    except Exception as error:
        # Never stringify arbitrary exceptions: driver/native errors may embed secrets.
        tell('Launcher failed ('+type(error).__name__+'); see the private run directory for codes.')
        raise SystemExit(EXIT_FAILED)
