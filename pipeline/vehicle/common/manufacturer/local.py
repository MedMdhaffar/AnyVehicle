"""Owned, deadline-limited CUDA subprocesses using explicit verified artifacts.

``LocalRuntime`` (legacy, rollback) starts one native process per crop.
``PersistentRuntime`` keeps one owned native process with the model loaded and
serves crops from a bounded queue on a dispatcher thread, so the Engine frame
loop never waits for inference. Both share the verified lock, the isolated
interpreter (-I -B), the network-disabled worker and the per-call deadline.
"""
from collections import deque
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import queue
import select
import signal
import struct
import subprocess
import tempfile
import threading
import time

from ..db1_rows.core import Blocked, require
from ..gpu_lease import inherited_lease
from .engine import hashed

PREPROCESSING = 'original-bgr-to-rgb-png-v1'
MAX_CROP_BYTES = 8 * 1024 * 1024


class ManufacturerUnavailable(Blocked):
    """A required verified manufacturer asset or runtime is unavailable."""


def load_lock(lock_path):
    """Resolve a relocatable lock and verify every artifact; fail with its label."""
    path = Path(lock_path).expanduser()
    if not path.is_file():
        raise ManufacturerUnavailable('MANUFACTURER_LOCK_MISSING', f'manufacturer lock not found: {path}')
    path = path.resolve()
    lock = json.loads(path.read_text())
    # Relative paths are local to a relocated verified manifest.
    for item in lock['artifacts'].values():
        item_path = Path(item['path']).expanduser()
        item['path'] = str(item_path if item_path.is_absolute() else path.parent / item_path)
    runtime = Path(lock['runtime']).expanduser()
    lock['runtime'] = str(runtime if runtime.is_absolute() else path.parent / runtime)
    for label, item in lock['artifacts'].items():
        artifact = Path(item['path'])
        if not artifact.is_file():
            raise ManufacturerUnavailable('MANUFACTURER_ASSET_MISSING', f'manufacturer asset {label!r} not found: {artifact}')
        with artifact.open('rb') as stream:
            digest = __import__('hashlib').file_digest(stream, 'sha256').hexdigest()
        require(artifact.stat().st_size == item['bytes'] and digest == item['sha256'],
                'ARTIFACT_MISMATCH', f'manufacturer artifact {label!r} changed')
    return path, lock


def code_hashes():
    return {p.name: sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}


def encode_png(crop):
    from PIL import Image
    require(crop.dtype.name == 'uint8' and crop.ndim == 3 and crop.shape[2] == 3
            and crop.nbytes <= MAX_CROP_BYTES, 'REJECTED_CROP', 'bounded BGR crop required')
    with Image.fromarray(crop[:, :, ::-1]) as image, io.BytesIO() as buffer:
        image.save(buffer, format='PNG')
        return buffer.getvalue()


def visible_device():
    index = int(os.environ.get('T1_CONSOLE_DEVICE', 'cuda:0').split(':')[-1])
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    return visible.split(',')[index].strip() if visible else str(index)


def worker_environment(lock):
    return dict(PATH=lock['runtime'] + '/bin:/usr/bin:/bin', LD_LIBRARY_PATH=lock['runtime'] + '/lib',
                CUDA_VISIBLE_DEVICES=visible_device(), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                PYTHONDONTWRITEBYTECODE='1')


class LocalRuntime:
    mode = 'legacy'

    def __init__(self, lock_path, *, timeout=30, stop=None):
        require(type(timeout) in (float, int) and 0 < timeout <= 60,
                'INVALID_WINDOW', 'manufacturer timeout must be <=60 seconds')
        self.lock_path, self.lock = load_lock(lock_path)
        self.timeout = timeout
        self.stop = stop
        self.code = code_hashes()
        self.provenance = hashed(dict(lock=self.lock, code=self.code,
                                      timeout=timeout, preprocessing=PREPROCESSING))
        self.children = []
        self.metrics = []

    def infer(self, crop):
        require(crop.dtype.name == 'uint8' and crop.ndim == 3 and crop.shape[2] == 3
                and crop.nbytes <= MAX_CROP_BYTES, 'REJECTED_CROP', 'bounded BGR crop required')
        for name, digest in self.code.items():
            require(sha256((Path(__file__).parent/name).read_bytes()).hexdigest() == digest,
                    'ARTIFACT_MISMATCH', 'manufacturer code changed after preflight')
        pixels = encode_png(crop)
        env = worker_environment(self.lock)
        # No server, shared environment installation, provider, or CPU route.
        command = [self.lock['artifacts']['python']['path'], '-B', '-I',
                   str(Path(__file__).with_name('native_worker.py')), str(self.lock_path)]
        with tempfile.TemporaryFile() as output:
            child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=output,
                                     stderr=subprocess.DEVNULL, env=env, start_new_session=True,pass_fds=inherited_lease())
            self.children.append(child)
            # A dedicated feeder writes the whole PNG and closes stdin. Polling
            # communicate(input) cannot resume a partial write after its first
            # timeout, which deadlocked any PNG larger than the pipe buffer.
            def feed():
                try:
                    child.stdin.write(pixels)
                except (BrokenPipeError, OSError, ValueError):
                    pass
                finally:
                    try:child.stdin.close()
                    except (BrokenPipeError, OSError, ValueError):pass
            feeder = threading.Thread(target=feed, name='manufacturer-feed', daemon=True)
            feeder.start()
            try:
                deadline = time.monotonic()+self.timeout
                while True:
                    stop = getattr(self,'stop',None)
                    if stop is not None and stop.is_set():
                        raise RuntimeError('manufacturer controlled stop')
                    remaining = deadline-time.monotonic()
                    if remaining <= 0:raise subprocess.TimeoutExpired(command,self.timeout)
                    try:
                        child.wait(timeout=min(.1,remaining))
                        break
                    except subprocess.TimeoutExpired:
                        pass
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=5)
                raise TimeoutError('manufacturer owned process deadline') from None
            finally:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait(timeout=5)
                feeder.join(timeout=5)
            output.seek(0)
            raw = output.read(4097)
        require(child.returncode == 0 and len(raw) <= 4096,
                'MANUFACTURER_EXECUTION_FAILED', 'owned CUDA worker did not complete')
        result = json.loads(raw)
        require(set(result) in ({'outcome','text','gpu_memory'}, {'outcome','text','gpu_memory','timing'}) and result['outcome'] in {'completed','malformed'},
                'INVALID_MANUFACTURER', 'invalid native result')
        self.metrics.append(result['gpu_memory'])
        self.last_evaluation_timing=dict(result['timing'],worker_pid=child.pid) if result.get('timing') else None
        # A token-limit result remains a completed native call with malformed
        # bounded text. Policy will reject the deliberately empty response.
        return 'completed', result['text']

    @property
    def cleaned(self):
        return all(child.poll() is not None for child in self.children)

    visible_device = staticmethod(visible_device)


def _process_vram_mib(pid):
    """NVML per-process memory on the visible device/MIG instance, or None."""
    try:
        import pynvml
        pynvml.nvmlInit()
        handles = []
        for index in range(pynvml.nvmlDeviceGetCount()):
            parent = pynvml.nvmlDeviceGetHandleByIndex(index)
            try:
                if pynvml.nvmlDeviceGetMigMode(parent)[0]:
                    for slot in range(pynvml.nvmlDeviceGetMaxMigDeviceCount(parent)):
                        try:handles.append(pynvml.nvmlDeviceGetMigDeviceHandleByIndex(parent, slot))
                        except pynvml.NVMLError:pass
                    continue
            except pynvml.NVMLError:
                pass
            handles.append(parent)
        for handle in handles:
            for process in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
                if process.pid == pid and isinstance(process.usedGpuMemory, int):
                    return process.usedGpuMemory / 2**20
    except Exception:
        return None
    return None


class PersistentRuntime:
    """One owned, model-resident native process per runtime instance.

    Thread model: the Engine owner thread calls ``can_accept``/``submit``/
    ``collect``/``wait``/``cancel``; a private dispatcher thread owns the child.
    Results are (request_id, outcome, text, seconds) with outcome in
    {completed, timed_out, failed, cancelled, execution_uncertain}.
    """
    mode = 'persistent'

    def __init__(self, lock_path, *, timeout=30, stop=None, queue_size=8, max_restarts=3,
                 ready_timeout=180, start=True, command=None):
        require(type(timeout) in (float, int) and 0 < timeout <= 60,
                'INVALID_WINDOW', 'manufacturer timeout must be <=60 seconds')
        require(type(queue_size) is int and 1 <= queue_size <= 64 and type(max_restarts) is int and 0 <= max_restarts <= 16,
                'INVALID_CAPACITY', 'bounded manufacturer queue and restart budget required')
        self.lock_path, self.lock = load_lock(lock_path)
        self.timeout, self.stop, self.ready_timeout = timeout, stop, ready_timeout
        self.code = code_hashes()
        self.provenance = hashed(dict(lock=self.lock, code=self.code, timeout=timeout,
                                      preprocessing=PREPROCESSING, runtime='persistent-v1'))
        self.command = command or [self.lock['artifacts']['python']['path'], '-B', '-I',
                                   str(Path(__file__).with_name('native_worker.py')), str(self.lock_path), '--serve']
        self.queue_size, self.max_restarts = queue_size, max_restarts
        self.requests = queue.Queue(maxsize=queue_size)
        self.results = deque()
        self.guard = threading.Lock()
        self.children, self.metrics, self.starts = [], [], []
        self.child = None
        self.restarts = 0
        self.failure = None          # runtime-level unavailability code
        self.inflight = None
        self.outstanding = 0
        self.stopping = threading.Event()
        self.closed = False
        if start:
            self._ensure_child()     # Load before the video loop; surface failures early.
        self.thread = threading.Thread(target=self._dispatch, name='manufacturer-dispatch', daemon=True)
        self.thread.start()

    # ---- Engine owner thread -------------------------------------------------
    def can_accept(self):
        return not self.closed and not self.stopping.is_set() and not self.requests.full()

    def submit(self, request_id, crop):
        require(self.can_accept(), 'MANUFACTURER_QUEUE_FULL', 'bounded manufacturer queue is full')
        require(crop.dtype.name == 'uint8' and crop.ndim == 3 and crop.shape[2] == 3
                and crop.nbytes <= MAX_CROP_BYTES, 'REJECTED_CROP', 'bounded BGR crop required')
        with self.guard:
            self.outstanding += 1
        if not hasattr(self,'evaluation_queued'):self.evaluation_queued={}
        self.evaluation_queued[request_id]=time.perf_counter_ns()
        self.requests.put_nowait((request_id, crop.copy(), time.monotonic()))

    def collect(self):
        items = []
        with self.guard:
            while self.results:
                items.append(self.results.popleft())
            self.outstanding -= len(items)
        return items

    def pending(self):
        with self.guard:
            return self.outstanding

    def wait(self, timeout):
        """Block (owner thread, never the frame loop) until all results are ready."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self.guard:
                if len(self.results) >= self.outstanding:
                    return True
            if self.stop is not None and self.stop.is_set():
                return False
            time.sleep(0.01)
        return False

    def cancel(self):
        """Controlled stop: kill the child; queued work is cancelled, in-flight is uncertain."""
        self.stopping.set()
        with self.guard:
            child = self.child
        if child is not None and child.poll() is None:
            self._kill(child)
        while True:
            try:
                request_id, _, started = self.requests.get_nowait()
            except queue.Empty:
                break
            self._finish(request_id, 'cancelled', '', started)

    def close(self):
        if self.closed:
            return
        self.stopping.set()
        self.closed = True
        try:
            # Queued items drain as cancelled once stopping is set; then the sentinel fits.
            self.requests.put(None, timeout=self.timeout + 10)
        except queue.Full:
            pass
        self.thread.join(timeout=self.timeout + 10)
        with self.guard:
            child = self.child
        if child is not None and child.poll() is None:
            try:
                child.stdin.write(struct.pack('>I', 0));child.stdin.flush()
                child.wait(timeout=10)
            except Exception:
                pass
            if child.poll() is None:
                self._kill(child)
        for child in self.children:
            if child.poll() is None:
                self._kill(child)

    @property
    def cleaned(self):
        return all(child.poll() is not None for child in self.children) and not self.thread.is_alive()

    # ---- Dispatcher thread ---------------------------------------------------
    def _finish(self, request_id, outcome, text, started, **metric):
        seconds = min(65.0, max(0.0, time.monotonic() - started))
        with self.guard:
            self.results.append((request_id, outcome, text, seconds))
            if metric:
                self.metrics.append(dict(metric, seconds=seconds, outcome=outcome))

    def _kill(self, child):
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass

    def _read(self, child, size, deadline):
        data = b''
        fd = child.stdout.fileno()
        while len(data) < size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('manufacturer persistent deadline')
            ready, _, _ = select.select([fd], [], [], min(0.1, remaining))
            if not ready:
                if child.poll() is not None:
                    raise EOFError('manufacturer worker exited')
                continue
            chunk = os.read(fd, size - len(data))
            if not chunk:
                raise EOFError('manufacturer worker exited')
            data += chunk
        return data

    def _message(self, child, deadline):
        size = struct.unpack('>I', self._read(child, 4, deadline))[0]
        require(0 < size <= 65536, 'INVALID_MANUFACTURER', 'bounded native response required')
        return json.loads(self._read(child, size, deadline))

    def _ensure_child(self):
        with self.guard:
            child = self.child
        if child is not None and child.poll() is None:
            return child
        if self.failure is not None:
            raise ManufacturerUnavailable(self.failure, 'manufacturer runtime unavailable')
        if self.starts:
            if self.restarts >= self.max_restarts:
                self.failure = 'MANUFACTURER_RESTART_BUDGET'
                raise ManufacturerUnavailable(self.failure, 'manufacturer restart budget exhausted')
            self.restarts += 1
        for name, digest in self.code.items():
            require(sha256((Path(__file__).parent/name).read_bytes()).hexdigest() == digest,
                    'ARTIFACT_MISMATCH', 'manufacturer code changed after preflight')
        started = time.monotonic()
        child = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                 stderr=subprocess.DEVNULL, env=worker_environment(self.lock),
                                 start_new_session=True, bufsize=0,pass_fds=inherited_lease())
        with self.guard:
            self.child = child
            self.children.append(child)
        try:
            ready = self._message(child, started + self.ready_timeout)
            require(ready.get('ready') is True, 'MANUFACTURER_EXECUTION_FAILED', 'native worker did not load')
        except Exception:
            self._kill(child)
            if len(self.starts) == 0:
                self.failure = 'MANUFACTURER_LOAD_FAILED'
            self.starts.append(dict(seconds=time.monotonic() - started, ready=False))
            raise
        self.starts.append(dict(seconds=time.monotonic() - started, ready=True, pid=child.pid,
                                gpu_memory=ready.get('gpu_memory'), process_vram_mib=_process_vram_mib(child.pid)))
        return child

    def _dispatch(self):
        while True:
            item = self.requests.get()
            if item is None:
                return
            request_id, crop, started = item
            dispatch_ns=time.perf_counter_ns()
            queue_wait_ms=max(0.0,(dispatch_ns-getattr(self,'evaluation_queued',{}).pop(request_id,dispatch_ns))/1e6)
            if self.stopping.is_set():
                self._finish(request_id, 'cancelled', '', started)
                continue
            try:
                png = encode_png(crop)
                child = self._ensure_child()
            except Exception:
                self._finish(request_id, 'failed', '', started)
                continue
            self.inflight = request_id
            try:
                child.stdin.write(struct.pack('>I', len(png)) + png)
                child.stdin.flush()
                result = self._message(child, time.monotonic() + self.timeout)
                require(set(result) in ({'outcome', 'text', 'gpu_memory'}, {'outcome','text','gpu_memory','timing'}) and result['outcome'] in {'completed', 'malformed'},
                        'INVALID_MANUFACTURER', 'invalid native result')
                if result.get('timing'):
                    if not hasattr(self,'evaluation_timings'):self.evaluation_timings={}
                    self.evaluation_timings[request_id]=dict(result['timing'],worker_pid=child.pid,queue_wait_ms=queue_wait_ms,
                        host_dispatch_ns=str(dispatch_ns),host_completion_ns=str(time.perf_counter_ns()))
                self._finish(request_id, 'completed', result['text'], started, pid=child.pid,
                             gpu_memory=result['gpu_memory'], process_vram_mib=_process_vram_mib(child.pid))
            except TimeoutError:
                self._kill(child)
                self._finish(request_id, 'timed_out', '', started, pid=child.pid)
            except Exception:
                self._kill(child)
                outcome = 'execution_uncertain' if self.stopping.is_set() else 'failed'
                self._finish(request_id, outcome, '', started, pid=child.pid)
            finally:
                self.inflight = None


def create_runtime(lock_path, *, mode=None, timeout=30, stop=None, **options):
    """Select the configured runtime; legacy remains an explicit rollback."""
    mode = mode or os.environ.get('T1_MANUFACTURER_RUNTIME', 'persistent')
    require(mode in {'persistent', 'legacy'}, 'INVALID_CONFIGURATION', 'T1_MANUFACTURER_RUNTIME must be persistent or legacy')
    if mode == 'legacy':
        return LocalRuntime(lock_path, timeout=timeout, stop=stop)
    return PersistentRuntime(lock_path, timeout=timeout, stop=stop, **options)
