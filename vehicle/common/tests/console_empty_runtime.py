"""Explicit server-side acceptance fixtures. Never selectable in an API body.

``empty``: no detections (or one tracked vehicle for RTSP).
``sequential``: N vehicles appear one after another on the real decoded frames,
with injected plate executors and an in-process persistent manufacturer fixture,
to exercise long-run lifetime release, progressive settlement and asynchronous
manufacturer closure deferral without a GPU.
"""
from collections import deque
from hashlib import sha256
import os
import tempfile
import threading
import time

import numpy as np


class EmptyVehicle:
    def __init__(self, tracked=False):
        self.prediction_calls=self.forward_calls=0
        self.closed=False
        self.placement=dict(profile='explicit empty detector fixture')
        self.tracked=tracked
    def process(self,pixels):
        self.prediction_calls+=1;self.forward_calls+=1
        return np.array([[10,10,80,80,7,.9,2,0]],dtype=np.float32) if self.tracked else np.empty((0,8),dtype=np.float32)
    def close(self):self.closed=True


class EmptyPlate:
    def process(self,pixels):
        from vehicle.common.real_video.plate_local import PlateDetections
        return PlateDetections((),())
    def close(self):pass


class SequentialVehicle:
    """Vehicle k (raw ID k) is visible for `visible` analyzed frames of each `period`."""
    def __init__(self, count, visible=4, period=20, box=(300, 150, 900, 550)):
        self.count, self.visible, self.period, self.box = count, visible, period, box
        self.prediction_calls = self.forward_calls = 0
        self.closed = False
        self.placement = dict(profile='explicit sequential detector fixture', vehicles=count)
    def process(self, pixels):
        index = self.prediction_calls
        self.prediction_calls += 1;self.forward_calls += 1
        vehicle = index // self.period + 1
        if vehicle > self.count or index % self.period >= self.visible:
            return np.empty((0, 8), dtype=np.float32)
        x1, y1, x2, y2 = self.box
        return np.array([[x1, y1, x2, y2, vehicle, .9, 2, 0]], dtype=np.float32)
    def close(self):self.closed = True


class SequentialPlate:
    """Plates for SequentialVehicle, reproducing ByteTrack's real first-sighting pattern.

    The first visible frame of each plate is tracker-tentative (untracked); later
    frames are tracked. Every fifth vehicle's plate is never confirmed.
    """
    def __init__(self, vehicles):
        self.vehicles, self.calls = vehicles, 0
    def process(self, pixels):
        from vehicle.common.real_video.plate_local import PlateDetections
        index = self.calls;self.calls += 1
        v = self.vehicles
        vehicle = index // v.period + 1
        if vehicle > v.count or index % v.period >= v.visible:
            return PlateDetections((), ())
        x1, y1, x2, y2 = v.box
        plate = (x1 + 200, y2 - 100, x1 + 350, y2 - 60, .9)
        if vehicle % 5 == 0 or index % v.period == 0:
            return PlateDetections((plate,), (), (plate,))
        return PlateDetections((plate,), ((vehicle, plate),))
    def close(self):pass


def sequential_bindings():
    """Injected local plate bindings: identical executor protocol, no native model."""
    from vehicle.common.engine_plate_artifacts import LocalArtifact, ArtifactFile
    from vehicle.common.engine_plate_local import LocalPlateBinding
    from vehicle.common.tests.test_plate_local_executors import Runtime
    directory = tempfile.mkdtemp(prefix='waycon-fixture-')
    weights = os.path.join(directory, 'injected-native')
    with open(weights, 'wb') as handle:handle.write(b'explicit injected console fixture')
    bindings = {}
    for name in ('ocr', 'detect'):
        identity = LocalArtifact('injected-console-native', name, (ArtifactFile.capture('fixture', weights),),
                                 (('injected-runtime', '1'),), ())
        runtime = Runtime(identity)
        bindings[name] = LocalPlateBinding(identity, runtime_factory=lambda ident, rt=runtime: rt)
    return bindings


class FixtureManufacturer:
    """In-process asynchronous runtime with the PersistentRuntime interface."""
    mode = 'persistent'
    timeout = 5
    provenance = sha256(b'explicit console manufacturer fixture').hexdigest()
    TEXT = '{"manufacturer":"Toyota","vehicle_type":"sedan","color":"white"}'

    def __init__(self, *, stop=None, latency=0.02, queue_size=8):
        self.stop, self.latency, self.queue_size = stop, latency, queue_size
        self.requests, self.results, self.metrics = deque(), deque(), []
        self.guard = threading.Lock();self.outstanding = 0
        self.stopping = threading.Event();self.closed = False
        self.starts = [dict(ready=True, seconds=0.0)];self.restarts = 0
        self.thread = threading.Thread(target=self._run, daemon=True);self.thread.start()
    def can_accept(self):
        with self.guard:return not self.closed and not self.stopping.is_set() and len(self.requests) < self.queue_size
    def submit(self, request_id, crop):
        with self.guard:self.requests.append((request_id, time.monotonic()));self.outstanding += 1
    def _run(self):
        while not self.closed:
            with self.guard:item = self.requests.popleft() if self.requests else None
            if item is None:
                time.sleep(.002);continue
            time.sleep(self.latency)
            with self.guard:
                outcome = 'execution_uncertain' if self.stopping.is_set() else 'completed'
                self.results.append((item[0], outcome, self.TEXT if outcome == 'completed' else '', self.latency))
                self.metrics.append(dict(seconds=self.latency, process_vram_mib=None, outcome=outcome))
    def collect(self):
        with self.guard:
            items = list(self.results);self.results.clear();self.outstanding -= len(items)
        return items
    def wait(self, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self.guard:
                if len(self.results) >= self.outstanding:return True
            time.sleep(.005)
        return False
    def cancel(self):
        self.stopping.set()
        with self.guard:
            while self.requests:
                request_id, _ = self.requests.popleft()
                self.results.append((request_id, 'cancelled', '', 0.0))
    def close(self):
        self.closed = True;self.thread.join(timeout=2)
    @property
    def cleaned(self):return not self.thread.is_alive()
