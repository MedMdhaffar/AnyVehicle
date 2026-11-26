"""Quality-gated manufacturer selection, persistent runtime and closure deferral."""
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest

import numpy as np

from contracts.db1.reference import canonical_json_bytes
from vehicle.common.db1_rows.core import Blocked
from vehicle.common.manufacturer.crop_quality import CropQualityAnalyzer, CropQualityConfig
from vehicle.common.manufacturer.engine import (ManufacturerLedger, SelectionConfig, validate_evidence, hashed)
from vehicle.common.manufacturer.local import (PersistentRuntime, ManufacturerUnavailable, load_lock, create_runtime)
from vehicle.common.manufacturer.policy import evaluate
from vehicle.common.manufacturer.streams import ReviewCodec
from vehicle.common.real_video.engine import Engine
from vehicle.common.real_video.input import DecodedFrame
from vehicle.common.frame_receipt import FrameReceiptClock, ClockPolicy
from vehicle.common.observation_evidence import RecognitionIdentity
from vehicle.common.redis_scalar.worker import load_codec
from vehicle.common.redis_scalar.synthetic import ORG, IDENTITY_NAMESPACE
from vehicle.common.tests.test_colour_aggregator import fixture
from vehicle.common.tests.test_manufacturer import Runtime
from vehicle.common.vehicle_observation import RunEnd

QUALITY = SelectionConfig(policy='quality_lifetime_v1')
TEXT = '{"manufacturer":"Toyota","vehicle_type":"sedan","color":"white"}'


def textured(width, height, seed=1):
    """Sharp, well-exposed synthetic vehicle crop (deterministic)."""
    rng = np.random.default_rng(seed)
    return rng.integers(30, 220, size=(height, width, 3), dtype=np.uint8)


class FakePersistent:
    """In-process asynchronous runtime with explicitly released results."""
    mode = 'persistent'
    provenance = 'b'*64
    timeout = 2

    def __init__(self, text=TEXT, capacity=8, auto=True):
        self.text, self.capacity, self.auto = text, capacity, auto
        self.queue, self.results, self.submitted, self.cancelled = [], [], [], False
        self.stop = None

    def can_accept(self):
        return not self.cancelled and len(self.queue) < self.capacity

    def submit(self, request_id, crop):
        self.submitted.append((request_id, sha256(crop.tobytes()).hexdigest()))
        self.queue.append(request_id)
        if self.auto:
            self.release()

    def release(self, outcome='completed'):
        while self.queue:
            self.results.append((self.queue.pop(0), outcome, self.text if outcome == 'completed' else '', 0.25))

    def collect(self):
        items, self.results = self.results, []
        return items

    def wait(self, timeout):
        return not self.queue

    def cancel(self):
        self.cancelled = True
        while self.queue:
            self.results.append((self.queue.pop(0), 'execution_uncertain', '', 0.0))


class CropQualityTests(unittest.TestCase):
    def setUp(self):
        self.analyzer = CropQualityAnalyzer()

    def test_accepts_large_sharp_exposed_crop_and_rejects_each_hard_failure(self):
        good = self.analyzer.analyze(textured(420, 280), bbox=(100, 100, 520, 380), frame_shape=(720, 1280, 3), label='car')
        self.assertTrue(good.eligible);self.assertEqual(good.reasons, ())
        cases = {
            'too_small': self.analyzer.analyze(textured(80, 60)),
            'insufficient_area': self.analyzer.analyze(textured(200, 130), label='bus'),
            'implausible_car_shape': self.analyzer.analyze(textured(200, 300), label='car'),
            'frame_boundary': self.analyzer.analyze(textured(420, 280), bbox=(0, 100, 420, 380), frame_shape=(720, 1280, 3)),
            'severely_blurred': self.analyzer.analyze(np.full((200, 320, 3), 120, np.uint8)),
            'essentially_black': self.analyzer.analyze(np.zeros((200, 320, 3), np.uint8)),
        }
        for reason, quality in cases.items():
            self.assertFalse(quality.eligible, reason)
            self.assertIn(reason, quality.reasons)
        # A truck may be taller than wide; only configured car labels get the shape check.
        self.assertTrue(self.analyzer.analyze(textured(200, 600), label='truck').eligible)

    def test_deterministic_rounded_record_survives_canonical_round_trip(self):
        crop = textured(300, 220, seed=7)
        first = self.analyzer.analyze(crop).record();second = self.analyzer.analyze(crop.copy()).record()
        self.assertEqual(first, second)
        raw = canonical_json_bytes(first)
        self.assertEqual(canonical_json_bytes(json.loads(raw)), raw)

    def test_v1_fingerprint_is_preserved_and_v2_changes_only_blur(self):
        from vehicle.common.manufacturer.crop_quality import QUALITY_V1, QUALITY_V2
        self.assertEqual(QUALITY_V1.fingerprint(), '0202982c4dbf9c3774a58885e0ee9944e39434beb7efacc615c6e6b86705c7bc')
        self.assertEqual(QUALITY_V2, CropQualityConfig())
        diff = {k for k, v in QUALITY_V1.value().items() if QUALITY_V2.value()[k] != v}
        self.assertEqual(diff, {'version', 'severe_blur_variance'})
        self.assertEqual((QUALITY_V1.severe_blur_variance, QUALITY_V2.severe_blur_variance), (2.0, 5.0))

    def test_moderately_blurred_crop_rejected_by_v2_only(self):
        import cv2
        from vehicle.common.manufacturer.crop_quality import QUALITY_V1, QUALITY_V2
        base = textured(420, 280, 11)
        crop = next(c for c in (cv2.GaussianBlur(base, (0, 0), s) for s in np.arange(1.5, 8, .25))
                    if 2.5 < CropQualityAnalyzer().analyze(c).sharpness_variance < 4.5)
        self.assertTrue(CropQualityAnalyzer(QUALITY_V1).analyze(crop).eligible)
        v2 = CropQualityAnalyzer(QUALITY_V2).analyze(crop)
        self.assertEqual((v2.eligible, v2.reasons), (False, ('severely_blurred',)))

    def test_invalid_bboxes_never_crash_and_give_no_edge_evidence(self):
        crop = textured(420, 280)
        for bbox in ((300, 100, 100, 300), (float('nan'), 0, 10, 10), (0, 0, 10), ('a', 0, 1, 1), None):
            q = self.analyzer.analyze(crop, bbox=bbox, frame_shape=(720, 1280, 3))
            self.assertIsNone(q.border_edges, bbox);self.assertTrue(q.eligible, bbox)
        self.assertIsNone(self.analyzer.analyze(crop, bbox=(0, 0, 10, 10), frame_shape=(0, 0, 3)).border_edges)

    def test_unusable_images_are_rejected_with_explicit_reasons(self):
        cases = {'empty_image': np.zeros((0, 0, 3), np.uint8),
                 'unsupported_array': np.zeros((300, 400, 3), np.float32)}
        for reason, image in cases.items():
            q = self.analyzer.analyze(image);self.assertEqual((q.eligible, q.reasons), (False, (reason,)))
        q = self.analyzer.analyze(np.zeros((300, 400, 4), np.uint8))
        self.assertEqual(q.reasons, ('unsupported_array',))
        self.assertFalse(self.analyzer.analyze(np.full((300, 420, 3), 255, np.uint8)).eligible)  # essentially white
        self.assertIn('essentially_white', self.analyzer.analyze(np.full((300, 420, 3), 255, np.uint8)).reasons)
        self.assertTrue(self.analyzer.analyze(textured(420, 280)[:, :, 0]).eligible)          # grayscale accepted

    def test_versioned_config_fingerprint_changes_with_any_threshold(self):
        base = CropQualityConfig()
        self.assertNotEqual(base.fingerprint(), replace(base, min_width=96).fingerprint())
        with self.assertRaises(ValueError):CropQualityConfig(size_weight=.9)
        with self.assertRaises(ValueError):CropQualityConfig(min_width=1.5)


class QualityLedgerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory();self.addCleanup(self.directory.cleanup)
        self.owner, self.measurements, self.classes = fixture()
        self.life = self.owner.lifetime

    def ledger(self, runtime, name='q.sqlite', **kwargs):
        ledger = ManufacturerLedger(enabled=True, runtime=runtime, journal=Path(self.directory.name)/name,
                                    selection=kwargs.pop('selection', QUALITY), **kwargs)
        self.addCleanup(ledger.close)
        return ledger

    def feed(self, ledger, crops, life=None):
        life = life or self.life
        for index, crop in enumerate(crops):
            receipt = replace(life.first_receipt, frame_index=index)
            h, w = crop.shape[:2]
            ledger.observe(life, receipt, (100, 100, 100+w, 100+h), crop, 'f'*64,
                           label='car', confidence=.9, frame_shape=(1080, 1920, 3))
        return life

    def test_first_crop_rejected_later_crop_accepted_and_best_submitted_once_at_closure(self):
        runtime = Runtime(TEXT);ledger = self.ledger(runtime)
        import cv2
        soft = lambda c:cv2.GaussianBlur(c, (0, 0), 1)  # eligible under v2, lower sharpness score
        crops = [textured(60, 40), soft(textured(420, 280, 2)), textured(520, 330, 3), soft(textured(440, 290, 4))]
        self.feed(ledger, crops)
        self.assertEqual(runtime.calls, 0)  # nothing is submitted before closure/window
        ledger.close_lifetime(self.life)
        evidence = ledger.terminal(self.life)
        validate_evidence(evidence)
        self.assertEqual(runtime.calls, 1)
        b = evidence['binding'];s = b['selection']
        self.assertEqual(b['crop_sha256'], sha256(crops[2].tobytes()).hexdigest())  # largest/best eligible
        self.assertEqual((s['observed'], s['eligible'], s['submission']), (4, 3, 'submitted'))
        self.assertEqual(s['rejected'], {'too_small': 1})
        self.assertEqual(b['receipt']['frame_index'], 2)
        self.assertEqual(evaluate(self.life, evidence).make, 'toyota')

    def test_all_crops_unusable_never_invoke_model_and_state_insufficient_quality(self):
        runtime = Runtime(TEXT);ledger = self.ledger(runtime)
        self.feed(ledger, [textured(50, 40), np.zeros((300, 400, 3), np.uint8), textured(150, 300)])
        ledger.close_lifetime(self.life)
        evidence = ledger.terminal(self.life)
        validate_evidence(evidence)
        decision = evaluate(self.life, evidence)
        self.assertEqual((runtime.calls, decision.status, decision.make, decision.reason), (0, 'unknown', None, 'insufficient_quality'))
        self.assertEqual(evidence['binding']['selection']['rejected'],
                         {'essentially_black': 1, 'implausible_car_shape': 1, 'severely_blurred': 1, 'too_small': 2})
        self.assertEqual(ledger.db.execute('SELECT count(*) FROM attempts').fetchone()[0], 0)

    def test_long_track_submits_after_bounded_wait_without_closure(self):
        runtime = Runtime(TEXT)
        ledger = self.ledger(runtime, selection=SelectionConfig(policy='quality_lifetime_v1', max_wait_observations=3))
        self.feed(ledger, [textured(420, 280, i) for i in range(6)])
        self.assertEqual(runtime.calls, 1)
        evidence = ledger.terminal(self.life)
        self.assertEqual(evidence['binding']['selection']['observed'], 4)

    def test_retry_safe_byte_identical_evidence_and_journal_replay(self):
        crops = [textured(60, 40), textured(440, 300, 9), textured(420, 290, 8)]
        first = self.ledger(Runtime(TEXT), name='a.sqlite');self.feed(first, crops);first.close_lifetime(self.life)
        second = self.ledger(Runtime(TEXT), name='b.sqlite');self.feed(second, crops);second.close_lifetime(self.life)
        a, b = first.terminal(self.life), second.terminal(self.life)
        # Independent executions: identical deterministic binding/request; latency is measured.
        self.assertEqual((canonical_json_bytes(a['binding']), a['request_id'], a['text']),
                         (canonical_json_bytes(b['binding']), b['request_id'], b['text']))
        self.assertEqual(canonical_json_bytes(first.terminal(self.life)), canonical_json_bytes(a))  # idempotent read
        # Restart against the same journal: no new call, identical bytes.
        runtime = Runtime(TEXT);restarted = self.ledger(runtime, name='a.sqlite')
        self.feed(restarted, crops);restarted.close_lifetime(self.life)
        self.assertEqual(runtime.calls, 0)
        self.assertEqual(canonical_json_bytes(restarted.terminal(self.life)), canonical_json_bytes(a))

    def test_receipt_frame_index_at_least_10_survives_canonical_decoding(self):
        """Regression: canonical JSON writes 22 as 2.2e1; json.loads made it 22.0."""
        from vehicle.common.real_video.engine import ClosedScalars
        from vehicle.common.console_projection import project_track
        for policy in ('first_eligible_v1', 'quality_lifetime_v1'):
            life = replace(self.life, last_receipt=replace(self.life.last_receipt, frame_index=40))
            ledger = self.ledger(Runtime(TEXT), name=policy+'.sqlite', selection=SelectionConfig(policy=policy))
            crop = textured(420, 280)
            ledger.observe(life, replace(life.first_receipt, frame_index=22), (100, 100, 520, 380), crop, 'f'*64,
                           label='car', confidence=.9, frame_shape=(1080, 1920, 3))
            ledger.close_lifetime(life)
            raw = canonical_json_bytes(ledger.terminal(life)).decode()
            self.assertIn('2.2e1', raw)
            decision = evaluate(life, json.loads(raw))           # tolerant of a plain decoder
            self.assertEqual(decision.make, 'toyota')
            decision.validate(life)

    def test_legacy_policy_evidence_is_unchanged_v1(self):
        ledger = self.ledger(Runtime(TEXT), selection=SelectionConfig(policy='first_eligible_v1'))
        self.feed(ledger, [textured(60, 50), textured(420, 280)])
        evidence = ledger.terminal(self.life)
        self.assertNotIn('selection', evidence['binding'])
        self.assertEqual(evidence['binding']['crop_sha256'], sha256(textured(60, 50).tobytes()).hexdigest())

    def test_asynchronous_result_and_stop_cancellation_never_fabricate(self):
        runtime = FakePersistent(auto=False);ledger = self.ledger(runtime)
        self.feed(ledger, [textured(440, 300)])
        ledger.close_lifetime(self.life)
        self.assertTrue(ledger.waiting_for(self.life))
        with self.assertRaises(Blocked):ledger.terminal(self.life)
        runtime.cancel();ledger.poll()
        self.assertFalse(ledger.waiting_for(self.life))
        evidence = ledger.terminal(self.life)
        validate_evidence(evidence)
        self.assertEqual((evidence['outcome'], evidence['text']), ('execution_uncertain', ''))
        self.assertIsNone(evaluate(self.life, evidence).make)

    def test_full_queue_defers_closure_without_blocking_then_submits_fifo(self):
        runtime = FakePersistent(auto=False, capacity=1);ledger = self.ledger(runtime)
        other = replace(self.life, service_track_id=2, raw_track_id=20)
        self.feed(ledger, [textured(440, 300, 1)]);self.feed(ledger, [textured(440, 300, 2)], life=other)
        started = time.monotonic()
        ledger.close_lifetime(self.life);ledger.close_lifetime(other)
        self.assertLess(time.monotonic() - started, 0.5)               # never waits for a slot
        self.assertEqual(len(runtime.submitted), 1)
        self.assertTrue(ledger.waiting_for(other))                      # deferred closure stays unfrozen
        runtime.release();ledger.poll()                                 # slot frees -> deferred submitted
        self.assertEqual(len(runtime.submitted), 2)
        runtime.release();ledger.poll()
        self.assertFalse(ledger.waiting_for(self.life) or ledger.waiting_for(other))
        self.assertEqual({evaluate(l, ledger.terminal(l)).make for l in (self.life, other)}, {'toyota'})

    def test_stop_before_submission_is_explicit(self):
        stop = threading.Event()
        runtime = FakePersistent();ledger = self.ledger(runtime, stop=stop)
        self.feed(ledger, [textured(440, 300)])
        stop.set();ledger.close_lifetime(self.life)
        evidence = ledger.terminal(self.life)
        self.assertEqual((runtime.submitted, evidence['outcome']), ([], 'unknown'))
        self.assertEqual(evaluate(self.life, evidence).reason, 'stopped_before_submission')

    def test_review_codec_accepts_v2_evidence(self):
        from vehicle.common.real_video.engine import ClosedScalars
        ledger = self.ledger(Runtime(TEXT));self.feed(ledger, [textured(440, 300)]);ledger.close_lifetime(self.life)
        closed = ClosedScalars(self.owner, self.classes, self.measurements,
                               manufacturer_json=canonical_json_bytes(ledger.terminal(self.life)).decode())
        codec = ReviewCodec(load_codec(ORG, IDENTITY_NAMESPACE))
        self.assertEqual(codec.join(codec.encode(closed)).manufacturer_review.make, 'toyota')


class EngineDeferralTests(unittest.TestCase):
    """The shared Engine keeps a closure unfrozen until its single request finishes."""
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory();self.addCleanup(self.directory.cleanup)
        self.owner, _, _ = fixture()
        self.context = self.owner.lifetime.context

    def run_engine(self, runtime, *, frames, rows, stop=None, after_frame=None):
        ledger = ManufacturerLedger(enabled=True, runtime=runtime, journal=Path(self.directory.name)/'e.sqlite',
                                    selection=QUALITY, stop=stop, max_lifetimes=4096)
        self.addCleanup(ledger.close)
        class Detector:
            calls = 0
            def process(inner, pixels):
                inner.calls += 1
                return np.array(rows(inner.calls - 1), dtype=float).reshape(-1, 8)
            def close(inner):pass
        detector = Detector();tick = [0]
        def mono():
            tick[0] += 1;return 100.0 + tick[0]
        from datetime import timedelta
        clock = FrameReceiptClock(self.context, utc_clock=lambda:self.context.started_at+timedelta(seconds=tick[0]),
                                  monotonic_clock=mono, policy=ClockPolicy())
        engine = Engine(self.context, clock, detector, algorithm=RecognitionIdentity('test-hsv',1,'classify'),
                        quality_algorithm='test-quality', estimate=lambda c:SimpleNamespace(label='red',confidence=.8),
                        quality=lambda *a:SimpleNamespace(combined=.9), manufacturer=ledger, streaming=True, max_vehicles=16)
        self.addCleanup(engine.close)
        image = textured(1280, 720, 5)
        for index in range(frames):
            engine.process(DecodedFrame(clock.capture(index, float(index)), image))
            if after_frame is not None:after_frame(engine)
        return engine, ledger, detector

    def test_eof_with_pending_work_waits_and_more_than_16_sequential_lifetimes(self):
        # 40 sequential vehicles (raw IDs 1..40), each visible 3 frames then gone.
        def rows(i):
            vid = i // 20 + 1
            return [] if i % 20 >= 3 or vid > 40 else [[200, 200, 650, 500, vid, .9, 2, 0]]
        runtime = FakePersistent(auto=False);peak = [0]
        def settle(engine):
            # Console-style progressive settlement: results arrive, closures are acknowledged.
            runtime.release();engine.poll_manufacturer()
            peak[0] = max(peak[0], len(engine.working) + len(engine.closed_ids))
            for sid in tuple(engine.pending):engine.acknowledge(sid)
        engine, ledger, detector = self.run_engine(runtime, frames=40*20, rows=rows, after_frame=settle)
        self.assertGreaterEqual(engine.released, 38)
        self.assertLessEqual(peak[0], 2)                       # bounded active state
        engine.finish(RunEnd.END_OF_INPUT)
        runtime.release();engine.poll_manufacturer()
        for sid in tuple(engine.pending):engine.acknowledge(sid)
        self.assertEqual(engine.released, 40)
        self.assertEqual(len(runtime.submitted), 40)           # exactly one request per lifetime
        self.assertEqual(len({r for r, _ in runtime.submitted}), 40)
        self.assertEqual(detector.calls, 800)                  # one detector, no duplicate tracking
        self.assertEqual((engine.working, engine.pending, engine.closed_ids), ({}, {}, set()))
        self.assertEqual((ledger.tracks, ledger.waiting, ledger.retained_bytes), ({}, {}, 0))

    def test_closure_is_not_frozen_until_result_then_has_result(self):
        def rows(i):return [[200, 200, 650, 500, 1, .9, 2, 0]] if i < 3 else []
        runtime = FakePersistent(auto=False)
        engine, ledger, _ = self.run_engine(runtime, frames=25, rows=rows)
        self.assertEqual(engine.pending, {})                   # closed but awaiting the model
        self.assertTrue(ledger.waiting)
        runtime.release();engine.poll_manufacturer()
        closed = next(iter(engine.pending.values()))
        evidence = json.loads(closed.manufacturer_json)
        self.assertEqual((evidence['outcome'], evaluate(closed.owner.lifetime, evidence).make), ('completed', 'toyota'))
        receipt = evidence['binding']['receipt']
        self.assertTrue(any(s.receipt.frame_index == receipt['frame_index'] for s in closed.classes))

    def test_stop_during_pending_cancels_and_settles_without_fabrication(self):
        stop = threading.Event()
        def rows(i):return [[200, 200, 650, 500, 1, .9, 2, 0]] if i < 3 else []
        runtime = FakePersistent(auto=False)
        engine, ledger, _ = self.run_engine(runtime, frames=25, rows=rows, stop=stop)
        stop.set();engine.finish(RunEnd.SERVICE_STOP)
        closed = next(iter(engine.pending.values()))
        evidence = json.loads(closed.manufacturer_json)
        self.assertEqual((evidence['outcome'], evidence['text']), ('execution_uncertain', ''))


FAKE_WORKER = r'''
import json, os, struct, sys, time
control = sys.argv[1]
out = sys.stdout.buffer
def send(value):
    data = json.dumps(value).encode(); out.write(struct.pack('>I', len(data)) + data); out.flush()
with open(control + '.starts', 'a') as f: f.write('x')
send(dict(ready=True, gpu_memory=None))
while True:
    header = sys.stdin.buffer.read(4)
    if len(header) < 4: break
    size = struct.unpack('>I', header)[0]
    if size == 0: break
    png = sys.stdin.buffer.read(size)
    actions = open(control).read().split()
    action = actions[0] if actions else 'ok'
    open(control, 'w').write(' '.join(actions[1:]))
    if action == 'hang': time.sleep(60)
    if action == 'crash': os._exit(3)
    send(dict(outcome='completed', text=str(len(png)), gpu_memory=None))
'''


class PersistentRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp());self.addCleanup(__import__('shutil').rmtree, self.directory)
        artifacts = {}
        for name in ('model', 'projector', 'python'):
            path = self.directory/name;path.write_bytes(name.encode()*16)
            artifacts[name] = dict(path=name, bytes=path.stat().st_size, sha256=sha256(path.read_bytes()).hexdigest())
        (self.directory/'runtime').mkdir()
        self.lock = self.directory/'lock.json'
        self.lock.write_text(json.dumps(dict(artifacts=artifacts, runtime='runtime')))
        self.script = self.directory/'worker.py';self.script.write_text(FAKE_WORKER)
        self.control = self.directory/'control';self.control.write_text('')

    def runtime(self, **kwargs):
        options = dict(timeout=1, max_restarts=2, ready_timeout=20)
        options.update(kwargs)
        runtime = PersistentRuntime(self.lock, command=[sys.executable, '-I', str(self.script), str(self.control)], **options)
        self.addCleanup(runtime.close)
        return runtime

    def call(self, runtime, request_id, crop=None):
        runtime.submit(request_id, textured(64, 48) if crop is None else crop)
        self.assertTrue(runtime.wait(15))
        return runtime.collect()

    def test_one_model_load_serves_many_requests_then_cleans_up(self):
        runtime = self.runtime()
        results = [self.call(runtime, f'r{i}')[0] for i in range(5)]
        self.assertEqual([r[1] for r in results], ['completed']*5)
        self.assertEqual((len(runtime.starts), (self.control.parent/'control.starts').read_text()), (1, 'x'))
        runtime.close()
        self.assertTrue(runtime.cleaned)

    def test_timeout_and_crash_restart_then_budget_exhausts(self):
        runtime = self.runtime(max_restarts=1)
        self.control.write_text('hang ok')
        self.assertEqual(self.call(runtime, 'slow')[0][1], 'timed_out')
        self.assertEqual(self.call(runtime, 'after')[0][1], 'completed')   # restarted once
        self.assertEqual(runtime.restarts, 1)
        self.control.write_text('crash')
        self.assertEqual(self.call(runtime, 'boom')[0][1], 'failed')
        self.assertEqual(self.call(runtime, 'refused')[0][1], 'failed')    # budget exhausted: no spawn
        self.assertEqual(runtime.failure, 'MANUFACTURER_RESTART_BUDGET')
        runtime.close();self.assertTrue(runtime.cleaned)

    def test_bounded_queue_and_cancel_marks_queued_and_inflight(self):
        runtime = self.runtime(timeout=30, queue_size=2)
        self.control.write_text('hang')
        runtime.submit('inflight', textured(64, 48))
        deadline = time.monotonic()+10
        while runtime.inflight is None and time.monotonic() < deadline:time.sleep(.01)
        runtime.submit('queued-1', textured(64, 48));runtime.submit('queued-2', textured(64, 48))
        self.assertFalse(runtime.can_accept())
        with self.assertRaises(Blocked):runtime.submit('overflow', textured(64, 48))
        runtime.cancel();self.assertTrue(runtime.wait(15))
        outcomes = dict((r[0], r[1]) for r in runtime.collect())
        self.assertEqual(outcomes, {'inflight':'execution_uncertain','queued-1':'cancelled','queued-2':'cancelled'})
        runtime.close();self.assertTrue(runtime.cleaned)

    def test_unavailable_assets_fail_clearly_before_any_process(self):
        (self.directory/'model').unlink()
        with self.assertRaises(ManufacturerUnavailable) as error:load_lock(self.lock)
        self.assertEqual(error.exception.code, 'MANUFACTURER_ASSET_MISSING')
        with self.assertRaises(ManufacturerUnavailable):create_runtime(self.lock, mode='persistent')
        with self.assertRaises(ManufacturerUnavailable) as missing:load_lock(self.directory/'absent.json')
        self.assertEqual(missing.exception.code, 'MANUFACTURER_LOCK_MISSING')

    def test_worker_that_never_becomes_ready_fails_construction(self):
        self.script.write_text('import time; time.sleep(60)')
        with self.assertRaises(Exception):
            PersistentRuntime(self.lock, command=[sys.executable, '-I', str(self.script)], timeout=1, ready_timeout=1)

    def test_legacy_runtime_feeds_large_png_without_deadlock(self):
        """Regression: polling communicate(input) never resumed a partial write."""
        from vehicle.common.manufacturer.local import LocalRuntime
        from unittest.mock import patch
        import subprocess
        runtime = LocalRuntime.__new__(LocalRuntime)
        runtime.timeout, runtime.code, runtime.children, runtime.metrics, runtime.stop = 20, {}, [], [], None
        runtime.lock = {'runtime':'/usr', 'artifacts':{'python':{'path':sys.executable}}}
        runtime.lock_path = self.lock
        echo = ('import sys,json,time; time.sleep(.5); data=sys.stdin.buffer.read(); '
                'print(json.dumps(dict(outcome="completed",text=str(len(data)),gpu_memory=None)))')
        popen = subprocess.Popen
        with patch('vehicle.common.manufacturer.local.subprocess.Popen',
                   side_effect=lambda *a, **k: popen([sys.executable, '-c', echo], **k)):
            crop = np.random.default_rng(3).integers(0, 255, size=(640, 640, 3), dtype=np.uint8)
            outcome, text = runtime.infer(crop)
        self.assertEqual(outcome, 'completed');self.assertGreater(int(text), 65536)
        self.assertTrue(runtime.cleaned)


if __name__ == '__main__':
    unittest.main()
