"""Streaming release with plate OCR: bounded Engine/Plate/publisher state over many lifetimes.

Owned native Redis and real metadata worker processes; injected capture and
native plate executors (no model inference).
"""
import threading
import unittest

import numpy as np

from vehicle.common.tests.test_plate_requests_redis import RequestRedisTests
from vehicle.common.tests.test_redis_scalar_integration import eventually
from vehicle.common.tests import test_mp4_colour_plate as injected
from vehicle.common.redis_scalar.worker import load_request_codec
from vehicle.common.redis_scalar.synthetic import ORG, IDENTITY_NAMESPACE
from vehicle.common.redis_scalar.journal import Journal
from vehicle.common.plate_requests.host import V2Publisher
from vehicle.common.real_video.plate_engine import VideoPlateLimits
from vehicle.common.real_video.plate_local import PlateDetections
from vehicle.common.vehicle_observation import RunEnd


class SequentialVehicles:
    """Vehicle k (raw ID k) visible for 3 of every 20 frames."""
    def __init__(self, events):self.events, self.calls, self.closed = events, 0, False
    def visible(self, index):return index % 20 < 3
    def process(self, image):
        index = self.calls;self.calls += 1;self.events.append('vehicle')
        if not self.visible(index):
            return np.empty((0, 8), dtype=float)
        return np.array([[0, 0, 120, 90, index // 20 + 1, .9, 2, 0]], dtype=float)
    def close(self):self.closed = True


class SequentialPlates:
    def __init__(self, events, vehicles):self.events, self.vehicles, self.calls, self.closed = events, vehicles, 0, False
    def process(self, image):
        index = self.calls;self.calls += 1;self.events.append('plate')
        if not self.vehicles.visible(index):
            return PlateDetections((), ())
        return PlateDetections(((40, 60, 80, 72, .9),), ((1, (40, 60, 80, 72, .9)),))
    def close(self):self.closed = True


class StreamingReleaseTests(unittest.TestCase):
    # Reuse the owned native Redis fixture without re-running its tests.
    setUpClass = RequestRedisTests.__dict__['setUpClass']
    tearDownClass = RequestRedisTests.__dict__['tearDownClass']
    setUp, tearDown = RequestRedisTests.setUp, RequestRedisTests.tearDown
    journal, start, count = RequestRedisTests.journal, RequestRedisTests.start, RequestRedisTests.count

    def test_twenty_sequential_lifetimes_with_plates_release_all_retained_state(self):
        self.rc = load_request_codec(ORG, IDENTITY_NAMESPACE)
        owner = injected.CombinedVideoTests('runTest');owner.setUpClass();owner.setUp();self.addCleanup(owner.doCleanups)
        owner.vehicle = SequentialVehicles(owner.events);owner.plate = SequentialPlates(owner.events, owner.vehicle)
        engine = owner.engine(limits=VideoPlateLimits(streaming=True, max_frames=None, max_seconds=None,
                                                      max_api_calls=None, max_raw_total=None))
        self.start('colour');self.start('aggregator_plate');self.start('plate_requests')
        journal = Journal(None, namespace=self.ns, trusted_org=ORG, role='plate_requests', consumer='request-host')
        publisher = V2Publisher(self.client, journal, self.rc, engine)
        threads_before = threading.active_count()
        peak = dict(lives=0, plans=0, entries=0, active=0)
        releasing = set()
        def settle():
            for sid, closed in tuple(engine.pending.items()):
                if publisher.publish(closed):
                    engine.acknowledge(sid);releasing.add(sid)
            for sid in tuple(releasing):
                if publisher.release(sid):releasing.discard(sid)
            return not engine.pending and not releasing
        source = owner.source(20*20)
        with source:
            while True:
                packet = source.read()
                if packet is None:break
                engine.process(packet)
                if engine.pending:eventually(settle, seconds=30)
                peak.update(lives=max(peak['lives'], len(engine.plate.lives)), plans=max(peak['plans'], len(publisher.plans)),
                            entries=max(peak['entries'], len(publisher.writer.entries)),
                            active=max(peak['active'], len(engine.colour.working) + len(engine.colour.closed_ids)))
        engine.finish(RunEnd.END_OF_INPUT)
        eventually(settle, seconds=30)
        eventually(lambda:self.count('db1.delivery.v1') == 20, seconds=30)
        self.assertEqual(engine.colour.released, 20)
        self.assertEqual(len(publisher.released), 20)
        self.assertLessEqual(peak['lives'], 2);self.assertLessEqual(peak['plans'], 1);self.assertLessEqual(peak['active'], 2)
        self.assertEqual((engine.plate.lives, publisher.plans, publisher.writer.entries, publisher.input_progress), ({}, {}, {}, {}))
        self.assertEqual(engine.plate.attempts, 0)
        self.assertFalse(engine.plate.buffers)
        self.assertLessEqual(threading.active_count(), threads_before + 1)   # adapter executors retired
        # Stream IDs stay monotonic after releases (no reuse of fixed IDs).
        rows = self.client.xrange(publisher.journal.key('engine.lifecycle.plate'))
        ids = [int(r[0].split(b'-')[0]) for r in rows]
        self.assertEqual(ids, sorted(set(ids)));self.assertEqual(len(ids), 20)
        self.assertEqual(self.count('diagnostics.plate_requests'), 0)
        self.assertEqual(self.count('diagnostics.aggregator_plate'), 0)
        engine.close()


if __name__ == '__main__':
    unittest.main()
