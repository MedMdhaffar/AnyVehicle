"""Plate state must outlive every vehicle that still owns its evidence."""
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import cv2
import numpy as np

from vehicle.common.evaluation_capture import EvaluationRecorder
from vehicle.common.frame_receipt import FrameReceiptClock, ClockPolicy
from vehicle.common.real_video.recognition_engine import RecognitionEngine
from vehicle.common.recognition_transport import RecognitionCodec, aggregate
from vehicle.common.redis_scalar.codec import unpack_life
from vehicle.common.redis_scalar.synthetic import ORG, IDENTITY_NAMESPACE
from vehicle.common.tests.test_recognition_v3 import evidence_fixture
from vehicle.common.vehicle_observation import VehicleLifecycle
from vehicle.recognition.attributes.manufacturer.tracking import TrackRecognition
from vehicle.recognition.plate.run_video_prototype import RecognitionAttempt, TrackState
from vehicle.recognition.plate.schemas import UnifiedPlateResult


class PlateOwnershipTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.directory = Path(temp.name)
        template = unpack_life(evidence_fixture()['life'])
        self.index = 0
        clock = FrameReceiptClock(template.context,
            utc_clock=lambda:template.first_receipt.utc+timedelta(seconds=self.index),
            monotonic_clock=lambda:template.first_receipt.monotonic+self.index,
            policy=ClockPolicy())
        engine = RecognitionEngine.__new__(RecognitionEngine)
        engine.context = template.context
        engine.directory = self.directory
        engine.lifecycle = VehicleLifecycle(template.context, clock)
        engine.receipts = {}
        for index in (0, 1, 10):
            self.index = index
            engine.receipts[index] = clock.capture(index)
            if index in (0, 1):engine.lifecycle.observe(index+11, engine.receipts[index])
        engine.closed = {1:engine.lifecycle.lose(11)}
        engine.parents = {11:1, 12:2}
        engine.profiles = {sid:SimpleNamespace(track_id=raw, recognition=TrackRecognition())
                           for sid,raw in ((1,11),(2,12))}
        engine.plate_parents = {(77,0):1, (77,1):2}
        engine.ready = []
        engine.models = {};engine.crops = {};engine.plate_diagnostics = {}
        engine.attribute_inputs = {};engine.plate_inputs = {};engine.attribute_service = None
        engine.evaluation = EvaluationRecorder(self.directory, template.context)
        self.images = [np.full((16,32,3), colour, np.uint8) for colour in (40,180)]
        attempts = []
        for frame,text in ((0,None),(1,'222 nt')):
            sample = f'plate-{frame}'
            result = UnifiedPlateResult(sample_id=sample,
                timestamp=engine.receipts[frame].utc.isoformat(), route='local',
                route_confidence=1, route_reason='category recognized', source='local_ocr',
                model_version='test-ocr', bbox=[0,0,32,16], status='accepted' if text else 'unresolved',
                predicted_category='nt', plate_category='temporary',
                recognition_status='recognized' if text else 'unresolved',
                registration_fields={'number':'222'} if text else {},
                normalized_number=text, text_confidence=.95 if text else .2,
                evidence=dict(engine_operations=[dict(kind='registration_ocr', outcome='succeeded',
                    model=dict(model_name='test-ocr', version=1, task='ocr'))]))
            path = self.directory/f'{sample}.png'
            self.assertTrue(cv2.imwrite(str(path), self.images[frame]))
            engine.plate_inputs[sample] = path
            attempts.append(RecognitionAttempt(result,1,crop_frame=frame,crop_score=.8+frame/10))
        self.state = TrackState(77,0,1,first_crop=self.images[0],best_crop=self.images[1],
            best_frame=1,submitted_frames=[0,1])
        # The real policy replaces A's unresolved result with B's accepted recheck.
        for attempt in attempts:self.state.record_attempt(attempt, .02)
        self.assertEqual(self.state.cached_attempt_index, 1)
        self.finalized = []
        engine.plate = SimpleNamespace(tracks={77:self.state}, disappearance_frames=5,
            finalize_track=lambda state:self.finalized.append(state.track_id),
            **{name:[] for name in ('pending_snapshots','detection_times_ms','tracking_times_ms',
                                  'crop_scoring_times_ms','detections_per_frame','tracked_outputs_per_frame')})
        self.engine = engine

    def close_second(self):
        self.engine.closed[2] = self.engine.lifecycle.lose(12)

    def test_collecting_old_vehicle_preserves_new_vehicle_selected_result_and_crop(self):
        engine = self.engine
        engine._collect()
        first = engine.drain()
        self.assertEqual([e['life']['service_track_id'] for e in first], [1])
        self.assertIn(77, engine.plate.tracks)
        self.assertIsNotNone(self.state.best_crop)
        self.assertEqual(self.finalized, [])
        self.close_second()
        engine._collect()
        second = engine.drain()
        self.assertEqual([e['life']['service_track_id'] for e in second], [2])
        self.assertEqual(second[0]['plates'][0]['result']['normalized_number'], '222 nt')
        self.assertEqual(second[0]['plates'][0]['receipt']['frame_index'], 1)
        np.testing.assert_array_equal(cv2.imread(str(self.directory/'crops/track-2-plate.png')), self.images[1])
        self.assertEqual(engine.plate.tracks, {})
        self.assertEqual(engine.plate_parents, {})
        self.assertEqual(self.finalized, [77])
        engine._collect()
        self.assertEqual(engine.drain(), [])
        self.verify_delivery(first+second)

    def verify_delivery(self, evidence):
        from vehicle.common.db1_consumer import NaturalConsumer
        from vehicle.common.db1_adapter import TrustedTenant
        from vehicle.common.tests.test_db1_consumer import supplied_test_registrations, TransactionAdapter
        codec = RecognitionCodec(ORG, IDENTITY_NAMESPACE)
        for item in evidence:
            sid = item['life']['service_track_id']
            raw = codec.make('db1.delivery.v1', item)
            facts = aggregate(item, codec)['facts']
            plate = next(f['delivery'] for f in facts if f['row_kind']=='fact_plate_read')
            self.assertEqual(plate['track_id'], sid)
            self.assertEqual(plate['selected_attempt']['plate_text'], None if sid==1 else '222 NT')
            self.assertEqual(plate['outcome'], 'unreadable' if sid==1 else 'read')
            res = supplied_test_registrations(SimpleNamespace(v1=codec), raw)
            context = codec.decode(raw).life.context
            consumer = NaturalConsumer(codec=codec, tenant=TrustedTenant(ORG,'ownership test'),
                trusted_run=context.run_id, trusted_camera=context.camera.camera_id,
                resolutions=res, connection_factory=lambda:nullcontext(None),
                environment='test', adapter=TransactionAdapter(codec.validation))
            self.assertEqual(consumer.process(raw, lambda:None).disposition, 'new')
            self.assertEqual(consumer.process(raw, lambda:None).disposition, 'identical')

    def test_reverse_collection_order_preserves_each_vehicles_attempts(self):
        first = self.engine.closed[1]
        self.close_second()
        self.engine.closed = {2:self.engine.closed[2], 1:first}
        self.engine._collect(final=True)
        emitted = self.engine.drain()
        self.assertEqual(len(emitted), 2)
        self.verify_delivery(emitted)
        self.assertEqual(self.finalized, [77])
        self.assertEqual(self.engine.plate.tracks, {})

    def test_latest_parent_is_retained_even_without_a_better_crop_or_new_attempt(self):
        self.state.attempts = self.state.attempts[:1]
        self.state.submitted_frames = [0]
        self.state.best_frame = 0
        self.state.cached_attempt_index = 0
        self.engine._prune()
        self.assertEqual(self.engine.plate_parents.get((77,1)), 2)
        self.assertIn(1, self.engine.receipts)
        self.engine._collect()
        self.assertIn(77, self.engine.plate.tracks)
        self.engine.drain()
        self.close_second()
        self.engine._collect()
        second = self.engine.drain()
        self.assertEqual(second[0]['plates'], [])  # Never copy A's reading to B.
        self.assertEqual(self.finalized, [77])
