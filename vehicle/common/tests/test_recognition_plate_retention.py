"""Bound plate crop retention without cancelling work or fabricating vehicles."""
from concurrent.futures import Future
import json
from pathlib import Path
import tempfile
from types import MethodType, SimpleNamespace
import unittest
import weakref

import numpy as np

from vehicle.common.real_video.recognition_engine import RecognitionEngine
from vehicle.recognition.plate.data_capture import ReviewStore
from vehicle.recognition.plate.run_video_prototype import RecognitionAttempt, TrackState, VideoPrototypeRunner


class PlateRetentionTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.directory = Path(temp.name)
        engine = RecognitionEngine.__new__(RecognitionEngine)
        engine.closed = {};engine.profiles = {};engine.ready = []
        engine.receipts = {};engine.plate_parents = {};engine.plate_inputs = {}
        engine.attribute_service = None;engine.manufacturer_evaluation = None
        engine.vehicle = None
        self.store = ReviewStore(self.directory/'plate-evidence')
        engine.plate = SimpleNamespace(tracks={}, disappearance_frames=5,
            tracker=SimpleNamespace(target=SimpleNamespace(max_time_lost=8)),
            pipeline=SimpleNamespace(store=self.store, marker_localizer=None),
            detector=None, close=lambda:None,
            **{name:[] for name in ('pending_snapshots','detection_times_ms','tracking_times_ms',
                                  'crop_scoring_times_ms','detections_per_frame','tracked_outputs_per_frame')})
        engine.plate.finalize_track = MethodType(VideoPrototypeRunner.finalize_track, engine.plate)
        self.engine = engine

    def add_track(self, raw=1, frame=0, attempted=True):
        image = np.full((16,32,3),100,np.uint8)
        state = TrackState(raw,frame,frame,best_frame=frame,first_crop=image,best_crop=image)
        if attempted:
            sample = f'plate-{raw}'
            result = SimpleNamespace(sample_id=sample)
            state.attempts = [RecognitionAttempt(result,1,crop_frame=frame)]
            state.cached_attempt_index = 0
            state.submitted_frames = [frame]
            path = self.directory/f'{sample}.png'
            path.write_bytes(b'retained evidence')
            self.engine.plate_inputs[sample] = path
        self.engine.plate.tracks[raw] = state
        self.engine.plate_parents[raw,frame] = None
        return state

    def collect_at(self, frame, **kwargs):
        self.engine.receipts[frame] = SimpleNamespace(frame_index=frame)
        self.engine._collect(**kwargs)

    def test_expired_orphan_releases_crops_and_references_but_preserves_saved_evidence(self):
        state = self.add_track()
        image = weakref.ref(state.best_crop)
        path = self.engine.plate_inputs['plate-1']
        self.collect_at(9)
        self.assertEqual(self.engine.plate.tracks, {})
        self.assertEqual(self.engine.plate_inputs, {})
        self.assertEqual(self.engine.plate_parents, {})
        self.assertIsNone(image())
        self.assertTrue(state.final_metadata_written)
        self.assertEqual(path.read_bytes(), b'retained evidence')
        events = [json.loads(line) for line in self.store.debug_path.read_text().splitlines()]
        self.assertEqual(sum(event['record_type']=='track_finalization' for event in events), 1)
        self.assertEqual(self.engine.drain(), [])

    def test_track_survives_loss_horizon_and_can_be_observed_again(self):
        state = self.add_track()
        self.collect_at(8)
        self.assertIs(self.engine.plate.tracks.get(1), state)
        state.last_frame = 8
        self.engine.plate_parents[1,8] = None
        self.collect_at(9)
        self.assertIs(self.engine.plate.tracks.get(1), state)
        self.collect_at(17)
        self.assertEqual(self.engine.plate.tracks, {})

    def test_pending_and_deferred_recognition_are_not_released_even_at_finish(self):
        state = self.add_track()
        state.pending = Future()
        self.collect_at(50,final=True)
        self.assertIs(self.engine.plate.tracks.get(1), state)
        state.pending.set_result(None)
        self.collect_at(51,final=True)
        self.assertIs(self.engine.plate.tracks.get(1), state)  # Runner must drain it first.
        state.pending = None
        state.deferred_reason = 'queue_capacity'
        self.collect_at(52,final=True)
        self.assertIs(self.engine.plate.tracks.get(1), state)
        state.deferred_reason = None
        self.collect_at(53,final=True)
        self.assertEqual(self.engine.plate.tracks, {})

    def test_active_vehicle_owner_keeps_its_plate_after_orphan_expiry(self):
        state = self.add_track()
        self.engine.profiles[7] = SimpleNamespace(recognition=SimpleNamespace(attempts=[],candidate=None))
        self.engine.plate_parents[1,0] = 7
        self.collect_at(50)
        self.assertIs(self.engine.plate.tracks.get(1), state)
        self.assertFalse(state.final_metadata_written)

    def test_sustained_unassociated_stream_has_bounded_track_and_crop_retention(self):
        for frame in range(1000):
            self.add_track(raw=frame+1,frame=frame,attempted=False)
            self.collect_at(frame)
            self.assertLessEqual(len(self.engine.plate.tracks), 9)
            self.assertLessEqual(len(self.engine.plate_parents), 9)
            self.assertLessEqual(len(self.engine.receipts), 9)
        self.collect_at(1000,final=True)
        self.assertEqual(self.engine.plate.tracks, {})
        self.assertEqual(self.engine.drain(), [])

    def test_close_releases_remaining_orphan_crops_after_joining_worker(self):
        state = self.add_track()
        plate = self.engine.plate
        joined = []
        plate.close = lambda:joined.append(True)
        self.engine.close()
        self.assertEqual(joined, [True])
        self.assertEqual(plate.tracks, {})
        self.assertIsNone(state.first_crop)
        self.assertIsNone(state.best_crop)
        self.assertEqual(self.engine.plate_inputs, {})
