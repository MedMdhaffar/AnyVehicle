"""CPU boundary checks; detector/OCR answers are controlled, queue and policy are real."""
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest

import numpy as np

from vehicle.common.plate_recovery import VehiclePlateRecovery
from vehicle.recognition.tracking import PlateTracker
from vehicle.recognition.plate.run_video_prototype import VideoPrototypeRunner, RecognitionAttempt, TrackState
from vehicle.recognition.plate.schemas import UnifiedPlateResult


class PlateRecoveryTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.engine = SimpleNamespace(directory=Path(temp.name),context=SimpleNamespace(run_id='test'),
            stop=threading.Event(),receipts={},plate_diagnostics={},plate_parents={},plate_inputs={})
        runner = VideoPrototypeRunner.__new__(VideoPrototypeRunner)
        runner.tracks={};runner.max_pending=1;runner.deferred_events=0
        runner.retry_improvement=1.5;runner.recheck_detail_improvement=1.25
        runner.replacement_confidence_gain=.02;runner._source_ref=None;runner.dedup_iou=.8
        runner.disappearance_frames=5
        for name in ('pending_snapshots','detection_times_ms','tracking_times_ms',
                     'crop_scoring_times_ms','detections_per_frame','tracked_outputs_per_frame'):
            setattr(runner,name,[])
        from vehicle.recognition.plate.data_capture import ReviewStore
        runner.pipeline=SimpleNamespace(store=ReviewStore(Path(temp.name)/'plate-evidence'))
        runner.tracker=PlateTracker(SimpleNamespace())
        runner.executor=ThreadPoolExecutor(max_workers=1)
        self.addCleanup(runner.executor.shutdown,True)
        self.calls=[];self.detector_calls=[];self.responses=[]
        def recognize(*args):
            self.calls.append(args)
            outcome = self.responses.pop(0) if self.responses else 'recognized'
            result = self.result(args[1].crop,args[5],outcome)
            import cv2
            path=Path(temp.name)/(result.sample_id+'.png')
            cv2.imwrite(str(path),args[1].crop)
            self.engine.plate_inputs[result.sample_id]=path
            return RecognitionAttempt(result,1,crop_frame=args[5],crop_score=args[2])
        runner._recognize=recognize
        runner.detector=SimpleNamespace(predict=self.detect)
        self.engine.plate=runner
        self.recovery=VehiclePlateRecovery(self.engine)
        self.image=np.random.default_rng(8).integers(30,220,(160,240,3),dtype=np.uint8)
        self.vehicles=[dict(track_id=1,bbox=[20,20,220,150])]
        self.roi_detections=[]

    def result(self, image, frame, outcome='recognized'):
        return UnifiedPlateResult(sample_id=f'sample-{frame}-{outcome}',timestamp='2026-11-24T00:00:00Z',
            route='local',route_confidence=1,route_reason='test',source='local_ocr',model_version='test',
            bbox=[0,0,40,15],status='accepted' if outcome=='recognized' else 'unresolved',
            plate_category='temporary',predicted_category='nt',recognition_status=outcome,
            unresolved_reason='insufficient_ocr_confidence' if outcome!='recognized' else None,
            normalized_number='123 NT' if outcome=='recognized' else None,text_confidence=.95 if outcome=='recognized' else .2,
            evidence=dict(engine_input=dict(sha256=sha256(image.tobytes()).hexdigest()),
                engine_operations=[dict(kind='registration_ocr',outcome='succeeded',
                                        model=dict(model_name='test',version=1,task='ocr'))]))

    def detect(self, image):
        self.detector_calls.append(image.shape)
        return self.roi_detections

    def observe(self, frame=0, detections=((80,100,140,120,.6),), image=None):
        receipt=SimpleNamespace(frame_index=frame)
        self.engine.receipts[frame]=receipt
        packet=SimpleNamespace(pixels=self.image if image is None else image,receipt=receipt)
        self.recovery.observe(packet,self.vehicles,list(detections),[])

    def states(self):
        return list(self.engine.plate.tracks.values())

    def drain(self):
        runner=self.engine.plate
        deadline=time.monotonic()+5
        while any(s.pending is not None or s.deferred_reason for s in self.states()):
            runner._drain('test',None)
            if time.monotonic()>deadline:self.fail('test OCR queue did not drain')
            time.sleep(.001)

    def test_detector_candidate_without_plate_tracker_schedules_once_at_closure_and_eof(self):
        self.observe()
        self.assertFalse(self.recovery.prepare(1,[]))
        self.drain()
        for _ in range(3):self.assertTrue(self.recovery.prepare(1,self.states()))
        self.assertEqual(len(self.calls),1)
        self.assertEqual(self.calls[0][6],'vehicle_closure_fallback')
        state=self.states()[0]
        self.assertEqual(self.engine.plate_parents[state.track_id,0],1)
        self.assertEqual(self.detector_calls,[])

    def test_existing_tracker_confidence_gap_drops_boxes_but_vehicle_recovery_keeps_them(self):
        from vehicle.recognition.config import paths
        from vehicle.recognition.plate.run_video_prototype import _build_tracker, TrackerDetections
        tracker=_build_tracker(paths()['plate_tracker'],30)
        for _ in range(10):
            rows=tracker.update(TrackerDetections.from_tuples([(80,100,140,120,.28)]),self.image)
            self.assertEqual(len(rows),0)
        # No global threshold is changed: the already accepted detector box is
        # retained under its unambiguous vehicle identity instead.
        self.observe(detections=((80,100,140,120,.28),))
        self.assertFalse(self.recovery.prepare(1,[]))
        self.drain()
        self.assertEqual(len(self.calls),1)
        self.assertEqual(self.calls[0][1].confidence,.28)

    def test_recovery_ids_do_not_collide_with_native_tracks_or_scene_resets(self):
        from vehicle.recognition.config import paths
        from vehicle.recognition.plate.run_video_prototype import _build_tracker, TrackerDetections
        tracker=PlateTracker(_build_tracker(paths()['plate_tracker'],30))
        first=tracker.allocate_id()
        native=int(tracker.update(TrackerDetections.from_tuples([(80,100,140,120,.9)]),self.image)[0,4])
        second=tracker.allocate_id()
        tracker.reset()
        third=tracker.allocate_id()
        next_native=int(tracker.update(TrackerDetections.from_tuples([(80,100,140,120,.9)]),self.image)[0,4])
        self.assertEqual(len({first,native,second,third,next_native}),5)
        self.assertGreater(third,second)

    def test_roi_detector_calls_are_bounded_and_boxes_map_back_to_original_frame(self):
        self.observe(detections=())
        self.observe(frame=1,detections=(),image=np.flip(self.image,axis=0).copy())
        self.roi_detections=[(10,70,80,90,.8)]
        self.assertFalse(self.recovery.prepare(1,[]))
        self.drain();self.assertTrue(self.recovery.prepare(1,self.states()))
        self.assertEqual(len(self.detector_calls),2)
        self.assertEqual(self.calls[0][1].bbox,(30,90,100,110))
        self.assertEqual(self.calls[0][8],240)  # Source width, not resized detector input.
        self.assertEqual(self.calls[0][9],160)
        for _ in range(3):self.assertTrue(self.recovery.prepare(1,self.states()))
        self.assertEqual(len(self.detector_calls),2)

    def test_roi_miss_is_recorded_without_an_ocr_execution(self):
        self.observe(detections=())
        self.assertTrue(self.recovery.prepare(1,[]))
        self.assertTrue(self.recovery.prepare(1,[]))
        self.assertEqual(len(self.detector_calls),1)
        self.assertEqual(self.calls,[])
        self.assertEqual(self.recovery.diagnostics(1)['stopping_stage'],'no_detection')

    def test_normal_accepted_result_does_not_schedule_recovery(self):
        self.observe()
        state=TrackState(5,0,0)
        state.attempts=[RecognitionAttempt(self.result(self.image,0),1,crop_frame=0)]
        self.engine.plate_parents[5,0]=1
        self.assertTrue(self.recovery.prepare(1,[state]))
        self.assertEqual(self.calls,[]);self.assertEqual(self.detector_calls,[])

    def test_one_low_confidence_recheck_uses_a_different_higher_quality_crop(self):
        self.observe(frame=1)
        state=TrackState(5,0,0)
        previous=RecognitionAttempt(self.result(np.zeros((4,4,3),np.uint8),0,'unresolved'),1,
                                    crop_frame=0,crop_score=1)
        state.attempts=[previous]
        self.engine.plate_parents[5,0]=1
        self.engine.plate.tracks[5]=state
        self.responses=['unresolved']
        self.assertFalse(self.recovery.prepare(1,[state]))
        self.drain()
        self.assertTrue(self.recovery.prepare(1,self.states()))
        self.assertEqual(len(self.calls),1)
        self.assertEqual(self.calls[0][6],'low_confidence_better_crop_recheck')
        self.assertEqual(self.calls[0][5],1)
        self.assertNotEqual(sha256(self.calls[0][1].crop.tobytes()).hexdigest(),
                            previous.result.evidence['engine_input']['sha256'])
        self.assertEqual(self.calls[0][-1]['retry_of'],previous.result.sample_id)
        self.assertTrue(all(a.result.recognition_status=='unresolved' for s in self.states() for a in s.attempts))

    def test_identical_image_is_not_rechecked_and_stop_does_not_start_ocr(self):
        self.observe()
        candidate=self.recovery.buffers[1].plates[0]
        state=TrackState(5,0,0)
        state.attempts=[RecognitionAttempt(self.result(candidate.plate.crop,0,'unresolved'),1,
                                         crop_frame=0,crop_score=1)]
        self.engine.plate_parents[5,0]=1
        self.assertTrue(self.recovery.prepare(1,[state]));self.assertEqual(self.calls,[])
        self.engine.stop.set()
        self.assertTrue(self.recovery.prepare(1,[]));self.assertEqual(self.calls,[])

    def test_ambiguous_vehicle_association_does_not_admit_a_plate(self):
        self.vehicles.append(dict(track_id=2,bbox=[21,20,221,150]))
        self.observe()
        self.assertEqual(self.recovery.buffers[1].plates,[])
        self.assertEqual(self.recovery.buffers[2].plates,[])
        self.roi_detections=[(60,80,120,100,.8)]
        self.assertTrue(self.recovery.prepare(1,[]));self.assertEqual(self.calls,[])

    def test_recovery_respects_existing_queue_capacity_and_releases_images(self):
        self.observe()
        blocker=threading.Event()
        blocked=TrackState(9,0,0)
        blocked.pending=self.engine.plate.executor.submit(blocker.wait,2)
        self.engine.plate.tracks[9]=blocked
        self.addCleanup(blocker.set)
        self.assertFalse(self.recovery.prepare(1,[]))
        recovery=next(s for s in self.states() if s.track_id!=9)
        self.assertIsNotNone(recovery.deferred_reason)
        self.assertIsNone(recovery.pending)
        blocker.set();blocked.pending.result();blocked.pending=None
        self.drain()
        self.assertEqual(len(self.calls),1)
        self.recovery.release(1)
        self.assertEqual(self.recovery.bytes,0)
        self.assertEqual(self.recovery.frames(),set())

    def test_eof_drains_recovery_before_closing_queue_without_duplicate_attempts(self):
        from vehicle.common.real_video.recognition_engine import RecognitionEngine
        from vehicle.common.tests.test_recognition_v3 import evidence_fixture
        from vehicle.common.redis_scalar.codec import unpack_life
        from vehicle.common.evaluation_capture import EvaluationRecorder
        from vehicle.recognition.attributes.manufacturer.tracking import TrackRecognition
        life=unpack_life(evidence_fixture()['life'])
        engine=RecognitionEngine.__new__(RecognitionEngine)
        engine.__dict__.update(self.engine.__dict__)
        engine.context=life.context
        engine.receipts={life.first_receipt.frame_index:life.first_receipt}
        engine.closed={};engine.ready=[];engine.models={};engine.crops={};engine.attribute_inputs={}
        engine.attribute_service=None;engine.manufacturer_evaluation=None
        engine.profiles={1:SimpleNamespace(track_id=life.raw_track_id,recognition=TrackRecognition())}
        engine.parents={life.raw_track_id:1}
        ends=[(life,),()]
        engine.lifecycle=SimpleNamespace(end=lambda reason:ends.pop(0),release_closed=lambda ids:None)
        engine.vehicle=SimpleNamespace(finish=lambda:None)
        engine.evaluation=EvaluationRecorder(engine.directory,engine.context)
        engine.plate_recovery=self.recovery
        self.engine=engine;self.recovery.engine=engine
        self.recovery.observe(SimpleNamespace(pixels=self.image,receipt=life.first_receipt),
            self.vehicles,[(80,100,140,120,.6)],[])
        closed=[]
        def finish(source):
            self.assertFalse(any(s.pending is not None for s in self.states()))
            closed.append(True)
        engine.plate.finish=finish
        engine.finish()
        engine._settle_closed()
        emitted=engine.drain()
        self.assertEqual(len(self.calls),1);self.assertEqual(closed,[True])
        self.assertEqual(len(emitted),1);self.assertEqual(len(emitted[0]['plates']),1)
        from vehicle.common.plate_status import project_plate
        self.assertEqual(project_plate(emitted[0])['text'],'123 NT')
        from vehicle.common.recognition_transport import RecognitionCodec
        from vehicle.common.redis_scalar.synthetic import ORG, IDENTITY_NAMESPACE
        codec=RecognitionCodec(ORG,IDENTITY_NAMESPACE)
        message=codec.decode(codec.make('db1.delivery.v1',emitted[0]))
        self.assertEqual(len(message.body['bundle']['facts']),2)
        self.assertEqual(self.recovery.bytes,0)
