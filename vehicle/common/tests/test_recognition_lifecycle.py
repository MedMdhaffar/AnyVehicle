"""Shared service identity survives source scene cuts and raw tracker ID reuse."""
from datetime import timedelta
from types import SimpleNamespace
import unittest

import numpy as np

from vehicle.common.real_video.recognition_engine import RecognitionEngine
from vehicle.common.real_video.input import DecodedFrame
from vehicle.common.vehicle_observation import VehicleLifecycle
from vehicle.common.frame_receipt import FrameReceiptClock, ClockPolicy
from vehicle.common.tests.test_recognition_v3 import evidence_fixture
from vehicle.common.redis_scalar.codec import unpack_life
from vehicle.recognition.attributes.pipeline import VehiclePipeline


class RecognitionLifecycleTests(unittest.TestCase):
    def test_scene_cut_closes_old_identity_before_reused_raw_id_is_observed(self):
        life=unpack_life(evidence_fixture()['life']);counter=[0]
        clock=FrameReceiptClock(life.context,utc_clock=lambda:life.first_receipt.utc+timedelta(seconds=counter[0]),
            monotonic_clock=lambda:life.first_receipt.monotonic+counter[0],policy=ClockPolicy())
        class Detector:
            class_names={0:'car'}
            tracker=SimpleNamespace(update=lambda *args:None)
            def process(self,frame):return np.array([[1,1,10,10,1,.9,0]])
            def reset(self):pass
        pipeline=VehiclePipeline(Detector())
        pipeline.scene_detector=SimpleNamespace(update=lambda frame:counter[0]==1)
        engine=RecognitionEngine.__new__(RecognitionEngine)
        engine.context=life.context;engine.vehicle=pipeline;engine.attribute_service=None
        engine.lifecycle=VehicleLifecycle(life.context,clock)
        for field in ('receipts','parents','profiles','plate_parents','closed','crops','models','plate_diagnostics','attribute_inputs','plate_inputs'):
            setattr(engine,field,{})
        engine.ready=[]
        engine.plate=SimpleNamespace(tracks={},tracker=SimpleNamespace(reset=lambda:None),
            process_frame=lambda *args:[],frame_plate_detections=[],disappearance_frames=5,
            finish=lambda *args:None,minimum_exit_candidates=1,
            **{k:[] for k in ('pending_snapshots','detection_times_ms','tracking_times_ms','crop_scoring_times_ms',
                             'detections_per_frame','tracked_outputs_per_frame')})
        for index in (0,1):
            counter[0]=index
            engine.process(DecodedFrame(clock.capture(index),np.zeros((20,20,3),dtype=np.uint8)))
        first=unpack_life(engine.drain()[0]['life'])
        engine.finish()
        second=unpack_life(engine.drain()[0]['life'])
        self.assertEqual(first.raw_track_id,second.raw_track_id)
        self.assertNotEqual(first.service_track_id,second.service_track_id)
        self.assertEqual((first.tracker_epoch,second.tracker_epoch),(0,1))
        self.assertEqual((first.last_receipt.frame_index,second.first_receipt.frame_index),(0,1))


class ByteTrackLifecycleIntegrationTests(unittest.TestCase):
    """Exercise real CPU ByteTrack and real scene detection without model inference."""
    def setUp(self):
        from pathlib import Path
        import tempfile
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.directory=Path(self.temp.name)
        life=unpack_life(evidence_fixture()['life']);self.context=life.context;self.index=0
        self.clock=FrameReceiptClock(self.context,utc_clock=lambda:life.first_receipt.utc+timedelta(seconds=self.index/25),
            monotonic_clock=lambda:life.first_receipt.monotonic+self.index/25,policy=ClockPolicy())
        from ultralytics.trackers.byte_tracker import BYTETracker
        import yaml
        from vehicle.recognition.attributes.config import TRACKER_CONFIG
        config=SimpleNamespace(**yaml.safe_load(TRACKER_CONFIG.read_text()))
        owner=self
        class Detector:
            class_names={2:'car',8:'van'}
            def __init__(self):self.tracker=BYTETracker(config,frame_rate=25)
            def process(self,frame):
                from ultralytics.engine.results import Boxes
                from vehicle.recognition.attributes.detector import suppress_duplicate_boxes
                data=np.asarray(owner.detections,dtype=np.float32).reshape(-1,6)
                boxes=Boxes(data,frame.shape[:2]).cpu().numpy()
                return self.tracker.update(suppress_duplicate_boxes(boxes),frame)
            def reset(self):self.tracker.reset()  # Real reset permits raw ID reuse.
        detector=Detector()
        engine=RecognitionEngine.__new__(RecognitionEngine)
        engine.context=self.context;engine.directory=self.directory;engine.attribute_service=None
        engine.lifecycle=VehicleLifecycle(self.context,self.clock);engine.vehicle=VehiclePipeline(detector)
        for name in ('receipts','parents','profiles','plate_parents','closed','crops','models','plate_diagnostics','attribute_inputs','plate_inputs'):
            setattr(engine,name,{})
        engine.ready=[];engine.finished=False
        engine.plate=SimpleNamespace(tracks={},tracker=SimpleNamespace(reset=lambda:None),
            process_frame=lambda *args:[],frame_plate_detections=[],disappearance_frames=5,
            finish=lambda *args:None,minimum_exit_candidates=1,
            **{k:[] for k in ('pending_snapshots','detection_times_ms','tracking_times_ms','crop_scoring_times_ms',
                             'detections_per_frame','tracked_outputs_per_frame')})
        from vehicle.common.evaluation_capture import EvaluationRecorder
        from vehicle.common.recognition_evaluation import ManufacturerEvaluation
        engine.evaluation=EvaluationRecorder(self.directory,self.context)
        engine.manufacturer_evaluation=ManufacturerEvaluation(engine,synchronize=lambda:None)
        self.engine=engine

    def step(self,detections,*,white=False):
        self.detections=detections
        image=np.full((480,640,3),255 if white else 0,dtype=np.uint8)
        receipt=self.clock.capture(self.index,media_pts_reader=lambda:self.index/25)
        result=self.engine.process(DecodedFrame(receipt,image));self.index+=1
        return result

    def test_real_scene_cut_and_bytetrack_reset_reuse_have_distinct_once_closed_lifetimes(self):
        box=[[20,20,220,200,.95,2]]
        before=self.step(box);old_sid=before[0]['track_id']
        after=self.step(box,white=True);new_sid=after[0]['track_id']
        closed=self.engine.drain()
        self.assertEqual(len(closed),1)
        old=unpack_life(closed[0]['life'])
        self.assertEqual((old.raw_track_id,old.tracker_epoch,old.first_receipt.frame_index,old.last_receipt.frame_index),(1,0,0,0))
        self.assertEqual(self.engine.vehicle.scene_cuts,[1])
        self.assertNotEqual(old_sid,new_sid)
        self.assertEqual(self.engine.parents[1],new_sid)
        for _ in range(2):self.assertEqual(self.step(box,white=True)[0]['track_id'],new_sid)
        self.assertEqual(self.engine.drain(),[])
        self.engine.finish();final=self.engine.drain()
        self.assertEqual(len(final),1)
        new=unpack_life(final[0]['life'])
        self.assertEqual((new.raw_track_id,new.tracker_epoch,new.first_receipt.frame_index,new.last_receipt.frame_index),(1,1,1,3))
        self.engine.finish();self.assertEqual(self.engine.drain(),[])
        from vehicle.common.evaluation import metrics
        summary=metrics(self.engine.evaluation.store.records(),self.engine.evaluation.store.calls())
        self.assertEqual(summary['manufacturer_completeness']['lifetimes'],2)
        self.assertTrue(summary['manufacturer_completeness']['balanced'])

    def test_delayed_old_closure_cannot_remove_reactivated_raw_id_parent(self):
        box=[[20,20,220,200,.95,2]]
        first=self.step(box)[0]['track_id']
        collect=self.engine._collect
        self.engine._collect=lambda **kwargs:None  # Simulate waiting on asynchronous plate work.
        for _ in range(16):self.step([])
        self.assertIn(first,self.engine.closed)
        resumed=self.step(box)[0]['track_id']
        self.assertNotEqual(first,resumed)
        self.assertEqual(self.engine.parents[1],resumed)
        self.engine._collect=collect;collect()
        old=self.engine.drain();self.assertEqual(len(old),1)
        self.assertEqual(self.engine.parents[1],resumed)
        self.assertEqual(self.step(box)[0]['track_id'],resumed)
        self.engine.finish();new=self.engine.drain();self.assertEqual(len(new),1)
        self.assertEqual(unpack_life(old[0]['life']).last_receipt.frame_index,0)
        self.assertEqual(unpack_life(new[0]['life']).first_receipt.frame_index,17)
        from vehicle.common.evaluation import metrics
        self.assertTrue(metrics(self.engine.evaluation.store.records(),self.engine.evaluation.store.calls())['manufacturer_completeness']['balanced'])

    def test_cross_class_duplicate_proposals_do_not_create_extra_logical_vehicles(self):
        proposals=[[20,20,220,200,.95,2],[21,21,221,201,.80,8],[320,20,520,200,.90,2]]
        ids=None
        for _ in range(4):
            rows=self.step(proposals)
            self.assertEqual(len(rows),2)
            current={r['track_id'] for r in rows}
            if ids is not None:self.assertEqual(current,ids)
            ids=current
        self.engine.finish();closed=self.engine.drain()
        self.assertEqual(len(closed),2)
        self.assertEqual(len({unpack_life(r['life']).service_track_id for r in closed}),2)
