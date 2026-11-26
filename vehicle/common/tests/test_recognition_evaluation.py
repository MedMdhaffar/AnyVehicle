"""Real v3 policy with cheap fake generation: completeness is not model accuracy."""
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
import json
import sqlite3
import tempfile
import threading
import unittest

import numpy as np

from vehicle.common.evaluation import metrics, public_records, EvaluationStore
from vehicle.common.evaluation_capture import EvaluationRecorder
from vehicle.common.frame_receipt import FrameReceiptClock, ClockPolicy
from vehicle.common.recognition_evaluation import ManufacturerEvaluation
from vehicle.common.real_video.recognition_engine import RecognitionEngine
from vehicle.common.real_video.input import DecodedFrame
from vehicle.common.redis_scalar.codec import unpack_life
from vehicle.common.tests.test_recognition_v3 import evidence_fixture
from vehicle.common.vehicle_observation import VehicleLifecycle, RunEnd
from vehicle.recognition.attributes.manufacturer.config import RecognitionConfig
from vehicle.recognition.attributes.manufacturer.service import ManufacturerService
from vehicle.recognition.attributes.manufacturer.tracking import TrackRecognizer, TrackRecognition
from vehicle.recognition.attributes.pipeline import VehiclePipeline


class Backend:
    def __init__(self, response='toyota'):
        self.response=response;self.inputs=[]
    def generate(self, png, prompt, **kwargs):
        self.inputs.append((png,prompt,kwargs))
        if isinstance(self.response,Exception):raise self.response
        return dict(choices=[dict(message=dict(content=json.dumps(dict(manufacturer=self.response,vehicle_type='sedan',color='black'))),finish_reason='stop')])


class ManufacturerEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.directory=Path(self.temp.name)
        life=unpack_life(evidence_fixture()['life']);self.context=life.context;self.index=0
        self.clock=FrameReceiptClock(self.context,utc_clock=lambda:life.first_receipt.utc+timedelta(seconds=self.index/25),
            monotonic_clock=lambda:life.first_receipt.monotonic+self.index/25,policy=ClockPolicy())
        self.lifecycle=VehicleLifecycle(self.context,self.clock)
        self.service=ManufacturerService.__new__(ManufacturerService)
        from vehicle.recognition.attributes.manufacturer.prompts import PROMPT
        self.service.config=RecognitionConfig();self.service.prompt=PROMPT
        self.service._lock=threading.RLock();self.service._backend=Backend()
        self.recognizer=TrackRecognizer(self.service,self.service.config)
        self.recognizer.output_dir=self.directory/'attribute-crops'
        self.engine=SimpleNamespace(context=self.context,receipts={},models={'attributes':{'model_name':'test-vlm'}},
            attribute_service=self.service,vehicle=SimpleNamespace(recognizer=self.recognizer),
            evaluation=EvaluationRecorder(self.directory,self.context),native_lane=threading.RLock(),stop=threading.Event(),calls={})
        self.syncs=[]
        self.observer=ManufacturerEvaluation(self.engine,synchronize=lambda:self.syncs.append(True))
        self.recognizer.decision_sink=self.observer.decision
        self.image=np.random.default_rng(7).integers(10,245,(480,640,3),dtype=np.uint8)

    def receipt(self):
        receipt=self.clock.capture(self.index,media_pts_reader=lambda:self.index/25)
        self.engine.receipts[self.index]=receipt
        self.index+=1
        return receipt

    def observe(self,state,life=None,image=None,frames=9):
        for n in range(frames):
            receipt=self.receipt()
            self.recognizer.observe(state,self.image if image is None else image,[20,20,420,320],receipt.frame_index,n+1,detector_label='car',detector_confidence=.9)
            life=self.lifecycle.observe(state.track_id,receipt)
            self.observer.bind(life,state)
        return life

    def state(self,raw):return TrackRecognition(track_id=raw,key=f'track_{raw}_{self.index}')

    def test_all_lifetimes_have_exclusive_outcomes_and_every_attempt_has_input_and_scoped_events(self):
        cases=[]
        for raw,response in [(1,'toyota'),(2,'unknown'),(3,RuntimeError('failed native')),(4,TimeoutError('native deadline'))]:
            state=self.state(raw);self.service._backend.response=response
            life=self.observe(state)
            self.observer.finalize(life,state);cases.append(state)
        state=self.state(5);life=self.observe(state,image=np.zeros_like(self.image));self.observer.finalize(life,state)
        state=self.state(6);life=self.observe(state,frames=1);self.observer.finalize(life,state)  # eligible, not yet scheduled
        state=self.state(7);self.engine.stop.set();life=self.observe(state);self.observer.finalize(life,state,incomplete=True)
        records=self.engine.evaluation.store.records();calls=self.engine.evaluation.store.calls();summary=metrics(records,calls)
        self.assertEqual(summary['manufacturer_completeness'],dict(lifetimes=7,finalized=7,pending=0,balanced=True,
            final_outcomes=dict(prediction_produced=1,unknown_manufacturer=1,inference_failure=2,no_eligible_crop=1,
                                skipped_unscheduled=1,stopped_incomplete=1)))
        self.assertEqual(summary['domains']['manufacturer']['total'],1)
        attempts=[r for r in records if r['kind']=='attempt'];self.assertEqual(len(attempts),5)
        native=[c for c in calls if c['stage']=='manufacturer'];self.assertEqual(len(native),4)
        self.assertEqual(len(self.syncs),8)
        self.assertEqual(summary['model_timeouts'],1);self.assertEqual(summary['model_failures'],1)
        for row in attempts:
            events=[c for c in calls if c.get('attempt_ordinal')==row['attempt_ordinal'] and c.get('track_key')==row['track_key']]
            self.assertTrue(any(c['stage']=='manufacturer_attempt' for c in events))
            if row['native_call_entered']:
                self.assertTrue(row['quality']['eligible']);self.assertIsNotNone(row['quality_time_ms'])
                self.assertIsNotNone(row['image']);self.assertTrue(self.engine.evaluation.store.crop(row['image']).exists())
                self.assertIsNotNone(row['inference_time_ms'])
                self.assertEqual(row['inference_time_ms'],(int(row['inference_completion_ns'])-int(row['inference_start_ns']))/1e6)
                self.assertGreater(row['attempt_time_ms'],row['inference_time_ms'])
            else:self.assertIsNone(row['inference_time_ms'])
        qualities=[r for r in records if r['kind']=='quality'];self.assertEqual(len(qualities),55)
        rejected=[r for r in qualities if r['outcome']=='quality_rejected'];self.assertTrue(any(r['image'] for r in rejected))
        self.assertEqual(sum(r['image'] is not None for r in rejected),1)
        self.assertTrue(all(r['quality_time_ms'] is not None for r in qualities))

    def test_pending_first_frame_submission_resolves_service_id_and_preserves_transport(self):
        self.recognizer.config=replace(self.recognizer.config,initial_frames=1)
        state=self.state(99);life=self.observe(state,frames=1)
        rows=self.engine.evaluation.store.records();attempt=next(r for r in rows if r['kind']=='attempt')
        self.assertEqual((attempt['track_id'],attempt['raw_track_id']),(life.service_track_id,99))
        import cv2
        actual=cv2.imread(str(self.engine.evaluation.store.crop(attempt['image'])))
        np.testing.assert_array_equal(actual,self.image[20:320,20:420])
        with self.engine.evaluation.store.connect() as db:
            self.assertEqual(db.execute('SELECT count(*) FROM manufacturer_pending WHERE projected=0').fetchone()[0],0)
        self.observer.finalize(life,state)
        prediction=next(r for r in rows if r['kind']=='attempt')
        self.assertEqual(prediction['predicted_label'],'toyota')

    def test_zero_calls_rejected_and_disabled_never_get_native_latency(self):
        state=self.state(3);life=self.observe(state,image=np.zeros_like(self.image));self.observer.finalize(life,state)
        self.engine.vehicle.recognizer=None;self.engine.attribute_service=None
        disabled=ManufacturerEvaluation(self.engine,synchronize=lambda:None)
        receipt=self.receipt();life=self.lifecycle.observe(4,receipt);disabled.finalize(life,TrackRecognition())
        summary=metrics(self.engine.evaluation.store.records(),self.engine.evaluation.store.calls())
        self.assertEqual(summary['total_model_calls'],0)
        self.assertEqual(summary['manufacturer_completeness']['lifetimes'],2)
        self.assertTrue(all(r.get('inference_time_ms') is None for r in self.engine.evaluation.store.records()))

    def test_inspection_does_not_modify_original_prediction_reviews_or_denominator(self):
        state=self.state(1);life=self.observe(state);self.observer.finalize(life,state)
        store=self.engine.evaluation.store
        predicted=next(r for r in store.records() if r['kind']=='prediction' and r['domain']=='manufacturer')
        rejected=next(r for r in store.records() if r['kind']=='quality')
        store.review(predicted['id'],'FALSE','citroen',revision=0)
        with store.connect() as db:before=db.execute('SELECT id,payload FROM evidence ORDER BY id').fetchall()
        store.inspect(rejected['id'],'NEEDS_ATTENTION','Inspect detail',revision=0)
        with store.connect() as db:after=db.execute('SELECT id,payload FROM evidence ORDER BY id').fetchall()
        self.assertEqual(before,after)
        result=metrics(store.records(),store.calls())
        self.assertEqual(result['domains']['manufacturer']['incorrect'],1)
        with self.assertRaises(ValueError):store.review(rejected['id'],'TRUE',revision=0)
        self.assertEqual(next(r for r in store.records() if r['id']==predicted['id'])['review']['correct_label'],'citroen')

    def test_historical_outer_duration_is_not_relabelled_as_native(self):
        store=self.engine.evaluation.store
        record=dict(id='prior',kind='prediction',domain='manufacturer',predicted_label='toyota',track_id=1,
            model_version='LFM2.5-VL-450M-sha256:old',inference_time_ms=420,image=None)
        store.record(record);store.review('prior','TRUE',revision=0)
        row=next(r for r in public_records(store,self.context.run_id) if r['id']=='prior')
        self.assertEqual(row['historical_producer_duration_ms'],420);self.assertIsNone(row['inference_time_ms'])
        self.assertEqual(store.records()[0]['inference_time_ms'],420)
        self.assertEqual(store.records()[0]['review']['verdict'],'TRUE')

    def test_failure_before_native_entry_does_not_fabricate_model_call_or_inference_time(self):
        def fail_sync():raise RuntimeError('sync before native failed')
        self.observer.sync=fail_sync
        state=self.state(1);life=self.observe(state);self.observer.finalize(life,state)
        store=self.engine.evaluation.store;summary=metrics(store.records(),store.calls())
        self.assertEqual(summary['total_model_calls'],0)
        attempt=next(r for r in store.records() if r['kind']=='attempt')
        self.assertFalse(attempt['native_call_entered']);self.assertIsNone(attempt['inference_time_ms'])
        self.assertTrue(attempt['image'])

    def test_quality_exception_is_retained_and_can_be_finalized_incomplete(self):
        def fail(*args):raise RuntimeError('quality failed')
        self.recognizer.selector.quality_analyzer.analyze=fail
        state=self.state(1);receipt=self.receipt()
        with self.assertRaisesRegex(RuntimeError,'quality failed'):
            self.recognizer.observe(state,self.image,[20,20,420,320],receipt.frame_index,1,detector_label='car')
        life=self.lifecycle.observe(1,receipt);self.observer.bind(life,state)
        self.observer.finalize(life,state,incomplete=True)
        rows=self.engine.evaluation.store.records()
        self.assertTrue(any(r['kind']=='quality' and r['outcome']=='quality_failure' for r in rows))
        self.assertTrue(metrics(rows,self.engine.evaluation.store.calls())['manufacturer_completeness']['balanced'])

    def test_known_attempt_prediction_can_be_reviewed_without_inflating_final_accuracy(self):
        state=self.state(1);life=self.observe(state);self.observer.finalize(life,state)
        store=self.engine.evaluation.store
        attempt=next(r for r in store.records() if r['kind']=='attempt')
        store.review(attempt['id'],'FALSE','citroen',revision=0)
        result=metrics(store.records(),store.calls())['domains']['manufacturer']
        self.assertEqual((result['total'],result['incorrect'],result['unreviewed']),(1,0,1))
        public=next(r for r in public_records(store,self.context.run_id) if r['id']==attempt['id'])
        self.assertTrue(public['reviewable_prediction']);self.assertFalse(public['accuracy_prediction'])

    def test_finish_guard_and_lifetime_binding_cannot_silently_drop_or_cross_identities(self):
        state=self.state(1);life=self.observe(state,frames=1)
        with self.assertRaisesRegex(ValueError,'manufacturer_evaluation_incomplete'):self.observer.ensure_complete()
        with self.assertRaisesRegex(ValueError,'manufacturer_lifetime_binding_changed'):
            self.observer.bind(replace(life,service_track_id=life.service_track_id+1),state)
        self.observer.finalize(life,state);self.observer.ensure_complete()


if __name__=='__main__':unittest.main()

class EngineEvaluationIntegrationTests(unittest.TestCase):
    setUp = ManufacturerEvaluationTests.setUp
    receipt = ManufacturerEvaluationTests.receipt
    def build_engine(self):
        # Use the production engine orchestration, pipeline and recognizer. Only
        # detector rows, plate work and native generation are cheap test doubles.
        self.service._backend=Backend()
        self.service.predict=ManufacturerService.predict.__get__(self.service)
        self.recognizer=TrackRecognizer(self.service,self.service.config)
        self.recognizer.output_dir=self.directory/'attribute-crops'
        class Detector:
            class_names={2:'car'}
            tracker=SimpleNamespace(update=lambda *args:None)
            def process(self,frame):return np.array([[20,20,420,320,1,.9,2]])
            def reset(self):pass
        engine=RecognitionEngine.__new__(RecognitionEngine)
        engine.context=self.context;engine.directory=self.directory;engine.lifecycle=self.lifecycle
        engine.attribute_service=self.service;engine.vehicle=VehiclePipeline(Detector(),recognizer=self.recognizer)
        for name in ('receipts','parents','profiles','plate_parents','closed','crops','models','plate_diagnostics','attribute_inputs','plate_inputs'):
            setattr(engine,name,{})
        engine.models={'attributes':{'model_name':'test-vlm','version':1,'task':'vlm'}};engine.ready=[];engine.finished=False
        engine.calls={};engine.stop=threading.Event();engine.native_lane=threading.RLock()
        engine.evaluation=EvaluationRecorder(self.directory,self.context)
        engine.plate=SimpleNamespace(tracks={},tracker=SimpleNamespace(reset=lambda:None),
            process_frame=lambda *args:[],frame_plate_detections=[],disappearance_frames=5,
            finish=lambda *args:None,close=lambda:None,minimum_exit_candidates=1,
            pipeline=SimpleNamespace(marker_localizer=None),detector=None,
            **{k:[] for k in ('pending_snapshots','detection_times_ms','tracking_times_ms','crop_scoring_times_ms',
                             'detections_per_frame','tracked_outputs_per_frame')})
        engine.manufacturer_evaluation=ManufacturerEvaluation(engine,synchronize=lambda:None)
        self.recognizer.decision_sink=engine._attribute_crop
        self.engine=engine
        return engine

    def test_completed_manufacturer_evidence_survives_plate_finalization_failure_before_delivery(self):
        engine=self.build_engine()
        for _ in range(9):engine.process(DecodedFrame(self.receipt(),self.image))
        def fail(*args):raise RuntimeError('plate_finalization_failure')
        engine.plate.finish=fail
        with self.assertRaisesRegex(RuntimeError,'plate_finalization_failure'):engine.finish()
        store=engine.evaluation.store;summary=metrics(store.records(),store.calls())
        self.assertTrue(summary['manufacturer_completeness']['balanced'])
        self.assertEqual(summary['domains']['manufacturer']['total'],1)
        self.assertEqual(engine.ready,[])  # No downstream delivery was needed.
        self.assertTrue(any(r['kind']=='attempt' and r['image'] for r in store.records()))

    def test_close_after_processing_failure_retains_lifetime_quality_and_incomplete_outcome(self):
        engine=self.build_engine()
        def fail(*args):raise RuntimeError('plate_processing_failure')
        engine.plate.process_frame=fail
        with self.assertRaises(RuntimeError):engine.process(DecodedFrame(self.receipt(),self.image))
        engine.attribute_service.close=lambda:None
        engine.close()
        summary=metrics(engine.evaluation.store.records(),engine.evaluation.store.calls())
        self.assertEqual(summary['manufacturer_completeness']['final_outcomes'],{'stopped_incomplete':1})
        self.assertTrue(summary['manufacturer_completeness']['balanced'])
        self.assertEqual(summary['total_model_calls'],0)
        self.assertEqual(len([r for r in engine.evaluation.store.records() if r['kind']=='quality']),1)

    def test_evaluation_storage_failure_still_releases_owned_models(self):
        engine=self.build_engine();engine.process(DecodedFrame(self.receipt(),self.image));closed=[]
        def fail(*args,**kwargs):raise ValueError('evaluation storage failed')
        engine.manufacturer_evaluation.finalize=fail
        engine.plate.close=lambda:closed.append('plate')
        engine.attribute_service.close=lambda:closed.append('attributes')
        with self.assertRaisesRegex(ValueError,'evaluation storage failed'):engine.close()
        self.assertEqual(closed,['plate','attributes']);self.assertIsNone(engine.vehicle)

    def test_scene_cut_reused_raw_id_keeps_attempt_crops_and_timings_on_the_correct_lifetime(self):
        engine=self.build_engine()
        first=np.random.default_rng(12).integers(20,110,self.image.shape,dtype=np.uint8)
        second=np.random.default_rng(13).integers(150,240,self.image.shape,dtype=np.uint8)
        engine.process(DecodedFrame(self.receipt(),first))
        engine.process(DecodedFrame(self.receipt(),second))
        before=engine.drain();self.assertEqual(len(before),1)
        engine.finish();after=engine.drain();self.assertEqual(len(after),1)
        lives=[unpack_life(v['life']) for v in before+after]
        self.assertEqual([life.raw_track_id for life in lives],[1,1])
        self.assertEqual([life.tracker_epoch for life in lives],[0,1])
        self.assertNotEqual(lives[0].service_track_id,lives[1].service_track_id)
        rows=engine.evaluation.store.records()
        attempts=sorted((r for r in rows if r['kind']=='attempt'),key=lambda r:r['frame_index'])
        self.assertEqual(len(attempts),2)
        import cv2
        for attempt,life,source in zip(attempts,lives,[first,second]):
            self.assertEqual(attempt['track_id'],life.service_track_id)
            self.assertEqual(attempt['frame_index'],life.first_receipt.frame_index)
            pixels=cv2.imread(str(engine.evaluation.store.crop(attempt['image'])))
            np.testing.assert_array_equal(pixels,source[20:320,20:420])
            self.assertIsNotNone(attempt['inference_time_ms'])
            self.assertEqual(attempt['inference_time_ms'],(int(attempt['inference_completion_ns'])-int(attempt['inference_start_ns']))/1e6)
        result=metrics(rows,engine.evaluation.store.calls())
        self.assertEqual(result['manufacturer_completeness']['lifetimes'],2)
        self.assertTrue(result['manufacturer_completeness']['balanced'])
        self.assertEqual(result['stages']['manufacturer']['total_calls'],2)
        # Exercise the actual codec, row construction and commit-before-ACK
        # consumer with captured scene lifetimes. No native PostgreSQL claim.
        from contextlib import nullcontext
        from vehicle.common.recognition_transport import RecognitionCodec
        from vehicle.common.redis_scalar.synthetic import ORG, IDENTITY_NAMESPACE
        from vehicle.common.tests.test_db1_consumer import supplied_test_registrations, TransactionAdapter
        from vehicle.common.db1_consumer import NaturalConsumer
        from vehicle.common.db1_adapter import TrustedTenant
        from vehicle.common.db1_rows import Dimension
        codec=RecognitionCodec(ORG,IDENTITY_NAMESPACE);adapter=TransactionAdapter(codec.validation);acks=[]
        for evidence in before+after:
            raw=codec.make('db1.delivery.v1',evidence)
            registration=supplied_test_registrations(SimpleNamespace(v1=codec),raw)
            registration=replace(registration,manufacturers=(Dimension('Toyota',41,True),))
            context=codec.decode(raw).life.context
            consumer=NaturalConsumer(codec=codec,tenant=TrustedTenant(ORG,'scene-cut integration fixture'),
                trusted_run=context.run_id,trusted_camera=context.camera.camera_id,resolutions=registration,
                connection_factory=lambda:nullcontext(None),environment='test',adapter=adapter)
            def ack():
                self.assertTrue(adapter.history);acks.append(True)
            self.assertEqual(consumer.process(raw,ack).disposition,'new')
            self.assertEqual(consumer.process(raw,ack).disposition,'identical')
        self.assertEqual(len(adapter.history),2)
        self.assertEqual(len(adapter.facts),6)
        self.assertEqual(len(acks),4)

    def test_first_frame_submission_after_cut_does_not_overwrite_previous_lifetime_thumbnail(self):
        engine=self.build_engine()
        # Supported scheduler configuration, test-only; quality thresholds unchanged.
        self.recognizer.config=replace(self.recognizer.config,initial_frames=1)
        first=np.random.default_rng(21).integers(20,110,self.image.shape,dtype=np.uint8)
        second=np.random.default_rng(22).integers(150,240,self.image.shape,dtype=np.uint8)
        writes=[];save=engine._save_crop
        def saving(sid,kind,image,frame):
            writes.append((sid,frame));return save(sid,kind,image,frame)
        engine._save_crop=saving
        old_sid=engine.process(DecodedFrame(self.receipt(),first))[0]['track_id']
        new_sid=engine.process(DecodedFrame(self.receipt(),second))[0]['track_id']
        self.assertNotEqual(old_sid,new_sid)
        self.assertNotIn((old_sid,1),writes)
        engine.finish()
        self.assertIn((old_sid,0),writes);self.assertIn((new_sid,1),writes)
        self.assertEqual(engine.crops[str(old_sid)]['vehicle']['frame'],0)
        self.assertEqual(engine.crops[str(new_sid)]['vehicle']['frame'],1)
        self.assertTrue(metrics(engine.evaluation.store.records(),engine.evaluation.store.calls())['manufacturer_completeness']['balanced'])
