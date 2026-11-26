import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

from vehicle.common.evaluation import (EvaluationStore, ReviewConflict, stable_id, latency,
    metrics, export_csv, public_records)
from vehicle.common.evaluation_catalog import EvaluationCatalog
from vehicle.common.evaluation_timing import measured


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.directory=Path(self.temp.name)/'run';self.directory.mkdir()
        self.store=EvaluationStore(self.directory,run_id='run-one',source_id='video-sha256:abc')

    def record(self,key,*,kind='prediction',domain='manufacturer',**fields):
        value=dict(id=stable_id('run-one',domain,key),run_id='run-one',track_id=1,domain=domain,kind=kind,
            predicted_label='Toyota',frame_index=2,video_timestamp=.08,model_version='test-model-v1',image=None)
        value.update(fields);self.store.record(value)
        return value['id']

    def test_initial_state_stable_identity_persistence_and_immutable_evidence(self):
        key=self.record('one');self.record('one')
        reopened=EvaluationStore(self.directory)
        self.assertEqual(len(reopened.records()),1)
        self.assertEqual(reopened.records()[0]['review']['verdict'],'UNREVIEWED')
        self.assertEqual(key,stable_id('run-one','manufacturer','one'))
        with self.assertRaises(ValueError):self.record('one',predicted_label='other')

    def test_review_changes_persist_and_keep_audit_history(self):
        key=self.record('one')
        self.store.review(key,'FALSE','Peugeot',revision=0)
        self.store.review(key,'NOT CLEAR',revision=1)
        self.store.review(key,'TRUE',revision=2)
        row=EvaluationStore(self.directory).records()[0]
        self.assertEqual(row['predicted_label'],'Toyota')
        self.assertEqual(row['review']['revision'],3)
        self.assertIsNone(row['review']['correct_label'])
        self.assertEqual([r['verdict'] for r in self.store.history()],['UNREVIEWED','FALSE','NOT CLEAR','TRUE'])
        with self.assertRaises(ReviewConflict):self.store.review(key,'FALSE',revision=0)

    def test_csv_excludes_unclear_unreviewed_and_quality_and_preserves_false_without_correction(self):
        for i,verdict in enumerate(['TRUE','FALSE','NOT CLEAR','UNREVIEWED']):
            key=self.record(str(i),domain='plate',predicted_label='123 تونس 4567, "x"\nline')
            if verdict!='UNREVIEWED':self.store.review(key,verdict,revision=0)
        quality=self.record('quality',kind='quality');self.store.inspect(quality,'INSPECTED',revision=0)
        text=export_csv(public_records(self.store,'run-one'))
        self.assertTrue(text.startswith('\ufeff'))
        rows=list(csv.DictReader(io.StringIO(text.lstrip('\ufeff'))))
        self.assertEqual(len(rows),2)
        self.assertEqual(rows[1]['review_verdict'],'FALSE')
        self.assertEqual(rows[1]['correct_label'],'')
        self.assertEqual(rows[0]['predicted_label'],'123 تونس 4567, "x"\nline')
        self.assertNotIn(self.temp.name,text)

    def test_metrics_denominators_empty_set_and_domains_are_separate(self):
        for i,verdict in enumerate(['TRUE','FALSE','NOT CLEAR','UNREVIEWED']):
            key=self.record(str(i))
            if verdict!='UNREVIEWED':self.store.review(key,verdict,revision=0)
        self.record('quality',kind='quality')
        result=metrics(self.store.records(),[])['domains']
        self.assertEqual(result['manufacturer'],dict(total=4,correct=1,incorrect=1,unclear=1,unreviewed=1,
            evaluated_accuracy=.5,review_coverage=.75,unclear_rate=.25))
        self.assertIsNone(result['plate']['evaluated_accuracy'])
        self.assertEqual(metrics([],[])['domains']['type']['total'],0)

    def test_missing_and_rejected_crops_are_explicit_and_never_deleted(self):
        folder=self.directory/'evaluation-images';folder.mkdir();name='a'*64+'.png';path=folder/name;path.write_bytes(b'png')
        key=self.record('rejected',kind='quality',image=name,outcome='rejected')
        self.store.inspect(key,'NEEDS_ATTENTION',revision=0)
        self.assertEqual(path.read_bytes(),b'png')
        self.assertTrue(public_records(self.store,'run-one')[0]['crop_reference'].endswith(name+'/'))
        path.unlink()
        self.assertEqual(public_records(self.store,'run-one')[0]['crop_status'],'unavailable')
        self.assertEqual(metrics(self.store.records(),[])['domains']['manufacturer']['total'],0)

    def test_path_traversal_and_symlink_rejected(self):
        with self.assertRaises(FileNotFoundError):self.store.crop('../database-private.json')
        folder=self.directory/'evaluation-images';folder.mkdir();(folder/('b'*64+'.png')).symlink_to('/etc/passwd')
        with self.assertRaises(FileNotFoundError):self.store.crop('b'*64+'.png')

    def test_source_frame_offset_is_explicit_and_not_assumed_for_missing_metadata(self):
        self.record('offset')
        self.assertIsNone(public_records(self.store,'run-one')[0]['source_frame_index'])
        (self.directory/'summary.json').write_text(json.dumps(dict(capture=dict(start_frame=250))))
        record=public_records(self.store,'run-one')[0]
        self.assertEqual(record['frame_index'],2)
        self.assertEqual(record['source_frame_index'],252)

    def test_historical_prediction_does_not_invent_crop_or_timing(self):
        old=Path(self.temp.name)/'historical';old.mkdir()
        (old/'summary.json').write_text(json.dumps(dict(run_id='old-run',predictions=[dict(track_id=2,
            make='toyota',status='accepted',model_text='original output')],plate_rows=[])))
        catalog=EvaluationCatalog(self.temp.name)
        store=catalog.open('old-run');record=public_records(store,'old-run')[0]
        self.assertEqual(record['predicted_label'],'toyota')
        self.assertEqual(record['crop_status'],'unavailable')
        self.assertIsNone(record['inference_time_ms'])
        self.assertFalse(store.calls())

    def test_timing_synchronizes_before_and_after_execution_and_records_failure(self):
        events=[];ticks=iter([100,200])
        value,timing=measured(lambda:events.append('invoke') or 7,
            synchronize=lambda:events.append('sync'),clock=lambda:next(ticks))
        self.assertEqual(events,['sync','invoke','sync']);self.assertEqual(value,7)
        self.assertEqual(timing['duration_ms'],.0001)
        def fail():raise TimeoutError('test')
        with self.assertRaises(TimeoutError) as caught:measured(fail,clock=iter([1000,2000]).__next__)
        self.assertEqual(caught.exception.evaluation_timing['completion_ns'],'2000')

    def test_latency_aggregation_excludes_failures_and_exposes_timeouts(self):
        values=[1,2,3,4,100]
        result=latency(values)
        self.assertEqual(result['mean'],22);self.assertEqual(result['median'],3)
        self.assertAlmostEqual(result['p95'],80.8);self.assertAlmostEqual(result['p99'],96.16)
        self.assertEqual(result['observed_wcet'],100)
        calls=[dict(stage='ocr',outcome='completed',duration_ms=2),dict(stage='ocr',outcome='timed_out',duration_ms=1000)]
        stage=metrics([],calls)['stages']['ocr']
        self.assertEqual((stage['count'],stage['timeouts'],stage['total_calls'],stage['mean']),(1,1,2,2))
        self.assertEqual(stage['unsuccessful_latency']['observed_wcet'],1000)

    def test_failed_gpu_sync_preserves_original_error_without_fabricating_completed_latency(self):
        events=[]
        def synchronize():
            events.append('sync')
            if len(events)>1:raise RuntimeError('sync failed')
        def fail():raise TimeoutError('original timeout')
        with self.assertRaisesRegex(TimeoutError,'original timeout') as caught:
            measured(fail,synchronize=synchronize)
        self.assertIsNone(caught.exception.evaluation_timing['duration_ms'])
        self.assertIsNone(caught.exception.evaluation_timing['completion_ns'])


if __name__=='__main__':unittest.main()
