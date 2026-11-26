"""Allowlisted media-run discovery and explicit historical evidence gaps."""
import json
import os
from pathlib import Path
import re
import sqlite3

from .evaluation import EvaluationStore, stable_id

ROOT=Path(__file__).resolve().parents[2]


class EvaluationCatalog:
    def __init__(self, root=None):
        self.root=Path(root or os.environ.get('T1_EVALUATION_ROOT',ROOT/'vehicle/media')).resolve()

    def directories(self):
        bases=[self.root,self.root/'t1-console']
        console=os.environ.get('T1_CONSOLE_RUN_ROOT')
        if console and Path(console).resolve() not in bases:bases.append(Path(console).resolve())
        seen=set()
        for base in bases:
            if not base.is_dir() or base.is_symlink():continue
            for p in sorted(base.iterdir()):
                if p in seen or not p.is_dir() or p.is_symlink():continue
                seen.add(p)
                yield p

    def runs(self):
        runs=[]
        for directory in self.directories():
            try:
                path=directory/'evaluation.sqlite3'
                if path.is_file() and not path.is_symlink():
                    with sqlite3.connect(path.as_uri()+'?mode=ro',uri=True) as db:
                        meta=dict(db.execute('SELECT key,value FROM metadata'))
                    run_id=meta['run_id'];source_id=meta.get('source_id',run_id)
                    legacy=source_id.startswith('historical-video:')
                else:
                    meta=self.legacy(directory)
                    if meta.get('source_kind') not in (None,'mp4','file'):continue
                    run_id=meta['run_id'];source_id='historical-video:'+run_id;legacy=True
                if not re.fullmatch(r'[A-Za-z0-9_-]{1,100}',run_id):continue
                try:summary=self.legacy(directory)
                except (OSError,ValueError):summary={}
                runs.append(dict(run_id=run_id,source_id=source_id,historical=legacy,
                    filename=summary.get('upload',{}).get('filename'),status=summary.get('status')))
            except (OSError,ValueError,KeyError,sqlite3.Error):continue
        return runs

    @staticmethod
    def legacy(directory):
        for name in ('status.json','summary.json'):
            p=directory/name
            if p.is_file() and not p.is_symlink():return json.loads(p.read_text())
        raise FileNotFoundError()

    def open(self, run_id):
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,100}',run_id):raise FileNotFoundError()
        for directory in self.directories():
            path=directory/'evaluation.sqlite3'
            if path.is_file() and not path.is_symlink():
                with sqlite3.connect(path.as_uri()+'?mode=ro',uri=True) as db:
                    metadata=dict(db.execute('SELECT key,value FROM metadata'))
                if metadata.get('run_id')==run_id:return EvaluationStore(directory)
            elif not path.is_symlink():
                try:summary=self.legacy(directory)
                except (OSError,ValueError):continue
                if summary.get('run_id')!=run_id:continue
                if summary.get('source_kind') not in (None,'mp4','file'):raise FileNotFoundError()
                store=EvaluationStore(directory,run_id=run_id,source_id='historical-video:'+run_id)
                records=[]
                for row in summary.get('rows',[]):
                    sid=row['track_id']
                    records.extend([(sid,'type',row.get('vehicle_type'),'completed'),
                        (sid,'colour',row.get('colour',{}).get('label'),row.get('colour',{}).get('status')),
                        (sid,'manufacturer',row.get('manufacturer',{}).get('make'),row.get('manufacturer',{}).get('status'))])
                for row in summary.get('predictions',[]):
                    if row.get('model_text'):records.append((row['track_id'],'manufacturer',row.get('make'),row.get('status')))
                for row in summary.get('plate_rows',[]):
                    records.append((row['track_id'],'plate',row.get('plate_text'),row.get('outcome')))
                for sid,domain,label,outcome in records:
                    if label is None and outcome not in {'unreadable','unknown'}:continue
                    store.record(dict(id=stable_id(run_id,domain,sid,'historical'),run_id=run_id,
                        source_id='historical-video:'+run_id,track_id=sid,domain=domain,kind='prediction',
                        predicted_label=label,outcome=outcome,frame_index=None,video_timestamp=None,
                        model_version=None,quality_policy_version=None,inference_time_ms=None,quality_time_ms=None,
                        image=None,historical=True,timing_status='unavailable',input_scope='Historical input crop unavailable; not reconstructed.'))
                return store
        raise FileNotFoundError()
