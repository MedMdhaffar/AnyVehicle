"""Run-local evaluation evidence and human ground truth, independent of DB1."""
import csv
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import sha256
import io
import json
import math
from pathlib import Path
import re
import sqlite3
import statistics
from uuid import uuid5, NAMESPACE_URL

VERDICTS = ('UNREVIEWED', 'TRUE', 'FALSE', 'NOT CLEAR')
DOMAINS = ('manufacturer', 'plate', 'plate_category', 'type', 'colour')
INSPECTIONS = ('UNINSPECTED', 'INSPECTED', 'NEEDS_ATTENTION')


def prediction_record(record):
    """Accuracy counts known final predictions, never attempts or abstentions."""
    label = record.get('predicted_label')
    return record['kind']=='prediction' and label is not None and str(label).strip().lower() not in ('','unknown')


def reviewable_record(record):
    """A known successful attempt may be reviewed without counting it twice."""
    label=record.get('predicted_label')
    return prediction_record(record) or (record['kind']=='attempt' and record.get('native_call_entered')
        and record.get('outcome')=='completed' and label is not None and str(label).strip().lower() not in ('','unknown'))


def stable_id(*parts):
    return uuid5(NAMESPACE_URL, json.dumps(parts, ensure_ascii=False, separators=(',', ':'))).hex


def latency(values):
    values = sorted(float(v) for v in values if v is not None and math.isfinite(float(v)) and float(v) >= 0)
    def percentile(p):
        index = (len(values) - 1) * p
        low = math.floor(index)
        return values[low] + (values[math.ceil(index)] - values[low]) * (index - low)
    if not values:
        return dict(count=0, mean=None, median=None, p95=None, p99=None, minimum=None,
                    observed_wcet=None, standard_deviation=None)
    return dict(count=len(values), mean=statistics.mean(values), median=statistics.median(values),
                p95=percentile(.95), p99=percentile(.99), minimum=values[0], observed_wcet=values[-1],
                standard_deviation=statistics.pstdev(values))


def metrics(records, calls):
    domains = {}
    for domain in DOMAINS:
        selected = [r for r in records if prediction_record(r) and r['domain'] == domain]
        counts = {v: sum(r['review']['verdict'] == v for r in selected) for v in VERDICTS}
        evaluated = counts['TRUE'] + counts['FALSE']
        total = len(selected)
        domains[domain] = dict(total=total, correct=counts['TRUE'], incorrect=counts['FALSE'],
            unclear=counts['NOT CLEAR'], unreviewed=counts['UNREVIEWED'],
            evaluated_accuracy=counts['TRUE']/evaluated if evaluated else None,
            review_coverage=(total-counts['UNREVIEWED'])/total if total else None,
            unclear_rate=counts['NOT CLEAR']/total if total else None)
    stages = {}
    for stage in sorted({c['stage'] for c in calls}):
        group = [c for c in calls if c['stage'] == stage]
        successful = [c for c in group if c['outcome'] == 'completed']
        stages[stage] = dict(**latency(c.get('duration_ms') for c in successful),
            scopes=sorted({c['scope'] for c in group if c.get('scope')}),
            total_calls=len(group), failures=sum(c['outcome'] == 'failed' for c in group),
            timeouts=sum(c['outcome'] == 'timed_out' for c in group),
            cancelled=sum(c['outcome'] in {'cancelled','execution_uncertain'} for c in group),
            unsuccessful_latency=latency(c.get('duration_ms') for c in group if c['outcome'] != 'completed'))
    plates=[r for r in records if r['domain']=='plate']
    model_calls=[c for c in calls if c.get('model_call',c['stage'] in {'vehicle_detection','plate_detection','manufacturer','ocr','plate_marker'})]
    request_outcomes=[c for c in calls if c['stage']=='manufacturer_request_outcome']
    lifetimes=[r for r in records if r['domain']=='manufacturer' and r['kind']=='lifetime']
    terminals=[r for r in records if r['domain']=='manufacturer' and r['kind']=='outcome']
    counts=dict(Counter(r['outcome'] for r in terminals))
    identities=[r['lifetime_id'] for r in terminals]
    completeness=dict(lifetimes=len(lifetimes),final_outcomes=counts,finalized=len(terminals),
        pending=len({r['lifetime_id'] for r in lifetimes}-set(identities)),
        balanced=len(lifetimes)==len(terminals) and len(set(identities))==len(terminals))
    return dict(domains=domains, stages=stages,total_model_calls=len(model_calls),
        manufacturer_completeness=completeness,
        model_failures=sum(c['outcome']=='failed' for c in model_calls+request_outcomes),
        model_timeouts=sum(c['outcome']=='timed_out' for c in model_calls+request_outcomes),
        unconfirmed_manufacturer_requests=len(request_outcomes),plate_outcomes=dict(
        reads=sum(r.get('outcome')=='read' for r in plates),
        unreadable=sum(r.get('outcome')=='unreadable' for r in plates),
        no_detection_observations=sum(r.get('detection_status')=='no_detection' for r in plates),
        human_confirmed_detection_failures=sum(r['kind']=='observation' and r.get('detection_status')=='no_detection'
            and r['review']['verdict']=='FALSE' for r in plates)),
        observed_wcet_note='Maximum observed successful latency; not a formal hard real-time WCET guarantee.')


class ReviewConflict(ValueError):
    pass


class EvaluationStore:
    def __init__(self, directory, *, run_id=None, source_id=None):
        self.directory = Path(directory)
        self.path = self.directory/'evaluation.sqlite3'
        if self.directory.is_symlink() or self.path.is_symlink():raise ValueError('unsafe_evaluation_path')
        with self.connect() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY,value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS evidence (id TEXT PRIMARY KEY,payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS calls (id TEXT PRIMARY KEY,payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS reviews (id TEXT PRIMARY KEY REFERENCES evidence,
                    verdict TEXT NOT NULL,correct_label TEXT,revision INTEGER NOT NULL,updated_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS review_changes (sequence INTEGER PRIMARY KEY,id TEXT NOT NULL,
                    previous TEXT,verdict TEXT NOT NULL,correct_label TEXT,revision INTEGER NOT NULL,
                    changed_at TEXT NOT NULL,reviewer TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS inspections (id TEXT PRIMARY KEY REFERENCES evidence,
                    status TEXT NOT NULL,note TEXT,revision INTEGER NOT NULL,updated_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS inspection_changes (sequence INTEGER PRIMARY KEY,id TEXT NOT NULL,
                    status TEXT NOT NULL,note TEXT,revision INTEGER NOT NULL,changed_at TEXT NOT NULL,reviewer TEXT NOT NULL);
            ''')
            if run_id is not None:
                for key,value in [('run_id',run_id),('source_id',source_id or run_id)]:
                    prior=db.execute('SELECT value FROM metadata WHERE key=?',(key,)).fetchone()
                    if prior and prior[0]!=value:raise ValueError('evaluation_identity_conflict')
                    db.execute('INSERT OR IGNORE INTO metadata VALUES (?,?)',(key,value))
        self.path.chmod(0o600)

    @contextmanager
    def connect(self):
        db=sqlite3.connect(self.path, timeout=10)
        db.execute('PRAGMA journal_mode=WAL')
        db.execute('PRAGMA synchronous=FULL')
        db.execute('PRAGMA foreign_keys=ON')
        try:
            with db:yield db
        finally:db.close()

    def metadata(self):
        with self.connect() as db:return dict(db.execute('SELECT key,value FROM metadata'))

    def record(self, record):
        payload=json.dumps(record,ensure_ascii=False,allow_nan=False,sort_keys=True)
        now=datetime.now(timezone.utc).isoformat()
        with self.connect() as db:
            prior=db.execute('SELECT payload FROM evidence WHERE id=?',(record['id'],)).fetchone()
            if prior:
                if prior[0]!=payload:raise ValueError('immutable_evaluation_evidence_changed')
                return
            db.execute('INSERT INTO evidence VALUES (?,?)',(record['id'],payload))
            db.execute('INSERT INTO reviews VALUES (?,?,NULL,0,?)',(record['id'],'UNREVIEWED',now))
            db.execute('INSERT INTO review_changes(id,previous,verdict,revision,changed_at,reviewer) VALUES (?,NULL,?,0,?,?)',
                       (record['id'],'UNREVIEWED',now,'producer'))

    def call(self, record):
        with self.connect() as db:
            payload=json.dumps(record,allow_nan=False,sort_keys=True)
            previous=db.execute('SELECT payload FROM calls WHERE id=?',(record['id'],)).fetchone()
            if previous and previous[0]!=payload:raise ValueError('immutable_timing_changed')
            db.execute('INSERT OR IGNORE INTO calls VALUES (?,?)',(record['id'],payload))

    def review(self, identity, verdict, corrected=None, *, revision, reviewer='local reviewer'):
        if verdict not in VERDICTS or type(revision) is not int or revision < 0:raise ValueError('invalid_review')
        if corrected is not None and (type(corrected) is not str or len(corrected)>2048):raise ValueError('invalid_corrected_label')
        if verdict!='FALSE':corrected=None
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            payload=db.execute('SELECT payload FROM evidence WHERE id=?',(identity,)).fetchone()
            if payload is None:raise FileNotFoundError('missing_prediction')
            if not reviewable_record(json.loads(payload[0])):raise ValueError('inspection_only')
            old=db.execute('SELECT verdict,revision FROM reviews WHERE id=?',(identity,)).fetchone()
            if old is None:raise FileNotFoundError('missing_prediction')
            if old[1]!=revision:raise ReviewConflict('review_changed_refresh_required')
            now=datetime.now(timezone.utc).isoformat()
            db.execute('UPDATE reviews SET verdict=?,correct_label=?,revision=?,updated_at=? WHERE id=?',
                (verdict,corrected,revision+1,now,identity))
            db.execute('INSERT INTO review_changes(id,previous,verdict,correct_label,revision,changed_at,reviewer) VALUES (?,?,?,?,?,?,?)',
                (identity,old[0],verdict,corrected,revision+1,now,reviewer[:100]))

    def inspect(self, identity, status, note=None, *, revision, reviewer='local reviewer'):
        if status not in INSPECTIONS or type(revision) is not int or revision<0:raise ValueError('invalid_inspection')
        if note is not None and (type(note) is not str or len(note)>2048):raise ValueError('invalid_note')
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            payload=db.execute('SELECT payload FROM evidence WHERE id=?',(identity,)).fetchone()
            if payload is None:raise FileNotFoundError('missing_evidence')
            if reviewable_record(json.loads(payload[0])):raise ValueError('prediction_requires_verdict')
            old=db.execute('SELECT revision FROM inspections WHERE id=?',(identity,)).fetchone()
            if (old[0] if old else 0)!=revision:raise ReviewConflict('inspection_changed_refresh_required')
            now=datetime.now(timezone.utc).isoformat()
            db.execute('INSERT OR REPLACE INTO inspections VALUES (?,?,?,?,?)',(identity,status,note,revision+1,now))
            db.execute('INSERT INTO inspection_changes(id,status,note,revision,changed_at,reviewer) VALUES (?,?,?,?,?,?)',
                       (identity,status,note,revision+1,now,reviewer[:100]))

    def records(self):
        with self.connect() as db:
            rows=db.execute('''SELECT e.payload,r.verdict,r.correct_label,r.revision,r.updated_at,
                i.status,i.note,i.revision,i.updated_at FROM evidence e JOIN reviews r USING(id)
                LEFT JOIN inspections i USING(id) ORDER BY e.rowid''').fetchall()
        return [dict(json.loads(p),review=dict(verdict=v,correct_label=c,revision=rev,updated_at=t),
            inspection=dict(status=s or 'UNINSPECTED',note=n,revision=ir or 0,updated_at=it))
            for p,v,c,rev,t,s,n,ir,it in rows]

    def calls(self):
        with self.connect() as db:return [json.loads(p) for (p,) in db.execute('SELECT payload FROM calls ORDER BY rowid')]

    def history(self):
        with self.connect() as db:
            db.row_factory=sqlite3.Row
            return [dict(row) for row in db.execute('SELECT * FROM review_changes ORDER BY sequence')]

    def inspection_history(self):
        with self.connect() as db:
            db.row_factory=sqlite3.Row
            return [dict(row) for row in db.execute('SELECT * FROM inspection_changes ORDER BY sequence')]

    def crop(self, image):
        if not re.fullmatch(r'[a-f0-9]{64}\.png',image):raise FileNotFoundError()
        folder=self.directory/'evaluation-images';path=folder/image
        if folder.is_symlink() or path.is_symlink() or not path.is_file():raise FileNotFoundError()
        return path


def public_records(store, run_id):
    rows=store.records()
    # CLI receipts count the bounded replay window; retain that identity and
    # expose the original source-video index separately when its offset is known.
    offset=None
    summary=store.directory/'summary.json'
    if summary.is_file() and not summary.is_symlink():
        try:offset=json.loads(summary.read_text()).get('capture',{}).get('start_frame')
        except (ValueError,OSError):pass
    elif (store.directory/'status.json').is_file():offset=0
    if type(offset) not in (int,float) or offset<0 or int(offset)!=offset:offset=None
    def image_reference(name):
        if name is None:return None
        try:store.crop(name)
        except (OSError,ValueError):return None
        return f'/api/t1/evaluation/runs/{run_id}/images/{name}/'
    for r in rows:
        r['reviewable_prediction']=reviewable_record(r)
        r['accuracy_prediction']=prediction_record(r)
        r['display_kind']=r['kind'] if r['kind']!='prediction' or r['reviewable_prediction'] else 'observation'
        if (str(r.get('model_version','')).startswith('LFM2.5-VL-450M-sha256:')
                and not r.get('timing_scope') and not r.get('inference_start_ns')):
            # Old v3 stored outer attempt latency in this field. Preserve it explicitly,
            # without rewriting immutable evidence or pretending it measured generation.
            r['historical_producer_duration_ms']=r.get('inference_time_ms')
            r['inference_time_ms']=None
            r['timing_status']='historical_scope_unverified'
        frame=r.get('frame_index')
        r['source_frame_index']=int(offset)+frame if offset is not None and frame is not None else None
        r['crop_reference']=image_reference(r.pop('image',None))
        r['original_frame_reference']=image_reference(r.pop('original_frame',None))
        r['supporting_crops']=[url for name in r.pop('inputs',[]) if (url:=image_reference(name))]
        r['crop_status']='available' if r['crop_reference'] else 'unavailable'
    return rows


def export_csv(records):
    columns=['run_id','source_id','camera_id','lifetime_id','track_id','frame_index','source_frame_index','video_timestamp','domain','predicted_label','correct_label',
        'review_verdict','is_correct','inference_time_ms','quality_time_ms','model_version',
        'quality_policy_version','crop_reference']
    output=io.StringIO(newline='');writer=csv.DictWriter(output,fieldnames=columns);writer.writeheader()
    for record in records:
        review=record['review']
        if not prediction_record(record) or review['verdict'] not in {'TRUE','FALSE'}:continue
        row={key:record.get(key) for key in columns}
        row.update(correct_label=review['correct_label'],review_verdict=review['verdict'],is_correct=review['verdict']=='TRUE')
        # Preserve CSV structure and prevent spreadsheet formula execution.
        for key,value in row.items():
            if isinstance(value,str) and value.lstrip().startswith(('=','+','-','@','\t','\r')):row[key]="'"+value
        writer.writerow(row)
    return '\ufeff'+output.getvalue()
