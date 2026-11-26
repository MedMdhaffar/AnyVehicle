"""Replay the baseline through HTTP admission, then compare plate evidence and cost."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def load(directory):
    from vehicle.common.redis_scalar.codec import bounded_json
    status = json.loads((directory/'status.json').read_text())
    evidence = [bounded_json(p.read_bytes(),max_bytes=16*1024*1024,max_depth=40,
        max_nodes=300000,max_string=8192)['body']['evidence'] for p in directory.glob('recognition-[0-9]*.json')]
    return status,evidence


def signature(evidence):
    life = evidence['life']
    return life['raw_track_id'],life['tracker_epoch'],life['first']['frame_index']


def match_vehicle(original, current, used):
    exact = [e for e in current if signature(e)==signature(original) and e['life']['service_track_id'] not in used]
    if len(exact)==1:return exact[0],'raw tracker, epoch and original first frame'
    def box(e):
        attempts = e['attributes']['recognition_evidence']
        return next((a.get('engine_input',{}).get('bbox') for a in reversed(attempts)
                     if a.get('engine_input',{}).get('bbox')),None)
    def overlap(a,b):
        intersection=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
        union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
        return intersection/union if union else 0
    a=original['life'];scores=[]
    for candidate in current:
        b=candidate['life']
        if b['service_track_id'] in used or box(original) is None or box(candidate) is None:continue
        shared=max(0,min(a['last']['frame_index'],b['last']['frame_index'])-max(a['first']['frame_index'],b['first']['frame_index'])+1)
        span=max(a['last']['frame_index'],b['last']['frame_index'])-min(a['first']['frame_index'],b['first']['frame_index'])+1
        geometry=overlap(box(original),box(candidate))
        if shared/span>=.7 and geometry>=.7:scores.append((float(shared/span)*geometry,candidate))
    scores.sort(key=lambda item:item[0],reverse=True)
    if scores and (len(scores)==1 or scores[0][0]-scores[1][0]>=.1):
        return scores[0][1],'source-frame interval and vehicle-crop bounding-box overlap'
    return None,'unmatched'


def compare(baseline, updated, output):
    from vehicle.common.plate_status import project_plate
    before,original = load(baseline)
    after,current = load(updated)
    used = set()
    comparisons = []
    for e in original:
        old = project_plate(e)
        sid = e['life']['service_track_id']
        if sid not in {17,18,19,21,23,33,35,41,49} and old['status']!='read':continue
        match,matched_by = match_vehicle(e,current,used)
        if match:used.add(match['life']['service_track_id'])
        new = project_plate(match) if match else None
        comparisons.append(dict(baseline_track=sid,
            updated_track=match['life']['service_track_id'] if match else None,
            matched_by=matched_by,
            baseline=old,updated=new,accepted_text_matches_visible_plate='unverified',
            previous_read_preserved=(new['status']=='read' and new['text']==old['text']) if old['status']=='read' and new is not None else None))
    def elapsed(s):return (datetime.fromisoformat(s['ended_at'])-datetime.fromisoformat(s['started_at'])).total_seconds()
    def attempts(path):return sum(1 for line in (path/'plate-evidence/predictions.jsonl').read_text().splitlines() if line.strip())
    cost = dict(baseline_elapsed_seconds=elapsed(before),updated_elapsed_seconds=elapsed(after),
        wall_time_delta_seconds=elapsed(after)-elapsed(before),
        extra_detector_calls=after['counters']['plate_detection']-before['counters']['plate_detection'],
        extra_registration_ocr_calls=after['counters'].get('registration_ocr',0)-before['counters'].get('registration_ocr',0),
        extra_plate_pipeline_attempts=attempts(updated)-attempts(baseline))
    report = dict(baseline=str(baseline),updated=str(updated),
        same_recorded_models_and_policies=before.get('models')==after.get('models'),
        same_input_sha256=before.get('upload',{}).get('sha256')==after.get('upload',{}).get('sha256'),
        same_analyzed_frame_count=before['analyzed_frames']==after['analyzed_frames'],
        updated_status=after['status'],cost=cost,vehicles=comparisons,
        note='Unmatched identities and visual text correctness require manual inspection. Wall time includes loading and delivery; this is not an isolated GPU benchmark.')
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2,default=float))
    print(json.dumps(dict(status=after['status'],report=str(output),cost=cost,
        changed_previous_reads=[c['baseline_track'] for c in comparisons if c['previous_read_preserved'] is False],
        unmatched=[c['baseline_track'] for c in comparisons if c['updated'] is None]),default=float))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,default=ROOT/'vehicle/media/t1-console/83a49b38-1deb-4539-b661-5a7d770dc60b')
    parser.add_argument('--output',type=Path,required=True,help='New isolated validation directory')
    parser.add_argument('--updated',type=Path,help='Compare an already completed run without inference')
    args = parser.parse_args()
    if args.updated:
        args.output.mkdir(parents=True,exist_ok=True)
        compare(args.baseline,args.updated,args.output/'comparison.json')
        return
    from vehicle.common.console_stack import configure
    configure()
    import torch
    if not torch.cuda.is_available():raise SystemExit('GPU replay blocked: CUDA unavailable; no inference or database writes performed.')
    config = json.loads((args.baseline/'config.json').read_text())
    if config['source_kind']!='mp4' or config['sample_every']!=1 or not config['manufacturer'] or config['max_seconds']:
        raise SystemExit('Baseline configuration is not supported by this replay wrapper.')
    command = [sys.executable,'-B',str(ROOT/'scripts/validate_recognition_console.py'),
        '--video',config['source'],'--database','--output',str(args.output.resolve())]
    subprocess.run(command,cwd=ROOT,check=True)
    admission = json.loads((args.output/'admission.json').read_text())
    compare(args.baseline,args.output/'runs'/admission['run_id'],args.output/'comparison.json')


if __name__=='__main__':main()
