"""Offline manufacturer crop-coverage audit (no VLM, production defaults unchanged).

Re-derives the console's vehicle observations deterministically (same MP4 decoder,
vehicle detector/tracker and shared Engine), scores every observed crop under the
production crop-quality policy and offline alternatives, and cross-checks track
identity/outcome against an existing console run. Evidence written to --output:

* tracks.json         per-lifetime observation, eligibility and rejection counts
* crops.jsonl.gz      per-crop scalar metrics (no pixels)
* evidence/           best eligible crop per policy, best rejected crop, and the
                      full source frame of each recovered crop (local only)
* summary.json        classification, recovery per policy, cross-check result
"""
import argparse
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def policies():
    from vehicle.common.manufacturer.crop_quality import QUALITY_V2
    v2 = QUALITY_V2
    alt = lambda name, **kw: (name, replace(v2, version='offline-' + name, **kw))
    return [('v2_production', v2),
            alt('area_49152', min_area=49152),
            alt('area_36864_min160x120', min_area=36864, min_width=160, min_height=120),
            alt('aspect_0.8', min_car_aspect_ratio=0.8),
            alt('aspect_off', min_car_aspect_ratio=0.0),
            alt('edges_1', max_border_edges=1),
            alt('combined_area49152_aspect0.8_edges1', min_area=49152, min_car_aspect_ratio=0.8, max_border_edges=1)]


class AuditLedger:
    """Engine manufacturer interface that records crop quality; never runs a model."""
    def __init__(self, output, crops_log):
        from vehicle.common.manufacturer.crop_quality import CropQualityAnalyzer
        self.policies = [(n, c, CropQualityAnalyzer(c)) for n, c in policies()]
        self.output, self.crops_log = output, crops_log
        self.tracks = {}

    def observe(self, life, receipt, bbox, crop, frame_sha256, *, label=None, confidence=None, frame_shape=None, frame=None):
        sid = life.service_track_id
        t = self.tracks.setdefault(sid, dict(service_track_id=sid, raw_track_id=life.raw_track_id, observed=0,
                                             first_frame=receipt.frame_index, labels=Counter(),
                                             eligible={n: 0 for n, _, _ in self.policies}, reasons=Counter(),
                                             reason_sets=Counter(), best={}, best_rejected=None, max_area=0))
        t['observed'] += 1;t['last_frame'] = receipt.frame_index;t['labels'][label] += 1
        h, w = crop.shape[:2];t['max_area'] = max(t['max_area'], w * h)
        record = dict(sid=sid, frame=receipt.frame_index, bbox=list(bbox), w=w, h=h, label=label)
        for name, config, analyzer in self.policies:
            q = analyzer.analyze(crop, bbox=bbox, frame_shape=frame_shape, label=label)
            record[name] = dict(eligible=q.eligible, reasons=list(q.reasons), score=round(q.score, 6))
            if name == 'v2_production':
                record.update(sharpness=round(q.sharpness_variance, 3), dark=round(q.dark_fraction, 4),
                              bright=round(q.bright_fraction, 4), edges=q.border_edges, aspect=round(q.aspect_ratio, 4))
                if not q.eligible:
                    t['reasons'].update(q.reasons);t['reason_sets'][';'.join(q.reasons)] += 1
                    if t['best_rejected'] is None or w * h > t['best_rejected'][0]:
                        t['best_rejected'] = (w * h, crop.copy(), receipt.frame_index, list(q.reasons))
            if q.eligible:
                t['eligible'][name] += 1
                old = t['best'].get(name)
                if old is None or q.score > old[0]:
                    t['best'][name] = (q.score, crop.copy(), receipt.frame_index, list(bbox), frame)
        self.crops_log.write(json.dumps(record) + '\n')

    def selected_class(self, life):
        return None

    def terminal(self, life):
        return dict(audit='coverage', service_track_id=life.service_track_id)

    def close(self):
        pass


def save_png(path, pixels):
    import cv2
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), pixels)


def classify(t):
    """Smallest single relaxation that would make any crop eligible, else 'multiple'."""
    if t['eligible']['v2_production']:
        return 'eligible'
    sets = [set(s.split(';')) for s in t['reason_sets']]
    groups = {'size': {'too_small', 'insufficient_area'}, 'frame_boundary': {'frame_boundary'},
              'aspect': {'implausible_car_shape'}, 'blur': {'severely_blurred'},
              'exposure': {'essentially_black', 'essentially_white'}}
    for name, members in groups.items():
        if any(s and s <= members for s in sets):
            return name + '_only'
    return 'multiple'


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--video', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--compare-run', type=Path, help='console run directory to cross-check')
    p.add_argument('--sample-every', type=int, default=1)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True);os.chmod(a.output, 0o700)
    os.environ.update(YOLO_OFFLINE='true', YOLO_AUTOINSTALL='false', YOLO_CONFIG_DIR=str(a.output/'model-settings'))
    from vehicle.common.console_stack import configure, model_paths, console_device
    configure(load_dsn=False)
    from uuid import UUID
    from vehicle.common.db1_context import CameraRegistration, ModelRegistration, ResolutionInfo, RunContext
    from vehicle.common.frame_receipt import FrameReceiptClock, ClockPolicy
    from vehicle.common.observation_evidence import RecognitionIdentity
    from vehicle.common.real_video.input import MP4Input
    from vehicle.common.real_video.local import LocalDetector, pixel_operations, fingerprint
    from vehicle.common.real_video.engine import Engine
    from vehicle.common.engine_plate import native_access
    from vehicle.common.vehicle_observation import RunEnd
    import cv2, torch
    torch.set_num_threads(1);cv2.setNumThreads(1)
    paths = model_paths();org = UUID('12345678-1234-4234-8234-123456789012');now = datetime.now(timezone.utc)
    context = RunContext(org, CameraRegistration(org, 'coverage-audit', None, 'Africa/Tunis', now, None), a.output.name, 'file', now,
                         'vehicle-coverage-audit-v1', ResolutionInfo(None, None, 'vehicle', ModelRegistration(
                             None, 'local-yolo26-sha256:' + fingerprint(paths['vehicle_model'])['sha256'], 1, 'detect'),
                             None, 'offline coverage audit', False), 'development')
    clock = FrameReceiptClock(context, utc_clock=lambda: datetime.now(timezone.utc), monotonic_clock=time.monotonic, policy=ClockPolicy())
    crops_log = gzip.open(a.output/'crops.jsonl.gz', 'wt')
    ledger = AuditLedger(a.output, crops_log)
    source = MP4Input(a.video, clock);source.__enter__()
    with native_access():
        vehicle = LocalDetector(paths['vehicle_model'], paths['vehicle_tracker'], device=console_device(),
                                frame_rate=source.metadata['fps'], threads=1)
    estimate, quality = pixel_operations()
    engine = Engine(context, clock, vehicle, algorithm=RecognitionIdentity('t1-local-hsv-console', 1, 'classify'),
                    quality_algorithm='t1-colour-quality', estimate=estimate, quality=quality, manufacturer=ledger,
                    streaming=True, max_vehicles=128)
    observe = ledger.observe
    current = {}
    ledger.observe = lambda *args, **kw: observe(*args, frame=current.get('pixels'), **kw)
    started = time.monotonic();frames = analyzed = 0
    try:
        while True:
            packet = source.read()
            if packet is None:
                break
            frames += 1
            if packet.receipt.frame_index % a.sample_every:
                continue
            vehicle.source_index = packet.receipt.frame_index
            current['pixels'] = packet.pixels
            engine.process(packet);analyzed += 1
            for sid in tuple(engine.pending):
                engine.acknowledge(sid)
        engine.finish(RunEnd.END_OF_INPUT, trigger='eof')
        for sid in tuple(engine.pending):
            engine.acknowledge(sid)
    finally:
        source.close();crops_log.close();engine.close()
    # Evidence and per-track records.
    names = [n for n, _ in policies()]
    out_tracks = []
    for sid, t in sorted(ledger.tracks.items()):
        category = classify(t)
        entry = dict(service_track_id=sid, raw_track_id=t['raw_track_id'], first_frame=t['first_frame'], last_frame=t['last_frame'],
                     observed=t['observed'], labels=dict(t['labels']), max_crop_area=t['max_area'], eligible=t['eligible'],
                     rejection_reasons=dict(t['reasons']), reason_sets=dict(t['reason_sets']), category=category, evidence={})
        if t['best_rejected'] is not None:
            _, pixels, frame, reasons = t['best_rejected']
            name = f'evidence/track-{sid:04d}-largest-rejected-f{frame}.png';save_png(a.output/name, pixels)
            entry['evidence']['largest_rejected'] = dict(file=name, frame=frame, reasons=reasons)
        for policy, (score, pixels, frame, bbox, full) in t['best'].items():
            if policy != 'v2_production' and 'v2_production' in t['best']:
                continue  # Recovery evidence only for tracks v2 rejects entirely.
            name = f'evidence/track-{sid:04d}-best-{policy}-f{frame}.png';save_png(a.output/name, pixels)
            item = dict(file=name, frame=frame, bbox=bbox, score=round(score, 6), width=pixels.shape[1], height=pixels.shape[0])
            if policy != 'v2_production' and full is not None:
                fname = f'evidence/frames/frame-{frame:06d}.jpg'
                if not (a.output/fname).exists():
                    (a.output/fname).parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(a.output/fname), full, [cv2.IMWRITE_JPEG_QUALITY, 92])
                item['source_frame'] = fname
            entry['evidence'][policy] = item
        out_tracks.append(entry)
    (a.output/'tracks.json').write_text(json.dumps(out_tracks, indent=1))
    rejected = [t for t in out_tracks if not t['eligible']['v2_production']]
    recovery = {n: dict(recovered=sum(1 for t in rejected if t['eligible'][n]),
                        still_rejected=sum(1 for t in rejected if not t['eligible'][n]),
                        recovered_best_areas=sorted(t['evidence'][n]['width'] * t['evidence'][n]['height']
                                                    for t in rejected if n in t['evidence'])) for n in names[1:]}
    summary = dict(video=str(a.video), frames=frames, analyzed=analyzed, wall_seconds=round(time.monotonic() - started, 1),
                   tracks=len(out_tracks), v2_eligible_tracks=len(out_tracks) - len(rejected), v2_rejected_tracks=len(rejected),
                   categories=dict(Counter(t['category'] for t in rejected)),
                   rejection_reason_totals=dict(sum((Counter(t['rejection_reasons']) for t in rejected), Counter())),
                   crops_observed=sum(t['observed'] for t in out_tracks), recovery=recovery,
                   policies={n: c.value() for n, c in policies()})
    if a.compare_run:
        status = json.loads((a.compare_run/'status.json').read_text())
        rows = {r['track_id']: r for r in status['rows']}
        mismatches = []
        for t in out_tracks:
            r = rows.get(t['service_track_id'])
            submitted = r is not None and r['manufacturer']['reason'] != 'insufficient_quality'
            if (r is None or (r['first_frame'], r['last_frame'], r['raw_track_id']) != (t['first_frame'], t['last_frame'], t['raw_track_id'])
                    or submitted != bool(t['eligible']['v2_production'])):
                mismatches.append(t['service_track_id'])
        summary['cross_check'] = dict(console_tracks=len(rows), audit_tracks=len(out_tracks),
                                      missing_in_audit=sorted(set(rows) - {t['service_track_id'] for t in out_tracks}),
                                      mismatched=mismatches)
    (a.output/'summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k not in ('policies',)}, indent=1)[:4000])


if __name__ == '__main__':
    main()
