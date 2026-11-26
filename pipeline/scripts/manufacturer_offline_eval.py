"""Offline manufacturer evaluation through Aziz's runtime, parser and quality gate.

Reproduces the labeled still-image evaluation with the production manufacturer
model, prompt, preprocessing and parsing (vehicle/common/manufacturer). Every
CSV row is reported: missing/undecodable files are listed with an explicit
status and never silently dropped. Metrics:

* accuracy_answered = correct / answered (an accepted, supported make)
* coverage          = answered / evaluated
* accuracy_overall  = correct / evaluated  (Mohamed's original definition)

Optional controlled degradations (--scales, --blur) re-run the same labeled
images at lower resolution or with blur, to calibrate quality-gate thresholds
against measured accuracy. Outputs (predictions.csv, summary.json) stay local;
images are never copied. No network, provider or model download is used.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class VramSampler:
    """NVML per-process peak for runtime children (MIG-aware); device peak too."""
    def __init__(self, interval=0.05):
        self.interval, self.peak_process, self.peak_device, self.error = interval, {}, None, None
        self.stopped = threading.Event()
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nvml = pynvml
            self.handles = []
            for index in range(pynvml.nvmlDeviceGetCount()):
                parent = pynvml.nvmlDeviceGetHandleByIndex(index)
                try:
                    if pynvml.nvmlDeviceGetMigMode(parent)[0]:
                        for slot in range(pynvml.nvmlDeviceGetMaxMigDeviceCount(parent)):
                            try:self.handles.append(pynvml.nvmlDeviceGetMigDeviceHandleByIndex(parent, slot))
                            except pynvml.NVMLError:pass
                        continue
                except pynvml.NVMLError:
                    pass
                self.handles.append(parent)
            self.baseline = {p.pid for h in self.handles for p in pynvml.nvmlDeviceGetComputeRunningProcesses(h)}
        except Exception as error:
            self.nvml, self.error = None, type(error).__name__
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        while self.nvml and not self.stopped.wait(self.interval):
            for handle in self.handles:
                try:
                    for p in self.nvml.nvmlDeviceGetComputeRunningProcesses(handle):
                        if p.pid not in self.baseline and isinstance(p.usedGpuMemory, int):
                            self.peak_process[p.pid] = max(self.peak_process.get(p.pid, 0), p.usedGpuMemory / 2**20)
                    used = self.nvml.nvmlDeviceGetMemoryInfo(handle).used / 2**20
                    self.peak_device = max(self.peak_device or 0, used)
                except self.nvml.NVMLError:
                    pass

    def __enter__(self):
        self.thread.start();return self

    def __exit__(self, *exc):
        self.stopped.set();self.thread.join(timeout=2)

    def summary(self):
        values = list(self.peak_process.values())
        return dict(error=self.error, processes_observed=len(values),
                    peak_process_vram_mib=max(values) if values else None,
                    peak_device_used_mib=self.peak_device,
                    scope='NVML usedGpuMemory per new runtime PID on the visible device/MIG instance; sampled every 50 ms')


def variants(args):
    items = [('original', 1.0, 0.0)]
    items += [(f'scale_{s:g}', s, 0.0) for s in args.scales]
    items += [(f'blur_{b:g}', 1.0, b) for b in args.blur]
    return items


def degrade(image, scale, sigma):
    import cv2
    if scale != 1.0:
        h, w = image.shape[:2]
        image = cv2.resize(image, (max(1, round(w*scale)), max(1, round(h*scale))), interpolation=cv2.INTER_AREA)
    if sigma:
        image = cv2.GaussianBlur(image, (0, 0), sigma)
    return image


def metrics(records):
    evaluated = [r for r in records if r['status'] == 'evaluated']
    answered = [r for r in evaluated if r['outcome'] == 'accepted']
    correct = [r for r in answered if r['correct']]
    def ratio(a, b):return round(a/b, 4) if b else None
    return dict(evaluated=len(evaluated), answered=len(answered), correct=len(correct),
                accuracy_answered=ratio(len(correct), len(answered)), coverage=ratio(len(answered), len(evaluated)),
                accuracy_overall=ratio(len(correct), len(evaluated)),
                unknown=sum(r['outcome'] == 'unknown' for r in evaluated),
                malformed=sum(r['outcome'] == 'malformed' for r in evaluated),
                execution_failed=sum(r['outcome'] not in {'accepted', 'unknown', 'malformed'} for r in evaluated))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--csv', type=Path, required=True)
    parser.add_argument('--images', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--lock', type=Path, default=ROOT/'vehicle/common/manufacturer/assets.json')
    parser.add_argument('--runtime', choices=('persistent', 'legacy'), default='persistent')
    parser.add_argument('--timeout', type=float, default=30)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--scales', type=lambda v:[float(x) for x in v.split(',') if x], default=[])
    parser.add_argument('--blur', type=lambda v:[float(x) for x in v.split(',') if x], default=[])
    parser.add_argument('--reference-predictions', type=Path, help='Mohamed predictions.csv for label agreement')
    args = parser.parse_args()
    import cv2
    from vehicle.common.manufacturer.crop_quality import CropQualityAnalyzer, CropQualityConfig
    from vehicle.common.manufacturer.local import create_runtime
    from vehicle.common.manufacturer.policy import interpret_text
    from vehicle.common.manufacturer.taxonomy import normalize_manufacturer
    args.output.mkdir(parents=True, exist_ok=True)
    os.chmod(args.output, 0o700)
    with args.csv.open(newline='', encoding='utf-8-sig') as handle:
        rows = list(csv.DictReader(handle))
    if args.limit:
        rows = rows[:args.limit]
    analyzer = CropQualityAnalyzer(CropQualityConfig())
    records, missing, undecodable = [], [], []
    started = time.monotonic()
    with VramSampler() as sampler:
        load_started = time.monotonic()
        runtime = create_runtime(args.lock, mode=args.runtime, timeout=args.timeout)
        load_seconds = time.monotonic() - load_started
        try:
            for index, row in enumerate(rows, 1):
                image_id = row['image_id'].strip()
                expected = normalize_manufacturer(row['manufacturer'])
                base = dict(image_id=image_id, label=row['manufacturer'], expected=expected,
                            human_reviewed=row.get('human_reviewed', ''), view=row.get('view', ''))
                path = args.images/f'{image_id}.jpg'
                if not path.is_file():
                    missing.append(image_id)
                    records.append(dict(base, variant='original', status='missing_image'))
                    continue
                original = cv2.imread(str(path))
                if original is None:
                    undecodable.append(image_id)
                    records.append(dict(base, variant='original', status='decode_failed'))
                    continue
                for name, scale, sigma in variants(args):
                    image = degrade(original, scale, sigma)
                    quality = analyzer.analyze(image)  # no bbox/label: size, sharpness, exposure only
                    record = dict(base, variant=name, status='evaluated', width=image.shape[1], height=image.shape[0],
                                  gate_eligible=quality.eligible, gate_reasons=';'.join(quality.reasons),
                                  gate_score=round(quality.score, 4), sharpness=round(quality.sharpness_variance, 2))
                    call = time.monotonic()
                    if args.runtime == 'legacy':
                        try:
                            outcome, text = runtime.infer(image)
                        except TimeoutError:
                            outcome, text = 'timed_out', ''
                        except Exception:
                            outcome, text = 'failed', ''
                        seconds = time.monotonic() - call
                    else:
                        runtime.submit(f'{image_id}:{name}', image)
                        runtime.wait(args.timeout + 5)
                        (_, outcome, text, seconds), = runtime.collect()
                    parsed, make, reason = interpret_text(text) if outcome == 'completed' else (outcome, None, outcome)
                    record.update(outcome=parsed, predicted=make or 'unknown', reason=reason,
                                  correct=make == expected, seconds=round(seconds, 4), raw_output=text)
                    records.append(record)
                print(f'[{index}/{len(rows)}] {image_id}', flush=True)
        finally:
            if hasattr(runtime, 'close'):
                runtime.close()
    runtime_metrics = getattr(runtime, 'metrics', [])
    with (args.output/'predictions.csv').open('w', newline='', encoding='utf-8') as handle:
        fields = ['image_id', 'variant', 'status', 'label', 'expected', 'predicted', 'outcome', 'correct', 'reason',
                  'width', 'height', 'gate_eligible', 'gate_reasons', 'gate_score', 'sharpness', 'seconds',
                  'human_reviewed', 'view', 'raw_output']
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader();writer.writerows(records)
    by_variant = {}
    for name, _, _ in variants(args):
        group = [r for r in records if r['variant'] == name]
        eligible = [r for r in group if r.get('gate_eligible')]
        rejected = [r for r in group if r['status'] == 'evaluated' and not r.get('gate_eligible')]
        by_variant[name] = dict(all=metrics(group), gate_eligible=metrics(eligible), gate_rejected=metrics(rejected),
                                width=group[0].get('width') if group else None)
    seconds = sorted(r['seconds'] for r in records if r['status'] == 'evaluated')
    agreement = None
    if args.reference_predictions and args.reference_predictions.is_file():
        with args.reference_predictions.open(newline='', encoding='utf-8') as handle:
            reference = {r['image_id']: r['predicted'] for r in csv.DictReader(handle)}
        ours = {r['image_id']: r['predicted'] for r in records if r['variant'] == 'original' and r['status'] == 'evaluated'}
        common = sorted(set(reference) & set(ours))
        agreement = dict(compared=len(common), same_prediction=sum(reference[i] == ours[i] for i in common),
                         differing=[dict(image_id=i, reference=reference[i], ours=ours[i]) for i in common if reference[i] != ours[i]])
    summary = dict(
        csv_rows=len(rows), missing_images=missing, undecodable_images=undecodable,
        labels_human_reviewed=sorted({r['human_reviewed'] for r in records}),
        denominator_note='Missing/undecodable rows are listed, not evaluated; every evaluated row stays in the denominators.',
        runtime=args.runtime, model_load_seconds=round(load_seconds, 3),
        model_loads=len(getattr(runtime, 'starts', [])) or len(getattr(runtime, 'children', [])),
        latency_seconds=dict(calls=len(seconds), median=seconds[len(seconds)//2] if seconds else None,
                             p95=seconds[int(len(seconds)*0.95)] if seconds else None,
                             mean=round(statistics.fmean(seconds), 4) if seconds else None, max=seconds[-1] if seconds else None),
        runtime_process_vram_mib=[m.get('process_vram_mib') for m in runtime_metrics if isinstance(m, dict)][:1] +
                                 [m.get('process_vram_mib') for m in runtime_metrics if isinstance(m, dict)][-1:],
        vram=sampler.summary(), wall_seconds=round(time.monotonic() - started, 2),
        quality_config_sha256=CropQualityConfig().fingerprint(), variants=by_variant, reference_agreement=agreement)
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'variants'}, indent=1)[:3000])
    for name, item in by_variant.items():
        print(name, json.dumps(item['all']))


if __name__ == '__main__':
    main()
