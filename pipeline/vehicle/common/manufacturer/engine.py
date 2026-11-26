"""Engine-owned manufacturer crop selection, with durable at-most-once admission.

Two explicit selection policies:

* ``first_eligible_v1`` (legacy, rollback): the first clipped crop >=48 px per
  axis and <=8 MiB is submitted immediately. Evidence bytes are unchanged.
* ``quality_lifetime_v1``: every observed crop is scored by the CPU-only
  crop-quality gate. Ineligible crops are counted by reason and never reach the
  model. The highest-scoring eligible crop is retained (one bounded copy per
  active lifetime) and submitted exactly once, at closure or after a bounded
  number of observations for long-lived tracks. Selection never depends on any
  inference result, so asynchronous execution cannot change the chosen crop.

The SQLite journal is a single host execution fence, not a distributed queue:
the reservation row is committed before dispatch, and an uncertain start stays
uncertain. Pixel bytes never enter it or Redis.
"""
from dataclasses import dataclass, field
from hashlib import sha256
from decimal import Decimal
import json
from pathlib import Path
import sqlite3
import time

from contracts.db1.reference import canonical_json_bytes
from ..db1_rows.core import require
from ..redis_scalar.codec import pack_life, pack_receipt
from .crop_quality import CropQualityAnalyzer, CropQualityConfig, REASONS

OUTCOMES = {'completed', 'timed_out', 'failed', 'execution_uncertain', 'disabled', 'unknown', 'cancelled'}
POLICIES = ('first_eligible_v1', 'quality_lifetime_v1')
SUBMISSIONS = ('submitted', 'none_eligible', 'not_observed', 'stopped_before_submission')
EXTRA_REASONS = ('crop_bytes', 'retention_budget')
MAX_CROP_BYTES = 8 * 1024 * 1024
BINDING_V1 = {'vehicle', 'receipt', 'bbox', 'shape', 'crop_sha256', 'frame_sha256', 'provenance'}


def hashed(value):
    return sha256(canonical_json_bytes(value)).hexdigest()


def load_canonical(text):
    """Decode canonical evidence exactly. Canonical JSON writes integers >= 10 in
    exponent form (22 -> 2.2e1); plain json.loads would turn them into floats."""
    from decimal import Decimal as D
    def number(token):
        value = D(token)
        # Evidence numbers originate as Python floats, so float() reproduces the
        # same shortest repr and therefore identical canonical bytes.
        return int(value) if value == int(value) else float(token)
    return json.loads(text, parse_float=number)


@dataclass(frozen=True)
class SelectionConfig:
    policy: str = 'quality_lifetime_v1'
    quality: CropQualityConfig = field(default_factory=CropQualityConfig)
    max_wait_observations: int = 48
    max_retained_bytes: int = 64 * 1024 * 1024

    def __post_init__(self):
        require(self.policy in POLICIES, 'INVALID_CONFIGURATION', 'unknown manufacturer selection policy')
        require(isinstance(self.quality, CropQualityConfig), 'INVALID_CONFIGURATION', 'versioned quality config required')
        require(type(self.max_wait_observations) is int and 1 <= self.max_wait_observations <= 100000
                and type(self.max_retained_bytes) is int and MAX_CROP_BYTES <= self.max_retained_bytes <= 1024**3,
                'INVALID_CAPACITY', 'bounded selection window and retained crop bytes required')

    def value(self):
        return dict(policy=self.policy, quality=self.quality.value(),
                    max_wait_observations=self.max_wait_observations, max_retained_bytes=self.max_retained_bytes)

    def fingerprint(self):
        return hashed(self.value())


LEGACY_SELECTION = SelectionConfig(policy='first_eligible_v1')


def vehicle_binding(life):
    c = life.context
    require(not c.database_keys_resolved and c.environment == 'development',
            'RESOLVED_CONTEXT', 'manufacturer review requires explicit unresolved context')
    return dict(context=dict(org_id=str(c.org_id), camera_id=c.camera.camera_id,
        camera_timezone=c.camera.timezone, valid_from=c.camera.valid_from.isoformat(),
        valid_to=c.camera.valid_to.isoformat() if c.camera.valid_to else None,
        run_id=c.run_id, source_kind=c.source_kind, started_at=c.started_at.isoformat(),
        service_version=c.service_version, detector=dict(model_name=c.resolution.detector.model_name,
            version=c.resolution.detector.version, task='detect'), resolution_provenance=c.resolution.provenance),
        service_track_id=life.service_track_id, tracker_epoch=life.tracker_epoch,
        raw_track_id=life.raw_track_id, lifetime_sequence=life.lifetime_sequence)


def _count(value):
    return type(value) is int and 0 <= value < 2**53


def validate_selection(s, attempted):
    import re
    require(type(s) is dict and set(s) == {'policy', 'config_sha256', 'observed', 'eligible', 'rejected', 'selected', 'submission'},
            'INVALID_MANUFACTURER', 'closed selection summary required')
    require(s['policy'] == 'quality_lifetime_v1' and type(s['config_sha256']) is str
            and re.fullmatch('[a-f0-9]{64}', s['config_sha256']) is not None,
            'INVALID_MANUFACTURER', 'versioned selection policy required')
    require(_count(s['observed']) and _count(s['eligible']) and s['eligible'] <= s['observed'],
            'INVALID_MANUFACTURER', 'bounded observation counts required')
    require(type(s['rejected']) is dict and all(k in REASONS + EXTRA_REASONS and _count(v) and v > 0
                                                for k, v in s['rejected'].items()),
            'INVALID_MANUFACTURER', 'explicit rejection reasons required')
    require(s['submission'] in SUBMISSIONS and (s['submission'] == 'submitted') == attempted,
            'INVALID_MANUFACTURER', 'submission status differs from crop binding')
    require((s['selected'] is not None) == attempted, 'INVALID_MANUFACTURER', 'selected quality only for a submitted crop')
    if s['selected'] is not None:
        q = s['selected']
        require(type(q) is dict and q.get('eligible') is True and q.get('reasons') == [],
                'INVALID_MANUFACTURER', 'only an eligible crop can be submitted')
    if s['submission'] == 'none_eligible':
        require(s['observed'] > 0 and s['eligible'] == 0, 'INVALID_MANUFACTURER', 'none_eligible requires rejected observations')
    if s['submission'] == 'not_observed':
        require(s['observed'] == 0, 'INVALID_MANUFACTURER', 'not_observed requires no observations')


def validate_evidence(value):
    require(type(value) is dict and set(value) == {'binding', 'request_id', 'outcome', 'text', 'seconds'},
            'INVALID_MANUFACTURER', 'closed manufacturer evidence required')
    b = value['binding']
    require(type(b) is dict and set(b) in (BINDING_V1, BINDING_V1 | {'selection'}),
            'INVALID_MANUFACTURER', 'closed crop binding required')
    require(value['request_id'] == hashed(b) and value['outcome'] in OUTCOMES,
            'MANUFACTURER_BINDING_MISMATCH', 'request digest/outcome differs')
    require(value['outcome'] != 'cancelled' or 'selection' in b,
            'INVALID_MANUFACTURER', 'cancelled outcome requires a versioned selection')
    require(type(value['text']) is str and len(value['text'].encode()) <= 2048
            and (value['outcome'] == 'completed' or value['text'] == ''),
            'INVALID_MANUFACTURER', 'bounded model text only on completed execution')
    import math
    require(type(value['seconds']) in (int, float, Decimal) and math.isfinite(value['seconds'])
            and 0 <= value['seconds'] <= 65, 'INVALID_MANUFACTURER', 'bounded latency required')
    if 'selection' in b:
        validate_selection(b['selection'], b['receipt'] is not None)
    if b['receipt'] is not None:
        require(type(b['bbox']) is list and len(b['bbox']) == 4
                and all(type(x) in (int,float,Decimal) and math.isfinite(x) and x >= 0 and x == int(x) for x in b['bbox'])
                and b['bbox'][2] > b['bbox'][0] and b['bbox'][3] > b['bbox'][1],
                'INVALID_MANUFACTURER', 'original crop geometry required')
        require(b['shape'] == [b['bbox'][3]-b['bbox'][1], b['bbox'][2]-b['bbox'][0], 3],
                'INVALID_MANUFACTURER', 'crop shape differs from bbox')
        import re
        require(all(type(b[k]) is str and re.fullmatch('[a-f0-9]{64}', b[k]) for k in ('crop_sha256','frame_sha256','provenance')),
                'INVALID_MANUFACTURER', 'artifact/crop/frame fingerprints required')
    else:
        require(value['outcome'] in {'disabled','unknown'} and b['bbox'] == [] and b['shape'] == []
                and b['crop_sha256'] is None and b['frame_sha256'] is None,
                'INVALID_MANUFACTURER', 'nonattempt cannot claim crop execution')


@dataclass
class _Track:
    life: object
    observed: int = 0
    eligible: int = 0
    rejected: dict = field(default_factory=dict)
    first_eligible: int | None = None
    best: dict | None = None
    stopped: bool = False


class ManufacturerLedger:
    def __init__(self, *, enabled=False, runtime=None, journal=None, max_lifetimes=16, stop=None,
                 selection=LEGACY_SELECTION, settle_timeout=None):
        require(type(enabled) is bool and type(max_lifetimes) is int and 1 <= max_lifetimes <= 65536,
                'INVALID_CAPACITY', 'explicit bounded manufacturer capability')
        require(not enabled or (runtime is not None and journal is not None),
                'MISSING_MANUFACTURER_RUNTIME', 'opt-in requires local runtime and durable journal')
        require(isinstance(selection, SelectionConfig), 'INVALID_CONFIGURATION', 'explicit selection policy required')
        self.enabled, self.runtime = enabled, runtime
        self.stop = stop
        self.selection = selection
        self.analyzer = CropQualityAnalyzer(selection.quality)
        self.asynchronous = getattr(runtime, 'mode', 'legacy') == 'persistent'
        self.settle_timeout = settle_timeout
        self.max_lifetimes, self.results, self.calls = max_lifetimes, {}, 0
        self.tracks, self.waiting, self.requests, self.selections = {}, {}, {}, {}
        self.deferred = {}  # closed lifetimes whose retained best awaits a queue slot (FIFO)
        from collections import OrderedDict
        self.terminals = OrderedDict()
        self.retained_bytes = 0
        self.execute = self._execute
        self.submit_hook = None
        self.quality_seconds = 0.0
        # Run-level scalar counters (no pixels/text): crop observations and outcomes.
        self.stats = dict(observed=0, eligible=0, rejected={}, submitted=0, outcomes={})
        self.db = None
        if enabled:
            path = Path(journal)
            require(path.parent.is_dir(), 'MISSING_JOURNAL', 'explicit existing journal directory required')
            self.db = sqlite3.connect(path)
            self.db.execute('PRAGMA synchronous=FULL')
            self.db.execute('CREATE TABLE IF NOT EXISTS attempts (vehicle TEXT PRIMARY KEY, binding TEXT NOT NULL, result TEXT)')
            self.db.commit()

    # ---- observation -------------------------------------------------------
    def observe(self, life, receipt, bbox, crop, frame_sha256, *, label=None, confidence=None, frame_shape=None):
        if not self.enabled or (self.stop is not None and self.stop.is_set()):
            return
        self.poll()
        key = hashed(vehicle_binding(life))
        if key in self.results or key in self.waiting:
            return
        previous = self.db.execute('SELECT binding,result FROM attempts WHERE vehicle=?', (key,)).fetchone()
        if previous is not None:
            b = load_canonical(previous[0])
            self.results[key] = load_canonical(previous[1]) if previous[1] else dict(
                binding=b, request_id=hashed(b), outcome='execution_uncertain', text='', seconds=0.0)
            return
        if self.selection.policy == 'first_eligible_v1':
            # Deterministic selection: first observed clipped vehicle bbox >=48px
            # on each axis and <=8MiB. Later crops cannot replace or repeat it.
            if min(crop.shape[:2]) < 48 or crop.nbytes > MAX_CROP_BYTES:
                return
            if self.asynchronous and not self.runtime.can_accept():
                return  # Backpressure: never block the frame loop; a later eligible crop is used.
            b = dict(vehicle=vehicle_binding(life), receipt=pack_receipt(receipt), bbox=list(bbox),
                     shape=list(crop.shape), crop_sha256=sha256(crop.tobytes()).hexdigest(),
                     frame_sha256=frame_sha256, provenance=self.runtime.provenance)
            self.selections[key] = (receipt, label, confidence)
            self._dispatch(key, life, b, crop, receipt)
            return
        track = self.tracks.get(key)
        if track is None:
            track = self.tracks[key] = _Track(life)
        track.observed += 1
        self.stats['observed'] += 1
        started = time.perf_counter()
        quality = self.analyzer.analyze(crop, bbox=bbox, frame_shape=frame_shape, label=label)
        self.quality_seconds += time.perf_counter() - started
        if not quality.eligible:
            for reason in quality.reasons:
                track.rejected[reason] = track.rejected.get(reason, 0) + 1
                self.stats['rejected'][reason] = self.stats['rejected'].get(reason, 0) + 1
        elif crop.nbytes > MAX_CROP_BYTES:
            track.rejected['crop_bytes'] = track.rejected.get('crop_bytes', 0) + 1
        else:
            track.eligible += 1
            self.stats['eligible'] += 1
            if track.first_eligible is None:
                track.first_eligible = track.observed
            # Strictly higher score replaces; ties keep the earlier observation.
            if track.best is None or quality.score > track.best['quality'].score:
                old = track.best['crop'].nbytes if track.best is not None else 0
                if self.retained_bytes - old + crop.nbytes <= self.selection.max_retained_bytes:
                    self.retained_bytes += crop.nbytes - old
                    track.best = dict(crop=crop.copy(), receipt=receipt, bbox=list(bbox), frame_sha256=frame_sha256,
                                      quality=quality, label=label, confidence=confidence)
                else:
                    track.rejected['retention_budget'] = track.rejected.get('retention_budget', 0) + 1
        if (track.best is not None and track.first_eligible is not None
                and track.observed - track.first_eligible >= self.selection.max_wait_observations):
            self._submit_best(key, track)

    def _selection_summary(self, track, submission, selected=None):
        return dict(policy=self.selection.policy, config_sha256=self.selection.fingerprint(),
                    observed=track.observed if track else 0, eligible=track.eligible if track else 0,
                    rejected={k: track.rejected[k] for k in sorted(track.rejected)} if track else {},
                    selected=selected, submission=submission)

    def _submit_best(self, key, track):
        if track.best is None or key in self.waiting or key in self.results:
            return False
        if self.asynchronous and not self.runtime.can_accept():
            return False  # Backpressure: keep the retained best; retry on poll/closure.
        best = track.best
        crop = best['crop']
        b = dict(vehicle=vehicle_binding(track.life), receipt=pack_receipt(best['receipt']), bbox=list(best['bbox']),
                 shape=list(crop.shape), crop_sha256=sha256(crop.tobytes()).hexdigest(),
                 frame_sha256=best['frame_sha256'], provenance=self.runtime.provenance,
                 selection=self._selection_summary(track, 'submitted', best['quality'].record()))
        self.selections[key] = (best['receipt'], best['label'], best['confidence'])
        track.best = None
        self.retained_bytes -= crop.nbytes
        self._dispatch(key, track.life, b, crop, best['receipt'])
        return True

    def _dispatch(self, key, life, b, crop, receipt):
        require(len(self.waiting) < self.max_lifetimes, 'CAPACITY_EXCEEDED', 'manufacturer lifetime limit')
        # Exclusive transaction reserves the unique lifetime BEFORE model entry.
        # A second Engine/host sees the reservation and can never invoke it again.
        self.db.execute('BEGIN IMMEDIATE')
        try:
            count = self.db.execute('SELECT count(*) FROM attempts').fetchone()[0]
            require(count < self.max_lifetimes, 'CAPACITY_EXCEEDED', 'durable manufacturer lifetime bound')
            self.db.execute('INSERT INTO attempts VALUES (?,?,NULL)', (key, canonical_json_bytes(b).decode()))
            self.db.commit()
        except BaseException:
            self.db.rollback()
            raise
        self.calls += 1
        self.stats['submitted'] += 1
        if self.submit_hook is not None:
            self.submit_hook(life, receipt, crop)
        if self.asynchronous:
            request_id = hashed(b)
            self.requests[request_id] = key
            self.waiting[key] = b
            self.runtime.submit(request_id, crop)
            return
        started = time.monotonic()
        try:
            outcome, text = self.execute(lambda: self.runtime.infer(crop))
        except TimeoutError:
            outcome, text = 'timed_out', ''
        except Exception:
            outcome, text = 'failed', ''
        self._record(key, b, outcome, text, time.monotonic() - started)

    def _record(self, key, b, outcome, text, seconds):
        result = dict(binding=b, request_id=hashed(b), outcome=outcome, text=text if outcome == 'completed' else '',
                      seconds=min(65.0, seconds))
        validate_evidence(result)
        self.db.execute('UPDATE attempts SET result=? WHERE vehicle=?', (canonical_json_bytes(result).decode(), key))
        self.db.commit()
        self.results[key] = result
        self.stats['outcomes'][outcome] = self.stats['outcomes'].get(outcome, 0) + 1

    # ---- asynchronous completion --------------------------------------------
    def poll(self):
        if not self.enabled or not self.asynchronous:
            return 0
        done = 0
        for request_id, outcome, text, seconds in self.runtime.collect():
            key = self.requests.pop(request_id)
            b = self.waiting.pop(key)
            self._record(key, b, outcome, text, seconds)
            done += 1
        for key in tuple(self.deferred):
            if self.stop is not None and self.stop.is_set():
                break
            if not self._submit_best(key, self.deferred[key]):
                break  # Queue still full; keep FIFO order.
            del self.deferred[key]
        return done

    def waiting_for(self, life):
        """True while a closed lifetime's single manufacturer request is unfinished."""
        if not self.enabled:
            return False
        self.poll()
        key = hashed(vehicle_binding(life))
        return key in self.waiting or key in self.deferred

    def close_lifetime(self, life):
        """At closure, submit the lifetime's retained best crop (once)."""
        if not self.enabled or self.selection.policy == 'first_eligible_v1':
            return
        key = hashed(vehicle_binding(life))
        track = self.tracks.get(key)
        if track is None or track.best is None or key in self.deferred:
            return
        if self.stop is not None and self.stop.is_set():
            self._drop_best(track)
            return
        if not self._submit_best(key, track) and self.asynchronous:
            # Queue full: never block the frame loop. The closure stays unfrozen
            # (waiting_for) and poll() submits it once a slot frees, in FIFO order.
            self.deferred[key] = track

    def _drop_best(self, track):
        if track.best is not None:
            self.retained_bytes -= track.best['crop'].nbytes
            track.best = None
        track.stopped = True

    def selected_class(self, life):
        """(receipt, label, confidence) of the submitted crop, if any."""
        if not self.enabled:
            return None
        return self.selections.get(hashed(vehicle_binding(life)))

    def settle(self):
        """End of input/stop: finish or cancel outstanding work without fabricating results."""
        if not self.enabled or not self.asynchronous:
            return
        if self.stop is None or not self.stop.is_set():
            timeout = self.settle_timeout or (self.runtime.timeout + 1) * (len(self.waiting) + len(self.deferred) + 1)
            deadline = time.monotonic() + timeout
            while (self.waiting or self.deferred) and time.monotonic() < deadline:
                if self.stop is not None and self.stop.is_set():
                    break
                self.poll()
                self.runtime.wait(min(0.05, max(0.0, deadline - time.monotonic())))
        if self.waiting or self.deferred:
            # Stop or deadline: cancel without fabricating; unsubmitted bests are stopped.
            self.runtime.cancel()
            for track in self.deferred.values():
                self._drop_best(track)
            self.deferred.clear()
            self.runtime.wait(self.runtime.timeout + 10)
        self.poll()

    def cancel(self):
        if self.enabled and self.asynchronous:
            self.runtime.cancel()
            self.runtime.wait(self.runtime.timeout + 10)
            self.poll()
        self.deferred.clear()
        for track in self.tracks.values():
            self._drop_best(track)

    @staticmethod
    def _execute(call):
        from ..engine_plate import native_access
        with native_access():
            return call()

    # ---- terminal ------------------------------------------------------------
    def terminal(self, life):
        key = hashed(vehicle_binding(life))
        if key in self.terminals:
            return load_canonical(self.terminals[key])
        require(key not in self.waiting, 'MANUFACTURER_WAITING', 'manufacturer request unfinished at closure')
        track = self.tracks.pop(key, None)
        if track is not None and track.best is not None:
            self._drop_best(track)
        self.selections.pop(key, None)
        if key in self.results:
            # Prune active state: a closed lifetime is never observed again. The
            # durable journal row still fences any re-execution.
            result = self.results.pop(key)
        else:
            b = dict(vehicle=vehicle_binding(life), receipt=None, bbox=[], shape=[],
                     crop_sha256=None, frame_sha256=None,
                     provenance=self.runtime.provenance if self.enabled else None)
            if self.enabled and self.selection.policy != 'first_eligible_v1':
                submission = ('not_observed' if track is None or track.observed == 0 else
                              'stopped_before_submission' if track.stopped or track.eligible else 'none_eligible')
                b['selection'] = self._selection_summary(track, submission)
            result = dict(binding=b, request_id=hashed(b), outcome='unknown' if self.enabled else 'disabled', text='', seconds=0.0)
        # Bounded idempotence for repeated terminal reads of recent closures.
        self.terminals[key] = canonical_json_bytes(result).decode()
        while len(self.terminals) > 1024:
            self.terminals.popitem(last=False)
        return load_canonical(self.terminals[key])

    def close(self):
        if self.db is not None:
            self.db.close()
            self.db = None
