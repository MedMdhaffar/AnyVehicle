"""Producer-side v3 evaluation. Observes policy calls; never selects crops/results.

Manufacturer events precede service-ID allocation on a new track (and on cuts).
A small durable pending journal bridges that boundary without guessing an ID.
All public evidence is append-only and independent of Redis/catalogue delivery.
"""
from dataclasses import asdict
from copy import copy
from hashlib import sha256
import ctypes
import json
import threading
import time

from .evaluation import stable_id
from .evaluation_timing import measured

VERSION = 'recognition-evaluation-v1'
FINAL_OUTCOMES = ('prediction_produced', 'unknown_manufacturer', 'no_eligible_crop',
                  'inference_failure', 'skipped_unscheduled', 'stopped_incomplete', 'disabled')


def timing(start, end):
    return dict(start_ns=str(start), completion_ns=str(end), duration_ms=(end-start)/1e6)


def outcome(error):
    if isinstance(error, TimeoutError):return 'timed_out'
    if type(error).__name__ in ('InputStopped', 'KeyboardInterrupt'):return 'cancelled'
    return 'failed'


def manufacturer_sync(service):
    """Synchronize the backend's actual CUDA context, including llama.cpp work."""
    backend = service._backend
    if hasattr(backend, '_torch'):
        return lambda:backend._torch.cuda.synchronize(service.config.device)
    driver = ctypes.CDLL('libcuda.so.1')
    driver.cuCtxSynchronize.argtypes = []
    driver.cuCtxSynchronize.restype = ctypes.c_int
    def synchronize():
        status = driver.cuCtxSynchronize()
        if status:raise RuntimeError(f'manufacturer CUDA synchronization failed: {status}')
    return synchronize


class ManufacturerEvaluation:
    def __init__(self, engine, *, synchronize=None):
        self.engine, self.recorder = engine, engine.evaluation
        self.store = self.recorder.store
        self.source_id = self.store.metadata()['source_id']
        self.states, self.bound, self.finalized, self.attempts = {}, {}, set(), {}
        self.quality_times, self.rejected, self.retained_quality = {}, set(), {}
        self.local = threading.local()
        self.model = engine.models.get('attributes', {}).get('model_name')
        self.sync = synchronize
        recognizer = engine.vehicle.recognizer
        self.quality_config = asdict(recognizer.config.quality_config()) if recognizer else None
        self.quality_version = 'manufacturer-quality-sha256:'+sha256(
            json.dumps(self.quality_config,sort_keys=True).encode()).hexdigest()
        with self.store.connect() as db:
            db.execute('''CREATE TABLE IF NOT EXISTS manufacturer_pending
                (id TEXT PRIMARY KEY, track_key TEXT NOT NULL, payload TEXT NOT NULL, projected INTEGER NOT NULL DEFAULT 0)''')
        if recognizer is not None:self.install(recognizer)

    def _receipt(self, frame):
        receipt = self.engine.receipts.get(frame)
        return dict(frame_index=frame, video_timestamp=receipt.media_pts if receipt else None)

    def _event(self, state, category, key, **fields):
        event = dict(category=category, key=key, raw_track_id=state.track_id,
                     track_key=state.key, **fields)
        if state.key in self.bound:
            self._persist(event, self.bound[state.key])
            return
        identity = stable_id(self.recorder.run, VERSION, state.key, category, key)
        payload = json.dumps(event, sort_keys=True, allow_nan=False)
        with self.store.connect() as db:
            prior = db.execute('SELECT payload FROM manufacturer_pending WHERE id=?', (identity,)).fetchone()
            if prior and prior[0] != payload:raise ValueError('immutable_pending_evidence_changed')
            db.execute('INSERT OR IGNORE INTO manufacturer_pending(id,track_key,payload) VALUES (?,?,?)',
                       (identity, state.key, payload))

    def _persist(self, event, sid):
        value = {k:v for k,v in event.items() if k not in ('category', 'key')}
        value.update(id=stable_id(self.recorder.run, VERSION, event['track_key'], event['category'], event['key']),
                     run_id=self.recorder.run, track_id=sid, lifetime_id=f'{self.recorder.run}:{sid}')
        if event['category']=='call':
            self.store.call(value)
        else:
            defaults = dict(source_id=self.source_id,camera_id=self.engine.context.camera.camera_id,
                domain='manufacturer',kind='observation',predicted_label=None,image=None,original_frame=None,
                frame_index=None,video_timestamp=None,model_version=self.model,quality_policy_version=self.quality_version,
                quality_time_ms=None,inference_time_ms=None)
            self.store.record(dict(defaults, **value))

    def bind(self, life, state):
        if not state.key:
            state = copy(state)
            state.key, state.track_id = f'disabled:{life.service_track_id}', life.raw_track_id
        self.states[state.key] = state
        if state.key in self.bound:
            if self.bound[state.key]!=life.service_track_id:
                raise ValueError('manufacturer_lifetime_binding_changed')
            return state
        sid = life.service_track_id
        self.bound[state.key] = sid
        self._event(state, 'record', 'lifetime', kind='lifetime', outcome='processing',
                    **self._receipt(life.first_receipt.frame_index), attributes_enabled=self.engine.attribute_service is not None,
                    quality_config=self.quality_config)
        with self.store.connect() as db:
            pending = db.execute('SELECT id,payload FROM manufacturer_pending WHERE track_key=? AND projected=0 ORDER BY rowid',
                                 (state.key,)).fetchall()
        for identity, payload in pending:
            self._persist(json.loads(payload), sid)
            with self.store.connect() as db:
                db.execute('UPDATE manufacturer_pending SET projected=1 WHERE id=?', (identity,))
        return state

    def ensure_complete(self):
        with self.store.connect() as db:
            unbound=db.execute('SELECT count(*) FROM manufacturer_pending WHERE projected=0').fetchone()[0]
        if set(self.bound)!=self.finalized or unbound:
            raise ValueError('manufacturer_evaluation_incomplete')

    def _call(self, state, key, stage, scope, event, *, model_call=False, **fields):
        event = dict(event)
        event.setdefault('model_call',model_call)
        self._event(state, 'call', key, stage=stage, scope=scope, model_version=self.model,
                    **event, **fields)

    def install(self, recognizer):
        original_observe, original_skip = recognizer.observe, recognizer.resource_skip
        def observing(state, *args, **kwargs):
            self.states[state.key] = state
            previous = getattr(self.local, 'state', None)
            self.local.state = state
            try:return original_observe(state, *args, **kwargs)
            finally:self.local.state = previous
        def skipping(state, *args, **kwargs):
            self.states[state.key] = state
            return original_skip(state, *args, **kwargs)
        recognizer.observe, recognizer.resource_skip = observing, skipping
        select = recognizer.selector.select
        def selecting(candidates, **kwargs):
            state = self.local.state;frame = candidates[0].frame_index
            try:result, event = measured(lambda:select(candidates, **kwargs))
            except BaseException as error:
                event = dict(getattr(error,'evaluation_timing',{}),outcome=outcome(error))
                self._call(state,'quality-'+str(frame),'manufacturer_quality',
                    'CPU quality gate and single-candidate selection; excludes evidence I/O.',event,**self._receipt(frame))
                self._event(state,'record','quality-'+str(frame),kind='quality',outcome='quality_failure',
                    quality_time_ms=event.get('duration_ms'),error=type(error).__name__,**self._receipt(frame))
                raise
            self.quality_times[state.key,frame] = event
            self._call(state, 'quality-'+str(frame), 'manufacturer_quality',
                       'CPU quality gate and single-candidate selection; excludes evidence I/O.',
                       dict(event,outcome='completed'), **self._receipt(frame))
            return result
        recognizer.selector.select = selecting
        attempt = recognizer.attempt
        def attempting(state):
            if state.candidate is None or len(state.attempts)>=recognizer.config.max_attempts:
                return attempt(state)
            index = len(state.attempts)+1;frame = state.candidate.frame_index
            context = dict(state=state, ordinal=index, frame=frame, native=None, preprocessing=None,
                           queue=None, image=None, predict_entered=False)
            selected_quality=self.retained_quality.get(state.key,{})
            context['quality_time_ms']=(selected_quality.get('timing') or {}).get('duration_ms') if selected_quality.get('frame')==frame else None
            previous = getattr(self.local, 'attempt', None)
            self.local.attempt = context
            start = time.perf_counter_ns()
            try:return attempt(state)
            finally:
                end = time.perf_counter_ns()
                self.local.attempt = previous
                if len(state.attempts)>=index:
                    record = state.attempts[index-1]
                    success = 'result' in record
                    status = ('completed' if success else 'cancelled' if record.get('error')=='InputStopped'
                              else 'timed_out' if record.get('error')=='TimeoutError' else 'failed')
                    total = timing(start,end)
                    self.attempts[state.key,index] = dict(context, total=total)
                    self._call(state, f'attempt-{index}', 'manufacturer_attempt',
                        'Total scheduled attempt: evidence I/O, queue, preprocessing, native inference and parsing.',
                        dict(total,outcome=status), attempt_ordinal=index, **self._receipt(frame))
                    native = context['native'] or {}
                    self._event(state, 'record', f'attempt-{index}', kind='attempt', outcome=status,
                        attempt_ordinal=index, model_attempted=context['predict_entered'],
                        native_call_entered=bool(native.get('model_call')), image=context['image'],
                        predicted_label=record.get('result',{}).get('manufacturer'), result=record.get('result'),
                        error=record.get('error'), raw_output=record.get('raw_output'),
                        quality=record.get('source_quality'),quality_time_ms=context['quality_time_ms'],
                        quality_scope='Source crop before any model-input resizing.',
                        inference_time_ms=native.get('duration_ms'), inference_start_ns=native.get('start_ns'),
                        inference_completion_ns=native.get('completion_ns'), attempt_time_ms=total['duration_ms'],
                        timing_scope='native_generation_cuda_synchronized',
                        preprocessing_time_ms=(context['preprocessing'] or {}).get('duration_ms'),
                        queue_wait_ms=(context['queue'] or {}).get('duration_ms'),
                        input_scope='Exact PNG input to native manufacturer backend.' if native.get('model_call') else 'Scheduled attempt; native inference was not entered.',
                        **self._receipt(frame))
        recognizer.attempt = attempting
        service = self.engine.attribute_service
        if service is not None:self.install_service(service)

    def install_service(self, service):
        predict, generate = service.predict, service._backend.generate
        if self.sync is None:self.sync = manufacturer_sync(service)
        def predicting(*args, **kwargs):
            ctx = self.local.attempt;state = ctx['state'];index = ctx['ordinal'];frame = ctx['frame']
            queued = time.perf_counter_ns()
            with self.engine.native_lane, service._lock:
                acquired = time.perf_counter_ns();queue = timing(queued,acquired);ctx['queue'] = queue
                self._call(state, f'queue-{index}', 'manufacturer_queue', 'Wait for the shared native lane and manufacturer service lock.',
                           dict(queue,outcome='completed'), attempt_ordinal=index, **self._receipt(frame))
                if self.engine.stop is not None and self.engine.stop.is_set():
                    from .real_video.rtsp import InputStopped
                    raise InputStopped('controlled_stop')
                ctx['predict_entered'] = True
                self.engine.calls['attributes'] = self.engine.calls.get('attributes',0)+1
                ctx['preprocess_start'] = time.perf_counter_ns()
                error = None
                try:return predict(*args, **kwargs)
                except BaseException as exc:
                    error = exc;raise
                finally:
                    if ctx['preprocessing'] is None:
                        event = timing(ctx['preprocess_start'],time.perf_counter_ns())
                        event['outcome'] = outcome(error) if error else 'completed'
                        ctx['preprocessing'] = event
                        self._call(state, f'preprocess-{index}', 'manufacturer_preprocessing',
                            'Validation, BGR to RGB, PNG encoding and prompt preparation before native generation.',
                            event, attempt_ordinal=index, **self._receipt(frame))
        def generating(png, prompt, **kwargs):
            ctx = self.local.attempt;state = ctx['state'];index = ctx['ordinal'];frame = ctx['frame']
            pre = dict(timing(ctx['preprocess_start'],time.perf_counter_ns()),outcome='completed')
            ctx['preprocessing'] = pre
            self._call(state, f'preprocess-{index}', 'manufacturer_preprocessing',
                'Validation, BGR to RGB, PNG encoding and prompt preparation before native generation.',
                pre, attempt_ordinal=index, **self._receipt(frame))
            # Capture the actual transport pixels, including any configured resizing.
            import cv2
            import numpy as np
            image = cv2.imdecode(np.frombuffer(png,np.uint8),cv2.IMREAD_COLOR)
            if image is None:raise ValueError('invalid_native_input_png')
            ctx['image'] = self.recorder.image(image)
            event = dict(outcome='failed',duration_ms=None,start_ns=None,completion_ns=None,model_call=False)
            def invoke():
                event['model_call'] = True
                return generate(png,prompt,**kwargs)
            try:
                result, data = measured(invoke,synchronize=self.sync)
                event.update(data,outcome='completed')
                return result
            except BaseException as error:
                event.update(getattr(error,'evaluation_timing',{}),outcome=outcome(error))
                raise
            finally:
                ctx['native'] = event
                self._call(state, f'native-{index}', 'manufacturer',
                    'Native backend.generate including vision encoding and token generation; CUDA synchronized before and after.',
                    event, attempt_ordinal=index, **self._receipt(frame), image=ctx['image'],
                    transport_sha256=sha256(png).hexdigest(), input_sha256=sha256(image.tobytes()).hexdigest(),
                    prompt_sha256=sha256(prompt.encode()).hexdigest(), generation_parameters=kwargs)
        service.predict, service._backend.generate = predicting, generating

    def decision(self, record, image):
        if record.get('event')=='crop_submitted':
            ctx = getattr(self.local,'attempt',None)
            if ctx is not None and image is not None:ctx['image'] = self.recorder.image(image)
            return
        if record.get('event')!='crop_decision':return
        state = self.states[record['track_key']];frame = record['frame_index']
        quality = {k:record[k] for k in ('eligible','score','reasons','original_width','original_height','original_area',
            'sharpness_variance','dark_fraction','bright_fraction','border_edges','aspect_ratio') if k in record}
        reference = None
        reason = tuple(record.get('reasons',[]))
        if record.get('eligible') is False and (state.key,reason) not in self.rejected and image is not None:
            reference = self.recorder.image(image);self.rejected.add((state.key,reason))
        gate = self.quality_times.pop((state.key,frame),{})
        if record.get('retained'):self.retained_quality[state.key]=dict(frame=frame,timing=gate)
        self._event(state, 'record', 'quality-'+str(frame), kind='quality', quality=quality,
            quality_time_ms=gate.get('duration_ms'), image=reference,
            outcome='quality_rejected' if record.get('eligible') is False else 'eligible' if record.get('eligible') is True else 'skipped_unscheduled',
            scheduling_reason=record.get('decision_reason'), bbox=record['bbox'], retained=record.get('retained'),
            input_scope='Observed crop quality and scheduling decision; not a model prediction.', **self._receipt(frame))

    def finalize(self, life, state, *, incomplete=False):
        state = self.bind(life,state)
        if state.key in self.finalized:return
        successes = [(n,a) for n,a in enumerate(state.attempts,1) if 'result' in a]
        chosen = successes[-1] if successes else None
        label = chosen[1]['result']['manufacturer'] if chosen else None
        if label and label!='unknown':final = 'prediction_produced'
        elif chosen:final = 'unknown_manufacturer'
        elif incomplete or any(a.get('error')=='InputStopped' for a in state.attempts):final = 'stopped_incomplete'
        elif self.engine.attribute_service is None:final = 'disabled'
        elif state.attempts:final = 'inference_failure'
        elif state.rejected_crops and not state.accepted_crops:final = 'no_eligible_crop'
        else:final = 'skipped_unscheduled'
        info = self.attempts.get((state.key,chosen[0]),{}) if chosen else {}
        native = info.get('native') or {}
        self._event(state,'record','outcome',kind='outcome',outcome=final,
            predicted_label=label,processing_incomplete=incomplete,attempt_count=len(state.attempts),
            quality_counts=dict(observed=state.observed_crops,eligible=state.accepted_crops,rejected=state.rejected_crops,
                                reasons=state.rejection_reasons,scheduling=state.scheduling_reasons),
            image=info.get('image'), frame_index=life.last_receipt.frame_index,video_timestamp=life.last_receipt.media_pts)
        if chosen:
            ordinal, attempt = chosen
            for domain,key in (('manufacturer','manufacturer'),('type','vehicle_type'),('colour','color')):
                value = attempt['result'][key];known = value not in (None,'','unknown')
                if domain=='manufacturer' and not known:continue  # The lifetime outcome already records abstention.
                self._event(state,'record','final-'+domain,domain=domain,
                    kind='prediction' if known else 'observation',outcome='prediction_produced' if known else 'unknown_'+domain,
                    predicted_label=value,image=info.get('image'),attempt_ordinal=ordinal,
                    quality=attempt.get('source_quality'),quality_time_ms=info.get('quality_time_ms'),
                    quality_scope='Source crop before any model-input resizing.',
                    inference_time_ms=native.get('duration_ms'),inference_start_ns=native.get('start_ns'),
                    inference_completion_ns=native.get('completion_ns'),
                    timing_scope='native_generation_cuda_synchronized',
                    attempt_time_ms=(info.get('total') or {}).get('duration_ms'),
                    queue_wait_ms=(info.get('queue') or {}).get('duration_ms'),
                    preprocessing_time_ms=(info.get('preprocessing') or {}).get('duration_ms'),
                    input_scope='Exact input and original last successful result; latency is native backend generation.',
                    **self._receipt(attempt['frame_index']))
        self.finalized.add(state.key)
