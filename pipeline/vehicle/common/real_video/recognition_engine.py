"""Frame-in/evidence-out engine. No decoder, Redis, Django or SQL ownership."""
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import threading
import time

from ..vehicle_observation import VehicleLifecycle, RunEnd
from ..plate_association import associate_plate
from ..redis_scalar.codec import pack_life, pack_receipt


class RecognitionEngine:
    def __init__(self, context, clock, directory, *, attributes=True, fps=25, stop=None, source_id=None):
        from ...recognition.config import paths, plate_config
        from ...recognition.attributes.detector import VehicleDetectorTracker
        from ...recognition.attributes.pipeline import VehiclePipeline
        from ...recognition.attributes.manufacturer.config import RecognitionConfig
        from ...recognition.attributes.manufacturer.service import ManufacturerService
        from ...recognition.attributes.manufacturer.tracking import TrackRecognizer
        from ...recognition.plate.run_video_prototype import VideoPrototypeRunner, CountingReader, _build_tracker
        from ...recognition.plate.prototype_pipeline import PrototypePlatePipeline
        from ...recognition.plate.classification import create_classifier
        from ...recognition.plate.plate_detector import PlateDetector
        from ...recognition.plate.marker_localizer import TounesMarkerLocalizer
        from ...recognition.plate.ocr import PlateReader
        from ...recognition.plate.data_capture import ReviewStore
        from ..console_stack import console_device
        self.context, self.directory, self.stop = context, Path(directory), stop
        self.lifecycle = VehicleLifecycle(context, clock)
        self.receipts, self.parents, self.profiles, self.plate_parents = {}, {}, {}, {}
        self.closed, self.ready, self.crops = {}, [], {}
        self.plate_diagnostics = {}
        self.attribute_inputs = {}
        self.plate_inputs = {}
        from ..evaluation_capture import EvaluationRecorder
        self.evaluation = EvaluationRecorder(self.directory, context,source_id=source_id)
        self.finished = False
        self.native_lane = threading.RLock()
        self.calls = {}
        self.models = {}
        self.local = threading.local()
        self.attribute_service = self.vehicle = self.plate = None
        self._native_close = []
        self.manufacturer_evaluation = None
        self.plate_recovery = None
        try:
            p = paths()
            self.attribute_service = ManufacturerService(RecognitionConfig.from_env()) if attributes else None
            if self.attribute_service:
                self.models['attributes'] = self._attribute_identity()
            recognizer = TrackRecognizer(self.attribute_service, self.attribute_service.config) if attributes else None
            if recognizer:
                recognizer.output_dir = self.directory/'attribute-crops'
                recognizer.frame_rate = fps
            detector = VehicleDetectorTracker(console_device(), p['vehicle_model'], p['vehicle_tracker'], fps)
            self._instrument(detector, 'process', 'vehicle_detection', identity=self._identity(p['vehicle_model'], 'detect'))
            self.vehicle = VehiclePipeline(detector, recognizer=recognizer)
            from ..recognition_evaluation import ManufacturerEvaluation
            self.manufacturer_evaluation = ManufacturerEvaluation(self)
            if recognizer:
                recognizer.decision_sink = self._attribute_crop
            cfg = plate_config()
            self.models['plate_policy'] = {name:getattr(cfg,name) for name in (
                'TUNIS_USE_LEGACY_WHOLE_OCR','PRIVATE_ROUTE_MIN_OCR_CONF','PRIVATE_ROUTE_MIN_MARKER_CONF',
                'MARKER_REGION_OCR_CONF','WHOLE_PLATE_MIN_OCR_CONF')}
            reader = CountingReader(factory=lambda: PlateReader(configuration=cfg))
            marker = TounesMarkerLocalizer(str(p['marker_model']))
            original_classifier = create_classifier
            def classifier():
                value = original_classifier()
                self._instrument(value.ocr, 'recognize', 'category_ocr', identity=self._identity(value.config.paddle_model_dir, 'ocr'))
                self._instrument(value.encoder, 'cls_embedding', 'category_visual', identity=self._identity(value.config.dinov3_checkpoint, 'classify'))
                self.models['category_reference'] = dict(index_sha256=self._hash(value.config.reference_index),
                    compatibility_fingerprint=value.index.compatibility_fingerprint)
                return value
            pipeline = PrototypePlatePipeline(reader, marker, ReviewStore(self.directory/'plate-evidence'),
                                               classifier_factory=classifier, config=cfg)
            self._instrument(marker, 'locate', 'marker', identity=self._identity(p['marker_model'], 'detect'))
            # The copied service's own registration reader retains native scores.
            original_initialize = reader.initialize
            def initialize():
                original_initialize()
                self._instrument(reader.delegate, 'read_result', 'registration_ocr', identity=self._identity(p['ocr_dir'], 'ocr'))
            reader.initialize = initialize
            original_process = pipeline.process_crop
            def process_crop(*args, **kwargs):
                self.local.operations = []
                result = original_process(*args, **kwargs)
                result.evidence['engine_operations'] = list(self.local.operations)
                image = args[0].crop
                folder = self.directory/'plate-inputs'; folder.mkdir(mode=0o700,exist_ok=True)
                path = folder/(result.sample_id+'.png')
                import cv2
                if not cv2.imwrite(str(path), image): raise RuntimeError('plate_input_encoding_failed')
                path.chmod(0o600)
                self.plate_inputs[result.sample_id] = path
                result.evidence['engine_input'] = dict(sha256=sha256(image.tobytes()).hexdigest(),shape=list(image.shape))
                return result
            pipeline.process_crop = process_crop
            plate_detector = PlateDetector(str(p['plate_model']))
            self._instrument(plate_detector, 'predict', 'plate_detection', identity=self._identity(p['plate_model'], 'detect'))
            from ...recognition.tracking import PlateTracker
            self.plate = VideoPrototypeRunner(pipeline, plate_detector, PlateTracker(_build_tracker(p['plate_tracker'], fps)), reader,
                                              camera_id=context.camera.camera_id, dedup_iou=cfg.PLATE_DEDUP_IOU)
            from ..plate_recovery import VehiclePlateRecovery
            from ...recognition.plate.run_video_prototype import RUNNER_DEFAULTS
            self.plate_recovery = VehiclePlateRecovery(self)
            self.plate_recovery.trace(dict(stage='settings',detector_conf=cfg.PLATE_CONF,
                detector_input=cfg.DETECT_IMGSZ,tracker=vars(self.plate.tracker.target.args),
                admission={name:getattr(self.plate,name) for name in RUNNER_DEFAULTS if hasattr(self.plate,name)}))
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _hash(path):
        path = Path(path)
        if path.is_dir():
            h = sha256()
            for p in sorted(path.iterdir()):
                if p.is_file():
                    h.update(p.name.encode()); h.update(bytes.fromhex(RecognitionEngine._hash(p)))
            return h.hexdigest()
        with path.open('rb') as stream:
            return __import__('hashlib').file_digest(stream, 'sha256').hexdigest()

    def _identity(self, path, task):
        return dict(model_name=Path(path).name+'-sha256:'+self._hash(path), version=1, task=task)

    def _attribute_identity(self):
        c = self.attribute_service.config
        if c.engine == 'huggingface':
            material = self._hash(c.hf_model_path)
        else:
            material = sha256((self._hash(c.model_path)+self._hash(c.projector_path)).encode()).hexdigest()
        return dict(model_name='LFM2.5-VL-450M-sha256:'+material, version=1, task='vlm')

    def _instrument(self, target, method, stage, *, identity):
        original = getattr(target, method)
        self.models[stage] = identity
        def execute(*args, **kwargs):
            with self.native_lane:
                if self.stop is not None and self.stop.is_set():
                    from ..real_video.rtsp import InputStopped
                    raise InputStopped('controlled_stop')
                self.calls[stage] = self.calls.get(stage, 0)+1
                operation = dict(kind=stage, model=identity, outcome='failed', confidence=None)
                image = args[0] if args else None
                if hasattr(image, 'tobytes'):
                    operation.update(input_sha256=sha256(image.tobytes()).hexdigest(), shape=list(image.shape))
                def synchronize():
                    import torch
                    if torch.cuda.is_initialized():torch.cuda.synchronize()
                started = time.perf_counter_ns()
                entered = False
                def invoke():
                    nonlocal entered
                    entered = True
                    return original(*args, **kwargs)
                try:
                    result, _ = self.evaluation.measure(stage,identity['model_name'],invoke,
                        synchronize=synchronize,model_call_started=lambda:entered,
                        scope='Synchronized producer operation; includes its preprocessing and postprocessing.')
                    operation['outcome'] = 'succeeded'
                    return result
                finally:
                    operation['model_call'] = entered
                    operation['seconds'] = (time.perf_counter_ns()-started)/1e9
                    if hasattr(self.local, 'operations'):
                        self.local.operations.append(operation)
        setattr(target, method, execute)

    def _attribute_crop(self, record, image):
        if self.manufacturer_evaluation is not None:
            self.manufacturer_evaluation.decision(record,image)
        if record.get('submitted') and image is not None:
            import cv2
            folder = self.directory/'attribute-inputs'; folder.mkdir(mode=0o700,exist_ok=True)
            path = folder/f'{record["track_id"]}-{record["frame_index"]}.png'
            if not cv2.imwrite(str(path),image): raise RuntimeError('attribute_input_encoding_failed')
            path.chmod(0o600)
            self.attribute_inputs[record['track_id'],record['frame_index']] = dict(path=path,
                sha256=sha256(image.tobytes()).hexdigest(),bbox=record['bbox'],shape=list(image.shape))
            sid = self.parents.get(record['track_id'])
            profile = self.profiles.get(sid)
            # A first-frame submission can precede the new scene's ID binding.
            # The previous scene's same-raw parent does not own this crop.
            if profile is not None and profile.recognition.key==record['track_key']:
                self._save_crop(sid, 'vehicle', image, record['frame_index'])

    def _save_crop(self, sid, kind, image, frame):
        import cv2
        from ..console_runs import atomic_json
        folder = self.directory/'crops'; folder.mkdir(exist_ok=True, mode=0o700)
        path = folder/f'track-{sid}-{kind}.png'
        ok, encoded = cv2.imencode('.png', image)
        if not ok:
            raise RuntimeError('crop_encoding_failed')
        path.write_bytes(encoded.tobytes()); path.chmod(0o600)
        crop = dict(status='available', source='engine_selected', frame=frame, sha256=sha256(encoded).hexdigest())
        item = self.crops.setdefault(str(sid), {})
        item[kind] = crop if kind == 'vehicle' else dict(crop=crop)
        atomic_json(self.directory/'crops.json', dict(schema_version='console-crops-v1', tracks=self.crops))

    def process(self, packet):
        self.receipts[packet.receipt.frame_index] = packet.receipt
        # Scene handling belongs to the supplied pipeline, but close service IDs
        # before the replacement tracker admits observations from the new scene.
        previous_cuts = len(self.vehicle.scene_cuts)
        tracker = self.vehicle.detector_tracker.tracker
        update = tracker.update
        gap = packet.receipt.frame_index-getattr(self,'previous_vehicle_frame',packet.receipt.frame_index-1)
        def sampled(detections, frame, *args, **kwargs):
            from ..console_sampling import advance_tracker
            advance_tracker(tracker,gap,detections[:0],frame,update=update)
            return update(detections,frame,*args,**kwargs)
        tracker.update = sampled
        try: self.vehicle.process_frame(packet.pixels, packet.receipt.frame_index)
        finally: tracker.update = update
        self.previous_vehicle_frame = packet.receipt.frame_index
        if len(self.vehicle.scene_cuts) != previous_cuts:
            for life in self.lifecycle.reset():
                self.closed[life.service_track_id] = life
                self._finalize_manufacturer(life)
            self._settle_plate_tracks('scene_cut')
            self._settle_closed()
            self.plate.tracks.clear()
            self.plate.tracker.reset()
        # Retire the previous profiles before admitting any reused raw ID. On a
        # scene cut lifecycle.reset() already closed them, so lose() is a no-op;
        # doing this after observe() would close the replacement lifetime instead.
        for profile in self.vehicle.finalized_profiles:
            life = self.lifecycle.lose(profile.track_id)
            if life:
                self.closed[life.service_track_id] = life
                self._finalize_manufacturer(life)
        self.vehicle.finalized_profiles.clear()
        rows = []
        for raw, profile in self.vehicle.active_profiles.items():
            if profile.last_frame != packet.receipt.frame_index:
                continue
            life = self.lifecycle.observe(raw, packet.receipt)
            self.parents[raw] = life.service_track_id
            self.profiles[life.service_track_id] = profile
            observer = getattr(self,'manufacturer_evaluation',None)
            if observer:observer.bind(life,profile.recognition)
            rows.append(dict(track_id=life.service_track_id, bbox=profile.bbox))
        self.plate.tracker.frame_index = packet.receipt.frame_index
        annotations = self.plate.process_frame(packet.pixels, packet.receipt.frame_index, self.context.run_id)
        for detection in self.plate.frame_plate_detections:
            association = associate_plate(list(detection[:4]), rows)
            targets = [association['selected_track_id']] if association['selected_track_id'] is not None else [s['track_id'] for s in association['scores']]
            for sid in targets:
                diagnostic = self.plate_diagnostics.setdefault(sid, dict(detections=0, ambiguous=0))
                diagnostic['detections'] += 1
                diagnostic['ambiguous'] += association['status']=='ambiguous'
        for raw, bbox in annotations:
            association = associate_plate(list(bbox), rows)
            self.plate_parents[raw, packet.receipt.frame_index] = association['selected_track_id']
        if getattr(self,'plate_recovery',None) is not None:
            self.plate_recovery.observe(packet,rows,self.plate.frame_plate_detections,annotations)
        self._collect()
        return rows

    def _finalize_manufacturer(self, life, *, incomplete=False):
        observer = getattr(self,'manufacturer_evaluation',None)
        if observer and life.service_track_id in self.profiles:
            observer.finalize(life,self.profiles[life.service_track_id].recognition,incomplete=incomplete)

    def _settle_plate_tracks(self, trigger):
        for state in self.plate.tracks.values():
            if not state.attempts and state.pending is None and state.candidate_count >= self.plate.minimum_exit_candidates:
                self.plate._submit(state,self.context.run_id,trigger)
        while any(state.pending is not None or state.deferred_reason for state in self.plate.tracks.values()):
            self.plate._drain(self.context.run_id,None)
            time.sleep(.01)
        for state in self.plate.tracks.values(): self.plate.finalize_track(state)

    def _collect(self, *, final=False):
        from ..recognition_manufacturers import POLICY
        for sid, life in tuple(self.closed.items()):
            states = [state for raw,state in self.plate.tracks.items()
                      if any(parent == sid for (p,_),parent in self.plate_parents.items() if p == raw)]
            if any(state.pending is not None or state.deferred_reason for state in states):
                continue
            if not final and any(max(self.receipts)-state.last_frame < self.plate.disappearance_frames for state in states):
                continue
            recovery = getattr(self,'plate_recovery',None)
            if recovery is not None and not recovery.prepare(sid,states):continue
            plates = []
            for state in states:
                for ordinal, attempt in enumerate(state.attempts, 1):
                    if self.plate_parents.get((state.track_id, attempt.crop_frame)) != sid:
                        continue
                    result = attempt.result.model_dump(mode='json')
                    receipt = self.receipts[attempt.crop_frame]
                    plates.append(dict(ordinal=ordinal, plate_track_id=state.track_id, receipt=pack_receipt(receipt),
                        result=result, selected=attempt is state.cached_attempt, crop_quality=attempt.crop_score))
                    if attempt is state.cached_attempt:
                        import cv2
                        image = cv2.imread(str(self.plate_inputs[result['sample_id']]))
                        self._save_crop(sid, 'plate', image, attempt.crop_frame)
                        image_ref = self.evaluation.image(image)
                        self.evaluation.record('plate',sid,receipt,key=result['sample_id'],image=image_ref,
                            predicted_label=result['normalized_number'],outcome=result['recognition_status'],
                            model_version='plate-recognition-v3',confidence=result['text_confidence'],
                            attempts=len(state.attempts))
                        self.evaluation.record('plate_category',sid,receipt,key=result['sample_id'],image=image_ref,
                            predicted_label=result['predicted_category'],model_version='DINOv3-ArabicOCR-v3',
                            category_similarity=result['classification']['nearest_similarity'] if result.get('classification') else None)
            profile = self.profiles[sid]
            attrs = profile.recognition.to_dict()
            for attempt in attrs['recognition_evidence']:
                attempt['receipt'] = pack_receipt(self.receipts[attempt['frame_index']])
                source = self.attribute_inputs.get((profile.track_id,attempt['frame_index']))
                if source:
                    attempt['engine_input'] = {k:v for k,v in source.items() if k!='path'}
            successes = [a for a in attrs['recognition_evidence'] if 'result' in a]
            if successes:
                import cv2
                chosen = successes[-1]
                source = self.attribute_inputs[profile.track_id,chosen['frame_index']]
                image = cv2.imread(str(source['path']))
                receipt = self.receipts[chosen['frame_index']]
                self._save_crop(sid,'vehicle',image,receipt.frame_index)
            self.ready.append(dict(life=pack_life(life), attributes=attrs, plates=plates,
                models=dict(self.models), attributes_enabled=self.attribute_service is not None,
                manufacturer_policy=POLICY,
                plate_diagnostics=self.plate_diagnostics.pop(sid, dict(detections=0,ambiguous=0))))
            del self.closed[sid]
            self.lifecycle.release_closed((sid,))
            self.profiles.pop(sid, None)
            if recovery is not None:recovery.release(sid)
            # A lost ByteTrack ID can reactivate while the old lifetime waits on
            # plate work. Releasing that old closure must preserve its successor.
            if self.parents.get(life.raw_track_id)==sid:
                self.parents.pop(life.raw_track_id)
            for state in states:
                # One plate tracker ID can span multiple vehicle associations.
                # Release it only after every owning vehicle has been emitted.
                still_owned = any(raw == state.track_id and parent in self.profiles
                                  for (raw,_),parent in self.plate_parents.items())
                if state.pending is None and not state.deferred_reason and not still_owned:
                    self._release_plate_state(state)
        self._retire_unassociated_plates(final=final)
        self._prune()

    def _settle_closed(self):
        """Keep the OCR queue alive until closure/EOF recovery is also drained."""
        self._collect(final=True)
        while self.closed:
            self.plate._drain(self.context.run_id,None)
            self._collect(final=True)
            if self.closed:time.sleep(.01)

    def _release_plate_state(self, state):
        self.plate.finalize_track(state)
        for attempt in state.attempts:
            self.plate_inputs.pop(attempt.result.sample_id, None)
        state.first_crop = state.best_crop = None
        self.plate.tracks.pop(state.track_id, None)

    def _retire_unassociated_plates(self, *, final):
        owned = {raw for (raw,_),parent in self.plate_parents.items() if parent in self.profiles}
        tracker = getattr(self.plate, 'tracker', None)
        tracker = getattr(tracker, 'target', tracker)
        horizon = max(self.plate.disappearance_frames,
            getattr(tracker, 'max_time_lost', getattr(tracker, 'max_frames_lost', 0)))
        current = max(self.receipts, default=0)
        for raw, state in tuple(self.plate.tracks.items()):
            if raw in owned or state.pending is not None or state.deferred_reason:
                continue
            if final or current-state.last_frame > horizon:
                self._release_plate_state(state)

    def _prune(self):
        """Retain receipts/association entries only for selected or pending evidence."""
        frames = {max(self.receipts)} if self.receipts else set()
        for profile in self.profiles.values():
            frames.update(a['frame_index'] for a in profile.recognition.attempts)
            candidate = profile.recognition.candidate
            if candidate is not None: frames.add(candidate.frame_index)
        for state in self.plate.tracks.values():
            frames.update(state.submitted_frames)
            frames.add(state.best_frame)
            # The latest parent may not own the best crop or an OCR attempt yet.
            frames.add(state.last_frame)
        recovery = getattr(self,'plate_recovery',None)
        if recovery is not None:frames.update(recovery.frames())
        self.receipts = {index:receipt for index,receipt in self.receipts.items() if index in frames}
        self.plate_parents = {(raw,frame):parent for (raw,frame),parent in self.plate_parents.items()
                              if raw in self.plate.tracks and frame in frames}
        for name in ('pending_snapshots','detection_times_ms','tracking_times_ms','crop_scoring_times_ms',
                     'detections_per_frame','tracked_outputs_per_frame'):
            values = getattr(self.plate,name)
            del values[:-128]

    def drain(self):
        ready, self.ready = self.ready, []
        return ready

    def finish(self, reason=RunEnd.END_OF_INPUT):
        self.vehicle.finish()
        for life in self.lifecycle.end(reason):
            self.closed[life.service_track_id] = life
            self._finalize_manufacturer(life,incomplete=reason!=RunEnd.END_OF_INPUT)
        observer = getattr(self,'manufacturer_evaluation',None)
        if observer:observer.ensure_complete()
        ready_before_finish = len(self.ready)
        try:
            self._settle_plate_tracks('video_end')
            self._settle_closed()
            self.plate.finish(self.context.run_id)
        except BaseException:
            # Finalization failure must not expose a partly settled EOF batch.
            del self.ready[ready_before_finish:]
            raise
        self.finished = True

    def _retain_incomplete_evaluation(self):
        observer = getattr(self,'manufacturer_evaluation',None)
        if observer and self.vehicle is not None:
            # A detector/plate/broker failure must not hide profiles or completed calls.
            for profile in (*self.vehicle.active_profiles.values(),*self.vehicle.finalized_profiles):
                if profile.track_id not in self.parents and profile.last_frame in self.receipts:
                    receipt = self.receipts[profile.last_frame]
                    if receipt is self.lifecycle._clock.last_valid_receipt and self.lifecycle.end_reason is None:
                        life = self.lifecycle.observe(profile.track_id,receipt)
                        self.parents[profile.track_id] = life.service_track_id
                        self.profiles[life.service_track_id] = profile
                        observer.bind(life,profile.recognition)
            for life in self.lifecycle.lifetimes:
                self._finalize_manufacturer(life,incomplete=True)

    def close(self):
        try:
            self._retain_incomplete_evaluation()
        finally:
            # Storage/identity errors must fail visibly while still releasing models.
            try:
                if self.plate is not None:
                    self.plate.close()
                    for state in self.plate.tracks.values():
                        state.first_crop = state.best_crop = None
                    self.plate.tracks.clear()
                    self.plate.pipeline.marker_localizer = None
                    self.plate.detector = None
            finally:
                try:
                    if self.attribute_service is not None:self.attribute_service.close()
                finally:
                    self.vehicle = self.attribute_service = self.plate = None
                    self.profiles.clear(); self.receipts.clear(); self.plate_parents.clear()
                    self.plate_inputs.clear()
                    recovery = getattr(self,'plate_recovery',None)
                    if recovery is not None:recovery.close()
                    import gc
                    gc.collect()
