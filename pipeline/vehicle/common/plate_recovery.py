"""Bounded, engine-local plate recovery using the existing detector and OCR queue."""
from dataclasses import dataclass, field
from hashlib import sha256
import math

from .plate_association import associate_plate
from ..recognition.plate.plate_detector import PlateCrop, crop_detections, deduplicate_detections
from ..recognition.plate.run_video_prototype import TrackState, crop_quality_evidence


@dataclass
class Candidate:
    plate: PlateCrop
    frame: int
    shape: tuple
    score: float
    digest: str
    vehicles: list = field(default_factory=list)
    origin: str = 'full_frame'


@dataclass
class VehicleCandidates:
    plates: list = field(default_factory=list)
    vehicles: list = field(default_factory=list)
    initial_checked: bool = False
    retry_checked: bool = False


class VehiclePlateRecovery:
    MAX_VEHICLE_DETECTIONS = 2
    MAX_BYTES = 64 * 1024 * 1024

    def __init__(self, engine):
        self.engine = engine
        self.buffers = {}
        self.bytes = 0

    def diagnostics(self, sid):
        return self.engine.plate_diagnostics.setdefault(sid, dict(detections=0, ambiguous=0))

    def _retain(self, candidates, plate, receipt, shape, vehicles=(), origin='full_frame'):
        x1,y1,x2,y2 = plate.bbox
        score,_,_ = crop_quality_evidence(plate.crop, plate.confidence or 0,
            border_contact=x1<=0 or y1<=0 or x2>=shape[1] or y2>=shape[0])
        if not math.isfinite(score):return False
        if len(candidates)==2 and score<=candidates[-1].score:return False
        digest = sha256(plate.crop.tobytes()).hexdigest()
        if any(c.digest==digest for c in candidates):return False
        removed = candidates[-1].plate.crop.nbytes if len(candidates)==2 else 0
        if self.bytes-removed+plate.crop.nbytes>self.MAX_BYTES:return False
        candidate = Candidate(PlateCrop(plate.crop.copy(),plate.bbox,plate.confidence),
            receipt.frame_index,shape,score,digest,list(vehicles),origin)
        candidates.append(candidate)
        candidates.sort(key=lambda c:(-c.score,c.frame))
        self.bytes += candidate.plate.crop.nbytes
        if len(candidates)>2:self.bytes -= candidates.pop().plate.crop.nbytes
        return True

    def observe(self, packet, vehicles, detections, annotations):
        """Associate original detector boxes before independent plate tracking can drop them."""
        by_id = {row['track_id']:row for row in vehicles}
        for sid,row in by_id.items():
            buffer = self.buffers.setdefault(sid,VehicleCandidates())
            safe = crop_detections(packet.pixels,[(*row['bbox'],1.0)],min_width=4,min_height=4)
            if safe:self._retain(buffer.vehicles,safe[0],packet.receipt,packet.pixels.shape,vehicles,'vehicle_observation')
        trace = []
        for detection in detections:
            association = associate_plate(list(detection[:4]),vehicles)
            sid = association['selected_track_id']
            decision = dict(bbox=list(detection[:4]),confidence=detection[4],association=association)
            if sid is not None:
                safe = crop_detections(packet.pixels,[detection],min_width=4,min_height=4)
                diag = self.diagnostics(sid)
                if safe:
                    diag['eligible_crops'] = diag.get('eligible_crops',0)+1
                    decision['retained'] = self._retain(self.buffers[sid].plates,safe[0],packet.receipt,packet.pixels.shape)
                    if not self.buffers[sid].plates:diag['stopping_stage'] = 'crop_retention_limit'
                decision['eligible_crop'] = bool(safe)
            trace.append(decision)
        tracked_crops = []
        for raw,bbox in annotations:
            association = associate_plate(list(bbox),vehicles)
            sid = association['selected_track_id']
            tracked_crops.append(dict(track_id=raw,bbox=list(bbox),association=association))
            if sid is not None:
                diag = self.diagnostics(sid)
                diag['tracker_observations'] = diag.get('tracker_observations',0)+1
        self.trace(dict(frame=packet.receipt.frame_index,detections=trace,tracked_crops=tracked_crops,
            plate_tracks=[dict(track_id=s.track_id,candidates=s.candidate_count,
                best_frame=s.best_frame,best_score=s.best_score,pending=s.pending is not None,
                deferred=s.deferred_reason,attempts=len(s.attempts),
                stable_observations=s.observations_since_best,border_contact=s.best_border_contact,
                trigger=s.last_trigger_reason,recovery=bool(s.recovery_metadata)) for s in self.engine.plate.tracks.values()]))

    def trace(self, value):
        import json
        with (self.engine.directory/'plate-decisions.jsonl').open('a') as stream:
            stream.write(json.dumps(value,allow_nan=False)+'\n')

    def _detect_in_vehicle(self, sid, buffer):
        runner = self.engine.plate
        diag = self.diagnostics(sid)
        for name in ('fallback_detector_calls','fallback_detections','fallback_eligible_crops'):
            diag.setdefault(name,0)
        for vehicle in buffer.vehicles[:self.MAX_VEHICLE_DETECTIONS]:
            diag['fallback_detector_calls'] = diag.get('fallback_detector_calls',0)+1
            detections = deduplicate_detections(runner.detector.predict(vehicle.plate.crop),
                                              iou_threshold=runner.dedup_iou)
            diag['fallback_detections'] = diag.get('fallback_detections',0)+len(detections)
            recovered = 0
            ox,oy,_,_ = vehicle.plate.bbox
            for detection in detections:
                for local in crop_detections(vehicle.plate.crop,[detection],min_width=4,min_height=4):
                    x1,y1,x2,y2 = local.bbox
                    bbox = (x1+ox,y1+oy,x2+ox,y2+oy)
                    if associate_plate(list(bbox),vehicle.vehicles)['selected_track_id']!=sid:continue
                    recovered += 1
                    self._retain(buffer.plates,PlateCrop(local.crop,bbox,local.confidence),
                        self.engine.receipts[vehicle.frame],vehicle.shape,origin='vehicle_crop')
            diag['fallback_eligible_crops'] = diag.get('fallback_eligible_crops',0)+recovered
            self.trace(dict(vehicle=sid,stage='vehicle_crop_detection',frame=vehicle.frame,
                source_bbox=list(vehicle.plate.bbox),detections=detections,eligible=recovered))

    def _submit(self, sid, candidate, trigger, previous=None):
        runner = self.engine.plate
        raw = runner.tracker.allocate_id()
        state = TrackState(raw,candidate.frame,candidate.frame)
        state.consider(candidate.plate,candidate.frame,candidate.shape)
        state.recovery_metadata = dict(vehicle_id=sid,input_sha256=candidate.digest,
            detector_scope=candidate.origin,retry_of=previous.result.sample_id if previous else None)
        runner.tracks[raw] = state
        self.engine.plate_parents[raw,candidate.frame] = sid
        runner._submit(state,self.engine.context.run_id,trigger)
        diag = self.diagnostics(sid)
        diag['recovery_ocr_scheduled'] = diag.get('recovery_ocr_scheduled',0)+1
        self.trace(dict(vehicle=sid,stage='ocr_scheduled',trigger=trigger,plate_track=raw,
            frame=candidate.frame,source_bbox=list(candidate.plate.bbox),score=candidate.score,
            **state.recovery_metadata))

    def prepare(self, sid, states):
        """Called after normal admission settles, before a vehicle can be emitted."""
        buffer = self.buffers.get(sid)
        if buffer is None:return True
        if self.engine.stop is not None and self.engine.stop.is_set():
            self.diagnostics(sid)['stopping_stage'] = 'stopped_before_ocr'
            return True
        attempts = [a for s in states for a in s.attempts
                    if self.engine.plate_parents.get((s.track_id,a.crop_frame))==sid]
        self.trace(dict(vehicle=sid,stage='closure_admission',attempts=len(attempts),
            vehicle_crops=len(buffer.vehicles),
            plate_tracks=[dict(track_id=s.track_id,candidates=s.candidate_count,
                best_frame=s.best_frame,last_frame=s.last_frame,pending=s.pending is not None,
                deferred=s.deferred_reason,trigger=s.last_trigger_reason,
                stable_observations=s.observations_since_best,border_contact=s.best_border_contact,
                recovery=bool(s.recovery_metadata)) for s in states],
            retained=[dict(frame=c.frame,score=c.score,input_sha256=c.digest) for c in buffer.plates]))
        if any(a.result.recognition_status=='recognized' for a in attempts):return True
        diag = self.diagnostics(sid)
        if not attempts and not buffer.initial_checked:
            buffer.initial_checked = True
            if not buffer.plates:self._detect_in_vehicle(sid,buffer)
            if buffer.plates:
                self._submit(sid,buffer.plates[0],'vehicle_closure_fallback')
                return False
            diag.setdefault('stopping_stage', 'no_eligible_crop' if diag.get('detections',0) or diag.get('fallback_detections',0) else 'no_detection')
        if len(attempts)==1 and not buffer.retry_checked:
            buffer.retry_checked = True
            previous = attempts[0]
            reason = previous.result.unresolved_reason
            if reason in {'insufficient_ocr_confidence','insufficient_region_ocr_confidence'}:
                original = previous.result.evidence.get('engine_input',{}).get('sha256')
                for candidate in buffer.plates:
                    if original and candidate.score>previous.crop_score and candidate.digest!=original:
                        self._submit(sid,candidate,'low_confidence_better_crop_recheck',previous)
                        return False
        return True

    def frames(self):
        return {c.frame for b in self.buffers.values() for c in b.plates+b.vehicles}

    def release(self, sid):
        buffer = self.buffers.pop(sid,None)
        if buffer:self.bytes -= sum(c.plate.crop.nbytes for c in buffer.plates+buffer.vehicles)

    def close(self):
        self.buffers.clear()
        self.bytes = 0
