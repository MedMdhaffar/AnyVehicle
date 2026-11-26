"""Optional producer-side evidence capture. Never changes recognition decisions."""
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import threading

from .evaluation import EvaluationStore, stable_id
from .evaluation_timing import measured


class EvaluationRecorder:
    def __init__(self, directory, context, *, source_id=None):
        self.context=context
        self.store=EvaluationStore(directory,run_id=context.run_id,source_id=source_id)
        self.run=context.run_id;self.frame=None;self.observations={};self.selected={}
        self.colours={};self.plate_inputs={};self.quality={};self.rejected=set();self.local=threading.local()
        self.counter=0;self.lock=threading.RLock()
        self.original_frame_index=None;self.original_frame_image=None
        self.observed_counts={};self.detector_calls={}
        self.plate_quality_times={}

    def image(self, pixels):
        from PIL import Image
        import io
        if pixels is None:return None
        with io.BytesIO() as buffer:
            Image.fromarray(pixels[:,:,::-1]).save(buffer,format='PNG');data=buffer.getvalue()
        name=sha256(data).hexdigest()+'.png';folder=self.store.directory/'evaluation-images'
        folder.mkdir(mode=0o700,exist_ok=True);path=folder/name
        if not path.exists():
            import os
            try:
                fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
                with os.fdopen(fd,'wb') as target:target.write(data)
            except FileExistsError:pass
        return name

    def measure(self, stage, model, call, *, synchronize=lambda:None, identity=None, model_call_started=None, **details):
        with self.lock:
            self.counter+=1
            key=identity or str(self.counter)
        event=dict(id=stable_id(self.run,'call',stage,key),stage=stage,model_version=model,
                   run_id=self.run,**details)
        try:
            result,timing=measured(call,synchronize=synchronize)
        except Exception as error:
            event.update(getattr(error,'evaluation_timing',{}),outcome='timed_out' if isinstance(error,TimeoutError) else 'failed')
            if model_call_started is not None:event['model_call']=bool(model_call_started())
            self.store.call(event)
            raise
        event.update(timing,outcome='completed')
        if model_call_started is not None:event['model_call']=bool(model_call_started())
        self.store.call(event)
        return result,event

    def record(self, domain, track, receipt, *, kind='prediction', key='final', **fields):
        record=dict(id=stable_id(self.run,domain,track,key),run_id=self.run,
            source_id=self.store.metadata()['source_id'],track_id=track,
            camera_id=self.context.camera.camera_id,lifetime_id=f'{self.run}:{track}',
            frame_index=receipt.frame_index if receipt else None,
            video_timestamp=receipt.media_pts if receipt else None,domain=domain,kind=kind,
            predicted_label=None,image=None,original_frame=None,model_version=None,
            quality_policy_version=None,quality_time_ms=None,inference_time_ms=None)
        record.update(fields)
        self.store.record(record)

    def observation(self, life, receipt, box, crop, frame, label):
        sid=life.service_track_id
        self.local.observation=(life,receipt,crop)
        self.observed_counts[sid]=self.observed_counts.get(sid,0)+1
        if self.observed_counts[sid]>20:return  # Match the Engine's admitted evidence window.
        # Detector input is the full frame; the ROI is its contextual review crop.
        if self.original_frame_index!=receipt.frame_index:
            self.original_frame_image=self.image(frame);self.original_frame_index=receipt.frame_index
        item=dict(image=self.image(crop),original_frame=self.original_frame_image,receipt=receipt,
                  raw_track_id=life.raw_track_id,tracker_epoch=life.tracker_epoch,bbox=list(box))
        self.observations[sid,receipt.frame_index]=item

    def colour(self, life, receipt, crop, estimate, quality, inference, quality_timing):
        self.colours[life.service_track_id,receipt.frame_index]=dict(
            image=self.image(crop),quality_time_ms=quality_timing['duration_ms'],
            inference_time_ms=inference['duration_ms'],call_id=inference['id'],quality=float(quality.combined))
        self.colours[life.service_track_id,receipt.frame_index].update(
            inference_start_ns=inference['start_ns'],inference_completion_ns=inference['completion_ns'])

    def bind(self, engine):
        self.engine=engine;engine.colour.evaluation=self
        original_process=engine.process
        def process(packet):
            self.frame=packet.receipt
            return original_process(packet)
        engine.process=process
        vehicle=engine.colour.detector.target.target
        self.wrap_detector(vehicle,'vehicle_detection',self.context.resolution.detector.model_name,gpu=True)
        self.wrap_detector(engine.plate.detector,'plate_detection',getattr(getattr(engine.plate.detector,'identity',None),'digest','plate-detector'))
        engine.plate.evaluation=self
        if engine.colour.manufacturer is not None:self.bind_manufacturer(engine.colour.manufacturer)
        for binding in engine.plate.bindings.values():
            if hasattr(binding,'factory'):self.bind_plate(binding)

    def wrap_detector(self, detector, stage, model, *, gpu=False):
        original=detector.process
        def process(pixels):
            if not gpu and hasattr(detector,'initialize') and getattr(detector,'tracker',None) is None:
                self.measure(stage+'_load',model,detector.initialize)
            def sync():
                if gpu:
                    import torch
                    torch.cuda.synchronize(getattr(detector,'device','cuda:0'))
            result,event=self.measure(stage,model,lambda:original(pixels),synchronize=sync,
                identity=str(self.frame.frame_index),frame_index=self.frame.frame_index)
            if stage=='vehicle_detection':self.detector_calls[self.frame.frame_index]=event
            return result
        detector.process=process

    def bind_manufacturer(self, ledger):
        analyze=ledger.analyzer.analyze
        def analyzing(crop, **kwargs):
            result,event=self.measure('manufacturer_quality',ledger.selection.quality.version,
                lambda:analyze(crop,**kwargs))
            life,receipt,_=self.local.observation;sid=life.service_track_id
            self.quality[sid,receipt.frame_index]=(result.record(),event['duration_ms'])
            reason=tuple(result.reasons)
            image=None
            if not result.eligible and (sid,reason) not in self.rejected:
                image=self.image(crop);self.rejected.add((sid,reason))
            self.record('manufacturer',sid,receipt,kind='quality',key='quality-'+str(receipt.frame_index),
                image=image,original_frame=self.observations.get((sid,receipt.frame_index),{}).get('original_frame'),
                quality=result.record(),quality_time_ms=event['duration_ms'],
                quality_policy_version=ledger.selection.quality.version,outcome='eligible' if result.eligible else 'rejected')
            return result
        ledger.analyzer.analyze=analyzing
        previous=ledger.submit_hook
        def submitted(life,receipt,crop):
            if previous:previous(life,receipt,crop)
            self.selected[life.service_track_id]=dict(image=self.image(crop),receipt=receipt)
        ledger.submit_hook=submitted
        original_record=ledger._record
        def recorded(key,b,outcome,text,seconds):
            if not ledger.asynchronous and outcome=='completed':
                timing=getattr(ledger.runtime,'last_evaluation_timing',None)
                if timing:
                    if not hasattr(ledger.runtime,'evaluation_timings'):ledger.runtime.evaluation_timings={}
                    from .manufacturer.engine import hashed
                    ledger.runtime.evaluation_timings[hashed(b)]=timing
            return original_record(key,b,outcome,text,seconds)
        ledger._record=recorded
        execute=ledger.execute
        def executing(call):
            return self.measure('manufacturer_request',getattr(ledger.runtime,'provenance','manufacturer'),
                                lambda:execute(call))[0]
        ledger.execute=executing

    def bind_plate(self, binding):
        original=binding.invoke
        def invoke(kind,request,context):
            self.local.plate=request
            return original(kind,request,context)
        binding.invoke=invoke
        factory=binding.factory
        def runtime_factory(identity):
            runtime=factory(identity);initialize=runtime.initialize;predict=runtime.predict
            model=identity.model.model_name+':v'+str(identity.model.version)
            def loading():return self.measure('ocr_load' if identity.task=='ocr' else 'marker_load',model,initialize)[0]
            def predicting(image):
                request=self.local.plate
                sid=request.candidate.identity.service_track_id
                name=self.image(image)
                key=request.plan.operation_id
                result,event=self.measure('ocr' if identity.task=='ocr' else 'plate_marker',model,
                    lambda:predict(image),identity=key,track_id=sid,frame_index=request.candidate.receipt.frame_index,
                    operation= request.plan.kind,image=name)
                self.plate_inputs[key]=dict(image=name,timing=event,track_id=sid,
                    candidate_id=request.candidate.candidate_id,operation=request.plan.kind)
                return result
            runtime.initialize=loading;runtime.predict=predicting
            return runtime
        binding.factory=runtime_factory

    def plate_quality(self, candidate, pixels, timing):
        sid=candidate.identity.service_track_id;reason=[] if candidate.crop_accepted else ['size_or_association']
        self.plate_quality_times[candidate.candidate_id]=timing.get('duration_ms') if timing else None
        image=None
        if reason and (sid,'plate',tuple(reason)) not in self.rejected:
            self.rejected.add((sid,'plate',tuple(reason)));image=self.image(pixels)
        self.record('plate',sid,candidate.receipt,kind='quality',key=candidate.candidate_id,
            image=image,original_frame=self.observations.get((sid,candidate.receipt.frame_index),{}).get('original_frame'),
            quality=dict(eligible=candidate.crop_accepted,score=candidate.quality,reasons=reason),
            quality_policy_version=candidate.quality_algorithm,quality_time_ms=self.plate_quality_times[candidate.candidate_id],
            outcome='eligible' if candidate.crop_accepted else 'rejected')

    def finalize(self, closed, plate, bundle):
        from .colour_metadata import ColourMetadata
        from .manufacturer.policy import evaluate
        from .manufacturer.engine import load_canonical
        life=closed.owner.lifetime;sid=life.service_track_id
        facts={f['row_kind']:f['delivery'] for f in bundle['facts']}
        first=life.first_receipt;item=self.observations.get((sid,first.frame_index),{})
        detector_call=self.detector_calls.get(first.frame_index,{})
        self.record('type',sid,first,predicted_label=facts['fact_vehicle']['class']['name'],
            image=item.get('image'),original_frame=item.get('original_frame'),
            inputs=list(dict.fromkeys(v['original_frame'] for (track,_),v in self.observations.items() if track==sid and v.get('original_frame'))),
            model_version=self.context.resolution.detector.model_name,
            inference_time_ms=detector_call.get('duration_ms'),
            inference_start_ns=detector_call.get('start_ns'),inference_completion_ns=detector_call.get('completion_ns'),
            outcome='completed',input_scope='Full-frame detector; displayed ROI is context, not a second model call.',
            raw_track_id=life.raw_track_id,tracker_epoch=life.tracker_epoch)
        colour=ColourMetadata().finalize(life,closed.measurements).evidence
        info=self.colours.get((sid,colour.support_receipt.frame_index),{})
        self.record('colour',sid,colour.support_receipt,predicted_label=colour.label,
            original_frame=self.observations.get((sid,colour.support_receipt.frame_index),{}).get('original_frame'),
            inputs=[v['image'] for (track,_),v in self.colours.items() if track==sid],
            model_version=colour.executed_model.model_name+':v'+str(colour.executed_model.version),
            quality_policy_version=self.engine.colour.quality_algorithm,outcome=colour.state,
            input_scope='Temporal decision over admitted crops; latency is the supporting crop processing call.',**info)
        evidence=load_canonical(closed.manufacturer_json) if closed.manufacturer_json else None
        if evidence:
            decision=evaluate(life,evidence);selection=self.selected.get(sid,{})
            receipt=selection.get('receipt');image=selection.get('image')
            quality,quality_ms=self.quality.get((sid,receipt.frame_index if receipt else None),(None,None))
            native=getattr(self.engine.colour.manufacturer.runtime,'evaluation_timings',{}).get(evidence['request_id'])
            # No measured native response for a timeout/failure => no invented inference latency.
            if evidence['outcome']!='completed':native=None
            timing=native or {}
            self.record('manufacturer',sid,receipt,kind='prediction' if evidence['outcome']=='completed' else 'observation',
                predicted_label=decision.make,image=image,
                crop_sha256=evidence['binding']['crop_sha256'],
                original_frame=self.observations.get((sid,receipt.frame_index),{}).get('original_frame') if receipt else None,
                model_version=evidence['binding']['provenance'],quality=quality,quality_time_ms=quality_ms,
                quality_policy_version=self.engine.colour.manufacturer.selection.quality.version,
                outcome=decision.status,inference_time_ms=timing.get('duration_ms'),
                inference_start_ns=timing.get('start_ns'),inference_completion_ns=timing.get('completion_ns'),
                load_time_ms=timing.get('load_time_ms'),queue_wait_ms=timing.get('queue_wait_ms'),
                timing_status='measured' if native else 'unavailable',input_scope='Exact submitted manufacturer crop.')
            if native:
                self.store.call(dict(id=stable_id(self.run,'manufacturer',sid),run_id=self.run,stage='manufacturer',
                    model_version=evidence['binding']['provenance'],track_id=sid,
                    outcome='failed' if decision.status=='malformed' else 'completed',**native))
                if native.get('load_time_ms') is not None:
                    self.store.call(dict(id=stable_id(self.run,'manufacturer_load',native.get('worker_pid',sid)),
                        run_id=self.run,stage='manufacturer_load',model_version=evidence['binding']['provenance'],
                        outcome='completed',duration_ms=native['load_time_ms'],start_ns=native.get('load_start_ns'),
                        completion_ns=native.get('load_completion_ns')))
                if native.get('queue_wait_ms') is not None:
                    self.store.call(dict(id=stable_id(self.run,'manufacturer_queue',sid),run_id=self.run,
                        stage='manufacturer_queue',model_version=evidence['binding']['provenance'],
                        outcome='completed',duration_ms=native['queue_wait_ms']))
            elif evidence['binding']['receipt'] is not None:
                self.store.call(dict(id=stable_id(self.run,'manufacturer',sid),run_id=self.run,stage='manufacturer_request_outcome',
                    model_version=evidence['binding']['provenance'],track_id=sid,outcome=evidence['outcome'],duration_ms=None,
                    execution_status='Native completion not confirmed; no fabricated inference duration.'))
        if plate is not None:
            selected=plate.selected;receipt=selected.receipt if selected else None
            resources=[v for v in self.plate_inputs.values() if v['track_id']==sid and v['operation']!='marker']
            inputs=[v['image'] for v in resources]
            event=self.plate_inputs.get(selected.operation_id,{}) if selected else {}
            timing=event.get('timing',{})
            selected_attempt=next((a for a in plate.transcript.attempts if selected and a.attempt_id==selected.attempt_id),None)
            quality_ms=self.plate_quality_times.get(selected_attempt.candidate_id) if selected_attempt else None
            self.record('plate',sid,receipt,kind='prediction' if selected else 'observation',
                predicted_label=selected.text if selected else None,image=event.get('image'),inputs=inputs,
                original_frame=self.observations.get((sid,receipt.frame_index),{}).get('original_frame') if receipt else None,
                model_version=selected.model.model_name+':v'+str(selected.model.version) if selected else None,
                inference_time_ms=timing.get('duration_ms'),quality_time_ms=quality_ms,outcome=plate.outcome,
                inference_start_ns=timing.get('start_ns'),inference_completion_ns=timing.get('completion_ns'),
                input_scope='Exact selected OCR input; related region inputs are attached.',
                detection_status=plate.omission_reason if not selected else 'associated')
            if selected and selected.category:
                self.record('plate_category',sid,receipt,predicted_label=selected.category,
                    image=event.get('image'),inputs=inputs,model_version='local-plate-policy-v1',
                    outcome='completed',input_scope='Plate policy category; separate from full-text match.')

    def close(self):
        pass  # Each evidence transaction owns and closes its SQLite connection.
