"""Console host for the new shared recognition engine, MP4 and RTSP alike."""
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
from uuid import UUID

ROOT = Path(__file__).resolve().parents[2]


def public_row(evidence, bundle):
    from .redis_scalar.codec import unpack_life
    life = unpack_life(evidence['life'])
    facts = {f['row_kind']:f['delivery'] for f in bundle['facts']}
    attrs = evidence['attributes']
    colour = facts.get('fact_vehicle_colour')
    from .plate_status import project_plate
    return dict(run_id=life.context.run_id, track_id=life.service_track_id, raw_track_id=life.raw_track_id,
        tracker_epoch=life.tracker_epoch, first_frame=life.first_receipt.frame_index,
        last_frame=life.last_receipt.frame_index, frames_observed=life.frames_observed, state='settled',
        vehicle_type=facts['fact_vehicle']['class']['name'], type_confidence=None,
        colour=dict(status=colour['state'] if colour else attrs['recognition_status'],
                    label=colour['colour'] if colour else None, confidence=None, source='local_lfm'),
        manufacturer=dict(status='accepted' if attrs['manufacturer'] != 'unknown' else attrs['recognition_status'],
            make=attrs['manufacturer'] if attrs['manufacturer']!='unknown' else None,
            reason=attrs['recognition_failure'], review_only=False, confidence=None,
            attempts=attrs['recognition_attempts']),
        plate=project_plate(evidence),
        delivery=dict(postgres_commit='not_configured', redis_ack='not_attempted', review_ack='acknowledged'))


def run(directory):
    from .console_stack import configure, console_device
    from .console_runs import atomic_json, timestamp
    from .db1_context import CameraRegistration, ModelRegistration, ResolutionInfo, RunContext
    from .frame_receipt import FrameReceiptClock, ClockPolicy
    from .real_video.input import MP4Input
    from .real_video.latest_rtsp import LatestRTSPInput
    from .real_video.rtsp import private_camera, RTSPBounds, InputFault, InputStopped
    from .real_video.plate_engine import CombinedEngine
    from .vehicle_observation import RunEnd
    from .gpu_lease import GPULease
    from .mohamed_lpr_redis_runtime import OwnedRedis
    from .recognition_transport import RecognitionCodec, RecognitionWorker
    from .redis_scalar.journal import PRODUCE
    from .real_video.recognition_engine import RecognitionEngine
    from ..recognition.config import paths
    import cv2
    configure(load_dsn=False)
    directory = Path(directory).resolve(strict=True)
    config = json.loads((directory/'config.json').read_text())
    status = json.loads((directory/'status.json').read_text())
    stop, watcher_done = threading.Event(), threading.Event()
    source = engine = owned = gpu = db_child = worker = None
    rows, sequence, analyzed, skips = [], 0, 0, 0
    epoch, epoch_count = 0, 0
    reason, state, error_code = 'error', 'failed', None
    db1 = os.environ.get('T1_CONSOLE_DB1') == 'fixture'
    def save(**changes):
        status.update(changes); atomic_json(directory/'status.json', status)
    def watch():
        while not watcher_done.wait(.03):
            if (directory/'stop').exists():
                stop.set()
                if source is not None and hasattr(source,'halt'): source.halt.set()
                return
    threading.Thread(target=watch, daemon=True).start()
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, lambda *_:stop.set())
    def collect():
        nonlocal sequence, epoch, epoch_count, worker, namespace, rows
        for evidence in engine.drain():
            raw = codec.make('engine.recognition.v3', evidence)
            sequence += 1
            epoch_count += 1
            # Retain immutable input before broker publication; replay never
            # executes a recognizer or allocates another vehicle identity.
            target = directory/f'recognition-{sequence}.json'
            fd = os.open(target, os.O_CREAT|os.O_EXCL|os.O_WRONLY, 0o600)
            with os.fdopen(fd,'wb') as out: out.write(raw)
            owned.client.eval(PRODUCE, 1, worker.source, f'{sequence}-0', raw, 16384)
            while worker.step(): pass
            message = codec.decode(codec.make('db1.delivery.v1', evidence))
            rows.append(public_row(evidence, message.body['bundle']))
            status['delivery']['review_ack']='acknowledged'
            save(rows=rows)
            if epoch_count >= 1024:
                commit_epoch()
                # Only retire a fully aggregated and, if configured, committed
                # namespace. All original scalar inputs remain on disk.
                for key in owned.client.scan_iter(match=worker.journal.prefix+'*'):
                    if owned.client.type(key)==b'stream':
                        for group in owned.client.xinfo_groups(key):
                            if group['pending'] or group['lag']:raise RuntimeError('epoch_not_quiescent')
                keys=list(owned.client.scan_iter(match=worker.journal.prefix+'*'))
                for start in range(0,len(keys),500):owned.client.delete(*keys[start:start+500])
                epoch += 1;epoch_count = 0
                namespace=base_namespace+'-e'+str(epoch)
                worker=RecognitionWorker(owned.client,codec,namespace)
                rows=rows[-int(os.environ.get('WAYCON_CONSOLE_ROW_HISTORY','2000')):]
                save(rows=rows,redis_epoch=epoch)
    def commit_epoch():
        nonlocal db_child
        if worker.journal.token:
            from redis.exceptions import ResponseError
            try:worker.journal.release()
            except ResponseError as error:
                if str(error)!='STALE_OWNER':raise
                worker.journal.token=None
        if not db1 or not epoch_count:return
        control=dict(org=str(org),identity_namespace=str(identity_namespace),redis_socket=str(owned.socket),
                     namespace=namespace,source_media=media)
        atomic_json(directory/'recognition-db1-control.json',control)
        atomic_json(directory/f'recognition-db1-control-e{epoch}.json',control)
        db_child=subprocess.Popen([sys.executable,'-B',str(ROOT/'scripts/vehicle_console_db1.py'),str(directory)],
            stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        progress_path=directory/'recognition-db1-progress.json'
        while db_child.poll() is None:
            if progress_path.exists():
                progress=json.loads(progress_path.read_text())
                for row in rows:row['delivery'].update(progress['tracks'].get(str(row['track_id']),{}))
                save(rows=rows,phase='Committing PostgreSQL recognition results')
            time.sleep(.1)
        if db_child.returncode:raise RuntimeError('db1_delivery_failed')
        progress=json.loads(progress_path.read_text())
        for row in rows:row['delivery'].update(progress['tracks'].get(str(row['track_id']),{}))
        atomic_json(directory/f'recognition-db1-progress-e{epoch}.json',progress)
        status['delivery'].update(postgres_commit='committed',redis_ack='acknowledged')
    try:
        if (directory/'stop').exists(): stop.set()
        if stop.is_set(): raise InputStopped('controlled_stop')
        gpu = GPULease()
        import torch
        torch.set_num_threads(1); cv2.setNumThreads(1)
        os.environ.update(YOLO_OFFLINE='true', YOLO_AUTOINSTALL='false', PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK='True')
        org, identity_namespace = UUID('12345678-1234-4234-8234-123456789012'), UUID('98c2adf9-264e-4ac2-b6a1-3239fa02ecbb')
        now = datetime.now(timezone.utc)
        detector = ModelRegistration(None, 'vehicle_yolo26.pt-sha256:'+RecognitionEngine._hash(paths()['vehicle_model']), 1, 'detect')
        context = RunContext(org, CameraRegistration(org,'console-'+config['run_id'],None,'Africa/Tunis',now,None),
            config['run_id'], 'file' if config['source_kind']=='mp4' else 'rtsp', now, 'vehicle-recognition-v3',
            ResolutionInfo(None,None,'vehicle',detector,None,'explicit local recognition development run',False),'development')
        clock = FrameReceiptClock(context, utc_clock=lambda:datetime.now(timezone.utc), monotonic_clock=time.monotonic, policy=ClockPolicy())
        deadline = time.monotonic()+config['max_seconds'] if config.get('max_seconds') else float('inf')
        every = config['sample_every']
        source = MP4Input(config['source'],clock,strict_eof=status.get('upload',{}).get('frame_count_exact',True)) if config['source_kind']=='mp4' else LatestRTSPInput(
            private_camera(config['source']),clock,deadline=deadline,stop=stop,every=every,directory=directory,
            bounds=RTSPBounds(open_seconds=12,transport=config.get('rtsp_transport','tcp')))
        save(phase='Opening input', recognition_version='vehicle-recognition-v3')
        source.__enter__()
        media = None
        if config['source_kind']=='mp4':
            media = dict(sha256=RecognitionEngine._hash(config['source']), bytes=Path(config['source']).stat().st_size,
                width=int(source.metadata['width']),height=int(source.metadata['height']),content_type='video/mp4',extension='.video')
            if status.get('upload') and media['sha256'] != status['upload']['sha256']:
                raise RuntimeError('uploaded_video_changed')
        save(phase='Loading local plate and vehicle attribute models')
        engine = CombinedEngine.recognition(context, clock, directory, attributes=config['manufacturer'],
            fps=source.metadata.get('fps') or 25, stop=stop,
            source_id='video-sha256:'+media['sha256'] if media else None)
        deadline = time.monotonic()+config['max_seconds'] if config.get('max_seconds') else float('inf')
        if config['source_kind']=='rtsp': source.deadline = deadline
        owned = OwnedRedis(os.environ.get('T1_REDIS_SERVER') or shutil.which('redis-server')).__enter__()
        codec = RecognitionCodec(org, identity_namespace)
        base_namespace = 'recognition-'+config['run_id'].replace('-','')[:24]
        namespace = base_namespace+'-e0'
        worker = RecognitionWorker(owned.client, codec, namespace)
        save(status='running', phase='Processing shared frames', models=engine.models,
             attributes_enabled=config['manufacturer'], rows=rows)
        while not stop.is_set():
            if time.monotonic() >= deadline: reason='time_limit'; break
            try: packet = source.read()
            except InputStopped:
                stop.set(); reason='controlled_stop'; break
            except InputFault as error:
                reason='time_limit' if time.monotonic()>=deadline else 'camera_outage'
                status['input_code']=str(error)
                break
            if packet is None: reason='eof'; break
            if config['source_kind']=='mp4' and packet.receipt.frame_index % every:
                skips += 1; continue
            try: boxes = engine.process(packet)
            except InputStopped:
                stop.set(); reason='controlled_stop'; break
            analyzed += 1
            preview = packet.pixels.copy()
            for box in boxes:
                x1,y1,x2,y2 = box['bbox']; cv2.rectangle(preview,(x1,y1),(x2,y2),(145,230,180),2)
            ok, image = cv2.imencode('.jpg',preview)
            if ok:
                temp=directory/'preview.tmp'; temp.write_bytes(image.tobytes()); temp.chmod(0o600); temp.replace(directory/'preview.jpg')
            with (directory/'analyzed-source.jsonl').open('a') as out:
                out.write(json.dumps(dict(frame_index=packet.receipt.frame_index,utc=packet.receipt.utc.isoformat(),
                    monotonic=packet.receipt.monotonic,media_pts=packet.receipt.media_pts))+'\n')
            collect()
            counts=source.counts() if config['source_kind']=='rtsp' else dict(decoded_frames=source.decoded,sampling_skips=skips,overload_drops=0)
            save(frames=source.decoded, analyzed_frames=analyzed,**counts,
                preview_revision=status.get('preview_revision',0)+1,
                counters=dict(engine.calls,tracks=sequence,canonical_redis_deliveries=sequence))
        if stop.is_set(): reason='controlled_stop'
        source.close()
        save(status='settling', phase='Settling recognition requests')
        # Stop prevents new native work, while already-returned evidence remains
        # available. Normal EOF/outage follows the supplied finalization policy.
        engine.finish(RunEnd.END_OF_INPUT if reason=='eof' else RunEnd.CAMERA_OUTAGE if reason=='camera_outage' else RunEnd.SERVICE_STOP)
        collect()
        state = 'stopped' if stop.is_set() else 'completed' if reason=='eof' else 'partial'
        commit_epoch()
        status['counters'] = dict(engine.calls, tracks=sequence, canonical_redis_deliveries=sequence)
    except Exception as error:
        import traceback
        if isinstance(error, InputStopped) or stop.is_set():
            state, reason = 'stopped', 'controlled_stop'
        else:
            state, error_code = 'failed', 'db1_delivery_failed' if str(error)=='db1_delivery_failed' else 'backend_execution_failed'
        atomic_json(directory/'failure.json', dict(type=type(error).__name__,pipeline_code=getattr(error,'code',None),
            frames=[dict(file=Path(f.filename).name,line=f.lineno,function=f.name) for f in traceback.extract_tb(error.__traceback__)]))
    finally:
        watcher_done.set()
        cleanup=dict(decoder_closed=True,models_closed=True,redis_removed=True,private_camera_removed=True,native_children_exited=True)
        for value, key in ((source,'decoder_closed'),(engine,'models_closed')):
            if value:
                try:value.close()
                except Exception:cleanup[key]=False
        if owned:
            try:owned.__exit__(None,None,None)
            except Exception:cleanup['redis_removed']=False
        if config['source_kind']=='rtsp':
            try:shutil.rmtree(Path(config['source']).parent)
            except FileNotFoundError:pass
            except Exception:cleanup['private_camera_removed']=False
        if db_child and db_child.poll() is None: db_child.wait()
        if gpu: gpu.close()
        if not all(cleanup.values()): state,error_code='failed','cleanup_incomplete'
        counts=source.counts() if source and config['source_kind']=='rtsp' else dict(decoded_frames=source.decoded if source else 0,sampling_skips=skips,overload_drops=0)
        save(status=state,phase='Recognition '+state, rows=rows, error_code=error_code, ended_at=timestamp(),
            frames=source.decoded if source else 0,decoded_frames=source.decoded if source else 0,
            end_reason=reason, full_video_complete=config['source_kind']=='mp4' and state=='completed' and reason=='eof',
            cleanup=cleanup,stop_requested=stop.is_set(),analyzed_frames=analyzed,
            sampling_skips=counts['sampling_skips'],overload_drops=counts['overload_drops'])
    return 1 if state=='failed' else 0
