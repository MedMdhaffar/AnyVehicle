"""One owned console run, using shared Input, CombinedEngine and T1 v2 workers."""
from datetime import datetime, timezone
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def run(directory):
    if not os.environ.get('WAYCON_CONSOLE_TEST_PROFILE'):
        from vehicle.common.recognition_run import run as recognition_run
        return recognition_run(directory)
    from vehicle.common.console_stack import configure, model_paths, console_device
    configure(load_dsn=False)
    from vehicle.common.console_runs import atomic_json, timestamp, metadata_turn, review_acknowledgments
    directory = Path(directory).resolve(strict=True)
    config = json.loads((directory/'config.json').read_text())
    status = json.loads((directory/'status.json').read_text())
    source = None
    stop, watcher_done = threading.Event(), threading.Event()
    if (directory/'stop').exists():stop.set()
    def watch():
        while not watcher_done.wait(.03):
            if (directory/'stop').exists():
                stop.set()
                if source is not None and hasattr(source,'halt'):source.halt.set()
                return
    threading.Thread(target=watch,daemon=True).start()
    signal.signal(signal.SIGTERM,lambda *_:stop.set())
    signal.signal(signal.SIGINT,lambda *_:stop.set())
    def save(**changes):
        status.update(changes)
        atomic_json(directory/'status.json',status)
    source = engine = vehicle = plate_detector = owned = ledger = maker_runtime = publisher = crops = E = None
    db1_child = gpu_lease = None
    workers, bindings, originals, rows = [], {}, {}, {}
    private_camera_dir = Path(config['source']).parent if config['source_kind']=='rtsp' else None
    final_state, error_code, reason = 'completed', None, 'error'
    every = config.get('sample_every', 5 if config['source_kind']=='rtsp' else 1)
    analyzed = skips = 0
    os.environ.update(YOLO_OFFLINE='true',YOLO_AUTOINSTALL='false',COLOR_AGENT_ENABLED='false',
        PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK='True',YOLO_CONFIG_DIR=str(directory/'model-settings'))
    if config['source_kind']=='mp4':os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS']='protocol_whitelist;file,pipe'
    try:
        from vehicle.common.gpu_lease import GPULease
        gpu_lease = GPULease()
        db1_enabled = os.environ.get('T1_CONSOLE_DB1') == 'fixture'
        if db1_enabled:
            status['delivery'].update(postgres_commit='pending', redis_ack='not_attempted')
        if stop.is_set():
            final_state, reason = 'stopped', 'controlled_stop'
            return 0
        from vehicle.common.db1_context import CameraRegistration, ModelRegistration, ResolutionInfo, RunContext
        from vehicle.common.frame_receipt import FrameReceiptClock, ClockPolicy
        from vehicle.common.observation_evidence import RecognitionIdentity
        from vehicle.common.real_video.input import MP4Input
        from vehicle.common.real_video.rtsp import RTSPBounds, private_camera, InputStopped, InputFault
        from vehicle.common.real_video.latest_rtsp import LatestRTSPInput
        from vehicle.common.real_video.local import LocalDetector, pixel_operations, fingerprint
        from vehicle.common.real_video.plate_local import NativePlateDetector, detector_artifact
        from vehicle.common.real_video.plate_engine import CombinedEngine, VideoPlateLimits
        from vehicle.common.engine_plate import native_access
        from vehicle.common.engine_plate_local import LocalPlateBinding, artifact
        from vehicle.common.redis_scalar.worker import load_request_codec, Worker
        from vehicle.common.redis_scalar.journal import Journal
        from vehicle.common.plate_requests.worker import RequestWorker
        from vehicle.common.plate_requests.host import V2Publisher
        from vehicle.common.plate_metadata import PlateMetadata
        from vehicle.common.mohamed_lpr_redis_runtime import OwnedRedis
        from vehicle.common.manufacturer.engine import ManufacturerLedger, SelectionConfig
        from vehicle.common.manufacturer.local import create_runtime
        from vehicle.common.manufacturer.streams import ReviewCodec, ReviewWorker, publish, SOURCE, RESULT
        from vehicle.common.console_projection import project_track
        from vehicle.common.console_crops import ConsoleCrops
        from vehicle.common.vehicle_observation import RunEnd
        import cv2
        import torch
        torch.set_num_threads(1);cv2.setNumThreads(1)
        if stop.is_set():raise InputStopped('controlled_stop')
        org, namespace_id = UUID('12345678-1234-4234-8234-123456789012'), UUID('98c2adf9-264e-4ac2-b6a1-3239fa02ecbb')
        paths = model_paths()
        now = datetime.now(timezone.utc)
        model = paths['vehicle_model']
        context = RunContext(org,CameraRegistration(org,'console-'+config['run_id'],None,'Africa/Tunis',now,None),
            config['run_id'],'file' if config['source_kind']=='mp4' else 'rtsp',now,'vehicle-console-v1',
            ResolutionInfo(None,None,'vehicle',ModelRegistration(None,'local-yolo26-sha256:'+fingerprint(model)['sha256'],1,'detect'),
                           None,'console backend execution; no production DB1 registration',False),'development')
        clock = FrameReceiptClock(context,utc_clock=lambda:datetime.now(timezone.utc),monotonic_clock=time.monotonic,policy=ClockPolicy())
        deadline = time.monotonic()+config['max_seconds'] if config.get('max_seconds') else float('inf')
        source = MP4Input(config['source'],clock,strict_eof=status.get('upload',{}).get('frame_count_exact',True)) if config['source_kind']=='mp4' else LatestRTSPInput(
            private_camera(config['source']),clock,deadline=deadline,stop=stop,
            every=every, directory=directory,
            bounds=RTSPBounds(open_seconds=12, transport=config.get('rtsp_transport','tcp')))
        save(phase='Opening input')
        source.__enter__()
        source_fingerprint = fingerprint(config['source']) if config['source_kind']=='mp4' else None
        if source_fingerprint and status.get('upload') and source_fingerprint['sha256']!=status['upload']['sha256']:
            raise RuntimeError('uploaded_video_changed')
        if stop.is_set():raise InputStopped('controlled_stop')
        fixture = os.environ.get('WAYCON_CONSOLE_TEST_PROFILE')
        if fixture not in (None, 'empty', 'sequential'):raise RuntimeError('unknown_fixture_profile')
        if config['manufacturer'] and fixture == 'sequential':
            from vehicle.common.tests.console_empty_runtime import FixtureManufacturer
            maker_runtime = FixtureManufacturer(stop=stop)
        elif config['manufacturer']:
            from vehicle.common.manufacturer.local import ManufacturerUnavailable
            save(phase='Loading manufacturer model')
            try:
                maker_runtime = create_runtime(os.environ.get('T1_MANUFACTURER_LOCK',ROOT/'vehicle/common/manufacturer/assets.json'),
                                               timeout=30,stop=stop)
            except ManufacturerUnavailable:
                # Clear, path-free failure; never silently disable a requested capability.
                error_code = 'manufacturer_unavailable'
                raise
        policy = os.environ.get('T1_MANUFACTURER_SELECTION','quality_lifetime_v1')
        from vehicle.common.manufacturer.crop_quality import QUALITY_POLICIES
        quality_name = os.environ.get('T1_MANUFACTURER_QUALITY','v2')
        if quality_name not in QUALITY_POLICIES:raise RuntimeError('unknown_quality_policy')
        ledger = ManufacturerLedger(enabled=config['manufacturer'],runtime=maker_runtime,
                                    journal=directory/'manufacturer.sqlite',stop=stop,max_lifetimes=65536,
                                    selection=SelectionConfig(policy=policy,quality=QUALITY_POLICIES[quality_name]))
        crops = ConsoleCrops(directory)
        crops.bind_manufacturer(ledger, maker_runtime)
        fps = source.metadata['fps']
        import math
        fps = fps if math.isfinite(fps) and fps>0 else 25
        test_profile = fixture is not None
        retained_db1 = db1_enabled and config['source_kind']=='mp4' and not test_profile
        if retained_db1:
            from dataclasses import asdict
            from vehicle.common.db1_rows.make import MakeActivation
            from vehicle.common.manufacturer.test_database import CATALOGUE
            activation = MakeActivation('user-authorized test integration; supplied DB1 development catalogue v0',
                RecognitionIdentity('LFM2.5-VL-450M-runtime-sha256:'+maker_runtime.provenance,1,'vlm'),
                (maker_runtime.provenance,), CATALOGUE, 'selected_crop_receipt','submitted_crops','omit') if maker_runtime else None
            atomic_json(directory/'manufacturer-registration.json',dict(activation=asdict(activation) if activation else None,
                runtime_lock=maker_runtime.lock if maker_runtime else None, runtime_code=maker_runtime.code if maker_runtime else None))
        from vehicle.common.evaluation_capture import EvaluationRecorder
        evaluation = EvaluationRecorder(directory, context, source_id='video-sha256:'+source_fingerprint['sha256']) if config['source_kind']=='mp4' and not test_profile else None
        # Fixture data may reach only disposable DB1 fixtures, and only when a test asks explicitly.
        if test_profile and db1_enabled and not (fixture=='sequential' and os.environ.get('WAYCON_CONSOLE_TEST_DB1_FIXTURE')=='1'):
            raise RuntimeError('fixture_profile_requires_console_only')
        if fixture == 'empty':
            from vehicle.common.tests.console_empty_runtime import EmptyVehicle, EmptyPlate
            vehicle, plate_detector, bindings = EmptyVehicle(tracked=config['source_kind']=='rtsp'), EmptyPlate(), {}
        elif fixture == 'sequential':
            from vehicle.common.tests.console_empty_runtime import SequentialVehicle, SequentialPlate, EmptyPlate, sequential_bindings
            vehicle = SequentialVehicle(int(os.environ.get('WAYCON_CONSOLE_TEST_VEHICLES','40')))
            plate_detector = SequentialPlate(vehicle) if os.environ.get('WAYCON_CONSOLE_TEST_PLATES')=='tentative' else EmptyPlate()
            bindings = sequential_bindings()
        else:
            with native_access():
                if evaluation:
                    vehicle,_=evaluation.measure('vehicle_load',context.resolution.detector.model_name,
                        lambda:LocalDetector(model,paths['vehicle_tracker'],device=console_device(),frame_rate=fps,threads=1),
                        synchronize=lambda:torch.cuda.synchronize(console_device()))
                else:vehicle = LocalDetector(model,paths['vehicle_tracker'],device=console_device(),frame_rate=fps,threads=1)
            plate_detector = NativePlateDetector(detector_artifact(paths['plate_model'],
                                        paths['plate_tracker'],frame_rate=fps),frame_rate=fps)
            bindings = {kind:LocalPlateBinding(artifact(kind,path)) for kind,path in (
                ('ocr',paths['ocr_dir']), ('detect',paths['marker_model']))}
        estimate, quality = pixel_operations()
        engine = CombinedEngine(context,clock,vehicle,plate_detector,bindings,
            algorithm=RecognitionIdentity('t1-local-hsv-console',1,'classify'),quality_algorithm='t1-colour-quality',
            plate_quality_algorithm='t1-plate-crop-quality',estimate=estimate,quality_fn=quality,manufacturer=ledger,
            limits=VideoPlateLimits(streaming=True,max_frames=None,max_seconds=None,max_api_calls=None,max_raw_total=None))
        if evaluation:evaluation.bind(engine)
        crops.bind_plate(engine.plate)
        codec = load_request_codec(org,namespace_id)
        redis_server = os.environ.get('T1_REDIS_SERVER') or shutil.which('redis-server')
        if not redis_server:raise RuntimeError('missing_redis_server')
        owned = OwnedRedis(redis_server).__enter__()
        # Per-epoch streaming profile on the run-owned Redis (removed after the run):
        # bounded and fail-closed; quiescent epoch rotation keeps long runs inside it.
        from vehicle.common.console_epochs import open_epoch, usage, quiescent, retire, ROTATE_AT
        profile = dict(max_events=16384,max_vehicles=4096,max_stream=16384)
        if fixture and os.environ.get('WAYCON_CONSOLE_TEST_REDIS_PROFILE')=='small':
            profile = dict(max_events=64,max_vehicles=16,max_stream=64)  # explicit test-only fixture profile
        def new_epoch(index):
            return open_epoch(index,client=owned.client,codec=codec,org=org,run_id=config['run_id'],engine=engine,profile=profile)
        E = new_epoch(0)
        workers = E.workers  # cleanup reference; rebound on rotation
        epochs = []          # retired epoch summaries (scalars only)
        totals = dict(deliveries=0,review_results=0,tracks=0,pruned_rows=0)
        row_history = int(os.environ.get('WAYCON_CONSOLE_ROW_HISTORY','2000'))
        row_epoch = {}
        tracker = getattr(getattr(vehicle,'target',None),'tracker',None)
        initial_tracking = dict(frame_id=getattr(tracker,'frame_id',0),
            tracked=len(getattr(tracker,'tracked_stracks',())),lost=len(getattr(tracker,'lost_stracks',())),scope=config['run_id'])
        from collections import deque
        runtime_frames = deque(maxlen=128)
        def public_rows():
            return [rows[s] for s in sorted(rows)]
        def update_review_delivery():
            if not config['manufacturer']:return
            # Only unacknowledged reviews are re-queried; acknowledged state is final.
            states = review_acknowledgments(owned.client,E.review.journal,
                {sid:(entry,rows[sid]['manufacturer']['request_id']) for sid,entry in E.review_published.items() if sid not in E.review_done})
            for sid,state in states.items():
                rows[sid]['delivery']['review_ack'] = state
                if state=='acknowledged':E.review_done.add(sid)
            status['delivery']['review_ack'] = ('acknowledged' if (E.review_published or epochs) and
                len(E.review_done)==len(E.review_published) else 'pending')
        blocked = {}
        def settle_turn():
            """One bounded, non-native-overlapping settlement step for pending closures."""
            engine.poll_manufacturer()
            capture_closed()
            for worker in E.workers:metadata_turn(worker)
            update_review_delivery()
            for sid,closed in tuple(engine.pending.items()):
                blocked_reason = engine.plate.blocked_reason(closed) if sid not in E.publisher.plans else None
                if blocked_reason is not None:
                    # Fail closed for this lifetime only: no DB1/Redis delivery, explicit
                    # public state and counter, private crops released, run continues.
                    rows[sid]['state'] = 'blocked'
                    rows[sid]['plate'].update(status='blocked', diagnostic=dict(reason=blocked_reason.lower(), frame=None))
                    rows[sid]['delivery'].update(redis_ack='not_attempted')
                    if db1_enabled:rows[sid]['delivery']['postgres_commit']='not_attempted'
                    blocked[blocked_reason] = blocked.get(blocked_reason,0)+1
                    engine.plate.discard(sid)
                    engine.acknowledge(sid)
                    originals.pop(sid,None)
                    row_epoch[sid] = E.index
                    continue
                if E.publisher.publish(closed):
                    plan = E.publisher.plans[sid]
                    plate = PlateMetadata().evaluate(closed.owner.lifetime,plan.transcript).result
                    crops.plate(sid, plate)
                    row,bundle = project_track(closed,codec.v1.v0,plate=plate)
                    if evaluation:evaluation.finalize(closed,plate,bundle.payload.value())
                    if retained_db1 and config['manufacturer']:
                        fd=os.open(directory/f'manufacturer-evidence-{sid}.json',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
                        with os.fdopen(fd,'w') as retained:retained.write(closed.manufacturer_json)
                    row['delivery'].update(rows[sid]['delivery'])
                    rows[sid] = row
                    fd = os.open(directory/f'delivery-{sid}.json',os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600)
                    with os.fdopen(fd,'wb') as output:output.write(bundle.payload.canonical)
                    engine.acknowledge(sid)
                    originals.pop(sid,None)
                    E.releasing.add(sid);E.lifetimes += 1;row_epoch[sid] = E.index
            # Bounded active state: delivered lifetimes leave the publisher once
            # every retained publication byte is confirmed.
            for sid in tuple(E.releasing):
                if E.publisher.release(sid):E.releasing.discard(sid)
            save(rows=public_rows())
        def settle_pending(seconds):
            # Like drive_video: model work for closed lifetimes never overlaps the
            # next frame's detector. Waiting is bounded; evidence is never dropped.
            until = time.monotonic()+seconds
            while engine.pending and not stop.is_set():
                if time.monotonic()>=until:raise RuntimeError('settlement_deadline')
                settle_turn()
                if engine.pending:time.sleep(.01)
        def preview(packet, vehicles=()):
            pixels = packet.pixels.copy()
            for observation in vehicles:
                x1,y1,x2,y2 = observation.bbox
                cv2.rectangle(pixels,(x1,y1),(x2,y2),(145,230,180),2)
                cv2.putText(pixels,f'#{observation.service_track_id}',(max(0,x1),max(22,y1-8)),
                            cv2.FONT_HERSHEY_SIMPLEX,.8,(145,230,180),2,cv2.LINE_AA)
            ok, image = cv2.imencode('.jpg',pixels,[cv2.IMWRITE_JPEG_QUALITY,80])
            if ok:
                temp = directory/'preview.tmp'
                temp.write_bytes(image.tobytes());os.chmod(temp,0o600);temp.replace(directory/'preview.jpg')
                save(preview_revision=status['preview_revision']+1)
        def capture_closed():
            crops.diagnostics(engine.plate)
            for sid,closed in engine.pending.items():
                originals.setdefault(sid,closed)
                if sid not in rows:
                    rows[sid],_ = project_track(closed,codec.v1.v0,incomplete='stopped' if stop.is_set() else 'pending')
                    if db1_enabled:rows[sid]['delivery']['postgres_commit']='pending'
                if config['manufacturer'] and sid not in E.review_published and sid not in row_epoch:
                    raw = E.review.codec.encode(closed)
                    entry = f'{len(E.review_published)+1}-0'
                    publish(owned.client,E.review.journal,raw,entry)
                    E.review_published[sid] = entry
            metadata_turn(E.review)
            update_review_delivery()
            save(rows=public_rows())
        db1_totals = dict(committed=0,acknowledged=0)
        source_media_cache = {}
        def deliver_db1(epoch):
            """Commit-before-ACK delivery of one epoch's canonical stream (owned child)."""
            nonlocal db1_child, error_code
            if 'media' not in source_media_cache:
                source_media = None
                if config['source_kind']=='mp4':
                    source_media = {k:v for k,v in source_fingerprint.items() if k!='path'}
                    source_media.update(width=int(source.metadata['width']), height=int(source.metadata['height']))
                    source_media.update(content_type=status.get('upload',{}).get('content_type','video/mp4'), extension='.video')
                source_media_cache['media'] = source_media
            suffix = '' if epoch.index==0 else f'-e{epoch.index}'
            control_name, progress_name = f'db1-control{suffix}.json', f'db1-progress{suffix}.json'
            published = set()
            raws = owned.client.xrange(epoch.journal.key('db1.delivery.v1'))
            for _,fields in raws:
                raw = fields[b'json']
                sid = codec.v1.decode(raw).life.service_track_id
                published.add(sid)
                fd = os.open(directory/f'db1-wire-{sid}.json', os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o600)
                with os.fdopen(fd,'wb') as retained:retained.write(raw)
            members = {sid for sid,e in row_epoch.items() if e==epoch.index}
            for sid in members:
                if sid in rows:
                    state = 'pending' if sid in published else 'not_attempted'
                    rows[sid]['delivery'].update(postgres_commit=state,redis_ack=state)
            atomic_json(directory/control_name, dict(org=str(org), identity_namespace=str(namespace_id),
                redis_socket=str(owned.socket), namespace=epoch.namespace, deliveries=len(raws),
                source_media=source_media_cache['media']))
            if db1_child is None:
                db1_child = subprocess.Popen([sys.executable, '-B', str(ROOT/'scripts/vehicle_console_db1.py'), str(directory),
                    control_name, progress_name], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            def db1_progress():
                path = directory/progress_name
                if not path.exists():return
                progress = json.loads(path.read_text())
                for sid in members:
                    if sid in rows:rows[sid]['delivery'].update(progress['tracks'].get(str(sid), {}))
                for key in ('postgres_commit','redis_ack'):
                    values = [rows[sid]['delivery'][key] for sid in published if sid in rows]
                    target = 'committed' if key=='postgres_commit' else 'acknowledged'
                    status['delivery'][key] = target if values and all(v==target for v in values) else (
                        'no_deliveries' if not values and not epochs else status['delivery'].get(key) if not values else 'pending')
                save(rows=public_rows())
            # Stop never kills a writer mid-transaction or releases its run slot.
            while db1_child.poll() is None:
                db1_progress()
                if retained_db1 and (directory/progress_name).exists():
                    progress=json.loads((directory/progress_name).read_text())
                    if progress['status']=='completed':break
                    if progress['status']=='failed':
                        error_code='db1_delivery_failed'
                        raise RuntimeError('db1_delivery_failed')
                time.sleep(.03)
            db1_progress()
            if db1_child.returncode:
                error_code='db1_delivery_failed'
                raise RuntimeError('db1_delivery_failed')
            if not retained_db1:db1_child = None
            return published
        def prune_rows():
            """Bounded public history: drop the oldest finished rows of retired epochs."""
            if len(rows) <= row_history:return
            retired = {e['index'] for e in epochs}
            for sid in sorted(rows):
                if len(rows) <= row_history:break
                if row_epoch.get(sid) in retired and rows[sid]['state'] in ('settled','blocked'):
                    del rows[sid];row_epoch.pop(sid,None);totals['tracks'] += 1;totals['pruned_rows'] += 1
        def maybe_rotate():
            """Quiescent epoch rotation: deliver, retire and replace a half-used epoch."""
            nonlocal E, workers
            if fixture and os.environ.get('WAYCON_CONSOLE_TEST_NO_ROTATION')=='1':return  # test-only: prove the old ceiling
            if stop.is_set() or usage(owned.client,E,profile) < ROTATE_AT:return
            if not quiescent(owned.client,E,engine,metadata_turn):return
            update_review_delivery()
            if len(E.review_done) != len(E.review_published):return
            if db1_enabled:
                published = deliver_db1(E)
                members = [sid for sid,e in row_epoch.items() if e==E.index]
                db1_totals['committed'] += sum(rows[s]['delivery']['postgres_commit']=='committed' for s in members if s in rows)
                db1_totals['acknowledged'] += sum(rows[s]['delivery']['redis_ack']=='acknowledged' for s in members if s in rows)
            totals['deliveries'] += owned.client.xlen(E.journal.key('db1.delivery.v1'))
            totals['review_results'] += owned.client.xlen(E.review.journal.key(RESULT))
            summary = dict(index=E.index, lifetimes=E.lifetimes, usage=round(usage(owned.client,E,profile),4),
                           deliveries=owned.client.xlen(E.journal.key('db1.delivery.v1')))
            summary['deleted_keys'] = retire(owned.client,E)
            epochs.append(summary)
            E = new_epoch(E.index+1);workers = E.workers
            prune_rows()
            save(redis_epoch=E.index)
        declared=source.metadata.get('declared_frames',0)
        save(status='running',phase='Processing shared vehicle frames',tracker_fps=fps,
             total_frames=status.get('upload',{}).get('total_frames',int(declared) if math.isfinite(declared) and declared>0 else None))
        def counts():
            value = source.counts() if config['source_kind']=='rtsp' else dict(decoded_frames=source.decoded,sampling_skips=skips,overload_drops=0)
            return dict(value,frames=source.decoded,analyzed_frames=analyzed)
        while True:
            if stop.is_set():reason='controlled_stop';break
            if time.monotonic()>=deadline:reason='time_limit';break
            try:
                packet = source.read()
            except InputStopped:
                reason='controlled_stop';break  # Same semantics as the Stop check above.
            except InputFault as error:
                if config['source_kind']!='rtsp':raise
                # Mirror the standalone RTSP host: a lost camera ends this run with its own
                # cause; lifetimes already observed are closed, settled and delivered.
                # Reconnection is a new run (new Input, clock, Engine and tracker).
                reason='time_limit' if time.monotonic()>=deadline else 'camera_outage'
                status['input_code']=str(error)  # fixed, non-sensitive InputFault code
                break
            if packet is None:reason='eof';break
            if stop.is_set():reason='controlled_stop';break
            if config['source_kind']=='mp4' and packet.receipt.frame_index % every:
                skips += 1
                if skips % 100 == 0:save(**counts())
                continue
            # Preview is derived from the same Input; never a browser decoder or
            # another backend video reader. First frame is visible before VLM work.
            preview(packet)
            vehicle.source_index = plate_detector.source_index = packet.receipt.frame_index
            if test_profile:time.sleep(float(os.environ.get('WAYCON_CONSOLE_TEST_DELAY','0')))
            engine.process(packet)
            analyzed += 1
            # Disk evidence records actual successful Engine handoffs. It never
            # reconstructs selected indexes from a total or retains frame pixels.
            with (directory/'analyzed-source.jsonl').open('a') as evidence:
                evidence.write(json.dumps(dict(frame_index=packet.receipt.frame_index,
                    utc=packet.receipt.utc.isoformat(),monotonic=packet.receipt.monotonic,
                    media_pts=packet.receipt.media_pts,connection_id=config['run_id']))+'\n')
            runtime_frames.append(dict(frame=packet.receipt.frame_index,
                vehicle_rows=len(engine.colour.detector.rows),
                observed_vehicles=len(engine.colour.pipeline.observation_frame.vehicles)))
            atomic_json(directory/'native-runtime.json', dict(source_fps=fps, imported_opencv=cv2.__version__,
                vehicle=vehicle.placement, frames=list(runtime_frames), initial_tracking=initial_tracking, evidence_admission=dict(
                    colour_not_admitted=engine.colour.evidence_not_admitted,plate_not_admitted=engine.plate.evidence_not_admitted),
                vehicle_prediction_calls=vehicle.prediction_calls, vehicle_forward_calls=vehicle.forward_calls))
            preview(packet,engine.colour.pipeline.observation_frame.vehicles)
            save(**counts())
            capture_closed()
            # Progressive settlement keeps active state bounded for long runs.
            if engine.pending:settle_pending(300)
            maybe_rotate()
        source.close()
        save(**counts(),end_reason=reason)
        save(status='settling',phase='Joining Plate, Colour and manufacturer decisions')
        engine.finish({'eof':RunEnd.END_OF_INPUT,'camera_outage':RunEnd.CAMERA_OUTAGE}.get(reason,RunEnd.SERVICE_STOP),trigger=reason)
        capture_closed()
        settle_pending(300)
        # Confirm the real canonical Redis delivery, independent of review ACK.
        for _ in range(64):
            progress = [metadata_turn(worker) for worker in E.workers]
            if not any(progress):break
        update_review_delivery()
        if config['manufacturer'] and not rows:status['delivery']['review_ack']='not_attempted'
        deliveries = totals['deliveries']+owned.client.xlen(E.journal.key('db1.delivery.v1'))
        status['counters'] = dict(vehicle_detector_calls=engine.audit.calls.get('vehicle',0),
            plate_detector_calls=engine.audit.calls.get('plate',0),manufacturer_calls=ledger.calls,
            plate_model_calls={kind:dict(binding.calls) for kind,binding in bindings.items()},
            tracks=totals['tracks']+len(rows),canonical_redis_deliveries=deliveries,
            manufacturer_review_results=totals['review_results']+owned.client.xlen(E.review.journal.key(RESULT)),
            released_lifetimes=engine.colour.released, blocked_lifetimes=dict(blocked),
            redis_epochs=len(epochs)+1, retired_epochs=list(epochs), public_rows_pruned=totals['pruned_rows'])
        if maker_runtime is not None:
            # Scalar runtime measurements only; no crop, text or path.
            metrics = list(getattr(maker_runtime,'metrics',[]))
            seconds = sorted(m['seconds'] for m in metrics if isinstance(m,dict) and 'seconds' in m)
            vram = [m['process_vram_mib'] for m in metrics if isinstance(m,dict) and m.get('process_vram_mib') is not None]
            status['manufacturer_runtime'] = dict(mode=getattr(maker_runtime,'mode','legacy'),
                selection=ledger.selection.policy, quality_version=ledger.selection.quality.version,
                quality_config_sha256=ledger.selection.quality.fingerprint(),
                model_loads=len(getattr(maker_runtime,'starts',[])) or len(getattr(maker_runtime,'children',[])),
                restarts=getattr(maker_runtime,'restarts',None), calls=ledger.calls,
                median_call_seconds=seconds[len(seconds)//2] if seconds else None,
                max_call_seconds=seconds[-1] if seconds else None,
                peak_process_vram_mib=max(vram) if vram else None,
                quality_cpu_seconds=round(ledger.quality_seconds,3), crops=ledger.stats)
        if stop.is_set():
            final_state='stopped'
            for row in rows.values():
                if row['state'] not in ('settled','blocked'):row['plate']['status']='stopped';row['state']='incomplete'
        elif status['counters']['tracks']-sum(blocked.values())!=deliveries:raise RuntimeError('canonical_delivery_incomplete')
        if reason in ('time_limit','camera_outage') and final_state!='stopped':final_state='partial'
        if db1_enabled:
            save(phase='Committing DB1 test database', rows=public_rows())
            deliver_db1(E)
            status['counters']['db1_committed_tracks'] = db1_totals['committed']+sum(row['delivery']['postgres_commit']=='committed' for sid,row in rows.items() if row_epoch.get(sid)==E.index)
            status['counters']['db1_acknowledged_tracks'] = db1_totals['acknowledged']+sum(row['delivery']['redis_ack']=='acknowledged' for sid,row in rows.items() if row_epoch.get(sid)==E.index)
        save(rows=public_rows(),counters=status['counters'])
    except Exception as error:
        import traceback
        from vehicle.common.real_video.rtsp import InputFault
        # Enumerated journal Lua replies carry no private data; record them as codes.
        lua_codes = {'CAPACITY_EXCEEDED','STALE_OWNER','PAYLOAD_CONFLICT','REPAIR_REQUIRED','REVISION_CONFLICT',
                     'OUTPUT_MISSING','INTENT_CHANGED','PUBLICATION_CONFLICT'}
        pipeline_code = getattr(error,'code',None) or (str(error) if type(error).__name__=='ResponseError' and str(error) in lua_codes else None)
        atomic_json(directory/'failure.json',dict(type=type(error).__name__,
            pipeline_code=pipeline_code,
            input_code=str(error) if isinstance(error, InputFault) else None,
            frames=[dict(file=Path(f.filename).name,line=f.lineno,function=f.name)
                    for f in traceback.extract_tb(error.__traceback__)]))
        from vehicle.common.real_video.rtsp import InputStopped
        if isinstance(error,InputStopped) or stop.is_set():final_state='stopped'
        else:
            final_state='failed'
            # Never stringify decoder/native exceptions or request configuration.
            error_code=error_code or ('input_unavailable' if source is not None and engine is None else 'backend_execution_failed')
        reason = 'controlled_stop' if stop.is_set() else ('camera_outage' if config['source_kind']=='rtsp' and isinstance(error,InputFault) else 'error')
    finally:
        if db1_child is not None:
            (directory/'db1-finish').touch(mode=0o600,exist_ok=True)
            db1_child.wait()
        cleanup = dict(decoder_closed=True,models_closed=True,redis_removed=True,
                       private_camera_removed=True,native_children_exited=True)
        if source is not None:
            try:source.close()
            except Exception:cleanup['decoder_closed']=False
        if engine is not None:
            for life in engine.plate.lives.values():
                if life.adapter is not None:
                    try:life.adapter.stop();life.adapter.cleanup(discard=True)
                    except Exception:cleanup['models_closed']=False
            if engine.audit.active is None:
                for resource in engine.plate.buffers:resource.release()
            try:engine.close();cleanup['models_closed'] = cleanup['models_closed'] and engine.closed
            except Exception:cleanup['models_closed']=False
        elif vehicle is not None:
            try:vehicle.close()
            except Exception:cleanup['models_closed']=False
        if ledger is not None:
            if stop.is_set():
                try:ledger.cancel()
                except Exception:pass
            ledger.close()
        if maker_runtime is not None:
            try:
                if hasattr(maker_runtime,'close'):maker_runtime.close()
            except Exception:pass
            cleanup['native_children_exited']=maker_runtime.cleaned
        for worker in (E.workers if E is not None else workers):
            if worker.journal.token:
                try:worker.journal.release()
                except Exception:pass
        if owned is not None:
            try:owned.__exit__(None,None,None);cleanup['redis_removed']=owned.cleaned
            except Exception:cleanup['redis_removed']=False
        if private_camera_dir is not None:
            try:shutil.rmtree(private_camera_dir)
            except FileNotFoundError:pass
            except Exception:cleanup['private_camera_removed']=False
        watcher_done.set()
        if crops is not None:
            try:crops.close()
            except Exception:cleanup['models_closed']=False
        if not all(cleanup.values()):final_state,error_code='failed','cleanup_incomplete'
        if gpu_lease is not None:gpu_lease.close()
        if source is not None:
            status.update(frames=source.decoded,decoded_frames=source.decoded,analyzed_frames=analyzed,sampling_skips=skips)
            if config['source_kind']=='rtsp' and hasattr(source,'counts'):status.update(source.counts())
        save(status=final_state,phase={'completed':'Full video complete' if config['source_kind']=='mp4' else 'Stream ended',
             'partial':'Partial run: camera outage' if reason=='camera_outage' else 'Partial run: timeout','stopped':'Stopped safely: partial run','failed':'Run failed: partial run'}[final_state],
             end_reason=reason,full_video_complete=(config['source_kind']=='mp4' and reason=='eof' and final_state=='completed'),
             error_code=error_code,ended_at=timestamp(),cleanup=cleanup,stop_requested=stop.is_set())
    return 0 if final_state in {'completed','stopped','partial'} else 1


if __name__=='__main__':
    launch_fd = os.environ.pop('T1_CONSOLE_LAUNCH_FD', None)
    if launch_fd is not None:
        with os.fdopen(int(launch_fd), 'rb') as gate:
            if gate.read(1) != b'1':
                raise SystemExit(1)
    raise SystemExit(run(sys.argv[1]))
