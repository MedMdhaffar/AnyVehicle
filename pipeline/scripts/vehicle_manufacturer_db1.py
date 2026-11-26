"""Bounded prerecorded real-model pipeline -> durable outbox -> test PostgreSQL.

Uses only the configured existing local P4 test server. Retains the new random
test database, outbox, wire evidence and private reconnect configuration.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import threading
import time
from uuid import UUID, uuid4

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video',type=Path,default=ROOT/'services/license_plate_service/run_video_pipeline/test_video.mp4')
    p.add_argument('--start-frame',type=int,default=250)
    p.add_argument('--max-frames',type=int,default=8,choices=range(1,21))
    p.add_argument('--max-seconds',type=float,default=240)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.start_frame<0 or not 0<a.max_seconds<=300:p.error('bounded start/frame/time required')
    os.umask(0o077)
    from vehicle.common.gpu_lease import GPULease
    gpu_lease = GPULease()
    a.output.mkdir(parents=True,exist_ok=False)
    from vehicle.common.console_stack import configure,model_paths,database_probe
    configure()
    if os.environ.get('WAYCON_POSTGRES_MODE')!='existing':
        raise RuntimeError('configured existing test PostgreSQL required')
    os.environ['DB1_P4_EXISTING_SERVER']='1'
    database_probe()
    os.environ.update(YOLO_OFFLINE='true',YOLO_AUTOINSTALL='false',COLOR_AGENT_ENABLED='false',
        PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK='True',YOLO_CONFIG_DIR=str(a.output/'model-settings'))
    import cv2,torch
    torch.set_num_threads(1);cv2.setNumThreads(1)
    from vehicle.common.db1_context import CameraRegistration,ModelRegistration,ResolutionInfo,RunContext
    from vehicle.common.frame_receipt import FrameReceiptClock,ClockPolicy
    from vehicle.common.observation_evidence import RecognitionIdentity
    from vehicle.common.real_video.input import MP4Input
    from vehicle.common.real_video.local import LocalDetector,pixel_operations,fingerprint
    from vehicle.common.real_video.plate_local import NativePlateDetector,detector_artifact
    from vehicle.common.real_video.plate_engine import CombinedEngine,VideoPlateLimits
    from vehicle.common.engine_plate import native_access
    from vehicle.common.engine_plate_local import LocalPlateBinding,artifact
    from vehicle.common.manufacturer.engine import ManufacturerLedger,SelectionConfig
    from vehicle.common.manufacturer.local import LocalRuntime
    from vehicle.common.manufacturer.delivery import ManufacturerDeliveryCodec
    from vehicle.common.manufacturer.test_database import ManufacturerTestDatabase,CATALOGUE
    from vehicle.common.db1_rows.make import MakeActivation
    from vehicle.common.db1_consumer import NaturalConsumer,RedisDB1Consumer
    from vehicle.common.db1_outbox import LocalOutbox
    from vehicle.common.redis_scalar.worker import load_request_codec,Worker
    from vehicle.common.redis_scalar.journal import Journal,PRODUCE
    from vehicle.common.plate_requests.worker import RequestWorker
    from vehicle.common.plate_requests.host import V2Publisher
    from vehicle.common.mohamed_lpr_redis_runtime import OwnedRedis
    from vehicle.common.vehicle_observation import RunEnd
    from contracts.db1.reference import canonical_json_bytes
    stop=threading.Event()
    signal.signal(signal.SIGTERM,lambda *_:stop.set())
    signal.signal(signal.SIGINT,lambda *_:stop.set())
    started=time.monotonic();deadline=started+a.max_seconds
    now=datetime.now(timezone.utc)
    org=UUID('12345678-1234-4234-8234-123456789012')
    namespace_id=UUID('98c2adf9-264e-4ac2-b6a1-3239fa02ecbb')
    run_id='manufacturer-db1-'+uuid4().hex
    paths=model_paths()
    context=RunContext(org,CameraRegistration(org,run_id,None,'Africa/Tunis',now,None),run_id,'file',now,
        'manufacturer-db1-v2',ResolutionInfo(None,None,'vehicle',ModelRegistration(None,
        'local-yolo26-sha256:'+fingerprint(paths['vehicle_model'])['sha256'],1,'detect'),None,
        'bounded real-model test; host replay receipts',False),'development')
    clock=FrameReceiptClock(context,utc_clock=lambda:datetime.now(timezone.utc),monotonic_clock=time.monotonic,policy=ClockPolicy())
    source=MP4Input(a.video,clock)
    from vehicle.common.evaluation_capture import EvaluationRecorder
    evaluation=EvaluationRecorder(a.output,context,source_id='video-sha256:'+fingerprint(a.video)['sha256'])
    engine=runtime=ledger=owned=outbox=db=None
    workers=[];report=dict(status='running',run_id=run_id,predictions=[],commits=[],cleanup={})
    result=1
    try:
        source.__enter__()
        # Replay receipt timeline starts at this explicitly selected source frame.
        # Decode the short prefix without inference: this HEVC source cannot
        # reliably seek to arbitrary P frames without its earlier references.
        for _ in range(a.start_frame):
            if not source.capture.grab():raise RuntimeError('source window beyond EOF')
        source.metadata['declared_frames']-=a.start_frame
        runtime=LocalRuntime(ROOT/'vehicle/common/manufacturer/assets.json',timeout=30)
        activation=MakeActivation('user-authorized test integration; supplied DB1 development catalogue v0',
            RecognitionIdentity('LFM2.5-VL-450M-runtime-sha256:'+runtime.provenance,1,'vlm'),
            (runtime.provenance,),CATALOGUE,'selected_crop_receipt','submitted_crops','omit')
        (a.output/'manufacturer-registration.json').write_bytes(canonical_json_bytes(dict(
            activation=json.loads(json.dumps(asdict(activation))),runtime_lock=runtime.lock,runtime_code=runtime.code)))
        ledger=ManufacturerLedger(enabled=True,runtime=runtime,journal=a.output/'manufacturer.sqlite',
            selection=SelectionConfig(policy='quality_lifetime_v1'))
        with native_access():
            vehicle,_=evaluation.measure('vehicle_load',context.resolution.detector.model_name,
                lambda:LocalDetector(paths['vehicle_model'],paths['vehicle_tracker'],device='cuda:0',
                                  frame_rate=source.metadata['fps'],threads=1),
                synchronize=lambda:torch.cuda.synchronize('cuda:0'))
        plate=NativePlateDetector(detector_artifact(paths['plate_model'],paths['plate_tracker'],
            frame_rate=source.metadata['fps']),frame_rate=source.metadata['fps'])
        bindings={kind:LocalPlateBinding(artifact(kind,path)) for kind,path in [('ocr',paths['ocr_dir']),('detect',paths['marker_model'])]}
        estimate,quality=pixel_operations()
        engine=CombinedEngine(context,clock,vehicle,plate,bindings,algorithm=RecognitionIdentity('t1-local-hsv-console',1,'classify'),
            quality_algorithm='t1-colour-quality',plate_quality_algorithm='t1-plate-crop-quality',estimate=estimate,
            quality_fn=quality,manufacturer=ledger,limits=VideoPlateLimits(streaming=True,max_frames=None,
            max_seconds=None,max_api_calls=None,max_raw_total=None))
        evaluation.bind(engine)
        codec=load_request_codec(org,namespace_id)
        delivery=ManufacturerDeliveryCodec(codec.v1,activation)
        owned=OwnedRedis(os.environ.get('T1_REDIS_SERVER') or shutil.which('redis-server')).__enter__()
        namespace='make-'+uuid4().hex[:16]
        workers=[Worker(owned.client,codec=codec.v1.v0,namespace=namespace,role='colour',consumer='make-colour',lease_ms=60000),
            Worker(owned.client,codec=codec.v1,namespace=namespace,role='aggregator_plate',consumer='make-join',lease_ms=60000),
            RequestWorker(owned.client,codec=codec,namespace=namespace,consumer='make-plate',lease_ms=60000)]
        journal=Journal(owned.client,namespace=namespace,trusted_org=org,role='plate_requests',consumer='make-engine')
        publisher=V2Publisher(owned.client,journal,codec,engine)
        outbox=LocalOutbox(path=a.output/'outbox.sqlite',namespace=namespace_id,validation=codec.validation)
        reason='frame_limit'
        for _ in range(a.max_frames):
            if stop.is_set():reason='controlled_stop';break
            if time.monotonic()>=deadline:reason='time_limit';break
            packet=source.read()
            if packet is None:reason='eof';break
            engine.process(packet)
        source.close()
        engine.finish(RunEnd.END_OF_INPUT if reason=='eof' else RunEnd.SERVICE_STOP,trigger=reason)
        for w in workers:w.setup()
        report['capture']=dict(reason=reason,frames=source.decoded,start_frame=a.start_frame,decoder_opens=source.opens)
        originals=dict(engine.pending)
        for sid,closed in originals.items():
            while True:
                if time.monotonic()>=deadline:raise RuntimeError('metadata settlement deadline')
                for w in workers:w.step()
                if publisher.publish(closed):break
                time.sleep(.005)
        # Native completion precedes the aggregator's final delivery turn.
        # Wait for every original closure, including the last terminal join.
        while owned.client.xlen(journal.key('db1.delivery.v1'))<len(originals):
            if time.monotonic()>=deadline:raise RuntimeError('final delivery join deadline')
            for w in workers:w.step()
            time.sleep(.005)
        # Preserve existing v1 rows; publish each manufacturer's new v2 envelope
        # on a separate run-owned partition, using the existing durable DB reader.
        target=Journal(owned.client,namespace=namespace+'-db1',trusted_org=org,role='aggregator_plate',consumer='make-db1')
        base_rows=owned.client.xrange(journal.key('db1.delivery.v1'))
        reader=consumer=None
        for index,(_,fields) in enumerate(base_rows,1):
            base=fields[b'json'];m=codec.v1.decode(base);sid=m.life.service_track_id
            from vehicle.common.plate_metadata import PlateMetadata
            plate_result=PlateMetadata().evaluate(m.life,publisher.plans[sid].transcript).result
            evaluation.finalize(originals[sid],plate_result,m.body['bundle'])
            raw=delivery.encode(base,originals[sid].manufacturer_json)
            message=delivery.decode(raw)
            maker=message.body['manufacturer']
            report['predictions'].append(dict(track_id=sid,raw_track_id=m.life.raw_track_id,
                status=maker['status'],make=maker['make'],delivery_reason=maker['delivery_reason'],
                model_text=maker['evidence']['text'],selection=maker['evidence']['binding'].get('selection')))
            (a.output/f'delivery-{sid}.json').write_bytes(raw)
            if db is None:
                video=fingerprint(a.video);video.update(width=int(source.metadata['width']),height=int(source.metadata['height']))
                db=ManufacturerTestDatabase(codec.v1,base,activation,runtime,video,a.output,
                    deliveries=[f[b'json'] for _,f in base_rows])
                consumer=NaturalConsumer(codec=delivery,tenant=db.fixture.tenant,trusted_run=run_id,
                    trusted_camera=context.camera.camera_id,resolutions=db.fixture.res,
                    connection_factory=db.fixture.connect,environment='test',outbox=outbox)
                reader=RedisDB1Consumer(owned.client,target,consumer,name='make-db1')
            entry_id=f'{2*index-1}-0'
            owned.client.eval(PRODUCE,1,target.key('db1.delivery.v1'),entry_id,raw,target.max_stream)
            boundaries=[]
            def hook(phase):
                if phase=='committed':
                    assert outbox.load(consumer.prepare(raw).logical_identity) is not None
                    assert len(owned.client.xpending_range(target.key('db1.delivery.v1'),reader.group,min=entry_id,max=entry_id,count=1))==1
                    boundaries.append('durable_outbox_and_postgres_before_ack')
            consumer.hook=hook
            first=reader.step()
            assert first.disposition=='new' and boundaries and reader.last_delivery.ack_reply==1
            consumer.hook=lambda phase:None
            # Redeliver the exact bytes under a fresh transport ID; no inference.
            before=dict(engine.audit.calls)
            owned.client.eval(PRODUCE,1,target.key('db1.delivery.v1'),f'{2*index}-0',raw,target.max_stream)
            retry=reader.step()
            assert retry.disposition=='identical' and before==engine.audit.calls
            report['commits'].append(dict(track_id=sid,first=first.disposition,retry=retry.disposition,
                fact_count=first.fact_count,acceptance_sha256=first.acceptance_sha256,boundary=boundaries))
            engine.acknowledge(sid)
        if db is None:raise RuntimeError('no completed deliveries')
        with db.fixture.connect() as connection:
            with connection.transaction(),connection.cursor() as q:
                q.execute("SELECT set_config('app.org_id',%s,true)",(str(org),))
                counts={}
                for table in ('fact_vehicle','fact_vehicle_colour','fact_plate_read','fact_vehicle_make'):
                    from psycopg import sql
                    q.execute(sql.SQL('SELECT count(*) FROM public.{}').format(sql.Identifier(table)));counts[table]=q.fetchone()[0]
                q.execute('SELECT f.uid,f.track_id,f.manufacturer_key,m.name,f.model_key,f.manufacturer_conf,f.observations FROM public.fact_vehicle_make f JOIN public.dim_manufacturer m USING(manufacturer_key) ORDER BY f.track_id')
                report['make_rows']=[dict(zip(('uid','track_id','manufacturer_key','name','model_key','manufacturer_conf','observations'),[str(x) if isinstance(x,UUID) else x for x in r])) for r in q.fetchall()]
                q.execute('SELECT track_id,outcome,plate_text,attempts FROM public.fact_plate_read ORDER BY track_id')
                report['plate_rows']=[dict(zip(('track_id','outcome','plate_text','attempts'),r)) for r in q.fetchall()]
                report['counts']=counts
        report.update(database=db.fixture.db,native_calls=engine.audit.calls,manufacturer_calls=ledger.calls,
            pending=owned.client.xpending(target.key('db1.delivery.v1'),reader.group)['pending'],
            retained_closures=len(engine.pending),original_closures=len(originals))
        assert report['make_rows'] and report['pending']==0 and not engine.pending
        assert counts['fact_vehicle']==len(base_rows) and counts['fact_vehicle_colour']>0 and counts['fact_plate_read']>0
        report['status']='passed';result=0
    except Exception as error:
        report.update(status='failed',error_type=type(error).__name__,error_code=getattr(error,'code',None),
                      sqlstate=getattr(error,'sqlstate',None))
        # Safe traceback gives code location; message may contain SQL/plate data.
        import traceback
        report['error_locations']=[dict(file=Path(f.filename).name,line=f.lineno,function=f.name) for f in traceback.extract_tb(error.__traceback__)]
        print(json.dumps({k:report[k] for k in ('status','error_type','error_code','error_locations')}))
    finally:
        source.close()
        if engine is not None:
            engine.close();report['cleanup']['engine_closed']=engine.closed
        if runtime is not None:report['cleanup']['manufacturer_children_reaped']=runtime.cleaned
        for w in workers:
            if w.journal.token:
                try:w.journal.release()
                except Exception as error:report['cleanup'].setdefault('lease_release_errors',[]).append(type(error).__name__)
        if owned is not None:
            owned.__exit__(None,None,None);report['cleanup']['redis_cleaned']=owned.cleaned
        if outbox is not None:
            report['outbox_entries']=len(outbox.inspect_pending());outbox.close()
        if db is not None:db.close()
        gpu_lease.close()
        report['elapsed_seconds']=time.monotonic()-started
        (a.output/'summary.json').write_bytes(canonical_json_bytes(report))
    print(json.dumps(dict(status=report['status'],counts=report.get('counts'),output=str(a.output))))
    return result


if __name__=='__main__':raise SystemExit(main())
