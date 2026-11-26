"""CPU DB1 host for full console videos. One retained test DB/outbox per run.

Each epoch reuses the same natural identities and registrations. CUDA never
receives provisioning credentials. Unmapped manufacturer labels remain omitted.
"""
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace


def run(directory):
    from .console_runs import atomic_json
    from .console_stack import configure, database_probe
    from .db1_consumer import NaturalConsumer, RedisDB1Consumer
    from .db1_outbox import LocalOutbox
    from .db1_rows.make import MakeActivation
    from .observation_evidence import RecognitionIdentity
    from .manufacturer.delivery import ManufacturerDeliveryCodec
    from .manufacturer.test_database import ManufacturerTestDatabase
    from .redis_scalar.worker import load_request_codec
    from .redis_scalar.journal import Journal, PRODUCE
    import redis
    configure();database_probe()
    registration = json.loads((directory/'manufacturer-registration.json').read_text())
    a = registration['activation']
    activation = MakeActivation(**{**a, 'model':RecognitionIdentity(**a['model']),
        'runtime_provenance':tuple(a['runtime_provenance']), 'brand_mapping':tuple(tuple(v) for v in a['brand_mapping'])}) if a else None
    db = outbox = client = None
    index = 0
    try:
        while True:
            suffix = '' if index==0 else f'-e{index}'
            control_path = directory/f'db1-control{suffix}.json'
            while not control_path.exists():
                if (directory/'db1-finish').exists():return
                # Parent death ends this host after the current transaction.
                if os.getppid()==1:return
                time.sleep(.05)
            control = json.loads(control_path.read_text())
            progress_path = directory/f'db1-progress{suffix}.json'
            progress = dict(status='running', tracks={}, evidence=[])
            def save():atomic_json(progress_path, progress)
            save()
            try:
                base = load_request_codec(control['org'], control['identity_namespace']).v1
                codec = ManufacturerDeliveryCodec(base, activation) if activation else base
                if client is None:client=redis.Redis(unix_socket_path=control['redis_socket'], socket_timeout=2)
                source = Journal(client, namespace=control['namespace'], trusted_org=base.org,
                                 role='aggregator_plate', consumer='console-source')
                raws = [f[b'json'] for _,f in client.xrange(source.key('db1.delivery.v1'))]
                if len(raws)!=control['deliveries']:raise RuntimeError('delivery_count_mismatch')
                if raws and db is None:
                    db = ManufacturerTestDatabase(base, raws[0], activation, SimpleNamespace(lock=registration.get('runtime_lock')),
                        control['source_media'], directory, deliveries=raws)
                    outbox = LocalOutbox(path=directory/'outbox.sqlite', namespace=base.identity_policy.namespace, validation=base.validation)
                elif raws:db.extend(base, raws)
                target = Journal(client, namespace=control['namespace']+'-db1', trusted_org=base.org,
                                 role='aggregator_plate', consumer='console-db1')
                for number, raw in enumerate(raws, 1):
                    message=base.decode(raw);sid=message.life.service_track_id
                    if activation:
                        evidence=(directory/f'manufacturer-evidence-{sid}.json').read_text()
                        raw=codec.encode(raw, evidence)
                    retained=directory/f'db1-v2-wire-{sid}.json'
                    if retained.exists():
                        if retained.read_bytes()!=raw:raise RuntimeError('retained_wire_changed')
                    else:
                        fd=os.open(retained, os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o600)
                        with os.fdopen(fd,'wb') as stream:stream.write(raw)
                    natural=NaturalConsumer(codec=codec, tenant=db.fixture.tenant, trusted_run=message.life.context.run_id,
                        trusted_camera=message.life.context.camera.camera_id, resolutions=db.fixture.res,
                        connection_factory=db.fixture.connect, environment='test', outbox=outbox)
                    reader=RedisDB1Consumer(client, target, natural, name='console-db1')
                    entry_id=f'{2*number-1}-0'
                    client.eval(PRODUCE, 1, target.key('db1.delivery.v1'), entry_id, raw, target.max_stream)
                    state=dict(postgres_commit='pending', redis_ack='pending')
                    if activation:state['manufacturer_db1']=codec.decode(raw).body['manufacturer']['delivery_reason']
                    progress['tracks'][str(sid)]=state;save()
                    boundaries=[]
                    def hook(phase):
                        if phase=='committed':
                            assert outbox.load(natural.prepare(raw).logical_identity) is not None
                            assert len(client.xpending_range(target.key('db1.delivery.v1'), reader.group, min=entry_id,max=entry_id,count=1))==1
                            state['postgres_commit']='committed';boundaries.append('outbox_and_postgres_before_ack');save()
                    natural.hook=hook
                    result=reader.step()
                    assert result.disposition=='new' and reader.last_delivery.ack_reply==1 and boundaries
                    state['redis_ack']='acknowledged'
                    retry=None
                    if os.environ.get('WAYCON_CONSOLE_VERIFY_RETRY')=='1':
                        natural.hook=lambda phase:None
                        client.eval(PRODUCE,1,target.key('db1.delivery.v1'),f'{2*number}-0',raw,target.max_stream)
                        retry=reader.step().disposition
                        assert retry=='identical' and reader.last_delivery.ack_reply==1
                    progress['evidence'].append(dict(track_id=sid,fact_count=result.fact_count,retry=retry,
                        acceptance_sha256=result.acceptance_sha256,boundary=boundaries,ack_reply=1))
                    save()
                progress.update(status='completed',pending=0, database=db.fixture.db if db else None)
                if db:
                    with db.fixture.connect() as connection, connection.transaction(), connection.cursor() as cursor:
                        cursor.execute("SELECT set_config('app.org_id',%s,true)",(str(base.org),))
                        cursor.execute('SELECT f.uid,f.track_id,m.name,f.manufacturer_key,f.model_key,f.manufacturer_conf FROM public.fact_vehicle_make f JOIN public.dim_manufacturer m USING(manufacturer_key) ORDER BY f.track_id')
                        progress['make_rows']=[dict(zip(('uid','track_id','name','manufacturer_key','model_key','confidence'),
                            (str(row[0]), *row[1:]))) for row in cursor.fetchall()]
                        progress['counts']={}
                        from psycopg import sql
                        for table in ('fact_vehicle','fact_vehicle_colour','fact_plate_read','fact_vehicle_make'):
                            cursor.execute(sql.SQL('SELECT count(*) FROM public.{}').format(sql.Identifier(table)))
                            progress['counts'][table]=cursor.fetchone()[0]
                save();index+=1
            except Exception as error:
                import traceback
                progress.update(status='failed', error_type=type(error).__name__, error_code=getattr(error,'code',None),
                    locations=[dict(file=Path(f.filename).name,line=f.lineno,function=f.name) for f in traceback.extract_tb(error.__traceback__)])
                save();raise
    finally:
        if outbox:outbox.close()
        if db:db.close()
        if client:client.close()
