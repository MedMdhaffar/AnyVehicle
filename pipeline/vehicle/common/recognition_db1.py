"""CPU delivery host reusing the outbox, RLS adapter and Redis commit/ACK boundary."""
import json
from pathlib import Path


def deliver(directory):
    from .console_stack import configure, database_probe
    from .recognition_transport import RecognitionCodec
    from .manufacturer.test_database import ManufacturerTestDatabase
    from .db1_consumer import NaturalConsumer, RedisDB1Consumer
    from .db1_outbox import LocalOutbox
    from .redis_scalar.journal import Journal
    from .console_runs import atomic_json
    import redis
    configure(); database_probe()
    directory = Path(directory)
    control = json.loads((directory/'recognition-db1-control.json').read_text())
    codec = RecognitionCodec(control['org'], control['identity_namespace'])
    client = redis.Redis(unix_socket_path=control['redis_socket'], socket_timeout=5)
    journal = Journal(client, namespace=control['namespace'], trusted_org=codec.org,
                      role='aggregator_plate', consumer='console-recognition-db1')
    raws = [fields[b'json'] for _,fields in client.xrange(journal.key('db1.delivery.v1'))]
    progress = dict(status='running', tracks={}, evidence=[])
    def save(): atomic_json(directory/'recognition-db1-progress.json', progress)
    db = outbox = None
    try:
        save()
        if not raws:
            progress['status'] = 'completed'; save(); return
        # Immutable stream bytes are reused on retry. Never provision a second
        # target for an already accepted run: reconnect its retained fixture.
        private = directory/'database-private.json'
        if private.exists():
            from .recognition_reconnect import retained_database
            db = retained_database(directory, codec, raws)
        else:
            db = ManufacturerTestDatabase(codec, raws[0], None, None, control['source_media'], directory, deliveries=raws)
            from .recognition_reconnect import save_registrations
            save_registrations(directory, db.fixture.res)
        outbox = LocalOutbox(path=directory/'outbox.sqlite', namespace=codec.identity_policy.namespace, validation=codec.validation)
        run = db.fixture.res.run
        natural = NaturalConsumer(codec=codec, tenant=db.fixture.tenant, trusted_run=run.run_id,
            trusted_camera=run.camera.camera_id, resolutions=db.fixture.res,
            connection_factory=db.fixture.connect, environment='test', outbox=outbox)
        reader = RedisDB1Consumer(client, journal, natural, name='console-recognition-db1')
        while True:
            def committed(phase):
                if phase == 'committed':
                    sid = str(codec.decode(reader.last_delivery.raw).life.service_track_id)
                    entry = natural.prepare(reader.last_delivery.raw)
                    from psycopg import sql
                    # Independent normal-role connection observes committed
                    # rows while the exact stream entry is still pending.
                    with db.fixture.connect() as check, check.transaction(), check.cursor() as q:
                        q.execute("SELECT set_config('app.org_id',%s,true)",(str(codec.org),))
                        for fact in entry.construction.ready_rows:
                            q.execute(sql.SQL('SELECT count(*) FROM public.{} WHERE uid=%s AND track_id=%s').format(
                                sql.Identifier(fact.row_kind)),(fact.sql.value()['uid'],int(sid)))
                            if q.fetchone()!=(1,):raise RuntimeError('independent_commit_check_failed')
                    pending = client.xpending_range(journal.key('db1.delivery.v1'),reader.group,
                        min=reader.last_delivery.entry_id,max=reader.last_delivery.entry_id,count=1)
                    if len(pending)!=1:raise RuntimeError('delivery_not_pending_before_ack')
                    progress['tracks'][sid] = dict(postgres_commit='committed', redis_ack='pending')
                    save()
            natural.hook = committed
            result = reader.step()
            if result is None: break
            sid = str(codec.decode(reader.last_delivery.raw).life.service_track_id)
            progress['tracks'][sid] = dict(postgres_commit='committed', redis_ack='acknowledged')
            progress['evidence'].append(dict(track_id=int(sid), disposition=result.disposition,
                acceptance_sha256=result.acceptance_sha256, ack_reply=reader.last_delivery.ack_reply,
                independent_rows_visible_before_ack=True))
            save()
        progress['status'] = 'completed'; save()
    except Exception as error:
        import traceback
        progress.update(status='failed', error_type=type(error).__name__, code=getattr(error,'code',None),
            locations=[dict(file=Path(f.filename).name,line=f.lineno,function=f.name) for f in traceback.extract_tb(error.__traceback__)])
        save(); raise
    finally:
        if outbox: outbox.close()
        if db: db.close()
        client.close()
