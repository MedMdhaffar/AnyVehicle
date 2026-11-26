"""Configured local P4 test server only; retained real-model smoke database.

Uses the existing guarded fixture's random database and restricted application
role. Manufacturer names use the dev seed plus explicit recognition deliveries.
Models use generated keys and actual executed identities, never seed models.
"""
from dataclasses import replace
from datetime import timedelta
import json
from pathlib import Path
from types import SimpleNamespace

from contracts.db1.reference import derive_date_key
from ..db1_adapter import TrustedTenant
from ..db1_rows import Dimension, ResolvedModel
from ..real_video.test_database import fixture_module, ROOT
from ..tests.test_db1_consumer import supplied_test_registrations

CATALOGUE = (('renault','Renault'),('peugeot','Peugeot'),('toyota','Toyota'),('volkswagen','Volkswagen'))


def register_manufacturers(cursor, names):
    """Resolve names in this owned fixture; keys always come from PostgreSQL."""
    for name in sorted(set(names)):
        cursor.execute('INSERT INTO public.dim_manufacturer(name) VALUES (%s) ON CONFLICT (name) DO NOTHING', (name,))
    cursor.execute('SELECT name,manufacturer_key FROM public.dim_manufacturer ORDER BY name')
    return tuple(Dimension(name,key,True) for name,key in cursor.fetchall())


class ManufacturerTestDatabase:
    def __init__(self, base_codec, raw, activation, runtime, video, directory, *, deliveries=()):
        module = fixture_module()
        module.provisioning_params(__import__('os').environ)
        supplied = supplied_test_registrations(SimpleNamespace(v1=base_codec),raw)
        all_res=[supplied_test_registrations(SimpleNamespace(v1=base_codec),v) for v in deliveries] or [supplied]
        # Whole bounded run is closed before provisioning; register exactly its
        # observed type/colour/category/manufacturer and executed model identities.
        fields={}
        for field in ('classes','colours','plate_categories','models','manufacturers'):
            values={}
            for r in all_res:
                for v in getattr(r,field):
                    identity=(v.identity.model_name,v.identity.version) if field=='models' else v.name
                    values[identity]=v
            fields[field]=tuple(replace(v,key=i+1) for i,v in enumerate(values.values()))
        # Temporary keys are supplied records only; database registration below
        # replaces every one with PostgreSQL-generated keys.
        supplied=replace(supplied,**fields)
        self.directory = Path(directory)
        class Fixture(module.PostgreSQL16Tests):
            delivery_validation = base_codec.validation
            @classmethod
            def seed_registrations(cls):
                r=supplied.run; c=r.camera
                models=[]
                with cls.setup.cursor() as q:
                    q.execute('INSERT INTO public.dim_org(org_id,name) VALUES (%s,%s)',(r.org_id,'WAYCON real-model test'))
                    q.execute('INSERT INTO public.dim_camera(camera_id,org_id,timezone,valid_from) VALUES (%s,%s,%s,%s) RETURNING camera_key',
                              (c.camera_id,r.org_id,c.timezone,c.valid_from)); camera=replace(c,camera_key=q.fetchone()[0])
                    identities=[(r.detector.model_name,r.detector.version,r.detector.task)] + [
                        (m.identity.model_name,m.identity.version,m.identity.task) for m in supplied.models]
                    if activation:identities.append((activation.model.model_name,activation.model.version,activation.model.task))
                    keys={}
                    for name,version,task in dict.fromkeys(identities):
                        maker=bool(activation and name==activation.model.model_name)
                        q.execute('INSERT INTO public.dim_model(model_name,version,task,base_model,weights_path) VALUES (%s,%s,%s,%s,%s) RETURNING model_key',
                            (name,version,task,'LFM2.5-VL-450M' if maker else None,
                             runtime.lock['artifacts']['model']['path'] if maker else None))
                        keys[name,version]=q.fetchone()[0]
                    detector=replace(r.detector,model_key=keys[r.detector.model_name,r.detector.version])
                    for m in supplied.models+((ResolvedModel(activation.model,1),) if activation else ()):
                        models.append(replace(m,key=keys[m.identity.model_name,m.identity.version]))
                    # Retain the supplied seed, then resolve recognized names in
                    # this project-owned test catalogue using generated keys.
                    seed=(ROOT/'db1_contract/03_dev_seed.sql').read_text(encoding='utf-8')
                    statement=next(s.strip() for s in seed.split(';') if s.strip().startswith('INSERT INTO dim_manufacturer '))
                    q.execute(statement)
                    q.execute('SELECT name,manufacturer_key FROM public.dim_manufacturer ORDER BY name')
                    makers=tuple(Dimension(name,key,True) for name,key in q.fetchall())
                    assert {m.name for m in makers}=={name for _,name in CATALOGUE}
                    makers=register_manufacturers(q, (d.name for d in supplied.manufacturers))
                    # Existing P4 test semantics for type/colour/category;
                    # these are test registrations, not production approvals.
                    dimensions=[]
                    for table,key,values in [('dim_class','class_key',supplied.classes),
                        ('dim_colour','colour_key',supplied.colours),('dim_plate_category','plate_category_key',supplied.plate_categories)]:
                        out=[]
                        for d in values:
                            if table=='dim_class':
                                q.execute('INSERT INTO public.dim_class(name,class_group) VALUES (%s,%s) RETURNING class_key',(d.name,d.group))
                            else:
                                from psycopg import sql
                                q.execute(sql.SQL('INSERT INTO public.{}(name) VALUES (%s) RETURNING {}').format(sql.Identifier(table),sql.Identifier(key)),(d.name,))
                            out.append(replace(d,key=q.fetchone()[0]))
                        dimensions.append(tuple(out))
                    clip = None
                    if video is not None:
                        clip=supplied.media[0]
                        object_key='clips/'+video['sha256']+video.get('extension','.mp4')
                        content_type=video.get('content_type','video/mp4')
                        q.execute('INSERT INTO public.media_object(org_id,camera_key,ts,kind,bucket,object_key,content_type,bytes,width,height,sha256) VALUES (%s,%s,%s,\'clip\',%s,%s,%s,%s,%s,%s,%s) RETURNING media_key',
                            (r.org_id,camera.camera_key,r.started_at,'test-source-registration',object_key,content_type,video['bytes'],video['width'],video['height'],video['sha256']))
                        clip=replace(clip,media_key=q.fetchone()[0],camera_key=camera.camera_key,ts=r.started_at,
                            bucket='test-source-registration',object_key=object_key,content_type=content_type,
                            bytes=video['bytes'],width=video['width'],height=video['height'],sha256=video['sha256'])
                    q.execute("SELECT lane_key FROM public.dim_lane WHERE name='vehicle'");lane=q.fetchone()[0]
                    q.execute('INSERT INTO public.dim_run(run_id,org_id,lane_key,camera_key,model_key,source_kind,source_media_key,started_at,service_version) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING run_key',
                        (r.run_id,r.org_id,lane,camera.camera_key,detector.model_key,'file' if video else 'stream',clip.media_key if clip else None,r.started_at,r.service_version))
                    run_key=q.fetchone()[0]
                dates=tuple(sorted({derive_date_key((r.started_at+timedelta(days=i)).isoformat(),camera.timezone) for i in (0,1)}))
                run=replace(r,run_key=run_key,camera=camera,detector=detector,lane_key=lane,
                            source_media_key=clip.media_key if clip else None,date_keys=tuple(sorted(set(dates)|set(r.date_keys))),provenance='real-model test runtime registrations; DB1 dev manufacturer catalogue')
                cls.res=replace(supplied,run=run,models=tuple(models),manufacturers=makers,media=(clip,) if clip else (),
                                classes=dimensions[0],colours=dimensions[1],plate_categories=dimensions[2])
                cls.tenant=TrustedTenant(r.org_id,'configured local P4 real-model test')
        self.fixture=Fixture
        try:Fixture.setUpClass()
        except BaseException:
            Fixture.doClassCleanups()
            raise
        # Contains credentials: private local configuration only, never printed.
        self.private=self.directory/'database-private.json'
        fd=__import__('os').open(self.private,__import__('os').O_WRONLY|__import__('os').O_CREAT|__import__('os').O_EXCL,0o600)
        with __import__('os').fdopen(fd,'w') as stream:
            json.dump(dict(application_params=Fixture.application_params,database=Fixture.db,
                           owner_role=Fixture.owner_role,app_role=Fixture.app_role),stream)

    def close(self):
        # Retain committed test rows and private reconnect configuration.
        # No changes to existing databases, roles, or server configuration.
        self.fixture.setup.close()
        self.fixture.admin.close()

    def extend(self, base_codec, deliveries):
        """Register later epochs, including newly observed manufacturer names."""
        from psycopg import sql
        fixture = self.fixture
        res = fixture.res
        incoming = [supplied_test_registrations(SimpleNamespace(v1=base_codec), raw) for raw in deliveries]
        fields = {}
        with fixture.setup.transaction(), fixture.setup.cursor() as cursor:
            for field, table, key in (('classes','dim_class','class_key'), ('colours','dim_colour','colour_key'),
                                     ('plate_categories','dim_plate_category','plate_category_key'), ('models','dim_model','model_key')):
                values = list(getattr(res, field))
                identity = (lambda v:(v.identity.model_name,v.identity.version)) if field=='models' else (lambda v:v.name)
                known = {identity(v) for v in values}
                for r in incoming:
                    for v in getattr(r, field):
                        if identity(v) in known:continue
                        if field=='models':
                            cursor.execute('INSERT INTO public.dim_model(model_name,version,task) VALUES (%s,%s,%s) RETURNING model_key',
                                (v.identity.model_name,v.identity.version,v.identity.task))
                        elif field=='classes':
                            cursor.execute('INSERT INTO public.dim_class(name,class_group) VALUES (%s,%s) RETURNING class_key', (v.name,v.group))
                        else:
                            cursor.execute(sql.SQL('INSERT INTO public.{}(name) VALUES (%s) RETURNING {}').format(sql.Identifier(table),sql.Identifier(key)), (v.name,))
                        values.append(replace(v, key=cursor.fetchone()[0]));known.add(identity(v))
                fields[field] = tuple(values)
            fields['manufacturers'] = register_manufacturers(cursor,
                (d.name for r in incoming for d in r.manufacturers))
        dates = tuple(sorted(set(res.run.date_keys).union(*(r.run.date_keys for r in incoming))))
        fixture.res = replace(res, run=replace(res.run, date_keys=dates), **fields)
