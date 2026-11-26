"""Opt-in, GPU-free delivery check using fresh disposable PG16 databases/roles."""
from contextlib import ExitStack
from dataclasses import replace
import os
from pathlib import Path
import tempfile
import unittest

from vehicle.common.tests.test_recognition_v3 import evidence_fixture
from vehicle.common.recognition_transport import RecognitionCodec
from vehicle.common.recognition_manufacturers import POLICY
from vehicle.common.redis_scalar.codec import pack_life, unpack_life
from vehicle.common.redis_scalar.synthetic import ORG, IDENTITY_NAMESPACE


@unittest.skipUnless(os.environ.get('T1_RECOGNITION_POSTGRES_TEST') == '1',
                     'explicit disposable PostgreSQL check required')
class ManufacturerPostgresTests(unittest.TestCase):
    def test_new_and_later_manufacturers_commit_with_identical_retry(self):
        from vehicle.common.console_stack import configure, database_probe
        from vehicle.common.console_postgres import OwnedPostgres
        from vehicle.common.manufacturer.test_database import ManufacturerTestDatabase
        from vehicle.common.db1_consumer import NaturalConsumer
        configure()
        codec = RecognitionCodec(ORG, IDENTITY_NAMESPACE)

        def delivery(identifier, sid):
            evidence = evidence_fixture()
            life = unpack_life(evidence['life'])
            evidence['life'] = pack_life(replace(life, service_track_id=sid,
                raw_track_id=sid, context=replace(life.context, source_kind='rtsp')))
            evidence['manufacturer_policy'] = POLICY
            evidence['attributes']['manufacturer'] = identifier
            evidence['attributes']['recognition_evidence'][0]['result']['manufacturer'] = identifier
            return codec.make('db1.delivery.v1', evidence)

        first = delivery('mercedes', 1)
        later = delivery('volvo', 2)
        unknown = delivery('unknown', 3)
        with tempfile.TemporaryDirectory(prefix='manufacturer-pg-check-') as temp, ExitStack() as resources:
            directory = Path(temp)
            if os.environ.get('WAYCON_POSTGRES_MODE', 'owned') == 'existing':
                os.environ['DB1_P4_EXISTING_SERVER'] = '1'
                os.environ.pop('DB1_P4_ISOLATED', None)
                database_probe()
            else:
                resources.enter_context(OwnedPostgres(directory))
            database = ManufacturerTestDatabase(codec, first, None, None, None,
                directory, deliveries=(first,))
            try:
                def consume(raw):
                    fixture = database.fixture
                    run = fixture.res.run
                    consumer = NaturalConsumer(codec=codec, tenant=fixture.tenant,
                        trusted_run=run.run_id, trusted_camera=run.camera.camera_id,
                        resolutions=fixture.res, connection_factory=fixture.connect,
                        environment='test')
                    phases = []
                    consumer.hook = phases.append
                    result = consumer.process(raw, lambda:phases.append('ack'))
                    self.assertLess(phases.index('committed'), phases.index('ack'))
                    return result

                self.assertEqual(consume(first).disposition, 'new')
                original_key = next(d.key for d in database.fixture.res.manufacturers
                                    if d.name == 'Mercedes-Benz')
                database.extend(codec, (later, unknown))
                self.assertEqual(consume(later).disposition, 'new')
                self.assertEqual(consume(unknown).disposition, 'new')
                self.assertEqual(consume(first).disposition, 'identical')
                database.extend(codec, (later,))
                self.assertEqual(next(d.key for d in database.fixture.res.manufacturers
                    if d.name == 'Mercedes-Benz'), original_key)
                with database.fixture.connect() as connection, connection.transaction(), connection.cursor() as q:
                    q.execute('SELECT rolsuper,rolbypassrls FROM pg_roles WHERE rolname=current_user')
                    self.assertEqual(q.fetchone(), (False, False))
                    q.execute("SELECT set_config('app.org_id',%s,true)", (str(ORG),))
                    q.execute('SELECT f.track_id,m.name,f.manufacturer_conf FROM public.fact_vehicle_make f '
                              'LEFT JOIN public.dim_manufacturer m ON m.manufacturer_key=f.manufacturer_key '
                              'ORDER BY f.track_id')
                    self.assertEqual(q.fetchall(), [(1,'Mercedes-Benz',None),(2,'Volvo',None),(3,None,None)])
                    q.execute('SELECT count(*) FROM db1_delivery.acceptance')
                    self.assertEqual(q.fetchone(), (3,))
                    q.execute("SELECT set_config('app.org_id',%s,true)", ('00000000-0000-4000-8000-000000000002',))
                    q.execute('SELECT count(*) FROM public.fact_vehicle_make')
                    self.assertEqual(q.fetchone(), (0,))
            finally:
                database.fixture.doClassCleanups()
                self.assertFalse(database.fixture.tearDown_exceptions)


if __name__ == '__main__':
    unittest.main()
