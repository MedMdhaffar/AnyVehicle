"""Opt-in CPU replay of retained REAL envelopes across four DB1 epochs.

No models or video decoding. The source run's verified registrations and exact
evidence are replayed, without changing its outbox, database or review records.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import unittest
from uuid import uuid4

from vehicle.common.console_runs import atomic_json
from vehicle.common.console_stack import configure, database_probe
from vehicle.common.mohamed_lpr_redis_runtime import OwnedRedis
from vehicle.common.redis_scalar.journal import Journal
from vehicle.common.redis_scalar.worker import load_request_codec

ROOT=Path(__file__).resolve().parents[3]


@unittest.skipUnless(os.environ.get('WAYCON_DB1_REPLAY_RUN'),'explicit retained real-model run required')
class RetainedDeliveryTests(unittest.TestCase):
    def test_epochs_keep_one_database_outbox_and_retry_identity(self):
        configure();database_probe()
        source=Path(os.environ['WAYCON_DB1_REPLAY_RUN']).resolve(strict=True)
        directory=ROOT/'vehicle/media'/('universal-epoch-replay-'+uuid4().hex)
        directory.mkdir(mode=0o700)
        atomic_json(directory/'manufacturer-registration.json',json.loads((source/'manufacturer-registration.json').read_text()))
        control=json.loads((source/'db1-control.json').read_text())
        codec=load_request_codec(control['org'],control['identity_namespace']).v1
        raws=[p.read_bytes() for p in sorted(source.glob('db1-wire-*.json'))]
        self.assertEqual(len(raws),4)
        with OwnedRedis(os.environ['T1_REDIS_SERVER']) as owned:
            child=None;database=None
            try:
                for index,raw in enumerate(raws):
                    sid=codec.decode(raw).life.service_track_id
                    (directory/f'manufacturer-evidence-{sid}.json').write_bytes((source/f'manufacturer-evidence-{sid}.json').read_bytes())
                    namespace='epoch-replay-'+str(index)
                    journal=Journal(owned.client,namespace=namespace,trusted_org=codec.org,role='aggregator_plate',consumer='test')
                    owned.client.xadd(journal.key('db1.delivery.v1'),{'json':raw},id='1-0')
                    suffix='' if index==0 else f'-e{index}'
                    atomic_json(directory/f'db1-control{suffix}.json',control|dict(namespace=namespace,deliveries=1,redis_socket=str(owned.socket)))
                    if child is None:
                        child=subprocess.Popen([sys.executable,'-B',str(ROOT/'scripts/vehicle_console_db1.py'),str(directory)],
                            env={**os.environ,'WAYCON_CONSOLE_VERIFY_RETRY':'1'},stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
                    deadline=time.monotonic()+30
                    while time.monotonic()<deadline:
                        path=directory/f'db1-progress{suffix}.json'
                        if path.exists():
                            progress=json.loads(path.read_text())
                            if progress['status'] in ('completed','failed'):break
                        self.assertIsNone(child.poll(),'DB1 host exited early')
                        time.sleep(.05)
                    self.assertEqual(progress['status'],'completed',progress)
                    if database is None:database=progress['database']
                    self.assertEqual(progress['database'],database)
                    self.assertEqual(progress['counts']['fact_vehicle'],index+1)
                    self.assertEqual(progress['evidence'][0]['retry'],'identical')
                self.assertEqual(progress['counts'],dict(fact_vehicle=4,fact_vehicle_colour=4,fact_plate_read=3,fact_vehicle_make=2))
                self.assertTrue((directory/'outbox.sqlite').is_file())
                atomic_json(directory/'replay-result.json',dict(status='passed',epochs=4,database=database,counts=progress['counts'],source_run=source.name))
                print('Retained epoch replay evidence:',directory)
            finally:
                (directory/'db1-finish').touch(mode=0o600)
                if child:child.wait(timeout=30)
