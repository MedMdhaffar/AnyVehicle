"""Retained test registrations: restart delivery without re-registering facts."""
from dataclasses import asdict
from datetime import datetime
import json
from types import SimpleNamespace
from uuid import UUID


def save_registrations(directory, resolutions):
    from .console_runs import atomic_json
    atomic_json(directory/'recognition-registrations.json', json.loads(json.dumps(asdict(resolutions), default=str)))


def retained_database(directory, codec, raws):
    import psycopg
    from .console_stack import private_file
    from .db1_rows.records import Resolutions, RunResolution, Dimension, ResolvedModel, MediaResolution
    from .db1_context import CameraRegistration, ModelRegistration
    from .observation_evidence import RecognitionIdentity
    from .db1_adapter import TrustedTenant
    data = json.loads(private_file(directory/'recognition-registrations.json').read_text())
    access = json.loads(private_file(directory/'database-private.json').read_text())
    from .real_video.test_database import fixture_module
    from .recognition_policy_migration import migrate_empty_fixture
    params = fixture_module().provisioning_params(__import__('os').environ)
    with psycopg.connect(**{**params,'dbname':access['database']},autocommit=True) as connection:
        migrate_empty_fixture(connection,database=access['database'],validation=codec.validation)
    r = data['run']; c = r['camera']
    c.update(org_id=UUID(c['org_id']), valid_from=datetime.fromisoformat(c['valid_from']),
             valid_to=datetime.fromisoformat(c['valid_to']) if c['valid_to'] else None)
    r.update(org_id=UUID(r['org_id']), camera=CameraRegistration(**c), detector=ModelRegistration(**r['detector']),
             started_at=datetime.fromisoformat(r['started_at']), date_keys=tuple(r['date_keys']))
    media = []
    for m in data['media']:
        m.update(org_id=UUID(m['org_id']), ts=datetime.fromisoformat(m['ts']))
        media.append(MediaResolution(**m))
    res = Resolutions(run=RunResolution(**r), models=tuple(ResolvedModel(RecognitionIdentity(**v['identity']),v['key']) for v in data['models']),
        media=tuple(media), **{k:tuple(Dimension(**v) for v in data[k]) for k in ('classes','colours','plate_categories','manufacturers')})
    for raw in raws:
        message = codec.decode(raw)
        if message.life.context.run_id != res.run.run_id:
            raise ValueError('retained registration belongs to another run')
    fixture = SimpleNamespace(res=res,
        tenant=TrustedTenant(res.run.org_id,'retained local recognition test registrations'),
        connect=lambda:psycopg.connect(**access['application_params'],autocommit=True))
    from .manufacturer.test_database import ManufacturerTestDatabase
    with psycopg.connect(**{**params,'dbname':access['database']},autocommit=True) as connection:
        fixture.setup = connection
        ManufacturerTestDatabase.extend(SimpleNamespace(fixture=fixture),codec,raws)
    save_registrations(directory, fixture.res)
    return SimpleNamespace(fixture=fixture, close=lambda:None)
