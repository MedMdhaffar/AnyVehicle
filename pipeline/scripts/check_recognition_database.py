"""Read-only independent normal-role checks of retained recognition facts/RLS."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from uuid import UUID

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    import psycopg
    from psycopg import sql
    from vehicle.common.console_stack import private_file
    from vehicle.common.recognition_transport import RecognitionCodec
    from vehicle.common.console_runs import atomic_json
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path)
    directory=parser.parse_args().directory.resolve(strict=True)
    private=json.loads(private_file(directory/'database-private.json').read_text())
    control=json.loads((directory/'recognition-db1-control.json').read_text())
    codec=RecognitionCodec(control['org'],control['identity_namespace'])
    expected=[]
    for p in directory.glob('recognition-[0-9]*.json'):
        evidence=codec.decode(p.read_bytes()).body['evidence']
        expected.extend(codec.decode(codec.make('db1.delivery.v1',evidence)).body['bundle']['facts'])
    counts={}
    with psycopg.connect(**private['application_params'],autocommit=True) as connection, connection.transaction(), connection.cursor() as q:
        q.execute('SELECT rolsuper,rolbypassrls FROM pg_roles WHERE rolname=current_user')
        assert q.fetchone()==(False,False)
        q.execute("SELECT set_config('app.org_id',%s,true)",(control['org'],))
        for fact in expected:
            kind,row=fact['row_kind'],fact['delivery']
            q.execute(sql.SQL('SELECT track_id,ts FROM public.{} WHERE uid=%s').format(sql.Identifier(kind)),(row['uid'],))
            actual=q.fetchone()
            assert actual==(row['track_id'],datetime.fromisoformat(row['ts'].replace('Z','+00:00')))
            if kind!='fact_vehicle':
                model=row['selected_attempt']['model'] if kind=='fact_plate_read' else row['model']
                q.execute(sql.SQL('SELECT m.model_name,m.task FROM public.{} f JOIN public.dim_model m ON f.model_key=m.model_key WHERE uid=%s').format(sql.Identifier(kind)),(row['uid'],))
                assert q.fetchone()==(model['model_name'],model['task'])
            if kind=='fact_vehicle_colour':
                q.execute('SELECT conf FROM public.fact_vehicle_colour WHERE uid=%s',(row['uid'],));assert q.fetchone()==(None,)
            if kind=='fact_plate_read':
                q.execute('SELECT attempts,plate_text FROM public.fact_plate_read WHERE uid=%s',(row['uid'],))
                assert q.fetchone()==(row['attempts'],row['selected_attempt']['plate_text'])
            if kind=='fact_vehicle_make':
                q.execute('SELECT m.name,f.manufacturer_conf FROM public.fact_vehicle_make f '
                          'LEFT JOIN public.dim_manufacturer m ON m.manufacturer_key=f.manufacturer_key WHERE f.uid=%s',
                          (row['uid'],))
                assert q.fetchone()==(row['manufacturer'],None)
        for table in ('fact_vehicle','fact_vehicle_colour','fact_vehicle_make','fact_plate_read'):
            q.execute(sql.SQL('SELECT count(*) FROM public.{}').format(sql.Identifier(table)))
            counts[table]=q.fetchone()[0]
        q.execute('SELECT count(*) FROM db1_delivery.acceptance');acceptances=q.fetchone()[0]
        q.execute("SELECT set_config('app.org_id',%s,true)",(str(UUID(int=2)),))
        for table in counts:
            q.execute(sql.SQL('SELECT count(*) FROM public.{}').format(sql.Identifier(table)));assert q.fetchone()==(0,)
    report=dict(status='passed',counts=counts,acceptances=acceptances,expected_facts=len(expected),
        exact_timestamps_and_association=True,model_attribution_verified=True,missing_confidence_preserved=True,
        normal_role=True,cross_tenant_rows_visible=0,manufacturer_names_verified=True)
    atomic_json(directory/'recognition-readback.json',report)
    print(json.dumps(report))


if __name__=='__main__':main()
