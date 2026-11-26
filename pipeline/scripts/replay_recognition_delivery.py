"""Recover saved scalar deliveries without reopening video or executing models."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    from vehicle.common.console_stack import configure
    from vehicle.common.console_runs import atomic_json
    from vehicle.common.recognition_transport import RecognitionCodec, RecognitionWorker
    from vehicle.common.recognition_run import public_row
    from vehicle.common.mohamed_lpr_redis_runtime import OwnedRedis
    from vehicle.common.redis_scalar.journal import PRODUCE
    from vehicle.common.recognition_db1 import deliver
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path)
    args=p.parse_args();directory=args.directory.resolve(strict=True)
    configure()
    if os.environ.get('WAYCON_POSTGRES_MODE')=='existing':
        os.environ['DB1_P4_EXISTING_SERVER']='1';os.environ.pop('DB1_P4_ISOLATED',None)
    control=json.loads((directory/'recognition-db1-control.json').read_text())
    codec=RecognitionCodec(control['org'],control['identity_namespace'])
    files=sorted(directory.glob('recognition-[0-9]*.json'),key=lambda p:int(p.stem.split('-')[1]))
    if not files:raise ValueError('no retained engine evidence')
    rows=[]
    with OwnedRedis(os.environ.get('T1_REDIS_SERVER') or shutil.which('redis-server')) as redis:
        worker=RecognitionWorker(redis.client,codec,control['namespace'])
        try:
            for index,path in enumerate(files,1):
                raw=path.read_bytes();message=codec.decode(raw)
                if message.kind!='engine.recognition.v3':raise ValueError('wrong replay input')
                redis.client.eval(PRODUCE,1,worker.source,f'{index}-0',raw,16384)
                while worker.step():pass
                bundle=codec.decode(codec.make('db1.delivery.v1',message.body['evidence'])).body['bundle']
                rows.append(public_row(message.body['evidence'],bundle))
            control['redis_socket']=str(redis.socket)
            atomic_json(directory/'recognition-db1-control.json',control)
            worker.journal.release()
            deliver(directory)
        finally:
            if worker.journal.token:worker.journal.release()
    progress=json.loads((directory/'recognition-db1-progress.json').read_text())
    for row in rows:row['delivery'].update(progress['tracks'].get(str(row['track_id']),{}))
    status=json.loads((directory/'status.json').read_text())
    status.update(rows=rows,recovery=dict(recognition_reexecuted=False,events=len(files)),
                  delivery=dict(postgres_commit='committed',redis_ack='acknowledged',review_ack='acknowledged'))
    if status.get('error_code')=='db1_delivery_failed':
        status.update(status='completed' if status.get('end_reason')=='eof' else 'partial',
            error_code=None,phase='Recovered committed recognition delivery',
            full_video_complete=status.get('source_kind')=='mp4' and status.get('end_reason')=='eof')
    atomic_json(directory/'status.json',status)
    print(json.dumps(dict(status=progress['status'],tracks=len(rows),recognition_reexecuted=False)))


if __name__=='__main__':main()
