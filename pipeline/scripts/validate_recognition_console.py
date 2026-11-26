"""Exercise normal console HTTP admission and owned native worker, in an isolated run root."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--video',type=Path)
    parser.add_argument('--rtsp')
    parser.add_argument('--loopback-video',type=Path)
    parser.add_argument('--loopback-ready',action='store_true',help='Use an already prepared continuous H.264 test clip')
    parser.add_argument('--transport',default='tcp')
    parser.add_argument('--seconds',type=float)
    parser.add_argument('--stop-after',type=float)
    parser.add_argument('--sample-every',type=int,default=1)
    parser.add_argument('--database',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(mode=0o700,parents=True,exist_ok=False)
    camera = None
    if args.loopback_video:
        clip=args.loopback_video if args.loopback_ready else args.output/'loopback.mp4'
        if not args.loopback_ready:
            subprocess.run(['/usr/bin/ffmpeg','-v','error','-stream_loop','-1','-i',str(args.loopback_video),'-t','180','-an','-c:v','libx264',
                '-preset','ultrafast','-tune','zerolatency','-g','25','-x264-params','repeat-headers=1',str(clip)],check=True)
        with socket.socket() as probe:
            probe.bind(('127.0.0.1',0));camera_port=probe.getsockname()[1]
        args.rtsp=f'rtsp://127.0.0.1:{camera_port}/recognition'
        cfg=args.output/'vlc.conf'
        cfg.write_text('sout=#rtp{sdp='+args.rtsp+'}\nsout-keep=1\nrtsp-host=127.0.0.1\n')
        camera_log=(args.output/'camera.log').open('wb')
        camera=subprocess.Popen(['/usr/bin/cvlc','--config',str(cfg.resolve()),'--intf','dummy','--no-audio',
            '--loop','--no-media-library',str(clip.resolve())],stdout=camera_log,stderr=camera_log,
            env={**os.environ,'XDG_CONFIG_HOME':str(args.output.resolve()/'vlc-config'),
                 'XDG_CACHE_HOME':str(args.output.resolve()/'vlc-cache')})
        args.transport='udp'
    from vehicle.common.console_stack import configure
    configure()
    with socket.socket() as probe:
        probe.bind(('127.0.0.1',0));port=probe.getsockname()[1]
    env={**os.environ,'T1_CONSOLE_PYTHON':sys.executable,'T1_CONSOLE_RUN_ROOT':str(args.output.resolve()/'runs'),
        'T1_CONSOLE_DB1':'fixture' if args.database else 'console-only',
        'WAYCON_UPLOAD_ROOT':'/tmp/waycon-recognition-validation-uploads','PYTHONDONTWRITEBYTECODE':'1'}
    env.pop('WAYCON_LAUNCHER_DRAIN',None)
    env.pop('WAYCON_CONSOLE_TEST_PROFILE',None)
    if args.database:
        env['DB1_P4_EXISTING_SERVER']='1';env.pop('DB1_P4_ISOLATED',None)
    api=f'http://127.0.0.1:{port}'
    def request(path,body=None,headers=None):
        with urlopen(Request(api+path,data=body,headers=headers or {}),timeout=40) as response:
            return json.load(response)
    with (args.output/'django.log').open('wb') as log:
        server=subprocess.Popen([sys.executable,'-B',str(ROOT/'vehicle/manage.py'),'runserver',f'127.0.0.1:{port}','--noreload'],env=env,stdout=log,stderr=log)
        run=None
        try:
            deadline=time.monotonic()+40
            while time.monotonic()<deadline:
                try: request('/api/t1/runs/');break
                except OSError:time.sleep(.2)
            else:raise RuntimeError('validation backend failed to start')
            if args.video:
                with args.video.open('rb') as f:
                    upload=request('/api/t1/uploads/',f.read(),{'Content-Type':'application/octet-stream','X-Video-Filename':args.video.name})
                payload=dict(source_kind='mp4',upload_id=upload['upload_id'],manufacturer_enabled=True,sample_every=1)
            else:
                payload=dict(source_kind='rtsp',rtsp_uri=args.rtsp,rtsp_transport=args.transport,manufacturer_enabled=True,sample_every=args.sample_every)
            if args.seconds:payload['max_seconds']=args.seconds
            run=request('/api/t1/runs/',json.dumps(payload).encode(),{'Content-Type':'application/json'})
            (args.output/'admission.json').write_text(json.dumps(run))
            began=time.monotonic();stopped=False;processing_began=None
            print('Admitted run',run['run_id'],flush=True)
            while time.monotonic()-began<1800:
                status=request('/api/t1/runs/'+run['run_id']+'/')
                (args.output/'latest.json').write_text(json.dumps(status))
                if processing_began is None and status.get('analyzed_frames',0)>0:processing_began=time.monotonic()
                if args.stop_after and not stopped and processing_began is not None and time.monotonic()-processing_began>=args.stop_after:
                    request('/api/t1/runs/'+run['run_id']+'/stop/',b'');stopped=True
                if status['status'] in ('completed','failed','stopped','partial') and not status.get('active_slot'):
                    print(json.dumps({k:status.get(k) for k in ('status','error_code','analyzed_frames','counters','delivery','cleanup')}),flush=True)
                    return 1 if status['status']=='failed' else 0
                time.sleep(1)
            raise TimeoutError('native run did not finish within validation deadline')
        finally:
            if run:
                try:request('/api/t1/runs/'+run['run_id']+'/stop/',b'')
                except OSError:pass
            server.terminate()
            server.wait(timeout=15)
            if camera:
                camera.terminate();camera.wait(timeout=10);camera_log.close()


if __name__=='__main__':raise SystemExit(main())
