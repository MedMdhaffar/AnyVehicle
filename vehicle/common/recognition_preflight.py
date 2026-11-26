"""Probe the same model construction used by MP4 and RTSP, without an input."""
from datetime import datetime, timezone
from contextlib import closing
import os
from pathlib import Path
import tempfile
import time
from uuid import UUID


def check(directory, mode):
    from .console_runs import atomic_json
    from .console_stack import database_probe, console_device
    from .gpu_lease import GPULease
    from .db1_context import RunContext, CameraRegistration, ModelRegistration, ResolutionInfo
    from .frame_receipt import FrameReceiptClock, ClockPolicy
    from .real_video.plate_engine import CombinedEngine
    from .real_video.recognition_engine import RecognitionEngine
    from .real_video.input import DecodedFrame
    from .mohamed_lpr_redis_runtime import OwnedRedis
    from ..recognition.config import paths
    from .recognition_transport import RecognitionCodec
    import torch
    import numpy as np
    report = dict(status='checking', mode=mode, failures=[], manufacturer=dict(available=False))
    engine = None
    try:
        if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
        with closing(GPULease()), tempfile.TemporaryDirectory(prefix='recognition-preflight-') as temp:
            now=datetime.now(timezone.utc);org=UUID('12345678-1234-4234-8234-123456789012')
            detector=ModelRegistration(None,'vehicle_yolo26.pt-sha256:'+RecognitionEngine._hash(paths()['vehicle_model']),1,'detect')
            context=RunContext(org,CameraRegistration(org,'preflight',None,'Africa/Tunis',now,None),'preflight','file',now,
                'vehicle-recognition-v3',ResolutionInfo(None,None,'vehicle',detector,None,'preflight development probe',False),'development')
            clock=FrameReceiptClock(context,utc_clock=lambda:datetime.now(timezone.utc),monotonic_clock=time.monotonic,policy=ClockPolicy())
            engine=CombinedEngine.recognition(context,clock,Path(temp),attributes=True)
            engine.process(DecodedFrame(clock.capture(0),np.zeros((640,640,3),dtype=np.uint8)))
            engine.finish()
            report.update(cuda=dict(device=console_device(),name=torch.cuda.get_device_name(console_device())),
                          models=engine.models, manufacturer=dict(available=True), recognition_version='vehicle-recognition-v3')
            engine.close();engine=None
            with OwnedRedis(os.environ['T1_REDIS_SERVER']) as redis:
                report['redis']=dict(run_owned=True,ping=redis.client.ping())
            if mode=='fixture':
                report['database']=database_probe()
    except Exception as error:
        import traceback
        report['failures'].append(type(error).__name__+': '+str(error)[:300])
        report['locations']=[dict(file=Path(f.filename).name,line=f.lineno,function=f.name) for f in traceback.extract_tb(error.__traceback__)]
    finally:
        if engine is not None: engine.close()
    report['status']='failed' if report['failures'] else 'ready'
    atomic_json(Path(directory)/'preflight.json',report)
    return 1 if report['failures'] else 0
