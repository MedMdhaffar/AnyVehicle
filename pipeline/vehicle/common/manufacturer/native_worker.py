"""Private native worker; no Django, source checkout imports or video.

Default: one-shot (legacy runtime) — read one PNG from stdin, answer once.
``--serve``: persistent — load once, then answer length-prefixed PNG requests
until EOF or a zero-length frame. Each request resets native state (see
LlamaBackend.generate) so no image or KV state crosses requests.
"""
import hashlib
import json
import os
from pathlib import Path
import resource
import struct
import sys
import time
from types import SimpleNamespace

MAX_PNG = 8*1024*1024


def main():
    serve = sys.argv[2:] == ['--serve']
    # One-shot keeps the original 60 s CPU budget. A persistent worker's CPU
    # time accumulates across calls; the parent enforces each call's deadline.
    cpu = 6*3600 if serve else 60
    resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
    resource.setrlimit(resource.RLIMIT_FSIZE, (4096, 4096))
    resource.setrlimit(resource.RLIMIT_NOFILE, (128, 128))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    import socket
    def denied(*args, **kwargs):
        raise RuntimeError('manufacturer network disabled')
    socket.socket.connect = socket.socket.connect_ex = denied
    # The binding prints native diagnostics even with verbose=False. Keep those
    # off the result pipe, which admits only the bounded structured response.
    saved_stdout = os.dup(1)
    with open(os.devnull, 'wb') as sink:
        os.dup2(sink.fileno(), 1)
    lock_path=Path(sys.argv[1]).resolve()
    lock = json.loads(lock_path.read_text())
    for item in lock['artifacts'].values():
        path=Path(item['path']).expanduser()
        item['path']=str(path if path.is_absolute() else lock_path.parent/path)
    for key in ('model', 'projector'):
        item = lock['artifacts'][key]
        with Path(item['path']).open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != item['sha256']:
                raise RuntimeError('manufacturer asset changed')
    # -I excludes ambient PYTHONPATH; admit only the target-owned adapter directory.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from llama_backend import LlamaBackend
    from prompts import PROMPT
    import ctypes
    cuda = ctypes.CDLL(lock.get('cuda_runtime_library','libcudart.so.13'))
    def memory():
        free, total = ctypes.c_size_t(), ctypes.c_size_t()
        if cuda.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total)) != 0:
            return None
        return dict(used_mib=(total.value-free.value)/2**20, total_mib=total.value/2**20)
    def answer(png, loaded):
        if len(png) > MAX_PNG or not png.startswith(b'\x89PNG\r\n\x1a\n'):
            raise RuntimeError('invalid crop transport')
        if cuda.cudaDeviceSynchronize()!=0:raise RuntimeError('CUDA synchronization failed')
        start=time.perf_counter_ns()
        response = backend.generate(png, PROMPT, temperature=0.0, top_p=1.0, max_tokens=128)
        if cuda.cudaDeviceSynchronize()!=0:raise RuntimeError('CUDA synchronization failed')
        completion=time.perf_counter_ns()
        choice = response['choices'][0]
        text = choice['message']['content']
        valid = choice['finish_reason'] == 'stop' and type(text) is str and len(text.encode()) <= 2048
        return dict(outcome='completed' if valid else 'malformed', text=text if valid else '',
                    timing=dict(start_ns=str(start),completion_ns=str(completion),
                        duration_ms=(completion-start)/1e6,load_time_ms=load_ms,
                        load_start_ns=str(load_start),load_completion_ns=str(load_completion)),
                    gpu_memory=dict(after_load=loaded, after_inference=memory(),
                                    scope='CUDA-visible instance, includes other processes; not a peak'))
    if not serve:
        png = sys.stdin.buffer.read(MAX_PNG+1)
        if len(png) > MAX_PNG or not png.startswith(b'\x89PNG\r\n\x1a\n'):
            raise RuntimeError('invalid crop transport')
    load_start=time.perf_counter_ns()
    backend = LlamaBackend(SimpleNamespace(device='cuda:0', context_size=4096, max_new_tokens=128),
                           lock['artifacts']['model']['path'], lock['artifacts']['projector']['path'])
    if cuda.cudaDeviceSynchronize()!=0:raise RuntimeError('CUDA synchronization failed')
    load_completion=time.perf_counter_ns()
    load_ms=(load_completion-load_start)/1e6
    try:
        loaded = memory()
        if not serve:
            result = answer(png, loaded)
        else:
            out = os.fdopen(saved_stdout, 'wb', buffering=0)
            def send(value):
                data = json.dumps(value, allow_nan=False).encode()
                out.write(struct.pack('>I', len(data)) + data)
            send(dict(ready=True, gpu_memory=dict(after_load=loaded)))
            source = sys.stdin.buffer
            while True:
                header = source.read(4)
                if len(header) < 4:
                    break
                size = struct.unpack('>I', header)[0]
                if size == 0:
                    break
                if size > MAX_PNG:
                    raise RuntimeError('invalid crop transport')
                png = source.read(size)
                if len(png) != size:
                    break
                send(answer(png, loaded))
            return
    finally:
        backend.close()
    sys.stdout.flush()
    os.dup2(saved_stdout, 1)
    os.close(saved_stdout)
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()
