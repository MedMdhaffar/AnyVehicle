"""One inference host across console and manufacturer CLI, independent of browser."""
import fcntl
import os
from pathlib import Path


class GPUBusy(RuntimeError):
    pass


class GPULease:
    def __init__(self):
        path=Path(os.environ.get('WAYCON_GPU_LOCK', Path(__file__).resolve().parents[3]/'.waycon-gpu.lock'))
        self.fd=os.open(path, os.O_CREAT|os.O_RDWR|os.O_CLOEXEC, 0o600)
        try:fcntl.flock(self.fd, fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError:
            os.close(self.fd);self.fd=None
            raise GPUBusy('Another video inference host is using the GPU.') from None
        self.previous=os.environ.get('WAYCON_GPU_LEASE_FD')
        os.environ['WAYCON_GPU_LEASE_FD']=str(self.fd)

    def close(self):
        if self.fd is not None:
            if os.environ.get('WAYCON_GPU_LEASE_FD')==str(self.fd):
                if self.previous is None:os.environ.pop('WAYCON_GPU_LEASE_FD',None)
                else:os.environ['WAYCON_GPU_LEASE_FD']=self.previous
            os.close(self.fd);self.fd=None


def inherited_lease():
    """Native children retain the lease even if their inference host crashes."""
    value=os.environ.get('WAYCON_GPU_LEASE_FD')
    return (int(value),) if value is not None else ()
