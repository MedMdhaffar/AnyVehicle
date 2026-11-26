"""Bounded local video uploads. Media bytes never pass through request.data."""
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from uuid import UUID, uuid4

from .console_runs import atomic_json, timestamp

CHUNK_BYTES = 1024 * 1024


def server_video(value):
    """Validate a server-local video without copying or taking ownership of it."""
    if not isinstance(value, str) or not value.strip():
        raise UploadRejected('Enter an absolute video path on the server.')
    value = value.strip()
    path = Path(value)
    # Accept a path pasted from a shell, including quoted filename components.
    # Never execute it or expand variables; prefer a literal existing filename.
    if not path.exists():
        try:
            parts = shlex.split(value)
            if len(parts)==1:path=Path(parts[0])
        except ValueError:
            raise UploadRejected('The server video path has unmatched quotes.') from None
    if not path.is_absolute():raise UploadRejected('Enter an absolute video path on the server.')
    try:path = path.resolve(strict=True)
    except (FileNotFoundError, OSError, ValueError):
        raise UploadRejected('Server video not found. Check the path on the remote server.') from None
    if not path.is_file():raise UploadRejected('The server video path must point to a regular video file.')
    try:details = probe(path)
    except Exception:
        raise UploadRejected('The server video could not be decoded. Check file permissions and video format.') from None
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(CHUNK_BYTES), b''):digest.update(chunk)
    return path, dict(details, upload_id=None, filename=path.name, status='ready',
        bytes=path.stat().st_size, sha256=digest.hexdigest(), source='server_path')


class UploadRejected(ValueError):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def probe(path):
    """Inspect container contents and decode a real frame in a bounded CPU child."""
    result = subprocess.run([sys.executable, '-B', '-m', 'vehicle.common.video_uploads', str(path)],
        cwd=Path(__file__).resolve().parents[2], capture_output=True, timeout=35,
        env={**os.environ, 'OPENCV_FFMPEG_CAPTURE_OPTIONS':'protocol_whitelist;file,pipe',
             'OPENBLAS_NUM_THREADS':'1', 'OMP_NUM_THREADS':'1'})
    if result.returncode:
        raise UploadRejected('This file is not a video supported by the server decoder.')
    return json.loads(result.stdout)


def probe_local(path):
    # ffprobe excludes disguised still images and refuses network/playlist URLs.
    result = subprocess.run([shutil.which('ffprobe') or 'ffprobe', '-v', 'error',
        '-protocol_whitelist', 'file,pipe', '-show_format', '-show_streams', '-of', 'json', str(path)],
        capture_output=True, timeout=15, check=True)
    info = json.loads(result.stdout)
    fmt = info.get('format', {}).get('format_name', '')
    if not fmt or any(v in fmt for v in ('image2', '_pipe', 'hls', 'concat')):
        raise ValueError('unsupported_container')
    streams = [s for s in info['streams'] if s['codec_type']=='video' and not s.get('disposition', {}).get('attached_pic')]
    if not streams:raise ValueError('video_stream_required')
    import cv2
    cv2.setNumThreads(1)
    capture = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    try:
        ok, frame = capture.read()
        if not capture.isOpened() or not ok or frame is None:raise ValueError('undecodable_video')
        frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = capture.get(cv2.CAP_PROP_FPS)
        exact_frames = streams[0].get('nb_frames','')
        frame_count_exact = str(exact_frames).isdigit() and int(exact_frames)>0
        content_type = {'avi':'video/x-msvideo','matroska,webm':'video/x-matroska',
            'mpegts':'video/mp2t','asf':'video/x-ms-asf'}.get(fmt, 'video/'+fmt.split(',')[0])
        if 'mp4' in fmt:
            content_type = 'video/quicktime' if info.get('format',{}).get('tags',{}).get('major_brand','').strip()=='qt' else 'video/mp4'
        return dict(container=fmt, codec=streams[0]['codec_name'], width=frame.shape[1], height=frame.shape[0],
            content_type=content_type,
            frame_count_exact=frame_count_exact,
            total_frames=int(exact_frames) if frame_count_exact else int(frames) if math.isfinite(frames) and frames>0 else None,
            fps=fps if math.isfinite(fps) and fps>0 else None)
    finally:capture.release()


class UploadStore:
    def __init__(self, root=None):
        self.root = Path(root or os.environ.get('WAYCON_UPLOAD_ROOT',
            Path(__file__).resolve().parents[3]/'pipeline-uploads')).expanduser()
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.root = self.root.resolve(strict=True)
        repository = Path(__file__).resolve().parents[2]
        if self.root == repository or repository in self.root.parents:
            raise UploadRejected('WAYCON_UPLOAD_ROOT must be outside the source repository.', 503)
        self.max_bytes = int(os.environ.get('WAYCON_UPLOAD_MAX_BYTES', 8*1024**3))
        self.reserve_bytes = int(os.environ.get('WAYCON_UPLOAD_RESERVE_BYTES', 1024**3))

    def limits(self):
        return dict(max_bytes=self.max_bytes, available_bytes=max(0, shutil.disk_usage(self.root).free-self.reserve_bytes))

    def get(self, upload_id):
        if str(UUID(upload_id)) != upload_id:raise ValueError('invalid_upload_id')
        directory = self.root/upload_id
        if directory.is_symlink():raise ValueError('invalid_upload')
        metadata = json.loads((directory/'upload.json').read_text())
        path = directory/'source.video'
        if metadata['status']!='ready' or path.is_symlink() or path.stat().st_size!=metadata['bytes']:
            raise ValueError('upload_unavailable')
        return path, metadata

    def receive(self, stream, size, filename):
        if type(size) is not int or size<=0:raise UploadRejected('An explicit nonempty upload length is required.', 411)
        if size>self.max_bytes:raise UploadRejected('Video exceeds the configured upload size limit.', 413)
        # Serialize disk reservations across server processes and upload threads.
        with (self.root/'.upload.lock').open('a+b') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if size>self.limits()['available_bytes']:raise UploadRejected('Insufficient free upload disk space.', 507)
            upload_id = str(uuid4())
            directory = self.root/upload_id
            directory.mkdir(mode=0o700)
            # Original name is display metadata only; never a server path.
            name = filename.replace('\\', '/').rsplit('/', 1)[-1]
            name = ''.join(c for c in name if c.isprintable())[:240] or 'video'
            metadata = dict(upload_id=upload_id, filename=name, status='uploading', created_at=timestamp(), bytes=0)
            atomic_json(directory/'upload.json', metadata)
            digest, count = hashlib.sha256(), 0
            try:
                fd = os.open(directory/'source.video', os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o600)
                with os.fdopen(fd, 'wb') as target:
                    while count<size:
                        chunk = stream.read(min(CHUNK_BYTES, size-count))
                        if not chunk:raise UploadRejected('Upload disconnected before completion.')
                        if count+len(chunk)>size:raise UploadRejected('Upload length mismatch.')
                        if len(chunk)>self.limits()['available_bytes']:raise UploadRejected('Upload disk reserve reached.', 507)
                        target.write(chunk);digest.update(chunk);count+=len(chunk)
                    target.flush();os.fsync(target.fileno())
                details = probe(directory/'source.video')
                metadata.update(details, status='ready', bytes=count, sha256=digest.hexdigest())
                atomic_json(directory/'upload.json', metadata)
                return metadata
            except Exception:
                # Retain failed uploads as well; no automatic media deletion.
                metadata.update(status='rejected', bytes=count)
                atomic_json(directory/'upload.json', metadata)
                raise


if __name__=='__main__':
    try:print(json.dumps(probe_local(Path(sys.argv[1]).resolve(strict=True)), allow_nan=False))
    except Exception:raise SystemExit(1) from None
