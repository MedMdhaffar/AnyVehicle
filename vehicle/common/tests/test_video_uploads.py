"""Upload bounds, real containers and the shared GPU admission lease; no models."""
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from vehicle.common.video_uploads import UploadStore, UploadRejected, CHUNK_BYTES, probe
from vehicle.common.gpu_lease import GPULease, GPUBusy, inherited_lease

DETAILS=dict(container='mov,mp4',codec='h264',width=64,height=48,fps=25,total_frames=2)


class VideoUploadTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='waycon-upload-')
        self.addCleanup(self.tmp.cleanup)
        self.store=UploadStore(self.tmp.name)

    def test_large_stream_uses_bounded_reads_and_unique_paths(self):
        class Stream:
            remaining=CHUNK_BYTES*17+3
            reads=[]
            def read(s,n):
                self.assertLessEqual(n,CHUNK_BYTES);s.reads.append(n)
                size=min(s.remaining,n);s.remaining-=size
                return b'x'*size
        stream=Stream()
        with patch('vehicle.common.video_uploads.probe',return_value=DETAILS):
            first=self.store.receive(stream,stream.remaining,'../../unsafe.mkv')
            second=self.store.receive(io.BytesIO(b'other'),5,'unsafe.mkv')
        self.assertGreater(len(stream.reads),16)
        self.assertNotEqual(first['upload_id'],second['upload_id'])
        self.assertEqual(first['filename'],'unsafe.mkv')
        path,metadata=self.store.get(first['upload_id'])
        self.assertEqual(path.name,'source.video')
        self.assertEqual(path.stat().st_size,CHUNK_BYTES*17+3)
        self.assertEqual(metadata['sha256'],first['sha256'])
        self.assertEqual(path.stat().st_mode & 0o777,0o600)

    def test_size_and_disk_limits_reject_before_read(self):
        stream=SimpleNamespace(read=lambda _:self.fail('body must remain unread'))
        self.store.max_bytes=4
        with self.assertRaises(UploadRejected) as size:self.store.receive(stream,5,'a.mp4')
        self.assertEqual(size.exception.status,413)
        self.store.max_bytes=100
        with patch('vehicle.common.video_uploads.shutil.disk_usage',return_value=SimpleNamespace(free=1)):
            with self.assertRaises(UploadRejected) as disk:self.store.receive(stream,5,'a.mp4')
        self.assertEqual(disk.exception.status,507)
        self.assertEqual(list(Path(self.tmp.name).glob('*/upload.json')),[])

    def test_disconnect_keeps_rejected_bytes_and_cannot_start(self):
        with self.assertRaises(UploadRejected):self.store.receive(io.BytesIO(b'partial'),20,'a.mov')
        metadata=json.loads(next(Path(self.tmp.name).glob('*/upload.json')).read_text())
        self.assertEqual(metadata['status'],'rejected')
        with self.assertRaises(ValueError):self.store.get(metadata['upload_id'])
        self.assertEqual(next(Path(self.tmp.name).glob('*/source.video')).read_bytes(),b'partial')

    def test_bad_id_and_changed_size_rejected(self):
        with self.assertRaises(ValueError):self.store.get('../../private')
        with patch('vehicle.common.video_uploads.probe',return_value=DETAILS):
            item=self.store.receive(io.BytesIO(b'video'),5,'a.mp4')
        path,_=self.store.get(item['upload_id']);path.write_bytes(b'changed')
        with self.assertRaises(ValueError):self.store.get(item['upload_id'])

    def test_gpu_lease_excludes_other_hosts_and_releases(self):
        with patch.dict(os.environ,{'WAYCON_GPU_LOCK':str(Path(self.tmp.name)/'gpu.lock')}):
            first=GPULease()
            try:
                with self.assertRaises(GPUBusy):GPULease()
            finally:first.close()
            GPULease().close()

    def test_native_child_holds_gpu_lease_after_host_releases_it(self):
        import sys
        with patch.dict(os.environ,{'WAYCON_GPU_LOCK':str(Path(self.tmp.name)/'gpu.lock')}):
            lease=GPULease()
            child=subprocess.Popen([sys.executable,'-B','-c','import sys; sys.stdin.buffer.read()'],
                pass_fds=inherited_lease(),stdin=subprocess.PIPE)
            lease.close()
            try:
                with self.assertRaises(GPUBusy):GPULease()
            finally:
                child.stdin.close();child.wait(timeout=5)
            GPULease().close()

    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'),'requires existing decoder tools')
    def test_real_mp4_mov_avi_mkv_contents_with_generated_server_extension(self):
        for extension in ('mp4','mov','avi','mkv'):
            with self.subTest(extension=extension):
                video=Path(self.tmp.name)/('tiny.'+extension)
                subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','testsrc2=size=64x48:rate=25',
                    '-frames:v','2','-c:v','mpeg4',str(video)],check=True,timeout=15)
                with video.open('rb') as stream:metadata=self.store.receive(stream,video.stat().st_size,video.name)
                self.assertEqual(metadata['status'],'ready')
                self.assertEqual(metadata['total_frames'],2)
                self.assertEqual(metadata['width'],64)
        disguised=Path(self.tmp.name)/'fake.mp4';disguised.write_text('not a video')
        with self.assertRaises(UploadRejected):probe(disguised)
