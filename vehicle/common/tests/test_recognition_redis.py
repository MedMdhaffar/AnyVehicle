"""Real private Redis recovery for the v3 recognition aggregator."""
import os
import unittest

from vehicle.common.mohamed_lpr_redis_runtime import OwnedRedis
from vehicle.common.recognition_transport import RecognitionCodec, RecognitionWorker
from vehicle.common.redis_scalar.synthetic import ORG, IDENTITY_NAMESPACE
from vehicle.common.redis_scalar.journal import PRODUCE
from vehicle.common.tests.test_recognition_v3 import evidence_fixture


@unittest.skipUnless(os.environ.get('T1_REDIS_SERVER'), 'explicit Redis executable required')
class RedisRecognitionTests(unittest.TestCase):
    def test_death_after_publication_repairs_exact_output_before_ack(self):
        codec=RecognitionCodec(ORG,IDENTITY_NAMESPACE)
        with OwnedRedis(os.environ['T1_REDIS_SERVER']) as redis:
            worker=RecognitionWorker(redis.client,codec,'recognition-repair-test')
            raw=codec.make('engine.recognition.v3',evidence_fixture())
            redis.client.eval(PRODUCE,1,worker.source,'1-0',raw,16384)
            def die(phase):
                if phase=='published':raise RuntimeError('simulated owner death after publication')
            worker.journal.hook=die
            with self.assertRaises(RuntimeError):worker.step()
            self.assertEqual(redis.client.xpending(worker.source,worker.group)['pending'],1)
            output=worker.journal.key('db1.delivery.v1')
            original=redis.client.xrange(output)
            self.assertEqual(len(original),1)
            worker.journal.release()
            replacement=RecognitionWorker(redis.client,codec,'recognition-repair-test')
            replacement.step()
            self.assertEqual(redis.client.xrange(output),original)
            self.assertEqual(redis.client.xpending(worker.source,worker.group)['pending'],0)
            replacement.journal.release()

    def test_duplicate_evidence_does_not_duplicate_delivery(self):
        codec=RecognitionCodec(ORG,IDENTITY_NAMESPACE)
        with OwnedRedis(os.environ['T1_REDIS_SERVER']) as redis:
            worker=RecognitionWorker(redis.client,codec,'recognition-dedupe-test')
            raw=codec.make('engine.recognition.v3',evidence_fixture())
            for entry in ('1-0','2-0'):
                redis.client.eval(PRODUCE,1,worker.source,entry,raw,16384)
                self.assertTrue(worker.step())
            self.assertEqual(redis.client.xlen(worker.journal.key('db1.delivery.v1')),1)
            worker.journal.release()
