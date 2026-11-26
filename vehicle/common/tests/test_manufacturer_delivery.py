"""Focused v2 admission, durable retry, mapping and commit/ACK regressions."""
from contextlib import nullcontext
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from vehicle.common.db1_consumer import NaturalConsumer
from vehicle.common.db1_adapter import TrustedTenant
from vehicle.common.db1_outbox import LocalOutbox,AdmissionRejected
from vehicle.common.db1_rows import Dimension,ResolvedModel
from vehicle.common.db1_rows.core import Blocked
from vehicle.common.db1_rows.make import MakeActivation
from vehicle.common.manufacturer.delivery import ManufacturerDeliveryCodec
from vehicle.common.manufacturer.engine import ManufacturerLedger,SelectionConfig
from vehicle.common.observation_evidence import RecognitionIdentity
from vehicle.common.redis_scalar.worker import load_request_codec
from vehicle.common.redis_scalar.synthetic import ORG,IDENTITY_NAMESPACE
from vehicle.common.tests.test_db1_consumer import delivery_fixture,supplied_test_registrations,TransactionAdapter
from vehicle.common.tests.test_manufacturer import Runtime
from vehicle.common.tests.test_manufacturer_quality import textured


class ManufacturerDeliveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rc=load_request_codec(ORG,IDENTITY_NAMESPACE)
        cls.base=delivery_fixture(cls.rc)

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.model=RecognitionIdentity('synthetic-make-runtime',1,'vlm')
        self.activation=MakeActivation('test-only explicit mapping',self.model,(Runtime.provenance,),
            (('toyota','Toyota'),),'selected_crop_receipt','submitted_crops','omit')
        self.codec=ManufacturerDeliveryCodec(self.rc.v1,self.activation)
        self.life=self.rc.v1.decode(self.base).life
        self.res=replace(supplied_test_registrations(self.rc,self.base),
            manufacturers=(Dimension('Toyota',3,True),))
        self.res=replace(self.res,models=self.res.models+(ResolvedModel(self.model,84),))
        self.outbox=LocalOutbox(path=Path(self.tmp.name)/'outbox',namespace=IDENTITY_NAMESPACE,validation=self.rc.validation)
        self.addCleanup(self.outbox.close)
        self.adapter=TransactionAdapter(self.rc.validation)
        c=self.life.context
        self.consumer=NaturalConsumer(codec=self.codec,tenant=TrustedTenant(ORG,'test-only'),trusted_run=c.run_id,
            trusted_camera=c.camera.camera_id,resolutions=self.res,connection_factory=lambda:nullcontext(object()),
            environment='test',adapter=self.adapter,outbox=self.outbox)

    def wire(self,manufacturer='Toyota',error=None):
        ledger=ManufacturerLedger(enabled=True,runtime=Runtime(json.dumps(dict(manufacturer=manufacturer,
            vehicle_type='sedan',color='white')),error=error),journal=Path(self.tmp.name)/(manufacturer+'.sqlite'),
            selection=SelectionConfig(policy='quality_lifetime_v1'))
        try:
            crop=textured(440,300)
            ledger.observe(self.life,self.life.first_receipt,(10,10,450,310),crop,'f'*64,
                label='car',confidence=.9,frame_shape=(1080,1920,3))
            ledger.close_lifetime(self.life)
            evidence=ledger.terminal(self.life)
            return self.codec.encode(self.base,json.dumps(evidence))
        finally:ledger.close()

    def test_four_facts_outbox_reopen_retry_and_ack_after_commit(self):
        raw=self.wire();m=self.codec.decode(raw)
        before=self.rc.v1.decode(self.base).body['bundle']['facts']
        self.assertEqual(m.body['bundle']['facts'][:len(before)],before)
        def ack():
            self.assertEqual(len(self.adapter.facts),4)
            self.assertEqual(len(self.outbox.inspect_pending()),1)
        self.assertEqual(self.consumer.process(raw,ack).disposition,'new')
        self.outbox.close()
        self.outbox=LocalOutbox(path=Path(self.tmp.name)/'outbox',namespace=IDENTITY_NAMESPACE,validation=self.rc.validation)
        self.addCleanup(self.outbox.close);self.consumer.outbox=self.outbox
        self.assertEqual(self.consumer.process(raw,ack).disposition,'identical')
        row=self.outbox.inspect_pending()[0].construction.facts[-1].sql.value()
        self.assertEqual((row['manufacturer_key'],row['manufacturer_conf'],row['observations']),(3,None,1))

    def test_unmapped_unknown_and_failed_predictions_omit_without_losing_base_rows(self):
        for label,error in [('Kia',None),('unknown',None),('Toyota',RuntimeError('injected'))]:
            with self.subTest(label=label,error=error):
                message=self.codec.decode(self.wire(label,error))
                self.assertEqual(len(message.body['bundle']['facts']),3)
                self.assertNotEqual(message.body['manufacturer']['delivery_reason'],'mapped')

    def test_registration_missing_unapproved_and_provenance_mismatch_fail_closed(self):
        raw=self.wire()
        for makers in [(),(Dimension('Toyota',3,False),)]:
            self.consumer.resolutions=replace(self.res,manufacturers=makers)
            with self.assertRaises(Blocked):self.consumer.process(raw,lambda:self.fail('early ACK'))
        self.assertFalse(self.outbox.inspect_pending())
        wrong=ManufacturerDeliveryCodec(self.rc.v1,replace(self.activation,runtime_provenance=('c'*64,)))
        with self.assertRaises(Blocked):wrong.decode(raw)

    def test_changed_evidence_conflicts_in_durable_outbox_before_database_and_ack(self):
        raw=self.wire();self.consumer.process(raw,lambda:None)
        v=json.loads(raw)
        v['manufacturer']['seconds']+=.01
        changed=self.codec.encode(v['base_json'].encode(),json.dumps(v['manufacturer']))
        with self.assertRaises(AdmissionRejected):self.consumer.process(changed,lambda:self.fail('early ACK'))
        self.assertEqual(len(self.adapter.facts),4)

    def test_changed_hash_and_unversioned_consumer_reject(self):
        raw=self.wire();v=json.loads(raw);v['manufacturer']['text']='{}'
        with self.assertRaises(Blocked):self.codec.decode(json.dumps(v).encode())
        with self.assertRaises(Blocked):self.rc.v1.decode(raw)


if __name__=='__main__':unittest.main()
