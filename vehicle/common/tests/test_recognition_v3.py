"""New recognition boundaries: attribution, missing confidence and replay."""
from copy import deepcopy
from contextlib import nullcontext
import unittest

from contracts.db1.reference import canonical_json_bytes
from vehicle.common.recognition_transport import RecognitionCodec, aggregate
from vehicle.common.recognition_run import public_row
from vehicle.common.redis_scalar.worker import load_request_codec
from vehicle.common.redis_scalar.synthetic import ORG, IDENTITY_NAMESPACE
from vehicle.common.redis_scalar.codec import pack_life, pack_receipt
from vehicle.common.tests.test_db1_consumer import delivery_fixture, supplied_test_registrations, TransactionAdapter
from vehicle.common.db1_consumer import NaturalConsumer
from vehicle.common.db1_adapter import TrustedTenant
from vehicle.common.db1_rows.core import Blocked


def evidence_fixture():
    base = load_request_codec(ORG, IDENTITY_NAMESPACE)
    life = base.v1.decode(delivery_fixture(base)).life
    receipt = pack_receipt(life.first_receipt)
    model = dict(model_name='new-local-lfm',version=1,task='vlm')
    attrs = dict(manufacturer='renault',vlm_vehicle_type='sedan',color='white',recognition_status='recognized',
        recognition_failure='', recognition_attempts=1,
        recognition_evidence=[dict(frame_index=receipt['frame_index'],receipt=receipt,
            result=dict(manufacturer='renault',vehicle_type='sedan',color='white'))])
    return dict(life=pack_life(life), attributes=attrs, plates=[],models=dict(attributes=model),attributes_enabled=True)


class RecognitionTests(unittest.TestCase):
    def setUp(self):
        self.codec = RecognitionCodec(ORG, IDENTITY_NAMESPACE)
        self.evidence = evidence_fixture()

    def test_local_vlm_colour_and_type_preserve_absent_confidence(self):
        bundle=aggregate(self.evidence,self.codec)
        facts={f['row_kind']:f['delivery'] for f in bundle['facts']}
        self.assertEqual(facts['fact_vehicle']['class']['name'],'sedan')
        self.assertIsNone(facts['fact_vehicle']['class_conf'])
        self.assertEqual(facts['fact_vehicle_colour']['colour'],'white')
        self.assertIsNone(facts['fact_vehicle_colour']['conf'])
        self.assertEqual(facts['fact_vehicle_colour']['model']['task'],'vlm')
        self.assertNotIn('fact_plate_read',facts)
        self.assertIn(dict(domain='plate',reason='no_detection'),bundle['omissions'])

    def test_conflicting_bundle_and_tenant_are_rejected(self):
        raw=self.codec.make('db1.delivery.v1',self.evidence)
        decoded=self.codec.decode(raw)
        body=deepcopy(decoded.body)
        body['bundle']['facts'][0]['delivery']['class_conf']=1
        import json
        v=json.loads(raw);v['body']=body
        with self.assertRaises(Blocked):self.codec.decode(canonical_json_bytes(v))
        with self.assertRaises(Blocked):RecognitionCodec('00000000-0000-4000-8000-000000000001',IDENTITY_NAMESPACE).decode(raw)

    def test_new_delivery_reuses_adapter_commit_before_ack_and_identical_retry(self):
        raw=self.codec.make('db1.delivery.v1',self.evidence)
        res=supplied_test_registrations(type('Codec',(),dict(v1=self.codec))(),raw)
        from dataclasses import replace
        from vehicle.common.db1_rows import Dimension
        res=replace(res,manufacturers=(Dimension('Renault',41,True),))
        adapter=TransactionAdapter(self.codec.validation)
        c=self.codec.decode(raw).life.context
        consumer=NaturalConsumer(codec=self.codec,tenant=TrustedTenant(ORG,'test'),trusted_run=c.run_id,
            trusted_camera=c.camera.camera_id,resolutions=res,connection_factory=lambda:nullcontext(None),
            environment='test',adapter=adapter)
        acks=[]
        def ack():
            self.assertTrue(adapter.history);acks.append(True)
        self.assertEqual(consumer.process(raw,ack).disposition,'new')
        self.assertEqual(consumer.process(raw,ack).disposition,'identical')
        self.assertEqual(len(acks),2)

    def test_tunis_boolean_and_colour_vocabulary(self):
        from vehicle.recognition.config import boolean
        from vehicle.recognition.attributes.manufacturer.schemas import parse_result
        self.assertFalse(boolean('false'));self.assertTrue(boolean('true'))
        with self.assertRaises(ValueError):boolean('legacy')
        for raw,label in [('grey','white'),('silver','white'),('brown','black')]:
            self.assertEqual(parse_result('{"manufacturer":"renault","vehicle_type":"sedan","color":"'+raw+'"}').color,label)

    def test_disabled_attributes_do_not_create_inference_facts(self):
        self.evidence['attributes_enabled']=False
        self.evidence['attributes']['recognition_evidence']=[]
        bundle=aggregate(self.evidence,self.codec)
        self.assertEqual(len(bundle['facts']),1)

    def test_recognized_manufacturers_reach_sql_with_generated_catalogue_keys(self):
        cases = [('mercedes', 'Mercedes-Benz'), ('land_rover', 'Land Rover'),
                 ('bmw', 'BMW'), ('ford', 'Ford'), ('honda', 'Honda'),
                 ('mazda', 'Mazda'), ('volvo', 'Volvo'), ('renault', 'Renault'),
                 ('peugeot', 'Peugeot'), ('toyota', 'Toyota'), ('volkswagen', 'Volkswagen'),
                 ('iveco', 'iveco'), ('unknown', None)]
        for identifier, name in cases:
            with self.subTest(manufacturer=identifier):
                e = deepcopy(self.evidence)
                e['manufacturer_policy'] = 'recognized-name-v1'
                e['attributes']['manufacturer'] = identifier
                e['attributes']['recognition_evidence'][0]['result']['manufacturer'] = identifier
                bundle = aggregate(e, self.codec)
                make = next(f['delivery'] for f in bundle['facts'] if f['row_kind']=='fact_vehicle_make')
                self.assertEqual(make['manufacturer'], name)
                raw = self.codec.make('db1.delivery.v1', e)
                res = supplied_test_registrations(type('Codec', (), dict(v1=self.codec))(), raw)
                from dataclasses import replace
                from vehicle.common.db1_rows import Dimension
                res = replace(res, manufacturers=(Dimension(name, 937, True),) if name else ())
                c = self.codec.decode(raw).life.context
                adapter = TransactionAdapter(self.codec.validation)
                consumer = NaturalConsumer(codec=self.codec, tenant=TrustedTenant(ORG,'test'),
                    trusted_run=c.run_id, trusted_camera=c.camera.camera_id, resolutions=res,
                    connection_factory=lambda:nullcontext(None), environment='test', adapter=adapter)
                row = next(f for f in consumer.prepare(raw).construction.ready_rows if f.row_kind=='fact_vehicle_make')
                self.assertEqual(row.sql.value()['manufacturer_key'], 937 if name else None)
                self.assertIsNone(row.sql.value()['manufacturer_conf'])
                acks = []
                self.assertEqual(consumer.process(raw, lambda:acks.append(True)).disposition, 'new')
                self.assertEqual(consumer.process(raw, lambda:acks.append(True)).disposition, 'identical')
                self.assertEqual(len(acks), 2)

    def test_legacy_manufacturer_payload_remains_identical_on_retry(self):
        e = deepcopy(self.evidence)
        e['attributes']['manufacturer'] = 'mercedes'
        e['attributes']['recognition_evidence'][0]['result']['manufacturer'] = 'mercedes'
        raw = self.codec.make('db1.delivery.v1', e)
        message = self.codec.decode(raw)
        make = next(f['delivery'] for f in message.body['bundle']['facts'] if f['row_kind']=='fact_vehicle_make')
        self.assertIsNone(make['manufacturer'])
        self.assertEqual(self.codec.make(message.kind, message.body['evidence']), raw)

    def test_unregistered_known_manufacturer_is_rejected_instead_of_erased(self):
        from dataclasses import replace
        e = deepcopy(self.evidence)
        e['manufacturer_policy'] = 'recognized-name-v1'
        e['attributes']['manufacturer'] = 'mercedes'
        e['attributes']['recognition_evidence'][0]['result']['manufacturer'] = 'mercedes'
        raw = self.codec.make('db1.delivery.v1', e)
        res = supplied_test_registrations(type('Codec', (), dict(v1=self.codec))(), raw)
        res = replace(res, manufacturers=())
        c = self.codec.decode(raw).life.context
        consumer = NaturalConsumer(codec=self.codec, tenant=TrustedTenant(ORG,'test'),
            trusted_run=c.run_id, trusted_camera=c.camera.camera_id, resolutions=res,
            connection_factory=lambda:nullcontext(None), environment='test',
            adapter=TransactionAdapter(self.codec.validation))
        with self.assertRaises(Blocked) as error:
            consumer.prepare(raw)
        self.assertEqual(error.exception.code, 'MISSING_DIMENSION')

    def test_every_category_and_side_operation_reaches_db1_without_confidence_invention(self):
        from dataclasses import replace
        from vehicle.common.db1_rows import Dimension
        for category,text in [('private','123T4567'),('temporary','4567 nt'),('administrative','12 - 456'),('police','shorta 456')]:
            with self.subTest(category=category):
                e=deepcopy(self.evidence)
                receipt=e['attributes']['recognition_evidence'][0]['receipt']
                e['plates']=[dict(ordinal=1,plate_track_id=1,receipt=receipt,selected=True,crop_quality=.8,
                    result=dict(recognition_status='recognized',normalized_number=text,plate_category=category,
                        predicted_category=category,text_confidence=.95,evidence=dict(engine_operations=[
                            dict(kind=stage,outcome='succeeded',model=dict(model_name=stage,version=1,task='ocr' if stage=='registration_ocr' else 'detect'))
                            for stage in ('category_ocr','marker','registration_ocr','registration_ocr')])))]
                raw=self.codec.make('db1.delivery.v1',e)
                res=supplied_test_registrations(type('Codec',(),dict(v1=self.codec))(),raw)
                res=replace(res,manufacturers=(Dimension('Renault',41,True),))
                c=self.codec.decode(raw).life.context
                consumer=NaturalConsumer(codec=self.codec,tenant=TrustedTenant(ORG,'test'),trusted_run=c.run_id,
                    trusted_camera=c.camera.camera_id,resolutions=res,connection_factory=lambda:nullcontext(None),
                    environment='test',adapter=TransactionAdapter(self.codec.validation))
                row=next(f for f in consumer.prepare(raw).construction.ready_rows if f.row_kind=='fact_plate_read')
                self.assertEqual(row.delivery.value()['selected_attempt']['plate_text'],text.upper())
                self.assertEqual(row.sql.value()['conf'],row.delivery.value()['selected_attempt']['conf'])
                public = public_row(e, aggregate(e, self.codec))
                self.assertEqual(public['plate']['text'], text.upper())
                # Previously completed runs retain their text only in engine evidence.
                import tempfile
                from pathlib import Path
                from vehicle.common.console_crops import enrich_status
                with tempfile.TemporaryDirectory() as temp:
                    directory = Path(temp)/public['run_id']
                    directory.mkdir()
                    saved = directory/'recognition-1.json'
                    saved.write_bytes(self.codec.make('engine.recognition.v3', e))
                    original = saved.read_bytes()
                    del public['plate']['text']
                    status = enrich_status(directory, dict(rows=[public]))
                    self.assertEqual(status['rows'][0]['plate']['text'], text.upper())
                    self.assertEqual(saved.read_bytes(), original)
                    # An unreadable status must never inherit a retained number.
                    public['plate'].pop('text')
                    public['plate']['status'] = 'unreadable'
                    self.assertNotIn('text', enrich_status(directory, dict(rows=[public]))['rows'][0]['plate'])

    def test_unreadable_has_null_analytical_fields_and_retains_diagnostics(self):
        e=deepcopy(self.evidence)
        e['plates']=[dict(ordinal=1,plate_track_id=1,receipt=e['attributes']['recognition_evidence'][0]['receipt'],
            selected=True,crop_quality=.5,result=dict(recognition_status='unresolved',normalized_number=None,
            predicted_category='tunis',plate_category='private',text_confidence=.4,
            evidence=dict(engine_operations=[dict(kind='registration_ocr',outcome='succeeded',model=dict(model_name='ocr',version=1,task='ocr'))])))]
        plate=next(f['delivery'] for f in aggregate(e,self.codec)['facts'] if f['row_kind']=='fact_plate_read')
        self.assertEqual(plate['outcome'],'unreadable')
        self.assertIsNone(plate['selected_attempt']['conf'])
        self.assertIsNone(plate['selected_attempt']['plate_category'])
        self.assertIsNone(public_row(e, aggregate(e,self.codec))['plate']['text'])

    def test_native_tuple_metadata_is_scalarized_but_arrays_are_rejected(self):
        import numpy as np
        e=deepcopy(self.evidence)
        e['attributes']['recognition_evidence'][0]['quality']=(np.float32(.5),np.bool_(True))
        self.codec.make('engine.recognition.v3',e)
        e['attributes']['recognition_evidence'][0]['quality']=np.zeros((2,2))
        with self.assertRaises(Blocked):self.codec.make('engine.recognition.v3',e)


if __name__=='__main__':unittest.main()
