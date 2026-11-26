"""Opt-in v2 manufacturer delivery; existing plate/colour wire bytes are retained.

Only accepted, explicitly mapped labels produce make rows. No inference occurs
on decoding/retry. The trusted host supplies the verified runtime registration
and catalogue policy; its digest is bound into every envelope.
"""
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import json
from uuid import uuid5

from contracts.db1.reference import canonical_json_bytes
from ..db1_rows.core import require
from ..db1_rows.make import MakeActivation, make_natural
from ..redis_scalar.codec import Message, bounded_json
from .engine import load_canonical
from .policy import evaluate

VERSION = 'vehicle-manufacturer-db1-v2'


class ManufacturerDeliveryCodec:
    manufacturer_enabled = True

    def __init__(self, base, activation):
        require(isinstance(activation, MakeActivation) and activation.unmapped == 'omit',
                'INVALID_CONFIGURATION', 'conservative mapped-only manufacturer policy required')
        self.base, self.activation = base, activation
        self.org, self.identity_policy, self.validation = base.org, base.identity_policy, base.validation
        self.plate_reader = getattr(base, 'plate_reader', None)
        self.policy_sha256 = sha256(canonical_json_bytes(json.loads(json.dumps(asdict(activation))))).hexdigest()

    def _body(self, base_raw, evidence):
        original = self.base.decode(base_raw)
        require(original.kind == 'db1.delivery.v1', 'WRONG_STREAM', 'completed plate/colour delivery required')
        life = original.life
        decision = evaluate(life, evidence)
        receipt = evidence['binding']['receipt']
        if receipt is not None:
            require(any(canonical_json_bytes(c['receipt']) == canonical_json_bytes(receipt)
                        for c in original.body['closed']['classes']),
                    'RECEIPT_MISMATCH', 'manufacturer crop requires original same-track class receipt')
        natural = make_natural(decision, life, activation=self.activation,
            identity_policy=self.identity_policy, validation=self.validation)
        bundle = deepcopy(original.body['bundle'])
        require(not any(f['row_kind'] == 'fact_vehicle_make' for f in bundle['facts']),
                'INVALID_DELIVERY', 'base delivery already contains make row')
        if natural is not None:
            require(decision.status == 'accepted' and natural['manufacturer'] is not None,
                    'INVALID_MANUFACTURER', 'only accepted mapped makers are admitted')
            bundle['facts'].append(dict(row_kind='fact_vehicle_make', delivery=natural))
            bundle['omissions'] = [o for o in bundle['omissions'] if o['domain'] != 'manufacturer']
            reason = 'mapped'
        else:
            reason = 'catalogue_unmapped' if decision.status == 'accepted' else decision.reason
        bundle['schema_version'] = 'natural-db1-bundle-manufacturer-v2'
        return dict(**{k:v for k,v in original.body.items() if k not in {'bundle','bundle_sha256'}},
            bundle=bundle,bundle_sha256=sha256(canonical_json_bytes(bundle)).hexdigest(),
            manufacturer=dict(evidence_version='manufacturer-evidence-v2',evidence=evidence,
                              status=decision.status,make=decision.make,delivery_reason=reason)), life

    def encode(self, base_raw, evidence_json):
        evidence = load_canonical(evidence_json)
        body, life = self._body(base_raw, evidence)
        event = str(uuid5(self.identity_policy.namespace,canonical_json_bytes([
            str(self.org),life.context.run_id,life.context.camera.camera_id,life.service_track_id,VERSION]).decode()))
        raw = canonical_json_bytes(dict(schema_version=VERSION,org_id=str(self.org),event_id=event,
            policy_sha256=self.policy_sha256,base_json=base_raw.decode(),manufacturer=evidence,
            body_sha256=sha256(canonical_json_bytes(body)).hexdigest()))
        self.decode(raw)
        return raw

    def decode(self, raw):
        v = bounded_json(raw,max_bytes=16*1024*1024,max_depth=40,max_nodes=250000,max_string=12*1024*1024)
        require(set(v)=={'schema_version','org_id','event_id','policy_sha256','base_json','manufacturer','body_sha256'}
                and v['schema_version']==VERSION and v['org_id']==str(self.org)
                and v['policy_sha256']==self.policy_sha256,
                'INVALID_MANUFACTURER','version, trusted org and exact delivery policy required')
        body, life = self._body(v['base_json'].encode(),v['manufacturer'])
        event = str(uuid5(self.identity_policy.namespace,canonical_json_bytes([
            str(self.org),life.context.run_id,life.context.camera.camera_id,life.service_track_id,VERSION]).decode()))
        require(v['event_id']==event and v['body_sha256']==sha256(canonical_json_bytes(body)).hexdigest(),
                'HASH_MISMATCH','versioned manufacturer delivery differs from evidence')
        return Message('db1.delivery.v1',event,body,canonical_json_bytes(v),life)
