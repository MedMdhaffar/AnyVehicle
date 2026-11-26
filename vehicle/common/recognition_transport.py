"""Versioned, pixel-free recognition evidence and deterministic aggregation."""
from hashlib import sha256
import json
from pathlib import Path
from uuid import UUID, uuid5

from contracts.db1.reference import canonical_json_bytes
from .db1_rows import UUID5Policy, ContractValidation
from .db1_rows.core import FrozenJSON, require, receipt_in_lifetime
from .db1_rows.natural_policy import NaturalPolicy
from .observation_evidence import VehicleIdentity
from .redis_scalar.codec import Message, bounded_json, unpack_life, unpack_receipt

ROOT = Path(__file__).resolve().parents[2]
VERSION = 'vehicle-recognition-v3'


def scalarize(value):
    """Convert native scalar/tuple metadata; pixel arrays are never admitted."""
    if isinstance(value, dict): return {k:scalarize(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [scalarize(v) for v in value]
    if type(value).__module__.startswith('numpy'):
        require(getattr(value,'shape',None)==(), 'PIXELS_IN_TRANSPORT', 'only native scalars may cross the engine boundary')
        return value.item()
    return value


class RecognitionValidation(ContractValidation):
    """Explicit local-VLM colour extension; legacy row validation is unchanged."""
    def __init__(self):
        from jsonschema import Draft202012Validator, FormatChecker
        rows = json.loads((ROOT/'contracts/db1/rows.v0.schema.json').read_text())
        diagnostics = json.loads((ROOT/'contracts/db1/diagnostics.v1.schema.json').read_text())
        super().__init__(rows, diagnostics)
        rows['$defs']['colour_support']['properties']['source']['enum'].append('local_vlm')
        self._row = Draft202012Validator(rows, format_checker=FormatChecker())
        self._delivery['fact_vehicle_colour'] = Draft202012Validator(
            {'$ref': '#/$defs/colour_delivery', '$defs': rows['$defs']}, format_checker=FormatChecker())

    def snapshot(self, snapshot, lookup):
        if snapshot['row_kind'] != 'fact_vehicle_colour' or snapshot['delivery_payload']['support']['source'] != 'local_vlm':
            return super().snapshot(snapshot, lookup)
        self._row.validate(snapshot)
        from contracts.db1.row_policies import validate_common, validate_parent, validate_model
        validate_common(snapshot, lookup)
        delivery, sql = snapshot['delivery_payload'], snapshot['resolved_sql_row']
        validate_parent(sql, lookup)
        validate_model(delivery['model'],sql['model_key'],lookup,{'vlm'})
        require(delivery['model']['task'] == 'vlm', 'MODEL_PROVENANCE_MISMATCH', 'local VLM model required')
        require(delivery['conf'] is None and sql['conf'] is None, 'FABRICATED_CONFIDENCE', 'VLM has no confidence')
        require((delivery['state']=='decided') == (delivery['colour'] is not None), 'COLOUR_DECISION_MISMATCH', 'colour state differs')
        require(sql['model_key'] == lookup['model']['model_key'], 'MODEL_RESOLUTION_MISMATCH', 'model differs')
        expected = lookup.get('catalogue')
        require(sql['colour_key'] == (expected['key'] if expected else None), 'MISSING_DIMENSION', 'colour differs')
        require(sql['samples'] == delivery['samples'] and sql['state'] == delivery['state'], 'COLOUR_COUNT_MISMATCH', 'colour support differs')
        for key in ('org_id','track_id','ts','uid'):
            require(sql[key] == delivery[key], 'PARENT_MISMATCH', 'colour identity differs')


def selected_plate_attempt(evidence):
    """Use the cached plate decisions and the delivery result-selection policy."""
    life = unpack_life(evidence['life'])
    candidates = []
    selected_tracks = set()
    for attempt in evidence['plates']:
        receipt_in_lifetime(unpack_receipt(attempt['receipt']), life)
        quality = attempt['result']['evidence'].get('quality', {})
        require(quality.get('frame_index',attempt['receipt']['frame_index'])==attempt['receipt']['frame_index'],
                'RECEIPT_MISMATCH','plate crop frame differs')
        if attempt['selected']:
            require(attempt['plate_track_id'] not in selected_tracks,'PAYLOAD_CONFLICT','multiple cached results for one plate track')
            selected_tracks.add(attempt['plate_track_id'])
        operations = attempt['result']['evidence'].get('engine_operations', [])
        executed = [o for o in operations if o['outcome']=='succeeded' and o['kind']=='registration_ocr']
        if executed:
            candidates.append(attempt)
    preferred = [a for a in candidates if a['selected']]
    pool = preferred or candidates
    if not pool:
        return None
    accepted = [a for a in pool if a['result']['recognition_status']=='recognized']
    return max(accepted or pool, key=lambda a:(a['crop_quality'], -a['receipt']['frame_index']))


def aggregate(evidence, codec):
    """Reproduce final rows solely from original, settled scalar evidence."""
    life = unpack_life(evidence['life'])
    require(not life.quarantined and life.closure is not None, 'CLOCK_QUARANTINE', 'closed valid lifetime required')
    identity = VehicleIdentity.from_lifetime(life)
    attributes = evidence['attributes']
    successes = [a for a in attributes['recognition_evidence'] if 'result' in a]
    selected = successes[-1] if successes else None
    result = selected['result'] if selected else {}
    if selected:
        require(attributes['manufacturer']==result['manufacturer'] and attributes['color']==result['color']
                and attributes.get('vlm_vehicle_type')==result['vehicle_type'],
                'PAYLOAD_CONFLICT', 'attribute claims differ from the last successful attempt')
    facts, omissions = [], []
    def append(kind, receipt, fields):
        receipt_in_lifetime(receipt, life)
        uid = str(codec.identity_policy.row_uid(identity, kind))
        facts.append(dict(row_kind=kind, delivery=NaturalPolicy.delivery(kind, identity, receipt, fields, uid, codec.validation)))
    for attempt in attributes['recognition_evidence']:
        receipt_in_lifetime(unpack_receipt(attempt['receipt']), life)
        require(attempt['frame_index']==attempt['receipt']['frame_index'],'RECEIPT_MISMATCH','attribute crop frame differs')
    append('fact_vehicle', life.first_receipt, dict(
        first_ts=life.first_receipt.utc.isoformat(), last_ts=life.last_receipt.utc.isoformat(),
        **{'class': dict(name=result.get('vehicle_type','unknown'), class_group='vehicle')},
        class_conf=None, frames_observed=life.frames_observed, best_media=None))
    if selected:
        receipt = unpack_receipt(selected['receipt'])
        model = evidence['models']['attributes']
        colour = result['color'] if result['color'] != 'unknown' else None
        append('fact_vehicle_colour', receipt, dict(model=model, colour=colour, conf=None,
            state='decided' if colour else 'undecided', samples=len(successes),
            support=dict(ts=receipt.utc.isoformat(), source='local_vlm')))
        from .recognition_manufacturers import POLICY, manufacturer_name
        policy = evidence.get('manufacturer_policy')
        require(policy in (None, POLICY), 'UNKNOWN_MANUFACTURER_POLICY', 'unsupported manufacturer naming policy')
        if policy == POLICY:
            maker = manufacturer_name(result['manufacturer'])
        else:
            # Historical accepted deliveries must remain byte-identical on retry.
            from .manufacturer.test_database import CATALOGUE
            maker = dict(CATALOGUE).get(result['manufacturer'])
        append('fact_vehicle_make', receipt, dict(model=model, manufacturer=maker, vehicle_model=None,
            manufacturer_conf=None, vehicle_model_conf=None, observations=len(attributes['recognition_evidence'])))
    else:
        reason = 'failure_before_model' if evidence['attributes_enabled'] else 'disabled_capability'
        omissions.extend(dict(domain=d, reason=reason) for d in ('colour','manufacturer'))
    chosen = selected_plate_attempt(evidence)
    if chosen is not None:
        r = chosen['result']
        operations = r['evidence']['engine_operations']
        identities = [o['model'] for o in operations if o['outcome']=='succeeded']
        binding = dict(models=identities, reference=evidence['models'].get('category_reference'),
                       policy=evidence['models'].get('plate_policy'))
        if 'category_policy' in evidence['models']:
            binding['category_policy'] = evidence['models']['category_policy']
        model = dict(model_name='plate-recognition-sha256:'+sha256(canonical_json_bytes(binding)).hexdigest(), version=1, task='ocr')
        receipt = unpack_receipt(chosen['receipt'])
        read = r['recognition_status']=='recognized'
        from contracts.db1.reference import normalize_plate_text
        number = normalize_plate_text(r['normalized_number']) if read else None
        confidence = r['text_confidence'] if any(o['kind']=='registration_ocr' and o['outcome']=='succeeded' for o in operations) else None
        append('fact_plate_read', receipt, dict(outcome='read' if read else 'unreadable', attempts=len(evidence['plates']),
            selected_attempt=dict(attempt_uid=str(uuid5(codec.identity_policy.namespace,
                f'{identity.context.run_id}:{identity.service_track_id}:plate:{chosen["plate_track_id"]}:{chosen["ordinal"]}')),
                submission_ordinal=evidence['plates'].index(chosen)+1, ts=receipt.utc.isoformat(), route='local_ocr',
                model=model, text_complete=read, structural_valid=True if read else None, plate_text=number,
                plate_category=r['plate_category'] if read and r.get('predicted_category') else None,
                conf=confidence if read else None, crop_score=chosen['crop_quality'], crop_media=None)))
    else:
        diagnostics = evidence.get('plate_diagnostics', {})
        reason = 'ambiguous_association' if diagnostics.get('ambiguous') else 'rejected_crop' if diagnostics.get('detections') and not evidence['plates'] else 'no_detection' if not evidence['plates'] else 'failure_before_model'
        omissions.append(dict(domain='plate', reason=reason))
    return dict(schema_version='natural-db1-recognition-v3', org_id=str(life.context.org_id), facts=facts, omissions=omissions)


class RecognitionCodec:
    recognition_enabled = True
    manufacturer_enabled = True

    def __init__(self, org, namespace):
        self.org = UUID(str(org))
        self.identity_policy = UUID5Policy(UUID(str(namespace)))
        self.validation = RecognitionValidation()
        from jsonschema import Draft202012Validator, FormatChecker
        self.validator = Draft202012Validator(json.loads((ROOT/'contracts/vehicle_recognition_v3.schema.json').read_text()),
                                             format_checker=FormatChecker())

    def make(self, kind, evidence):
        evidence = scalarize(evidence)
        body = dict(evidence=evidence, life=evidence['life'])
        if kind == 'db1.delivery.v1':
            from .aggregator import Aggregator
            body['bundle'] = Aggregator.join_recognition(evidence, self)
        life = unpack_life(body['life'])
        event_id = str(uuid5(self.identity_policy.namespace,
            f'{VERSION}:{self.org}:{life.context.run_id}:{life.service_track_id}:{kind}'))
        raw = canonical_json_bytes(dict(schema_version=VERSION, kind=kind, event_id=event_id, body=body))
        self.decode(raw)
        return raw

    def decode(self, raw):
        value = bounded_json(raw, max_bytes=16*1024*1024, max_depth=40, max_nodes=300000, max_string=8192)
        self.validator.validate(value)
        require(set(value)=={'schema_version','kind','event_id','body'} and value['schema_version']==VERSION,
                'INVALID_MESSAGE', 'recognition v3 envelope required')
        require(value['kind'] in ('engine.recognition.v3','db1.delivery.v1'), 'WRONG_STREAM', 'recognition kind required')
        body = value['body']; life = unpack_life(body['life'])
        require(life.context.org_id == self.org, 'ORG_MISMATCH', 'trusted org differs')
        expected = str(uuid5(self.identity_policy.namespace,
            f'{VERSION}:{self.org}:{life.context.run_id}:{life.service_track_id}:{value["kind"]}'))
        require(value['event_id']==expected, 'PAYLOAD_CONFLICT', 'event identity differs')
        require(body['life']==body['evidence']['life'], 'PARENT_MISMATCH', 'original lifetime differs')
        if value['kind']=='db1.delivery.v1':
            from .aggregator import Aggregator
            require(canonical_json_bytes(body['bundle'])==canonical_json_bytes(Aggregator.join_recognition(body['evidence'], self)),
                    'PAYLOAD_CONFLICT', 'delivery differs from original recognition evidence')
        return Message(value['kind'], value['event_id'], body, raw, life)


class RecognitionWorker:
    """Journal-fenced pure aggregation; native calls remain in the engine."""
    def __init__(self, client, codec, namespace):
        from .redis_scalar.journal import Journal
        self.client, self.codec = client, codec
        self.journal = Journal(client, namespace=namespace, trusted_org=codec.org,
            role='aggregator_plate', consumer='recognition-aggregator', max_events=16384, max_vehicles=4096,
            max_stream=16384)
        self.group = 'recognition-v3'; self.source = self.journal.key('engine.recognition.v3')
        self.journal.group = self.group
        from redis.exceptions import ResponseError
        try: client.xgroup_create(self.source, self.group, id='0-0', mkstream=True)
        except ResponseError as error:
            if not str(error).startswith('BUSYGROUP'): raise

    def step(self):
        if self.journal.token:
            from redis.exceptions import ResponseError
            try: self.journal.renew()
            except ResponseError as error:
                if str(error) != 'STALE_OWNER': raise
                self.journal.token = None
        if not self.journal.token and not self.journal.acquire():
            return False
        self.journal.repair()
        rows = self.client.xautoclaim(self.source, self.group, 'recognition-aggregator', 0, count=1)[1]
        if not rows:
            reply = self.client.xreadgroup(self.group, 'recognition-aggregator', {self.source:'>'}, count=1)
            rows = reply[0][1] if reply else []
        if not rows:
            return False
        entry, fields = rows[0]
        message = self.codec.decode(fields[b'json'])
        output = self.codec.make('db1.delivery.v1', message.body['evidence'])
        marker = 'seen:'+message.event_id
        digest = sha256(message.canonical).hexdigest()
        old = self.journal.seen(marker)
        require(old is None or old.decode()==digest, 'PAYLOAD_CONFLICT', 'changed original evidence')
        # Journal uses its role's group; use that same group for fenced XACK.
        self.journal.group = self.group
        self.journal.handoff(source='engine.recognition.v3', entry_id=entry, marker=marker,
                             digest=digest, outputs=[] if old else [('db1.delivery.v1', output)])
        return True
