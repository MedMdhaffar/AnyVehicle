"""Explicit manufacturer decision -> DB1 fact_vehicle_make projection.

The opt-in ManufacturerDeliveryCodec uses this projection. Existing v0/v1
closure contracts retain their original manufacturer omission semantics.

Every unresolved business decision is an explicit ``MakeActivation`` input with
no default: the approved model registration (bound to exact runtime provenance),
the approved brand -> ``dim_manufacturer`` mapping, the row timestamp basis, the
``observations`` basis and the treatment of answered-but-unmapped makes. Keys
are only ever read from supplied ``Resolutions``; none are invented.

Rows exist only for completed model calls, so ``model_key`` always names the
network that actually executed. Non-attempts and failed/uncertain executions
return ``None`` (the caller keeps its existing terminal omission).
"""
from dataclasses import dataclass
from uuid import UUID

from .core import require, receipt_in_lifetime
from .natural_policy import NaturalPolicy
from ..observation_evidence import RecognitionIdentity, VehicleIdentity

TS_BASES = ('selected_crop_receipt', 'lifetime_first_receipt')
OBSERVATION_BASES = ('submitted_crops', 'eligible_observations')
UNMAPPED = ('omit', 'null_manufacturer_row')


@dataclass(frozen=True, slots=True)
class MakeActivation:
    """Approval record. Constructing one is the activation decision itself."""
    approval_reference: str
    model: RecognitionIdentity
    runtime_provenance: tuple          # exact approved runtime provenance hashes
    brand_mapping: tuple               # ((canonical make id, dim_manufacturer name), ...)
    ts_basis: str
    observations_basis: str
    unmapped: str

    def __post_init__(self):
        import re
        require(type(self.approval_reference) is str and self.approval_reference.strip() != '',
                'MISSING_APPROVAL', 'explicit approval reference required')
        require(isinstance(self.model, RecognitionIdentity) and self.model.task in {'classify', 'vlm'},
                'MODEL_RESOLUTION_MISMATCH', 'approved make model must be a classify/vlm identity')
        require(type(self.runtime_provenance) is tuple and self.runtime_provenance
                and all(type(p) is str and re.fullmatch('[a-f0-9]{64}', p) for p in self.runtime_provenance),
                'MODEL_PROVENANCE_MISMATCH', 'approved runtime provenance hashes required')
        require(type(self.brand_mapping) is tuple and all(
                    type(m) is tuple and len(m) == 2 and all(type(v) is str and v.strip() for v in m) for m in self.brand_mapping)
                and len({m[0] for m in self.brand_mapping}) == len(self.brand_mapping),
                'MISSING_DIMENSION', 'unique approved brand -> catalogue mapping required')
        require(self.ts_basis in TS_BASES and self.observations_basis in OBSERVATION_BASES and self.unmapped in UNMAPPED,
                'INVALID_CONFIGURATION', 'explicit timestamp, observation and unmapped-make decisions required')

    def catalogue_name(self, make):
        return dict(self.brand_mapping).get(make)


def make_natural(decision, life, *, activation, identity_policy, validation):
    """Validated natural make_delivery payload, or None when no row is permitted."""
    from ..manufacturer.engine import load_canonical
    from ..manufacturer.policy import ManufacturerDecision
    from ..redis_scalar.codec import unpack_receipt
    require(isinstance(activation, MakeActivation), 'MISSING_APPROVAL', 'explicit activation record required')
    require(isinstance(decision, ManufacturerDecision), 'INVALID_MANUFACTURER', 'typed manufacturer decision required')
    decision.validate(life)  # reproduces the decision from its own evidence
    evidence = load_canonical(decision.evidence_json)
    binding = evidence['binding']
    if evidence['outcome'] != 'completed' or binding['receipt'] is None:
        return None  # non-attempt or failed/uncertain execution: no truthful model row
    require(binding['provenance'] in activation.runtime_provenance, 'MODEL_PROVENANCE_MISMATCH',
            'executed runtime is not the approved registration')
    name = activation.catalogue_name(decision.make) if decision.status == 'accepted' else None
    if name is None and activation.unmapped == 'omit':
        return None
    receipt = unpack_receipt(binding['receipt']) if activation.ts_basis == 'selected_crop_receipt' else life.first_receipt
    receipt_in_lifetime(receipt, life)
    if activation.observations_basis == 'submitted_crops':
        observations = 1  # at-most-once: exactly one submitted crop per lifetime
    else:
        require('selection' in binding, 'INVALID_MANUFACTURER', 'eligible-observation basis requires v2 selection evidence')
        observations = binding['selection']['eligible']
    identity = VehicleIdentity.from_lifetime(life)
    uid = str(UUID(str(identity_policy.row_uid(identity, 'fact_vehicle_make'))))
    fields = dict(model=activation.model.policy(), manufacturer=name, vehicle_model=None,
                  manufacturer_conf=None, vehicle_model_conf=None, observations=observations)
    return NaturalPolicy.delivery('fact_vehicle_make', identity, receipt, fields, uid, validation)


def project_make_sql(base, resolutions, context, life, *, validation, identity_policy):
    """SQL projection for a natural make payload; keys only from supplied resolutions."""
    from contracts.db1.reference import derive_date_key
    from .builder import RowBuilder
    builder = RowBuilder(identity_policy=identity_policy, validation=validation)
    validation.delivery('fact_vehicle_make', base)
    lookup = builder._resolution(resolutions, context, life)
    run = resolutions.run
    row = dict(org_id=base['org_id'], run_key=run.run_key, camera_key=run.camera.camera_key,
               date_key=derive_date_key(base['ts'], context.camera.timezone), ts=base['ts'], uid=base['uid'],
               track_id=base['track_id'])
    require(row['date_key'] in run.date_keys, 'MISSING_DATE', 'supporting receipt date is not registered')
    lookup['parent'] = {k: row[k] for k in ('org_id', 'run_key', 'camera_key', 'track_id')}
    model = builder._model(resolutions, base['model'])
    lookup['model'] = dict(**model.identity.policy(), model_key=model.key)
    maker = builder._dimension(resolutions.manufacturers, base['manufacturer'], 'manufacturer') \
        if base['manufacturer'] is not None else None
    lookup['manufacturer'] = dict(name=maker.name, manufacturer_key=maker.key) if maker else None
    row.update(model_key=model.key, manufacturer_key=maker.key if maker else None, vehicle_model_key=None,
               manufacturer_conf=base['manufacturer_conf'], vehicle_model_conf=base['vehicle_model_conf'],
               observations=base['observations'])
    validation.snapshot(dict(schema_version='db1-row-snapshot-v0', row_kind='fact_vehicle_make',
                             delivery_payload=base, resolved_sql_row=row), lookup)
    return row
