"""Manufacturer alone interprets VLM output. No score or car model is inferred."""
from dataclasses import dataclass
import json

from .brands import REGIONAL_MANUFACTURERS
from .taxonomy import normalize_manufacturer
from ..db1_rows.core import require, receipt_in_lifetime
from ..observation_evidence import VehicleIdentity
from ..redis_scalar.codec import pack_receipt


@dataclass(frozen=True)
class ManufacturerDecision:
    identity: VehicleIdentity
    evidence_json: str
    status: str
    make: str | None
    reason: str
    vehicle_model: None = None
    confidence: None = None

    def validate(self, life):
        from .engine import load_canonical
        evidence = load_canonical(self.evidence_json)
        require(self.identity == VehicleIdentity.from_lifetime(life),
                'PARENT_MISMATCH', 'manufacturer lifetime differs')
        require(evaluate(life, evidence) == self, 'MANUFACTURER_DECISION_MISMATCH',
                'manufacturer must reproduce its original evidence')


def strict_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError('duplicate JSON key')
        value[key] = item
    return value


def interpret_text(text):
    """Completed model text -> (outcome, make, reason). The single parsing rule."""
    try:
        text = text.strip()
        lines = text.splitlines()
        if len(lines) >= 3 and lines[0].lower() in ('```', '```json') and lines[-1] == '```':
            text = '\n'.join(lines[1:-1])
        data = json.loads(text, object_pairs_hook=strict_object)
        if (type(data) is not dict or set(data) != {'manufacturer', 'vehicle_type', 'color'}
                or any(type(v) is not str for v in data.values())):
            raise ValueError('invalid output shape')
        normalized = normalize_manufacturer(data['manufacturer'])
        if normalized in REGIONAL_MANUFACTURERS:
            return 'accepted', normalized, 'vocabulary_match_uncalibrated'
        return 'unknown', None, 'unknown_or_unsupported_make'
    except (ValueError, TypeError):
        return 'malformed', None, 'invalid_model_json'


def evaluate(life, evidence):
    from contracts.db1.reference import canonical_json_bytes
    from ..redis_scalar.codec import unpack_receipt
    from .engine import validate_evidence
    from ..db1_rows.core import FrozenJSON
    # Exact canonical numbers regardless of how the caller decoded the JSON.
    evidence = FrozenJSON.of(evidence).value()
    validate_evidence(evidence)
    identity = VehicleIdentity.from_lifetime(life)
    binding = evidence['binding']
    from .engine import vehicle_binding
    require(binding['vehicle'] == vehicle_binding(life), 'PARENT_MISMATCH',
            'manufacturer run/vehicle differs')
    if binding['receipt'] is not None:
        receipt = unpack_receipt(binding['receipt'])
        receipt_in_lifetime(receipt, life)
        # Original receipts with the same frame index cannot have different times.
        for original in (life.first_receipt, life.last_receipt):
            if receipt.frame_index == original.frame_index:
                require(pack_receipt(receipt) == pack_receipt(original),
                        'RECEIPT_MISMATCH', 'manufacturer receipt differs')
    outcome = evidence['outcome']
    make, reason = None, outcome
    if outcome == 'completed':
        outcome, make, reason = interpret_text(evidence['text'])
    elif binding['receipt'] is None and 'selection' in binding:
        # Quality-gated non-attempts state why no crop reached the model.
        reason = {'none_eligible': 'insufficient_quality', 'stopped_before_submission': 'stopped_before_submission',
                  'not_observed': 'not_observed'}.get(binding['selection']['submission'], outcome)
    return ManufacturerDecision(identity, canonical_json_bytes(evidence).decode(), outcome, make, reason)
