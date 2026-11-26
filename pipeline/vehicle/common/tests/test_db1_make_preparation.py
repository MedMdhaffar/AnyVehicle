"""Prepared (not activated) fact_vehicle_make projection: contract, keys, provenance, retry."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from contracts.db1.reference import canonical_json_bytes
from vehicle.common.db1_context import ModelRegistration, ResolutionInfo, RunContext
from vehicle.common.db1_consumer import registered_context
from vehicle.common.db1_rows import Dimension, ResolvedModel, RowBuilder, UUID5Policy
from vehicle.common.db1_rows.core import Blocked
from vehicle.common.db1_rows.make import MakeActivation, make_natural, project_make_sql
from vehicle.common.manufacturer.engine import ManufacturerLedger, SelectionConfig
from vehicle.common.manufacturer.policy import evaluate
from vehicle.common.observation_evidence import RecognitionIdentity
from vehicle.common.tests.test_db1_outbox import validation
from vehicle.common.tests.test_db1_rows import fixture, NAMESPACE
from vehicle.common.tests.test_manufacturer import Runtime
from vehicle.common.tests.test_manufacturer_quality import textured

MODEL = RecognitionIdentity('synthetic-approved-make-vlm', 1, 'vlm')


class MakePreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.validation = validation()
        cls.policy = UUID5Policy(NAMESPACE)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        completed, res = fixture()
        tested = completed.vehicle.lifetime
        c = tested.context
        # Original natural (unresolved) context with the fixture's natural identifiers.
        dev = RunContext(c.org_id, replace(c.camera, camera_key=None), c.run_id, c.source_kind, c.started_at,
                         c.service_version, ResolutionInfo(None, None, 'vehicle', replace(c.resolution.detector, model_key=None),
                                                           None, 'natural development context', False), 'development')
        self.life = replace(tested, context=dev)
        self.res = replace(res, models=res.models + (ResolvedModel(MODEL, 84),),
                           manufacturers=(Dimension('Toyota', 3, True), Dimension('Renault', 1, True)))
        self.resolved = replace(self.life, context=registered_context(dev, self.res, environment='test'))

    def activation(self, **changes):
        values = dict(approval_reference='TEST-ONLY synthetic approval', model=MODEL, runtime_provenance=(Runtime.provenance,),
                      brand_mapping=(('toyota', 'Toyota'), ('renault', 'Renault')), ts_basis='selected_crop_receipt',
                      observations_basis='submitted_crops', unmapped='omit')
        values.update(changes)
        return MakeActivation(**values)

    def decision(self, text='{"manufacturer":"Toyota","vehicle_type":"sedan","color":"white"}', crops=None, runtime=None):
        name = f'{len(list(Path(self.tmp.name).iterdir()))}.sqlite'
        ledger = ManufacturerLedger(enabled=True, runtime=runtime or Runtime(text), journal=Path(self.tmp.name)/name,
                                    selection=SelectionConfig(policy='quality_lifetime_v1'))
        self.addCleanup(ledger.close)
        for crop in crops or [textured(60, 40), textured(440, 300)]:
            h, w = crop.shape[:2]
            ledger.observe(self.life, self.life.first_receipt, (10, 10, 10+w, 10+h), crop, 'f'*64,
                           label='car', confidence=.9, frame_shape=(1080, 1920, 3))
        ledger.close_lifetime(self.life)
        return evaluate(self.life, ledger.terminal(self.life))

    def project(self, natural):
        return project_make_sql(natural, self.res, self.resolved.context, self.resolved,
                                validation=self.validation, identity_policy=self.policy)

    def test_accepted_mapped_make_builds_contract_valid_row_with_supplied_keys_only(self):
        decision = self.decision()
        natural = make_natural(decision, self.life, activation=self.activation(), identity_policy=self.policy, validation=self.validation)
        row = self.project(natural)
        self.assertEqual((natural['manufacturer'], natural['observations'], natural['manufacturer_conf']), ('Toyota', 1, None))
        self.assertEqual((row['model_key'], row['manufacturer_key'], row['vehicle_model_key']), (84, 3, None))
        self.assertEqual(natural['uid'], str(self.policy.row_uid(__import__('vehicle.common.observation_evidence', fromlist=['x']).VehicleIdentity.from_lifetime(self.life), 'fact_vehicle_make')))
        # Retry safety: identical evidence -> byte-identical payload and SQL row.
        again = make_natural(decision, self.life, activation=self.activation(), identity_policy=self.policy, validation=self.validation)
        self.assertEqual(canonical_json_bytes(again), canonical_json_bytes(natural))
        self.assertEqual(canonical_json_bytes(self.project(again)), canonical_json_bytes(row))

    def test_keys_and_registrations_are_never_invented(self):
        natural = make_natural(self.decision(), self.life, activation=self.activation(), identity_policy=self.policy, validation=self.validation)
        with self.assertRaises(Blocked) as missing_maker:
            project_make_sql(natural, replace(self.res, manufacturers=()), self.resolved.context, self.resolved,
                             validation=self.validation, identity_policy=self.policy)
        self.assertEqual(missing_maker.exception.code, 'MISSING_DIMENSION')
        with self.assertRaises(Blocked) as missing_model:
            project_make_sql(natural, replace(self.res, models=self.res.models[:-1]), self.resolved.context, self.resolved,
                             validation=self.validation, identity_policy=self.policy)
        self.assertEqual(missing_model.exception.code, 'MODEL_RESOLUTION_MISMATCH')
        unapproved = replace(self.res, manufacturers=(Dimension('Toyota', 3, False),))
        with self.assertRaises(Blocked):
            project_make_sql(natural, unapproved, self.resolved.context, self.resolved, validation=self.validation, identity_policy=self.policy)

    def test_model_provenance_must_be_the_approved_runtime(self):
        with self.assertRaises(Blocked) as error:
            make_natural(self.decision(), self.life, activation=self.activation(runtime_provenance=('c'*64,)),
                         identity_policy=self.policy, validation=self.validation)
        self.assertEqual(error.exception.code, 'MODEL_PROVENANCE_MISMATCH')

    def test_unmapped_answer_omits_or_emits_null_key_row_only_by_explicit_decision(self):
        decision = self.decision('{"manufacturer":"Kia","vehicle_type":"suv","color":"black"}')
        self.assertIsNone(make_natural(decision, self.life, activation=self.activation(), identity_policy=self.policy, validation=self.validation))
        natural = make_natural(decision, self.life, activation=self.activation(unmapped='null_manufacturer_row'),
                               identity_policy=self.policy, validation=self.validation)
        row = self.project(natural)
        self.assertEqual((natural['manufacturer'], row['manufacturer_key'], row['model_key']), (None, None, 84))

    def test_non_attempts_and_failed_execution_never_produce_rows(self):
        rejected = self.decision(crops=[textured(60, 40), textured(80, 60)])
        self.assertEqual(rejected.reason, 'insufficient_quality')
        failed = self.decision(runtime=Runtime(error=RuntimeError('injected')))
        for decision in (rejected, failed):
            self.assertIsNone(make_natural(decision, self.life, activation=self.activation(unmapped='null_manufacturer_row'),
                                           identity_policy=self.policy, validation=self.validation))

    def test_alternative_timestamp_and_observation_bases(self):
        natural = make_natural(self.decision(), self.life, activation=self.activation(
            ts_basis='lifetime_first_receipt', observations_basis='eligible_observations'),
            identity_policy=self.policy, validation=self.validation)
        self.assertEqual(natural['observations'], 1)  # one of two observed crops was eligible
        self.project(natural)

    def test_activation_requires_every_decision_explicitly(self):
        for field, value in (('approval_reference', ' '), ('ts_basis', 'now'), ('observations_basis', None),
                             ('unmapped', 'guess'), ('runtime_provenance', ()), ('brand_mapping', (('toyota', ''),))):
            with self.assertRaises(Blocked):self.activation(**{field: value})

    def test_draft_versioned_branch_describes_real_evidence_and_active_contracts_stay_unchanged(self):
        from jsonschema import Draft202012Validator
        root = Path(__file__).resolve().parents[3]
        draft = json.loads((root/'contracts/proposals/manufacturer_delivery_v1.draft.schema.json').read_text())
        self.assertEqual(draft['x-status'], 'proposal_not_approved')
        validator = Draft202012Validator(draft)
        decisions = [self.decision(), self.decision(crops=[textured(60, 40)]),
                     self.decision(runtime=Runtime(error=RuntimeError('injected')))]
        for decision in decisions:
            evidence = json.loads(decision.evidence_json)
            validator.validate(dict(capability='enabled', evidence_version='manufacturer-evidence-v2', evidence=evidence))
        validator.validate('unsupported')
        # The active canonical contracts still admit only `unsupported`.
        for name in ('vehicle_plate_requests_v2', 'vehicle_redis_plate_v1', 'vehicle_redis_scalar_v1'):
            text = (root/f'contracts/{name}.schema.json').read_text()
            self.assertIn('"manufacturer": {\n          "enum": [\n            "unsupported"\n          ]', text)
            self.assertNotIn('manufacturer-evidence-v2', text)

    def test_row_builder_projects_explicit_make_registration(self):
        builder = RowBuilder(identity_policy=self.policy, validation=self.validation)
        natural = make_natural(self.decision(), self.life, activation=self.activation(), identity_policy=self.policy, validation=self.validation)
        row=builder.project_natural('fact_vehicle_make', natural, self.res, self.resolved.context, self.resolved)
        self.assertEqual(row,self.project(natural))


if __name__ == '__main__':
    unittest.main()
