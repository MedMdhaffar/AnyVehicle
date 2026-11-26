-- DB1 — DEVELOPMENT SEED v0 (2026-10-08). Development only, never production.
--
-- The minimum to develop the vehicle service against 01_shared.sql + 02_vehicle.sql:
-- two orgs (so isolation can be tested), cameras, the classes, the models, one open
-- vehicle run, and starter vehicle dimensions. Fixed ids, so every developer has the same.
--
-- Run as the database owner. Row-level security filters the owner too (FORCE), so the
-- seed sets app.org_id before each org's rows.

BEGIN;

INSERT INTO dim_org (org_id, name) VALUES
    ('11111111-1111-4111-8111-111111111111', 'dev-org-a'),
    ('22222222-2222-4222-8222-222222222222', 'dev-org-b');

INSERT INTO dim_class (name, class_group) VALUES
    ('person', 'person'),
    ('bicycle', 'vehicle'), ('car', 'vehicle'), ('motorcycle', 'vehicle'),
    ('bus', 'vehicle'), ('truck', 'vehicle');

INSERT INTO dim_model (model_name, version, task) VALUES
    ('vehicle_detector', 1, 'detect'),
    ('plate_ocr',        1, 'ocr'),
    ('vehicle_colour',   1, 'classify'),
    ('vehicle_make',     1, 'classify');

INSERT INTO dim_plate_category (name) VALUES ('TN'), ('RS'), ('foreign'), ('diplomatic');
INSERT INTO dim_colour (name) VALUES
    ('white'), ('black'), ('grey'), ('silver'), ('red'), ('blue'), ('green'), ('yellow'), ('brown');
INSERT INTO dim_manufacturer (name) VALUES ('Renault'), ('Peugeot'), ('Toyota'), ('Volkswagen');
INSERT INTO dim_vehicle_model (manufacturer_key, name)
SELECT manufacturer_key, m.name
FROM (VALUES ('Renault', 'Clio'), ('Peugeot', '208'), ('Toyota', 'Corolla'), ('Volkswagen', 'Golf')) m(maker, name)
JOIN dim_manufacturer d ON d.name = m.maker;

-- org A: two cameras and one open vehicle run
SET LOCAL app.org_id = '11111111-1111-4111-8111-111111111111';
INSERT INTO dim_camera (camera_id, org_id, name, site, stream, timezone, valid_from) VALUES
    ('dev_gate_entry', '11111111-1111-4111-8111-111111111111', 'Gate entry', 'dev-site', 'main', 'Africa/Tunis', '2026-01-01'),
    ('dev_gate_exit',  '11111111-1111-4111-8111-111111111111', 'Gate exit',  'dev-site', 'main', 'Africa/Tunis', '2026-01-01');
INSERT INTO dim_run (run_id, org_id, lane_key, camera_key, model_key, service_version,
                     source_kind, started_at)
SELECT 'dev-vehicle-run-1', c.org_id, l.lane_key, c.camera_key, m.model_key, 'dev',
       'stream', '2026-10-08 06:00+00'
FROM dim_camera c, dim_lane l, dim_model m
WHERE c.camera_id = 'dev_gate_entry' AND l.name = 'vehicle'
  AND m.model_name = 'vehicle_detector' AND m.version = 1;

-- org B: one camera, to prove org A never sees it
SET LOCAL app.org_id = '22222222-2222-4222-8222-222222222222';
INSERT INTO dim_camera (camera_id, org_id, name, site, stream, timezone, valid_from) VALUES
    ('dev_other_org_cam', '22222222-2222-4222-8222-222222222222', 'Other org', 'dev-site-b', 'main', 'Africa/Tunis', '2026-01-01');

COMMIT;
