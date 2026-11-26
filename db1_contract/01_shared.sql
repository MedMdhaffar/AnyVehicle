-- DB1 shared schema — DEVELOPMENT CONTRACT v0 (2026-10-08)
--
-- The platform-owned part of the DB1 model (docs/sprints/DB1_MODEL.md §3, §4.12, §5.2):
-- the dimensions every service references. A service NEVER creates its own copy of these
-- tables: it applies this file, then its own tables on top.
--
-- ⚠ This is a contract to develop against, not the production schema. DB1 will ship the
--   real one through storage_v2's migrate; it replaces this file. What may change is
--   physical (partition layout, indexes, retention), not the tables, columns or keys below.
--
-- ⚠ Row-level security is ON for every org-scoped table. A superuser bypasses it, so a
--   test run as a superuser proves nothing about isolation. Connect as a normal role and
--   set the org first:   SET app.org_id = '<uuid>';
--
-- PostgreSQL 16. Apply on an empty database:  psql -v ON_ERROR_STOP=1 -f 01_shared.sql

BEGIN;

-- ── org ─────────────────────────────────────────────────────────────────────
CREATE TABLE dim_org (
    org_id      uuid PRIMARY KEY,
    name        text NOT NULL
);

-- ── lane: which service produced the data ───────────────────────────────────
CREATE TABLE dim_lane (
    lane_key    smallint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    name        text NOT NULL UNIQUE,          -- standard, face_push, punch, ar, vehicle
    service     text NOT NULL,
    description text
);

-- ── camera: one row per VERSION (SCD2) ──────────────────────────────────────
CREATE TABLE dim_camera (
    camera_key  bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    camera_id   text NOT NULL,                 -- natural id, e.g. hikvision_L45416138_14_sub
    org_id      uuid NOT NULL REFERENCES dim_org,
    name        text,
    site        text,
    stream      text,                          -- main | sub
    timezone    text NOT NULL,                 -- the site's; date_key is the local date
    valid_from  timestamptz NOT NULL,
    valid_to    timestamptz,                   -- NULL = current version
    CHECK (valid_to IS NULL OR valid_to > valid_from),
    UNIQUE (camera_id, valid_from),
    UNIQUE (camera_key, org_id)                -- target of dim_run's org check
);
-- at most one current version per camera
CREATE UNIQUE INDEX dim_camera_one_current ON dim_camera (camera_id) WHERE valid_to IS NULL;

-- ── ML model: one row per version ───────────────────────────────────────────
CREATE TABLE dim_model (
    model_key     bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    model_name    text NOT NULL,
    version       int  NOT NULL,
    task          text NOT NULL,               -- detect, classify, face, vlm, ocr, …
    base_model    text,
    weights_path  text,
    map50         real,
    registered_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (model_name, version)
);

-- ── detector classes ────────────────────────────────────────────────────────
CREATE TABLE dim_class (
    class_key   smallint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    name        text NOT NULL UNIQUE,          -- car, truck, bus, person, …
    class_group text NOT NULL                  -- vehicle, person, … (not "group": reserved word)
);

-- ── calendar ────────────────────────────────────────────────────────────────
CREATE TABLE dim_date (
    date_key    int  PRIMARY KEY,              -- yyyymmdd
    date        date NOT NULL UNIQUE,
    day_of_week int  NOT NULL CHECK (day_of_week BETWEEN 1 AND 7),   -- ISO, Monday = 1
    iso_week    int  NOT NULL,
    month       int  NOT NULL,
    quarter     int  NOT NULL,
    year        int  NOT NULL,
    is_weekend  bool NOT NULL,
    is_holiday  bool NOT NULL DEFAULT false
);
-- 2024 → 2035, generated: no service ever writes this table.
INSERT INTO dim_date
SELECT to_char(d, 'YYYYMMDD')::int, d::date,
       extract(isodow FROM d)::int, extract(week FROM d)::int,
       extract(month FROM d)::int, extract(quarter FROM d)::int, extract(year FROM d)::int,
       extract(isodow FROM d) IN (6, 7), false
FROM generate_series('2024-01-01'::date, '2035-12-31'::date, interval '1 day') d;

-- ── run: one lane processing one camera, from start to stop ─────────────────
-- media_object is created below; dim_run.source_media_key gets its FK after it.
CREATE TABLE dim_run (
    run_key          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    run_id           text NOT NULL UNIQUE,     -- natural id from the producing service
    org_id           uuid NOT NULL REFERENCES dim_org,
    lane_key         smallint NOT NULL REFERENCES dim_lane,
    camera_key       bigint NOT NULL,
    model_key        bigint REFERENCES dim_model,  -- the run's detector; NULL if none
    service_version  text,
    source_kind      text NOT NULL CHECK (source_kind IN ('stream', 'file')),
    source_media_key bigint,                   -- the clip, when source_kind = file
    started_at       timestamptz NOT NULL,
    ended_at         timestamptz,              -- NULL while running
    end_reason       text,
    CHECK (ended_at IS NULL OR ended_at >= started_at),
    CHECK (source_kind = 'stream' OR source_media_key IS NOT NULL),
    -- the run's org IS the camera's org
    FOREIGN KEY (camera_key, org_id) REFERENCES dim_camera (camera_key, org_id),
    -- target of every fact's composite FK (DB1_MODEL §5.2)
    UNIQUE (run_key, org_id, camera_key)
);

-- ── files kept in object storage (MinIO / S3) ───────────────────────────────
-- The database stores WHERE a file is (bucket, object_key), never a URL or a local path.
CREATE TABLE media_object (
    media_key    bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    org_id       uuid NOT NULL REFERENCES dim_org,
    run_key      bigint,                       -- NULL for a file no run produced (an input clip)
    camera_key   bigint,
    ts           timestamptz NOT NULL,
    kind         text NOT NULL,                -- vehicle_crop, plate_crop, face_crop, body_crop, clip, …
    bucket       text NOT NULL,
    object_key   text NOT NULL,
    content_type text NOT NULL,
    bytes        bigint NOT NULL CHECK (bytes >= 0),
    width        int,
    height       int,
    sha256       text NOT NULL,
    UNIQUE (bucket, object_key),
    FOREIGN KEY (run_key, org_id, camera_key) REFERENCES dim_run (run_key, org_id, camera_key)
);

ALTER TABLE dim_run
    ADD FOREIGN KEY (source_media_key) REFERENCES media_object;

-- ── lanes known today ───────────────────────────────────────────────────────
INSERT INTO dim_lane (name, service, description) VALUES
    ('standard',  'standard_service',             'object detection, tracking, rules'),
    ('face_push', 'video-analytics.push',         'face identification at doors'),
    ('punch',     'video-analytics.punch',        'punch verification on clips'),
    ('ar',        'action_recognition_pipeline',  'action recognition (VLM)'),
    ('vehicle',   'vehicle_service',              'plates, colour, make and model');

-- ── row-level security: fail-closed on app.org_id ───────────────────────────
-- Unset app.org_id → NULL → no rows. Never "everything".
ALTER TABLE dim_camera   ENABLE ROW LEVEL SECURITY;  ALTER TABLE dim_camera   FORCE ROW LEVEL SECURITY;
ALTER TABLE dim_run      ENABLE ROW LEVEL SECURITY;  ALTER TABLE dim_run      FORCE ROW LEVEL SECURITY;
ALTER TABLE media_object ENABLE ROW LEVEL SECURITY;  ALTER TABLE media_object FORCE ROW LEVEL SECURITY;

CREATE POLICY org_isolation ON dim_camera
    USING (org_id = NULLIF(current_setting('app.org_id', true), '')::uuid)
    WITH CHECK (org_id = NULLIF(current_setting('app.org_id', true), '')::uuid);
CREATE POLICY org_isolation ON dim_run
    USING (org_id = NULLIF(current_setting('app.org_id', true), '')::uuid)
    WITH CHECK (org_id = NULLIF(current_setting('app.org_id', true), '')::uuid);
CREATE POLICY org_isolation ON media_object
    USING (org_id = NULLIF(current_setting('app.org_id', true), '')::uuid)
    WITH CHECK (org_id = NULLIF(current_setting('app.org_id', true), '')::uuid);

COMMIT;
