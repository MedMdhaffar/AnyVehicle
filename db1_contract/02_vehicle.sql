-- DB1 vehicle lane — DEVELOPMENT CONTRACT v0 (2026-10-08)
--
-- The vehicle service's own tables (docs/sprints/DB1_MODEL.md §5.3), on top of
-- 01_shared.sql. Apply 01_shared.sql first.
--
-- Every fact opens with the six standard columns: org_id, run_key, camera_key, date_key,
-- ts, uid. The composite FK (run_key, org_id, camera_key) → dim_run makes a row's org and
-- camera always agree with its run.
--
-- ⚠ These facts are NOT partitioned, deliberately. A partitioned table's UNIQUE constraints
--   must include its partition column, which would forbid UNIQUE (run_key, track_id) on
--   fact_vehicle and so the FOREIGN KEY from each recognition to its vehicle. Vehicle volumes
--   are low (one row per vehicle, not per frame), so integrity wins; retention is a DELETE
--   by ts, served by the (org_id, ts) indexes.
--
-- ⚠ ts and date_key must agree: date_key is the LOCAL date of ts in the camera's timezone
--   (dim_camera.timezone). The ingest computes it; the database does not check it.

BEGIN;

-- ── vehicle dimensions (owned by the vehicle service) ───────────────────────
CREATE TABLE dim_plate_category (
    plate_category_key smallint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    name               text NOT NULL UNIQUE
);

CREATE TABLE dim_colour (
    colour_key  smallint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    name        text NOT NULL UNIQUE
);

CREATE TABLE dim_manufacturer (
    manufacturer_key int GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    name             text NOT NULL UNIQUE
);

CREATE TABLE dim_vehicle_model (
    vehicle_model_key int GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    manufacturer_key  int NOT NULL REFERENCES dim_manufacturer,
    name              text NOT NULL,
    UNIQUE (manufacturer_key, name),
    UNIQUE (vehicle_model_key, manufacturer_key)   -- target of fact_vehicle_make's check
);

-- ── fact_vehicle: one vehicle track ─────────────────────────────────────────
CREATE TABLE fact_vehicle (
    vehicle_id      bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    org_id          uuid   NOT NULL,
    run_key         bigint NOT NULL,
    camera_key      bigint NOT NULL REFERENCES dim_camera,
    date_key        int    NOT NULL REFERENCES dim_date,
    ts              timestamptz NOT NULL,              -- = first_ts
    uid             uuid   NOT NULL UNIQUE,
    track_id        bigint NOT NULL,
    first_ts        timestamptz NOT NULL,
    last_ts         timestamptz NOT NULL,
    class_key       smallint NOT NULL REFERENCES dim_class,
    class_conf      real CHECK (class_conf BETWEEN 0 AND 1),
    frames_observed int  NOT NULL CHECK (frames_observed > 0),
    best_media_key  bigint REFERENCES media_object,
    CHECK (ts = first_ts),
    CHECK (last_ts >= first_ts),
    FOREIGN KEY (run_key, org_id, camera_key) REFERENCES dim_run (run_key, org_id, camera_key),
    UNIQUE (run_key, track_id),                   -- the vehicle's identity
    UNIQUE (run_key, track_id, org_id, camera_key) -- target of the recognitions' FK
);

-- ── fact_plate_read: one plate recognition for one vehicle ──────────────────
CREATE TABLE fact_plate_read (
    plate_read_id      bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    org_id             uuid   NOT NULL,
    run_key            bigint NOT NULL,
    camera_key         bigint NOT NULL REFERENCES dim_camera,
    date_key           int    NOT NULL REFERENCES dim_date,
    ts                 timestamptz NOT NULL,           -- when the selected crop was taken
    uid                uuid   NOT NULL UNIQUE,
    track_id           bigint NOT NULL,                -- with run_key: → fact_vehicle
    model_key          bigint NOT NULL REFERENCES dim_model,   -- the OCR model
    outcome            text   NOT NULL CHECK (outcome IN ('read', 'unreadable', 'no_plate')),
    plate_text         text,                           -- ⚠ personal data
    plate_category_key smallint REFERENCES dim_plate_category,
    conf               real CHECK (conf BETWEEN 0 AND 1),
    attempts           int  NOT NULL CHECK (attempts >= 1),
    crop_media_key     bigint REFERENCES media_object,
    crop_score         real,
    -- a read has a text; a failed read has none
    CHECK ((outcome = 'read') = (plate_text IS NOT NULL)),
    FOREIGN KEY (run_key, org_id, camera_key) REFERENCES dim_run (run_key, org_id, camera_key),
    -- the vehicle this recognition belongs to: same run, same org, same camera
    FOREIGN KEY (run_key, track_id, org_id, camera_key)
        REFERENCES fact_vehicle (run_key, track_id, org_id, camera_key)
);

-- ── fact_vehicle_colour: one colour recognition ─────────────────────────────
CREATE TABLE fact_vehicle_colour (
    colour_read_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    org_id         uuid   NOT NULL,
    run_key        bigint NOT NULL,
    camera_key     bigint NOT NULL REFERENCES dim_camera,
    date_key       int    NOT NULL REFERENCES dim_date,
    ts             timestamptz NOT NULL,
    uid            uuid   NOT NULL UNIQUE,
    track_id       bigint NOT NULL,
    model_key      bigint NOT NULL REFERENCES dim_model,
    colour_key     smallint REFERENCES dim_colour,     -- NULL = no colour decided
    conf           real CHECK (conf BETWEEN 0 AND 1),
    state          text NOT NULL,
    samples        int  NOT NULL CHECK (samples >= 0),
    FOREIGN KEY (run_key, org_id, camera_key) REFERENCES dim_run (run_key, org_id, camera_key),
    -- the vehicle this recognition belongs to: same run, same org, same camera
    FOREIGN KEY (run_key, track_id, org_id, camera_key)
        REFERENCES fact_vehicle (run_key, track_id, org_id, camera_key)
);

-- ── fact_vehicle_make: one make / model recognition ─────────────────────────
CREATE TABLE fact_vehicle_make (
    make_read_id       bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    org_id             uuid   NOT NULL,
    run_key            bigint NOT NULL,
    camera_key         bigint NOT NULL REFERENCES dim_camera,
    date_key           int    NOT NULL REFERENCES dim_date,
    ts                 timestamptz NOT NULL,
    uid                uuid   NOT NULL UNIQUE,
    track_id           bigint NOT NULL,
    model_key          bigint NOT NULL REFERENCES dim_model,   -- the recognition network
    manufacturer_key   int REFERENCES dim_manufacturer,
    vehicle_model_key  int,
    manufacturer_conf  real CHECK (manufacturer_conf BETWEEN 0 AND 1),
    vehicle_model_conf real CHECK (vehicle_model_conf BETWEEN 0 AND 1),
    observations       int NOT NULL CHECK (observations >= 0),
    -- the make can be known without the model, never the reverse …
    CHECK (vehicle_model_key IS NULL OR manufacturer_key IS NOT NULL),
    -- … and a model is always stored under its own maker
    FOREIGN KEY (vehicle_model_key, manufacturer_key)
        REFERENCES dim_vehicle_model (vehicle_model_key, manufacturer_key),
    FOREIGN KEY (run_key, org_id, camera_key) REFERENCES dim_run (run_key, org_id, camera_key),
    -- the vehicle this recognition belongs to: same run, same org, same camera
    FOREIGN KEY (run_key, track_id, org_id, camera_key)
        REFERENCES fact_vehicle (run_key, track_id, org_id, camera_key)
);

-- ── indexes: org first, then time (every query is org-scoped under RLS) ─────
CREATE INDEX ON fact_vehicle        (org_id, camera_key, ts);
CREATE INDEX ON fact_plate_read     (org_id, camera_key, ts);
CREATE INDEX ON fact_plate_read     (run_key, track_id);
CREATE INDEX ON fact_plate_read     (org_id, plate_text, ts) WHERE plate_text IS NOT NULL;
CREATE INDEX ON fact_vehicle_colour (org_id, camera_key, ts);
CREATE INDEX ON fact_vehicle_colour (run_key, track_id);
CREATE INDEX ON fact_vehicle_make   (org_id, camera_key, ts);
CREATE INDEX ON fact_vehicle_make   (run_key, track_id);

-- ── row-level security: fail-closed on app.org_id ───────────────────────────
DO $$
DECLARE t text;
BEGIN
    FOREACH t IN ARRAY ARRAY['fact_vehicle', 'fact_plate_read', 'fact_vehicle_colour', 'fact_vehicle_make'] LOOP
        EXECUTE format('ALTER TABLE %I ENABLE ROW LEVEL SECURITY', t);
        EXECUTE format('ALTER TABLE %I FORCE ROW LEVEL SECURITY', t);
        EXECUTE format($p$CREATE POLICY org_isolation ON %I
            USING (org_id = NULLIF(current_setting('app.org_id', true), '')::uuid)
            WITH CHECK (org_id = NULLIF(current_setting('app.org_id', true), '')::uuid)$p$, t);
    END LOOP;
END $$;

COMMIT;
