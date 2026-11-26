# DB1 development contract — v0 (2026-10-08)

The database schema to develop the **vehicle service** against, before the platform's DB1
refactoring is built. The design and its reasons are in `../DB1_MODEL.md` (§5.3 for the
vehicle lane).

## Files

| File | What | Who owns it |
|---|---|---|
| `01_shared.sql` | the shared dimensions: `dim_org`, `dim_lane`, `dim_camera`, `dim_model`, `dim_class`, `dim_date`, `dim_run`, `media_object` | the platform: **never copy or edit it** |
| `02_vehicle.sql` | the vehicle tables: 4 dimensions, 4 facts | the vehicle service |
| `03_dev_seed.sql` | two dev orgs, two cameras, the classes, the models, one open vehicle run, starter colours / makes / models / plate categories | development only |
| `check_contract.py` | applies 01 → 03 to a throwaway database and checks every rule below (31 checks) | |

## Setup

```bash
createdb vehicle_dev
psql -d vehicle_dev -v ON_ERROR_STOP=1 -f 01_shared.sql
psql -d vehicle_dev -v ON_ERROR_STOP=1 -f 02_vehicle.sql
psql -d vehicle_dev -v ON_ERROR_STOP=1 -f 03_dev_seed.sql
```

PostgreSQL 16. Your service must connect as a **normal role, not a superuser**: a superuser
skips row-level security, and your code would then work in development and fail in
production. Every session sets its org first:

```sql
SET app.org_id = '11111111-1111-4111-8111-111111111111';   -- dev-org-a from the seed
```

No org set → every query returns 0 rows. That is intended (fail-closed).

Give that role what it needs, as the database owner:

```sql
CREATE ROLE vehicle_svc LOGIN PASSWORD '…';
GRANT USAGE ON SCHEMA public TO vehicle_svc;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO vehicle_svc;
GRANT INSERT ON fact_vehicle, fact_plate_read, fact_vehicle_colour, fact_vehicle_make,
                media_object, dim_run TO vehicle_svc;
GRANT UPDATE (ended_at, end_reason) ON dim_run TO vehicle_svc;
```

## The rules the database enforces

1. **Never create your own camera, date, run, model or vehicle-type table.** Use the shared ones.
   Vehicle type is `dim_class` (`class_group = 'vehicle'`).
2. **Every fact row carries** `org_id, run_key, camera_key, date_key, ts, uid`. The row's org
   and camera must be its run's: a composite foreign key refuses anything else.
3. **Your service opens a run** (`dim_run`, lane `vehicle`) when it starts on a camera and
   closes it (`ended_at`, `end_reason`) when it stops. In development you insert that row
   yourself; in production you send a run start / stop and the platform writes it. A recorded clip is a run with
   `source_kind = 'file'` and the clip in `media_object`.
4. **A vehicle is `(run_key, track_id)`**, unique. Each plate / colour / make recognition
   points at an existing vehicle of the same run, org and camera.
5. **Times are absolute `timestamptz`**, never frame numbers or seconds into a video.
   `fact_vehicle.ts` = `first_ts`. `date_key` is the **local date** of `ts` in the camera's
   timezone (`dim_camera.timezone`); the database does not check that, your code must.
6. **Write idempotently**: generate a `uid` per row and insert with
   `ON CONFLICT (uid) DO NOTHING`, so a retry never duplicates.
7. **A failed plate read is still a row**: `outcome = 'unreadable'` or `'no_plate'`, and then
   `plate_text` must be NULL. `outcome = 'read'` requires `plate_text`.
8. **Make and model**: the make can be stored without the model, never the reverse, and a
   model is always stored under its own maker (a Corolla can never be a Renault).
9. **Every recognition records its ML model** (`model_key` → `dim_model`).
10. **Images go to object storage** (MinIO / S3). Store only `(bucket, object_key)` in
    `media_object`, then reference its `media_key`. Never a URL, never a local path.
11. **`plate_text` is personal data.** It will get its own retention and access rule.

## What can still change when DB1 ships

The platform will deliver these tables through its own migrations; this folder is then
retired. **Tables, columns and keys are the contract.** Indexes, partitioning and retention
may change, and the service's insert code should not notice.

In production your service will not insert into the shared dimensions itself. It sends natural
ids (`run_id`, `camera_id`, class and model names) and the platform's ingest resolves them to
keys. Write your data-access code behind one small module, so that switch is one change.
