"""Apply the DB1 dev contract to a THROWAWAY database and prove every rule it claims.

    ADMIN_DSN=postgresql://<superuser>:<pw>@<host>:<port>/postgres python check_contract.py

Creates database `db1_contract_check` and role `db1_contract_check_rw` (NOT a superuser, so
row-level security really applies), runs 01 → 03, checks every rule, then drops both.
Requires: Python 3.10+, psycopg 3. Touches no other database.
"""
import os, sys, uuid
from pathlib import Path
import psycopg

DIR = Path(__file__).resolve().parent
ADMIN = os.environ["ADMIN_DSN"]
DB = "db1_contract_check"
ROLE, ROLE_PW = "db1_contract_check_rw", "db1_check_pw"
_base = ADMIN.rsplit("/", 1)[0]
OWNER = f"{_base}/{DB}"
RW = f"postgresql://{ROLE}:{ROLE_PW}@{_base.split('@', 1)[1]}/{DB}"
A, B = "11111111-1111-4111-8111-111111111111", "22222222-2222-4222-8222-222222222222"

ok = fail = 0
def check(name, cond):
    global ok, fail
    ok, fail = (ok + 1, fail) if cond else (ok, fail + 1)
    print(("PASS " if cond else "FAIL ") + name)

def admin(sql):
    with psycopg.connect(ADMIN, autocommit=True) as c:
        c.execute(sql)

def cleanup():
    admin(f'DROP DATABASE IF EXISTS "{DB}" WITH (FORCE)')
    admin(f"DROP ROLE IF EXISTS {ROLE}")

cleanup()
admin(f'CREATE DATABASE "{DB}"')
admin(f"CREATE ROLE {ROLE} LOGIN NOSUPERUSER NOBYPASSRLS PASSWORD '{ROLE_PW}'")
try:
    for f in ("01_shared.sql", "02_vehicle.sql", "03_dev_seed.sql"):
        try:
            with psycopg.connect(OWNER, autocommit=True) as c:
                c.execute((DIR / f).read_text())
            check(f"apply {f}", True)
        except psycopg.Error as e:
            check(f"apply {f}: {e}", False); raise SystemExit(1)
    with psycopg.connect(OWNER, autocommit=True) as c:
        c.execute(f"GRANT USAGE ON SCHEMA public TO {ROLE}")
        c.execute(f"GRANT SELECT, INSERT ON ALL TABLES IN SCHEMA public TO {ROLE}")
        r = c.execute("select count(*), min(date_key), max(date_key) from dim_date").fetchone()
        check(f"dim_date generated {r}", r[0] > 4000)

    def run(sql, params=(), org=A):
        """One statement as the non-superuser role, in its own transaction."""
        with psycopg.connect(RW) as c:
            if org:
                c.execute("select set_config('app.org_id', %s, false)", (org,))
            cur = c.execute(sql, params)
            out = cur.fetchall() if cur.description else None
            c.commit()
            return out

    def rejects(name, sql, params=(), org=A):
        try:
            run(sql, params, org); check(f"REJECTS {name}", False)
        except psycopg.Error as e:
            check(f"REJECTS {name}  [{type(e).__name__}]", True)

    # ── RLS ──
    check("RLS org A sees its 2 cameras", run("select count(*) from dim_camera")[0][0] == 2)
    check("RLS org B sees its 1 camera", run("select count(*) from dim_camera", org=B)[0][0] == 1)
    check("RLS no org → 0 rows (fail-closed)", run("select count(*) from dim_camera", org=None)[0][0] == 0)
    check("RLS org B sees no org-A run", run("select count(*) from dim_run", org=B)[0][0] == 0)

    rk, ck = run("select run_key, camera_key from dim_run where run_id='dev-vehicle-run-1'")[0]
    other_ck = run("select camera_key from dim_camera where camera_id='dev_gate_exit'")[0][0]
    cls = run("select class_key from dim_class where name='car'")[0][0]
    ocr, col, mk = (run("select model_key from dim_model where model_name=%s", (n,))[0][0]
                    for n in ("plate_ocr", "vehicle_colour", "vehicle_make"))
    renault = run("select manufacturer_key from dim_manufacturer where name='Renault'")[0][0]
    toyota = run("select manufacturer_key from dim_manufacturer where name='Toyota'")[0][0]
    corolla = run("select vehicle_model_key from dim_vehicle_model where name='Corolla'")[0][0]

    V = """insert into fact_vehicle (org_id, run_key, camera_key, date_key, ts, uid, track_id,
           first_ts, last_ts, class_key, class_conf, frames_observed)
           values (%s,%s,%s,20261008,%s,%s,%s,%s,%s,%s,0.9,40)"""
    t1 = "2026-10-08 08:00:00+00"
    run(V, (A, rk, ck, t1, uuid.uuid4(), 7, t1, "2026-10-08 08:00:04+00", cls))
    check("valid vehicle inserted", run("select count(*) from fact_vehicle")[0][0] == 1)

    P = """insert into fact_plate_read (org_id, run_key, camera_key, date_key, ts, uid, track_id,
           model_key, outcome, plate_text, conf, attempts) values (%s,%s,%s,20261008,%s,%s,%s,%s,%s,%s,%s,1)"""
    run(P, (A, rk, ck, t1, uuid.uuid4(), 7, ocr, "read", "123 TU 4567", 0.95))
    run(P, (A, rk, ck, t1, uuid.uuid4(), 7, ocr, "unreadable", None, None))
    check("plate read + failed read (all nulls) inserted", run("select count(*) from fact_plate_read")[0][0] == 2)
    run("""insert into fact_vehicle_colour (org_id, run_key, camera_key, date_key, ts, uid, track_id,
           model_key, colour_key, conf, state, samples) values (%s,%s,%s,20261008,%s,%s,7,%s,null,null,'undecided',0)""",
        (A, rk, ck, t1, uuid.uuid4(), col))
    check("colour with no colour decided inserted", True)
    M = """insert into fact_vehicle_make (org_id, run_key, camera_key, date_key, ts, uid, track_id,
           model_key, manufacturer_key, vehicle_model_key, observations) values (%s,%s,%s,20261008,%s,%s,7,%s,%s,%s,3)"""
    run(M, (A, rk, ck, t1, uuid.uuid4(), mk, renault, None))
    run(M, (A, rk, ck, t1, uuid.uuid4(), mk, toyota, corolla))
    check("make without model + Toyota Corolla inserted", run("select count(*) from fact_vehicle_make")[0][0] == 2)

    # ── the rules ──
    rejects("read without plate_text", P, (A, rk, ck, t1, uuid.uuid4(), 7, ocr, "read", None, 0.9))
    rejects("failed read WITH plate_text", P, (A, rk, ck, t1, uuid.uuid4(), 7, ocr, "no_plate", "X", None))
    rejects("unknown outcome", P, (A, rk, ck, t1, uuid.uuid4(), 7, ocr, "maybe", None, None))
    rejects("Corolla stored as a Renault", M, (A, rk, ck, t1, uuid.uuid4(), mk, renault, corolla))
    rejects("model without a maker", M, (A, rk, ck, t1, uuid.uuid4(), mk, None, corolla))
    rejects("recognition of a vehicle that does not exist", P, (A, rk, ck, t1, uuid.uuid4(), 999, ocr, "unreadable", None, None))
    rejects("recognition on another camera than its vehicle's run", P, (A, rk, other_ck, t1, uuid.uuid4(), 7, ocr, "unreadable", None, None))
    rejects("same track twice in a run", V, (A, rk, ck, t1, uuid.uuid4(), 7, t1, t1, cls))
    u = uuid.uuid4()
    run(V, (A, rk, ck, t1, u, 8, t1, t1, cls))
    rejects("same uid twice (retry is a no-op only with ON CONFLICT)", V, (A, rk, ck, t1, u, 9, t1, t1, cls))
    rejects("ts different from first_ts", V, (A, rk, ck, "2026-10-08 09:00+00", uuid.uuid4(), 10, t1, t1, cls))
    rejects("last_ts before first_ts", V, (A, rk, ck, t1, uuid.uuid4(), 11, t1, "2026-10-08 07:00+00", cls))
    rejects("date_key that is not a date", V.replace("20261008", "20261399"), (A, rk, ck, t1, uuid.uuid4(), 12, t1, t1, cls))
    # org integrity: org B writing onto org A's run
    rejects("org B row on org A's run (RLS WITH CHECK)", V, (B, rk, ck, t1, uuid.uuid4(), 13, t1, t1, cls), org=A)
    rejects("org B row on org A's run (composite FK)", V, (B, rk, ck, t1, uuid.uuid4(), 14, t1, t1, cls), org=B)
    rejects("run on a camera of another org",
            """insert into dim_run (run_id, org_id, lane_key, camera_key, source_kind, started_at)
               values ('x', %s, 1, %s, 'stream', now())""", (B, ck), org=B)
    rejects("file run without its clip",
            """insert into dim_run (run_id, org_id, lane_key, camera_key, source_kind, started_at)
               values ('y', %s, 1, %s, 'file', now())""", (A, ck))
    check("org B sees 0 vehicles", run("select count(*) from fact_vehicle", org=B)[0][0] == 0)
    check("no org sees 0 plate reads", run("select count(*) from fact_plate_read", org=None)[0][0] == 0)
    # idempotent retry, the way the ingest will write
    run(V + " on conflict (uid) do nothing", (A, rk, ck, t1, u, 8, t1, t1, cls))
    check("retry with ON CONFLICT (uid) DO NOTHING is a no-op", run("select count(*) from fact_vehicle")[0][0] == 2)
finally:
    cleanup()
print(f"\n{ok} passed, {fail} failed")
sys.exit(1 if fail else 0)
