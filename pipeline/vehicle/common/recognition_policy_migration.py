"""Explicit compatibility migration for empty, project-owned DB1 test targets.

SQL fact columns already allow null confidence. The extension permits a decided
local-VLM colour with absent confidence and records a distinct policy fingerprint.
Never change compatibility underneath any accepted delivery.
"""
import json


def migrate_empty_fixture(connection, *, database, validation):
    from .db1_adapter import DB1Adapter
    from .db1_rows import ContractValidation
    from .recognition_transport import ROOT
    legacy = ContractValidation(json.loads((ROOT/'contracts/db1/rows.v0.schema.json').read_text()),
                                json.loads((ROOT/'contracts/db1/diagnostics.v1.schema.json').read_text()))
    before, after = DB1Adapter(validation=legacy).compatibility, DB1Adapter(validation=validation).compatibility
    with connection.transaction(), connection.cursor() as cursor:
        cursor.execute('SELECT current_database()')
        if cursor.fetchone() != (database,) or not database.startswith('db1p4_db_'):
            raise ValueError('compatibility migration requires the exact retained owned test target')
        cursor.execute('SELECT metadata FROM db1_delivery.compatibility WHERE singleton=true FOR UPDATE')
        current=bytes(cursor.fetchone()[0])
        if current==after: return False
        if current!=before: raise ValueError('unknown compatibility version; migration refused')
        cursor.execute('SELECT count(*) FROM db1_delivery.acceptance')
        if cursor.fetchone()!=(0,):raise ValueError('accepted data exists; compatibility migration refused')
        cursor.execute('UPDATE db1_delivery.compatibility SET metadata=%s WHERE singleton=true',(after,))
    return True
