"""Public plate outcomes from executed stages, including historical v3 evidence."""
from .recognition_transport import selected_plate_attempt
from contracts.db1.reference import normalize_plate_text


def project_plate(evidence):
    attempts = evidence['plates']
    diagnostics = evidence.get('plate_diagnostics',{})
    detections = diagnostics.get('detections',0)+diagnostics.get('fallback_detections',0)
    ocr_attempts = sum(any(o['kind']=='registration_ocr' and o.get('model_call',o['outcome']=='succeeded')
                          for o in a['result'].get('evidence',{}).get('engine_operations',[]))
                       for a in attempts)
    chosen = selected_plate_attempt(evidence)
    result = chosen['result'] if chosen else {}
    read = result.get('recognition_status')=='recognized'
    errors = any(a['result']['recognition_status']=='error' for a in attempts)
    if read:
        status,reason = 'read','recognized'
    elif errors:
        status,reason = 'error','recognition_error'
    elif ocr_attempts:
        status,reason = 'unreadable','ocr_rejected'
    elif diagnostics.get('stopping_stage')=='stopped_before_ocr':
        status,reason = 'unresolved','stopped_before_ocr'
    elif detections or attempts:
        status,reason = 'unresolved',diagnostics.get('stopping_stage','ocr_not_attempted')
    else:
        status,reason = 'no_detection','no_plate_detected'
    return dict(status=status,text=normalize_plate_text(result['normalized_number']) if read else None,
        category=result['plate_category'] if read and result.get('predicted_category') else None,
        attempts=len(attempts),ocr_attempts=ocr_attempts,
        diagnostic=dict(version='plate-stages-v1',reason=reason,detections=detections,
            frame=chosen['receipt']['frame_index'] if chosen else None,
            rejection=(result.get('unresolved_reason') or (attempts[-1]['result'].get('unresolved_reason') if attempts else None)) if not read else None,
            stages=dict(diagnostics)))
