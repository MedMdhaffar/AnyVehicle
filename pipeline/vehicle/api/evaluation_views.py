"""Evaluation-only endpoints using existing local console media boundaries."""
from django.http import FileResponse, HttpResponse
from rest_framework.decorators import api_view
from rest_framework.response import Response
from common.evaluation import ReviewConflict, metrics, public_records, export_csv
from common.evaluation_catalog import EvaluationCatalog


def filtered(request, store, run_id):
    rows=public_records(store,run_id)
    model=request.query_params.get('model')
    return [r for r in rows if not model or r.get('model_version')==model]


@api_view(['GET'])
def runs(request):
    return Response(dict(runs=EvaluationCatalog().runs()),headers={'Cache-Control':'no-store'})


@api_view(['GET'])
def detail(request,run_id):
    try:
        store=EvaluationCatalog().open(run_id);rows=filtered(request,store,run_id)
        calls=store.calls();model=request.query_params.get('model')
        if model:calls=[c for c in calls if c.get('model_version')==model]
        return Response(dict(**store.metadata(),records=rows,calls=calls,metrics=metrics(rows,calls)),headers={'Cache-Control':'no-store'})
    except (FileNotFoundError,ValueError):return Response(dict(error='Evaluation run unavailable.'),status=404)


@api_view(['POST'])
def review(request,run_id,evidence_id):
    try:
        store=EvaluationCatalog().open(run_id)
        store.review(evidence_id,request.data.get('verdict'),request.data.get('correct_label'),
            revision=request.data.get('revision'),reviewer=request.user.get_username() if request.user.is_authenticated else 'local reviewer')
        return Response(dict(saved=True))
    except ReviewConflict:return Response(dict(error='This review changed. Refresh before saving.'),status=409)
    except FileNotFoundError:return Response(dict(error='Evaluation record unavailable.'),status=404)
    except (ValueError,TypeError):return Response(dict(error='Invalid review.'),status=400)


@api_view(['POST'])
def inspect(request,run_id,evidence_id):
    try:
        store=EvaluationCatalog().open(run_id)
        store.inspect(evidence_id,request.data.get('status'),request.data.get('note'),
            revision=request.data.get('revision'),reviewer=request.user.get_username() if request.user.is_authenticated else 'local reviewer')
        return Response(dict(saved=True))
    except ReviewConflict:return Response(dict(error='This inspection changed. Refresh before saving.'),status=409)
    except FileNotFoundError:return Response(dict(error='Evaluation record unavailable.'),status=404)
    except (ValueError,TypeError):return Response(dict(error='Invalid inspection.'),status=400)


@api_view(['GET'])
def image(request,run_id,image_name):
    try:
        store=EvaluationCatalog().open(run_id)
        # Only media actually referenced by this run's evidence/calls is served.
        names=set()
        for row in store.records():
            names.update(v for v in (row.get('image'),row.get('original_frame')) if v)
            names.update(row.get('inputs',[]))
        for call in store.calls():
            if call.get('image'):names.add(call['image'])
        if image_name not in names:raise FileNotFoundError()
        response=FileResponse(store.crop(image_name).open('rb'),content_type='image/png')
        response['X-Content-Type-Options']='nosniff';response['Cache-Control']='private, no-store'
        return response
    except (FileNotFoundError,ValueError):return Response(dict(error='Evidence image unavailable.'),status=404)


@api_view(['GET'])
def download(request,run_id):
    try:
        store=EvaluationCatalog().open(run_id)
        response=HttpResponse(export_csv(filtered(request,store,run_id)),content_type='text/csv; charset=utf-8')
        response['Content-Disposition']=f'attachment; filename="waycon-evaluation-{run_id}.csv"'
        response['Cache-Control']='no-store'
        return response
    except (FileNotFoundError,ValueError):return Response(dict(error='Evaluation run unavailable.'),status=404)


@api_view(['GET'])
def history(request,run_id):
    try:
        store=EvaluationCatalog().open(run_id)
        return Response(dict(records=public_records(store,run_id),changes=store.history(),inspection_changes=store.inspection_history()),headers={'Cache-Control':'no-store'})
    except (FileNotFoundError,ValueError):return Response(dict(error='Evaluation run unavailable.'),status=404)
