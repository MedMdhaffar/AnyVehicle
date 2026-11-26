"""Raw binary upload endpoint: bounded reads, no multipart memory buffering."""
from urllib.parse import unquote
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from common.video_uploads import UploadRejected, UploadStore


@csrf_exempt
@require_http_methods(['GET', 'POST'])
def videos(request):
    try:
        store = UploadStore()
        if request.method=='GET':result=store.limits()
        else:
            try:size=int(request.META.get('CONTENT_LENGTH', ''))
            except ValueError:raise UploadRejected('Upload length is required.', 411) from None
            result=store.receive(request, size, unquote(request.headers.get('X-Video-Filename', 'video')))
        response=JsonResponse(result, status=201 if request.method=='POST' else 200)
    except UploadRejected as error:response=JsonResponse({'error':str(error)}, status=error.status)
    except Exception:response=JsonResponse({'error':'Video upload or decoder validation failed.'}, status=400)
    response['Cache-Control']='no-store'
    return response
