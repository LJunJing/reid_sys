from django.http import JsonResponse, HttpResponse
from django.shortcuts import render
from django.conf import settings
import os
from PersonReID.inference import load
from django.views import View

def index(request):
    """Placeholder index view"""
    return render(request, 'reid_sys.html')

def upload_file(request):
    if request.method == 'POST':
        model_name = request.POST.get('model')
        dataset_name = request.POST.get('dataset')
        gallery_num = request.POST.get('gallery_num')
        if (model_name is None) or (dataset_name is None) or (gallery_num is None):
            return JsonResponse({'success': False, 'message': 'No model, dataset, or gallery_num selected'})
        print("model:", model_name)
        print("dataset:", dataset_name)
        print("gallery_num:", gallery_num)

        # query_files = request.FILES.getlist('query_files')  # 获取上传的文件列表，参数中的 'files' 对应前端发送的 formData 中的变量名
        # gallery_files = request.FILES.getlist('gallery_files')  # 获取上传的文件列表，参数中的 'files' 对应前端发送的 formData 中的变量名
        # if query_files is None or gallery_files is None:
        #     return JsonResponse({'success': False, 'message': '请上传图片'})
        # query = []
        # gallery = []
        # for myfile in query_files:
        #     print(myfile)
        #     query.append(myfile.name)
        #     filepath = os.path.join(settings.MEDIA_ROOT, 'query', str(myfile))
        #     # query.append(filepath)
        #     with open(filepath, 'wb+') as destination:
        #         for chunk in myfile.chunks():
        #             destination.write(chunk)
        # for myfile in gallery_files:
        #     print(myfile)
        #     gallery.append(myfile.name)
        #     filepath = os.path.join(settings.MEDIA_ROOT, 'gallery', str(myfile))
        #     # gallery.append(filepath)
        #     with open(filepath, 'wb+') as destination:
        #         for chunk in myfile.chunks():
        #             destination.write(chunk)

        query_list, gallery2dis = load(model_name, dataset_name, int(gallery_num))
        # query_list 和 gallery_list 中存放的是图片名称，gallery2dis是一个字典，gallery:distance
        print("query_list:", query_list)
        print("gallery2dis:", gallery2dis)

        return JsonResponse({'success': True, 'data': {'query_list': query_list, 'gallery2dis': gallery2dis}})
    else:
        return JsonResponse({'success': False, 'message': 'Only POST requests are allowed.'})

