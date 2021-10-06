import os
import json
from django.conf import settings
from django.db import transaction
from rest_framework.views import APIView
from rest_framework.response import Response
from .services.InferenceService import InferenceService

from aip import consts
from aip.auth_api_view import AuthAPIView
from aip.decorators import check_developer, check_developer_or_annotator

inferenceService = InferenceService()


class GetImageList(APIView):
    @check_developer
    def get(self, request):
        dataset_id = request.GET.get("dataset_id")
        image_list = inferenceService.get_image_list(dataset_id)
        data = {"image_list": image_list}
        return Response(data)


class GetTabularData(APIView):
    @check_developer
    def get(self, request):
        dataset_id = request.GET.get("dataset_id")
        tabular_column = inferenceService.get_tabular_info(dataset_id)
        data = {
            "column_num": len(tabular_column),
            "column_name": tabular_column,
            "tabular_list": inferenceService.get_tabular_data_from_storage(dataset_id),
        }
        return Response(data)


class GetImage(APIView):
    @check_developer
    def get(self, request):
        dataset_id = request.GET.get("dataset_id")
        data_name = request.GET.get("data_name")
        data = {"image_data": inferenceService.get_image(dataset_id, data_name)}
        return Response(data)


class StartInference(AuthAPIView):
    @transaction.atomic
    @check_developer_or_annotator
    def post(self, request):
        user_id = self.user_id
        file_list = request.FILES.getlist("files", "")
        project_id = request.POST.get("project_id", "")
        experiment_id = request.POST.get("experiment_id", "")
        dataset_id = request.POST.get("dataset_id", None)

        inference_info = {
            "user_id": user_id,
            "project_id": project_id,
            "experiment_id": experiment_id,
            "dataset_id": dataset_id,
            "create_user": user_id,
            "file_list": file_list,
            "annotation_yn": consts.NOT_ANNOTATION
        }
        inference_id = inferenceService.add_inference_id(inference_info)
        inferenceService.add_inference_log_info(inference_id, user_id)

        if file_list:
            inferenceService.upload_inference_file(inference_id, file_list)
            inferenceService.update_inference_local_filename(inference_id)

        inferenceService.start_inference(inference_info, inference_id)
        return Response({"inference_id": inference_id})


class StopInference(AuthAPIView):
    @check_developer
    def post(self, request):
        inference_id = json.loads(request.body.decode("utf-8"))["inference_id"]
        user_id = self.user_id
        inferenceService.update_stop_info(inference_id, user_id)
        return Response({})


class GetInferenceLog(APIView):
    @check_developer
    def get(self, request):
        inference_id = request.GET.get("inference_id")
        data = inferenceService.get_inference_log_progress(inference_id)
        return Response(data)


class GetInferenceResult(AuthAPIView):
    @check_developer
    def get(self, request):
        inference_id = request.GET.get("inference_id")
        user_id = self.user_id
        data = {
            "pod_status": inferenceService.get_inference_pod_status(inference_id, user_id),
            "result_data": inferenceService.get_result_dataset(inference_id),
        }
        return Response(data)


class DownloadInferenceResult(APIView):
    @check_developer
    def get(self, request):
        inference_id = request.GET.get("inference_id")
        save_format = request.GET.get("save_format")
        return inferenceService.download_result(inference_id, save_format)


class GetInferenceSetting(APIView):
    @check_developer
    def get(self, request):
        project_id = request.GET.get("project_id", "")
        experiment_id = json.loads(request.GET.get("experiment_id", ""))
        if not experiment_id:
            experiment_id = None
        return Response(
            inferenceService.get_inference_setting(project_id, experiment_id)
        )


class ServeResultMedia(APIView):
    def get(self, request, inference_id, results, file_name):
        relative_path = os.path.join("inference", inference_id, results, file_name)
        file_path = os.path.join(settings.MEDIA_ROOT, relative_path)
        data = inferenceService.serve_result_file(file_path)
        return data


class ServeMedia(APIView):
    def get(self, request, inference_id, file_name):
        relative_path = os.path.join("inference", inference_id, file_name)
        file_path = os.path.join(settings.MEDIA_ROOT, relative_path)
        data = inferenceService.serve_result_file(file_path)
        return data


class UploadMedia(APIView):
    def post(self, request, inference_id):
        files = request.FILES.getlist("files", "")
        relative_path = os.path.join("inference", inference_id)
        file_path = os.path.join(settings.MEDIA_ROOT, relative_path)
        inferenceService.upload_result_file(file_path, files)
        return Response({})


class HasOtherInferenceRunning(APIView):
    def get(self, request):
        project_id = request.GET.get("project_id")
        data = inferenceService.is_inference_running(project_id)
        return Response(data)


class GetCustomModelList(APIView):
    @check_developer_or_annotator
    def get(self, request):
        project_id = request.GET.get("project_id")
        return Response({"model_list": inferenceService.get_custom_model_list(project_id)})
