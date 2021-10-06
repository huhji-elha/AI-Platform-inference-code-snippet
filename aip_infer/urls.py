# -*- coding:utf-8 -*-
from django.urls import path
from . import views

app_name = "aip_infer"
urlpatterns = [
    path("getInferenceSetting", views.GetInferenceSetting.as_view(), name="getInferenceSetting"),

    path("getImageList", views.GetImageList.as_view(), name="getImageList"),
    path("getTabularData", views.GetTabularData.as_view(), name="getTabularData"),
    path("getImage", views.GetImage.as_view(), name="getImage"),

    path("startInference", views.StartInference.as_view(), name="startInference"),
    path("stopInference", views.StopInference.as_view(), name="stopInference"),

    path("getCustomModelList", views.GetCustomModelList.as_view(), name="getCustomModelList"),
    path("getInferenceLog", views.GetInferenceLog.as_view(), name="getInferenceLog"),
    path("getInferenceResult", views.GetInferenceResult.as_view(), name="getInferenceResult"),
    path("downloadInferenceResult", views.DownloadInferenceResult.as_view(), name="downloadInferenceResult"),

    path('media/inference/<str:inference_id>/<str:results>/<str:file_name>', views.ServeResultMedia.as_view()),
    path('media/inference/<str:inference_id>/<str:file_name>', views.ServeMedia.as_view()),
    path('upload/inference/<str:inference_id>', views.UploadMedia.as_view()),
    path('hasOtherInferenceRunning', views.HasOtherInferenceRunning.as_view()),
]
