"""URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/train/', include('aip_train.urls', namespace='aip_train')),
    path('api/models/', include('aip_deploy.urls', namespace='aip_deploy')),
    path('api/common/',  include('aip_common.urls', namespace='aip_common')),
    path('api/common/',  include('aip_prob_def.urls', namespace='aip_prob_def')),
    path('api/annotation/', include('aip_annotation.urls', namespace='aip_annotation')),
    path('api/datapre/', include("aip_data.urls", namespace='aip_data')),
    path('api/infer/', include('aip_infer.urls', namespace='aip_infer')),
]
