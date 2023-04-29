from django.urls import path
from object_detection import views

urlpatterns = [path('object_detection', views.object_detection),
                ]