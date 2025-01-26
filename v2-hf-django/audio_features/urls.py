from django.urls import path
from . import views

app_name = 'audio_features'

urlpatterns = [
    path('', views.audio_list, name='list'),
    path('scan/', views.scan_directory, name='scan'),
    path('extract/', views.extract_features, name='extract'),
] 