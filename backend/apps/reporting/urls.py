"""URL patterns for reporting app."""
from django.urls import path
from . import views

urlpatterns = [
    path("report/<str:session_id>/", views.download_report, name="download_report"),
]
