"""Root URL configuration for DeepCue."""
from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/", include("apps.sessions_app.urls")),
    path("api/", include("apps.reporting.urls")),
]
