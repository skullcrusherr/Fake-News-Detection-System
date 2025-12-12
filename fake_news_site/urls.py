from django.contrib import admin
from django.urls import path, include

from fake_news_site import settings
from django.conf.urls.static import static  # ✅ add this


urlpatterns = [
    path("admin/", admin.site.urls),
    path("", include("detector.urls")),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)