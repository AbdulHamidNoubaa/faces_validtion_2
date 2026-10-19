from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.shortcuts import redirect
from django.urls import include, path


def root_redirect(request):
    if request.user.is_authenticated:
        return redirect("accounts:dashboard")
    return redirect("accounts:login")


urlpatterns = [
    path("admin/", admin.site.urls),
    path("accounts/", include("web.accounts.urls")),
    path("checkpoint/", include("web.checkpoint.urls")),
    path("", root_redirect, name="root"),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
