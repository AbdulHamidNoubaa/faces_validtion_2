from django.urls import path

from web.checkpoint import views
from web.checkpoint import views_employees

app_name = "checkpoint"

urlpatterns = [
    path("verify/", views.checkpoint_home, name="home"),
    path("history/", views.verification_history, name="history"),
    path("api/verify/", views.checkpoint_api_verify, name="api_verify"),
    path("data/", views.dataset_overview, name="data_overview"),
    path("data/identities/", views.dataset_identities, name="data_identities"),
    path("data/identity/<str:identity>/", views.dataset_identity_detail, name="data_identity"),
    path("data/image/<path:relpath>", views.dataset_image, name="data_image"),
    path("employees/", views_employees.employee_list, name="employee_list"),
    path("employees/add/", views_employees.employee_create, name="employee_create"),
    path("employees/<int:pk>/", views_employees.employee_detail, name="employee_detail"),
    path("employees/<int:pk>/edit/", views_employees.employee_edit, name="employee_edit"),
    path("employees/<int:pk>/upload/", views_employees.employee_upload_photos, name="employee_upload"),
    path(
        "employees/photos/<int:photo_id>/reprocess/",
        views_employees.employee_reprocess_photo,
        name="employee_reprocess_photo",
    ),
]
