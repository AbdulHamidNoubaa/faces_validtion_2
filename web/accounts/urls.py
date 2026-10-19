from django.urls import path

from web.accounts import views

app_name = "accounts"

urlpatterns = [
    path("login/", views.login_view, name="login"),
    path("logout/", views.logout_view, name="logout"),
    path("dashboard/", views.dashboard_view, name="dashboard"),
    path("users/", views.user_list_view, name="user_list"),
    path("users/add/", views.user_create_view, name="user_create"),
    path("users/<int:user_id>/edit/", views.user_edit_view, name="user_edit"),
]
