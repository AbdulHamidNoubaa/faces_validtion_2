"""Authentication, dashboard, and user management views."""

from django.contrib import messages
from django.contrib.auth import login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.core.exceptions import PermissionDenied
from django.db.models import Count
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone
from django.utils.http import url_has_allowed_host_and_scheme
from django.views.decorators.http import require_http_methods

from web.accounts.forms import ArabicAuthenticationForm, UserCreateForm, UserUpdateForm
from web.accounts.models import UserProfile, get_user_profile
from web.accounts.permissions import can_manage_users, role_required
from web.checkpoint.models import VerificationLog
from web.checkpoint.services.dataset_catalog import get_dataset_summary
from web.checkpoint.services.verify_helpers import enrollment_stats


@require_http_methods(["GET", "POST"])
def login_view(request):
    if request.user.is_authenticated:
        return redirect("accounts:dashboard")
    form = ArabicAuthenticationForm(request, data=request.POST or None)
    if request.method == "POST" and form.is_valid():
        user = form.get_user()
        login(request, user)
        get_user_profile(user)
        messages.success(request, f"مرحباً، {user.username}")
        next_url = request.GET.get("next")
        if next_url and url_has_allowed_host_and_scheme(
            next_url, allowed_hosts={request.get_host()}
        ):
            return redirect(next_url)
        return redirect("accounts:dashboard")
    return render(request, "accounts/login.html", {"form": form})


@login_required
@require_http_methods(["POST"])
def logout_view(request):
    logout(request)
    messages.info(request, "تم تسجيل الخروج.")
    return redirect("accounts:login")


@login_required
def dashboard_view(request):
    profile = get_user_profile(request.user)
    ds = get_dataset_summary()
    log_qs = VerificationLog.objects.all()
    if not request.user.is_superuser and profile.role == UserProfile.Role.OPERATOR:
        log_qs = log_qs.filter(performed_by=request.user)
    stats = {
        "users_total": User.objects.filter(is_active=True).count(),
        "verifications_total": log_qs.count(),
        "verifications_accepted": log_qs.filter(accepted=True).count(),
    }
    today = timezone.localdate()
    stats["verifications_today"] = log_qs.filter(
        created_at__date=today
    ).count()
    return render(
        request,
        "accounts/dashboard.html",
        {
            "profile": profile,
            "stats": stats,
            "dataset": ds,
            "enrollment": enrollment_stats(),
            "recent_logs": log_qs.select_related("performed_by")[:8],
        },
    )


@role_required(UserProfile.Role.ADMIN)
def user_list_view(request):
    users = (
        User.objects.select_related("profile")
        .annotate(verification_count=Count("verification_logs"))
        .order_by("-date_joined")
    )
    return render(request, "accounts/user_list.html", {"users": users})


@role_required(UserProfile.Role.ADMIN)
@require_http_methods(["GET", "POST"])
def user_create_view(request):
    form = UserCreateForm(request.POST or None)
    if request.method == "POST" and form.is_valid():
        form.save()
        messages.success(request, "تم إنشاء المستخدم بنجاح.")
        return redirect("accounts:user_list")
    return render(request, "accounts/user_form.html", {"form": form, "title": "إضافة مستخدم"})


@role_required(UserProfile.Role.ADMIN)
@require_http_methods(["GET", "POST"])
def user_edit_view(request, user_id: int):
    user = get_object_or_404(User, pk=user_id)
    if user.is_superuser and not request.user.is_superuser:
        raise PermissionDenied
    form = UserUpdateForm(request.POST or None, instance=user)
    if request.method == "POST" and form.is_valid():
        form.save()
        messages.success(request, "تم تحديث المستخدم.")
        return redirect("accounts:user_list")
    return render(
        request,
        "accounts/user_form.html",
        {"form": form, "title": f"تعديل: {user.username}", "edit_user": user},
    )
