"""Template globals from config and auth."""

from django.conf import settings

from web.accounts.models import UserProfile, get_user_profile
from web.accounts.permissions import (
    can_manage_employees,
    can_manage_users,
    can_run_verification,
    can_view_dataset,
)


def site_settings(request):
    font = getattr(settings, "BFR_ARABIC_FONT", "tajawal")
    font_slug = "Tajawal" if font.lower() == "tajawal" else "Cairo"
    profile = get_user_profile(request.user) if request.user.is_authenticated else None
    return {
        "site_title_ar": settings.BFR_DEMO_TITLE_AR,
        "site_title_en": settings.BFR_DEMO_TITLE_EN,
        "arabic_font_slug": font_slug,
        "default_threshold": settings.BFR_DEFAULT_THRESHOLD,
        "model_choices": settings.BFR_MODELS,
        "current_profile": profile,
        "can_manage_users": can_manage_users(request.user) if request.user.is_authenticated else False,
        "can_verify": can_run_verification(request.user) if request.user.is_authenticated else False,
        "can_view_data": can_view_dataset(request.user) if request.user.is_authenticated else False,
        "can_manage_employees": can_manage_employees(request.user)
        if request.user.is_authenticated
        else False,
        "Role": UserProfile.Role,
    }
