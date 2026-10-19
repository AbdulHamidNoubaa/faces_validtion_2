"""Role-based access control."""

from functools import wraps

from django.contrib.auth.decorators import login_required
from django.core.exceptions import PermissionDenied

from web.accounts.models import UserProfile, get_user_profile


def role_required(*roles: str):
    """Require login and one of the given UserProfile.Role values."""

    def decorator(view_func):
        @login_required
        @wraps(view_func)
        def _wrapped(request, *args, **kwargs):
            profile = get_user_profile(request.user)
            if request.user.is_superuser:
                return view_func(request, *args, **kwargs)
            if profile and profile.role in roles:
                return view_func(request, *args, **kwargs)
            raise PermissionDenied("لا تملك صلاحية الوصول إلى هذه الصفحة.")

        return _wrapped

    return decorator


def can_manage_users(user) -> bool:
    if user.is_superuser:
        return True
    profile = get_user_profile(user)
    return profile is not None and profile.role == UserProfile.Role.ADMIN


def can_run_verification(user) -> bool:
    if user.is_superuser:
        return True
    profile = get_user_profile(user)
    return profile is not None and profile.role in (
        UserProfile.Role.ADMIN,
        UserProfile.Role.OPERATOR,
    )


def can_view_dataset(user) -> bool:
    if user.is_superuser:
        return True
    profile = get_user_profile(user)
    return profile is not None and profile.role in (
        UserProfile.Role.ADMIN,
        UserProfile.Role.ANALYST,
        UserProfile.Role.VIEWER,
    )


def can_manage_employees(user) -> bool:
    if user.is_superuser:
        return True
    profile = get_user_profile(user)
    return profile is not None and profile.role in (
        UserProfile.Role.ADMIN,
        UserProfile.Role.OPERATOR,
    )


def can_view_all_logs(user) -> bool:
    if user.is_superuser:
        return True
    profile = get_user_profile(user)
    return profile is not None and profile.role in (
        UserProfile.Role.ADMIN,
        UserProfile.Role.ANALYST,
    )
