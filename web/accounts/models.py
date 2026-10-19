"""User profile and roles for banking face recognition portal."""

from django.contrib.auth.models import User
from django.db import models


class UserProfile(models.Model):
    class Role(models.TextChoices):
        ADMIN = "admin", "مدير النظام"
        OPERATOR = "operator", "مشغّل نقطة التحقق"
        ANALYST = "analyst", "محلّل بيانات"
        VIEWER = "viewer", "قراءة فقط"

    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name="profile")
    role = models.CharField(
        max_length=16,
        choices=Role.choices,
        default=Role.OPERATOR,
    )
    full_name_ar = models.CharField(max_length=120, blank=True)
    employee_id = models.CharField(max_length=64, blank=True, verbose_name="رقم الموظف")
    branch = models.CharField(max_length=120, blank=True, verbose_name="الفرع")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "ملف مستخدم"
        verbose_name_plural = "ملفات المستخدمين"

    def __str__(self) -> str:
        return self.display_name

    @property
    def display_name(self) -> str:
        if self.full_name_ar:
            return self.full_name_ar
        return self.user.get_full_name() or self.user.username


def get_user_profile(user: User) -> UserProfile | None:
    if not user.is_authenticated:
        return None
    if user.is_superuser:
        profile, _ = UserProfile.objects.get_or_create(
            user=user,
            defaults={"role": UserProfile.Role.ADMIN, "full_name_ar": "مدير النظام"},
        )
        return profile
    try:
        return user.profile
    except UserProfile.DoesNotExist:
        return UserProfile.objects.create(user=user, role=UserProfile.Role.VIEWER)
