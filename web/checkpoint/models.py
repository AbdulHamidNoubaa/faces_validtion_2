"""SQLite models — employees, photos, verification audit."""

from django.contrib.auth.models import User
from django.db import models


def employee_photo_upload_to(instance, filename: str) -> str:
    code = instance.employee.employee_code
    return f"employees/{code}/{filename}"


class BankEmployee(models.Model):
    """موظف مصرف مسجّل في النظام."""

    employee_code = models.CharField(max_length=32, unique=True, verbose_name="رقم الموظف")
    full_name_ar = models.CharField(max_length=120, verbose_name="الاسم بالعربية")
    full_name_en = models.CharField(max_length=120, blank=True, verbose_name="الاسم بالإنجليزية")
    branch = models.CharField(max_length=120, blank=True, verbose_name="الفرع")
    department = models.CharField(max_length=120, blank=True, verbose_name="القسم")
    is_active = models.BooleanField(default=True)
    notes = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    created_by = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="employees_created",
    )

    class Meta:
        ordering = ["full_name_ar"]
        verbose_name = "موظف مصرف"
        verbose_name_plural = "موظفو المصرف"

    def __str__(self) -> str:
        return f"{self.full_name_ar} ({self.employee_code})"

    @property
    def folder_slug(self) -> str:
        return self.employee_code.replace(" ", "_")


class EmployeePhoto(models.Model):
    """صورة موظف — خام + معالجة بنفس pipeline مجموعة البيانات."""

    class Status(models.TextChoices):
        PENDING = "pending", "قيد الانتظار"
        PROCESSED = "processed", "تمت المعالجة"
        FAILED = "failed", "فشل"

    employee = models.ForeignKey(
        BankEmployee,
        on_delete=models.CASCADE,
        related_name="photos",
    )
    image = models.ImageField(upload_to=employee_photo_upload_to, verbose_name="الصورة الأصلية")
    apply_low_light = models.BooleanField(
        default=False,
        verbose_name="محاكاة إضاءة منخفضة (كما في LFW)",
    )
    status = models.CharField(
        max_length=16,
        choices=Status.choices,
        default=Status.PENDING,
        db_index=True,
    )
    processed_relpath = models.CharField(max_length=512, blank=True)
    face_detected = models.BooleanField(default=False)
    embedding_json = models.TextField(blank=True)
    error_message = models.TextField(blank=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    processed_at = models.DateTimeField(null=True, blank=True)
    uploaded_by = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="employee_photos_uploaded",
    )

    class Meta:
        ordering = ["-uploaded_at"]
        verbose_name = "صورة موظف"
        verbose_name_plural = "صور الموظفين"

    def __str__(self) -> str:
        return f"Photo #{self.pk} — {self.employee.employee_code}"


class VerificationLog(models.Model):
    """سجل عمليات التحقق (قاعدة بيانات SQLite3)."""

    class Source(models.TextChoices):
        UPLOAD = "upload", "رفع صورة"
        CAMERA = "camera", "كاميرا"

    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    source = models.CharField(max_length=16, choices=Source.choices, default=Source.UPLOAD)
    architecture = models.CharField(max_length=32)
    threshold = models.FloatField()
    brightness = models.FloatField()
    identity_predicted = models.CharField(max_length=128)
    confidence = models.FloatField()
    accepted = models.BooleanField()
    latency_ms = models.FloatField()
    face_detected = models.BooleanField(default=True)
    matched_employee = models.ForeignKey(
        BankEmployee,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="verification_matches",
    )
    match_source = models.CharField(
        max_length=32,
        blank=True,
        help_text="employee_gallery | classifier",
    )
    performed_by = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="verification_logs",
    )
    client_ip = models.GenericIPAddressField(null=True, blank=True)
    user_agent = models.CharField(max_length=255, blank=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "سجل تحقق"
        verbose_name_plural = "سجلات التحقق"

    def __str__(self) -> str:
        status = "مقبول" if self.accepted else "مرفوض"
        return f"{self.created_at:%Y-%m-%d %H:%M} — {self.identity_predicted} ({status})"
