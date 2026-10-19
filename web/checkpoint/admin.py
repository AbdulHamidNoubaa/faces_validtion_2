from django.contrib import admin

from web.checkpoint.models import BankEmployee, EmployeePhoto, VerificationLog


class EmployeePhotoInline(admin.TabularInline):
    model = EmployeePhoto
    extra = 0
    readonly_fields = ("status", "processed_relpath", "face_detected", "processed_at")


@admin.register(BankEmployee)
class BankEmployeeAdmin(admin.ModelAdmin):
    list_display = ("employee_code", "full_name_ar", "branch", "is_active", "created_at")
    list_filter = ("is_active", "branch")
    search_fields = ("employee_code", "full_name_ar")
    inlines = [EmployeePhotoInline]


@admin.register(EmployeePhoto)
class EmployeePhotoAdmin(admin.ModelAdmin):
    list_display = ("employee", "status", "apply_low_light", "face_detected", "uploaded_at")
    list_filter = ("status", "apply_low_light")


@admin.register(VerificationLog)
class VerificationLogAdmin(admin.ModelAdmin):
    list_display = (
        "created_at",
        "performed_by",
        "matched_employee",
        "identity_predicted",
        "match_source",
        "accepted",
        "confidence",
        "architecture",
        "source",
        "latency_ms",
    )
    list_filter = ("accepted", "architecture", "source", "face_detected", "match_source")
    search_fields = ("identity_predicted",)
    readonly_fields = (
        "created_at",
        "source",
        "architecture",
        "threshold",
        "brightness",
        "identity_predicted",
        "confidence",
        "accepted",
        "latency_ms",
        "face_detected",
        "client_ip",
        "user_agent",
        "match_source",
    )
    date_hierarchy = "created_at"
