"""Helpers for verification views."""

from web.checkpoint.models import BankEmployee, EmployeePhoto
from web.checkpoint.services.employee_matching import apply_verification_mode


def apply_verify_mode_from_form(result, form) -> None:
    mode = form.cleaned_data.get("verify_mode", "employees_only")
    apply_verification_mode(result, mode)


def enrollment_stats() -> dict:
    employees = BankEmployee.objects.filter(is_active=True).count()
    photos = EmployeePhoto.objects.filter(
        status=EmployeePhoto.Status.PROCESSED,
        employee__is_active=True,
    ).count()
    with_embedding = EmployeePhoto.objects.filter(
        status=EmployeePhoto.Status.PROCESSED,
        employee__is_active=True,
    ).exclude(embedding_json="").count()
    return {
        "active_employees": employees,
        "processed_photos": photos,
        "gallery_ready": with_embedding,
        "can_verify_employees": with_embedding > 0,
    }
