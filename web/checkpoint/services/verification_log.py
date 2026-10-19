"""Persist verification results to SQLite."""

from __future__ import annotations

from django.http import HttpRequest

from src.inference.pipeline import VerificationResult
from web.checkpoint.models import BankEmployee, VerificationLog


def _client_ip(request: HttpRequest | None) -> str | None:
    if request is None:
        return None
    forwarded = request.META.get("HTTP_X_FORWARDED_FOR")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.META.get("REMOTE_ADDR")


def log_verification(
    result: VerificationResult,
    request: HttpRequest | None = None,
    source: str = VerificationLog.Source.UPLOAD,
) -> VerificationLog:
    ua = ""
    if request is not None:
        ua = (request.META.get("HTTP_USER_AGENT") or "")[:255]
    user = None
    if request is not None and getattr(request, "user", None) and request.user.is_authenticated:
        user = request.user
    matched = None
    if result.employee_code:
        matched = BankEmployee.objects.filter(employee_code=result.employee_code).first()
    return VerificationLog.objects.create(
        source=source,
        architecture=result.architecture,
        threshold=result.threshold,
        brightness=result.brightness,
        identity_predicted=result.identity,
        confidence=result.confidence,
        accepted=result.accepted,
        latency_ms=result.latency_ms,
        face_detected=result.face_detected,
        matched_employee=matched,
        match_source=result.match_source or "classifier",
        performed_by=user,
        client_ip=_client_ip(request),
        user_agent=ua,
    )
