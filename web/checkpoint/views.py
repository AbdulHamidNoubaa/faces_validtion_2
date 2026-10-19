"""Checkpoint, dataset browser, and verification history."""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path

from django.http import FileResponse, Http404, JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_http_methods

from src.inference.pipeline import rgb_to_jpeg_bytes, verify_image_bytes
from src.utils.config_loader import project_root
from web.accounts.models import UserProfile
from web.accounts.permissions import can_view_all_logs, role_required
from web.checkpoint.forms import VerifyForm
from web.checkpoint.models import VerificationLog
from web.checkpoint.services.dataset_catalog import (
    get_dataset_summary,
    list_identities,
    samples_for_identity,
)
from web.checkpoint.services.verification_log import log_verification
from web.checkpoint.services.verify_helpers import apply_verify_mode_from_form, enrollment_stats


def _result_context(result, form: VerifyForm) -> dict:
    preview_b64 = None
    if result.preprocessed_rgb is not None:
        jpeg = rgb_to_jpeg_bytes(result.preprocessed_rgb)
        preview_b64 = base64.b64encode(jpeg).decode("ascii")
    return {
        "form": form,
        "has_result": True,
        "result": result,
        "preview_b64": preview_b64,
        "confidence_pct": round(result.confidence * 100, 2),
    }


@role_required(UserProfile.Role.ADMIN, UserProfile.Role.OPERATOR)
@require_http_methods(["GET", "POST"])
def checkpoint_home(request):
    form = VerifyForm(request.POST or None, request.FILES or None)
    enroll = enrollment_stats()
    context = {
        "form": form,
        "has_result": False,
        "enrollment": enroll,
    }

    if request.method == "POST" and form.is_valid():
        uploaded = form.cleaned_data.get("image")
        if not uploaded:
            form.add_error("image", "يرجى رفع صورة أو التقاطها من الكاميرا.")
        else:
            try:
                result = verify_image_bytes(
                    uploaded.read(),
                    architecture=form.cleaned_data["architecture"],
                    threshold=form.cleaned_data["threshold"],
                    brightness=form.cleaned_data["brightness"],
                )
                apply_verify_mode_from_form(result, form)
                log_verification(result, request, source=VerificationLog.Source.UPLOAD)
                context = _result_context(result, form)
                context["enrollment"] = enroll
            except FileNotFoundError as exc:
                form.add_error(None, str(exc))
            except ValueError as exc:
                form.add_error(None, str(exc))

    return render(request, "checkpoint/home.html", context)


@role_required(
    UserProfile.Role.ADMIN,
    UserProfile.Role.OPERATOR,
    UserProfile.Role.ANALYST,
)
@require_http_methods(["GET"])
def verification_history(request):
    qs = VerificationLog.objects.select_related("performed_by").all()
    if not can_view_all_logs(request.user):
        qs = qs.filter(performed_by=request.user)
    total_count = qs.count()
    logs = qs[:200]
    return render(
        request,
        "checkpoint/history.html",
        {"logs": logs, "total_count": total_count},
    )


@role_required(UserProfile.Role.ADMIN, UserProfile.Role.OPERATOR)
@require_http_methods(["POST"])
def checkpoint_api_verify(request):
    form = VerifyForm(request.POST, request.FILES)
    if not form.is_valid():
        return JsonResponse({"ok": False, "errors": form.errors}, status=400)
    uploaded = form.cleaned_data.get("image")
    if not uploaded:
        return JsonResponse({"ok": False, "error": "missing image"}, status=400)
    try:
        result = verify_image_bytes(
            uploaded.read(),
            architecture=form.cleaned_data["architecture"],
            threshold=form.cleaned_data["threshold"],
            brightness=form.cleaned_data["brightness"],
        )
        apply_verify_mode_from_form(result, form)
        log_verification(result, request, source=VerificationLog.Source.CAMERA)
        preview_b64 = None
        if result.preprocessed_rgb is not None:
            preview_b64 = base64.b64encode(
                rgb_to_jpeg_bytes(result.preprocessed_rgb)
            ).decode("ascii")
        return JsonResponse(
            {
                "ok": True,
                "identity": result.identity,
                "confidence": result.confidence,
                "confidence_pct": round(result.confidence * 100, 2),
                "accepted": result.accepted,
                "decision_ar": result.decision_ar,
                "decision_en": result.decision_en,
                "latency_ms": round(result.latency_ms, 1),
                "face_detected": result.face_detected,
                "preview_b64": preview_b64,
            }
        )
    except (FileNotFoundError, ValueError) as exc:
        return JsonResponse({"ok": False, "error": str(exc)}, status=400)
    except Exception:
        return JsonResponse({"ok": False, "error": "خطأ داخلي في الخادم"}, status=500)


@role_required(
    UserProfile.Role.ADMIN,
    UserProfile.Role.ANALYST,
    UserProfile.Role.VIEWER,
)
def dataset_overview(request):
    summary = get_dataset_summary()
    identities = list_identities(limit=50) if summary.get("ready") else []
    return render(
        request,
        "checkpoint/dataset_overview.html",
        {"summary": summary, "identities_preview": identities},
    )


@role_required(
    UserProfile.Role.ADMIN,
    UserProfile.Role.ANALYST,
    UserProfile.Role.VIEWER,
)
def dataset_identities(request):
    return render(
        request,
        "checkpoint/dataset_identities.html",
        {"identities": list_identities(limit=500)},
    )


@role_required(
    UserProfile.Role.ADMIN,
    UserProfile.Role.ANALYST,
    UserProfile.Role.VIEWER,
)
def dataset_identity_detail(request, identity: str):
    samples = {
        "train": samples_for_identity(identity, "train", 12),
        "val": samples_for_identity(identity, "val", 8),
        "test": samples_for_identity(identity, "test", 8),
    }
    return render(
        request,
        "checkpoint/dataset_identity.html",
        {"identity": identity, "samples": samples},
    )


@role_required(
    UserProfile.Role.ADMIN,
    UserProfile.Role.ANALYST,
    UserProfile.Role.VIEWER,
)
def dataset_image(request, relpath: str):
    safe = Path(relpath.replace("\\", "/"))
    if ".." in safe.parts:
        raise Http404
    full = (project_root() / safe).resolve()
    root = project_root().resolve()
    if not str(full).startswith(str(root)) or not full.is_file():
        raise Http404
    if "processed" not in full.parts and "employees" not in full.parts:
        raise Http404
    ctype = mimetypes.guess_type(str(full))[0] or "application/octet-stream"
    return FileResponse(open(full, "rb"), content_type=ctype)
