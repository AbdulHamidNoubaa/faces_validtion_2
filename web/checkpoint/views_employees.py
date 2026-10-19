"""Bank employee CRUD and photo enrollment."""

from django.contrib import messages
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_http_methods

from django.db.models import Count, Q

from web.accounts.models import UserProfile
from web.accounts.permissions import can_manage_employees, role_required
from web.checkpoint.forms_employees import BankEmployeeForm
from web.checkpoint.models import BankEmployee, EmployeePhoto
from web.checkpoint.services.employee_processing import process_employee_photo


@role_required(
    UserProfile.Role.ADMIN,
    UserProfile.Role.OPERATOR,
    UserProfile.Role.ANALYST,
    UserProfile.Role.VIEWER,
)
def employee_list(request):
    employees = (
        BankEmployee.objects.annotate(
            photo_count=Count("photos"),
            processed_count=Count(
                "photos", filter=Q(photos__status=EmployeePhoto.Status.PROCESSED)
            ),
        )
        .order_by("full_name_ar")
    )
    return render(request, "checkpoint/employees/list.html", {"employees": employees})


@role_required(UserProfile.Role.ADMIN, UserProfile.Role.OPERATOR)
@require_http_methods(["GET", "POST"])
def employee_create(request):
    form = BankEmployeeForm(request.POST or None)
    if request.method == "POST" and form.is_valid():
        emp = form.save(commit=False)
        emp.created_by = request.user
        emp.save()
        files = request.FILES.getlist("images")
        apply_ll = request.POST.get("apply_low_light") == "on"
        arch = request.POST.get("architecture", "mobilenetv2")
        n, errs = 0, 0
        for f in files:
            photo = EmployeePhoto.objects.create(
                employee=emp,
                image=f,
                apply_low_light=apply_ll,
                uploaded_by=request.user,
            )
            try:
                process_employee_photo(photo, architecture=arch)
                n += 1
            except Exception:
                errs += 1
        if errs:
            messages.warning(request, f"فشلت معالجة {errs} صورة.")
        if n:
            messages.success(request, f"تم إضافة الموظف ومعالجة {n} صورة للتحقق.")
        else:
            messages.success(request, "تم إضافة الموظف. ارفع صوراً للوجه لتفعيل التحقق.")
        return redirect("checkpoint:employee_detail", pk=emp.pk)
    return render(
        request,
        "checkpoint/employees/form.html",
        {"form": form, "title": "إضافة موظف", "show_photo_upload": True},
    )


@role_required(UserProfile.Role.ADMIN, UserProfile.Role.OPERATOR)
@require_http_methods(["GET", "POST"])
def employee_edit(request, pk: int):
    emp = get_object_or_404(BankEmployee, pk=pk)
    form = BankEmployeeForm(request.POST or None, instance=emp)
    if request.method == "POST" and form.is_valid():
        form.save()
        messages.success(request, "تم تحديث بيانات الموظف.")
        return redirect("checkpoint:employee_detail", pk=emp.pk)
    return render(
        request,
        "checkpoint/employees/form.html",
        {"form": form, "title": f"تعديل: {emp.full_name_ar}", "employee": emp},
    )


@role_required(
    UserProfile.Role.ADMIN,
    UserProfile.Role.OPERATOR,
    UserProfile.Role.ANALYST,
    UserProfile.Role.VIEWER,
)
def employee_detail(request, pk: int):
    emp = get_object_or_404(BankEmployee.objects.prefetch_related("photos"), pk=pk)
    can_upload = can_manage_employees(request.user)
    return render(
        request,
        "checkpoint/employees/detail.html",
        {
            "employee": emp,
            "can_upload": can_upload,
        },
    )


@role_required(UserProfile.Role.ADMIN, UserProfile.Role.OPERATOR)
@require_http_methods(["POST"])
def employee_upload_photos(request, pk: int):
    emp = get_object_or_404(BankEmployee, pk=pk)
    files = request.FILES.getlist("images")
    if not files:
        messages.error(request, "اختر ملفاً واحداً على الأقل.")
        return redirect("checkpoint:employee_detail", pk=pk)

    apply_ll = request.POST.get("apply_low_light") == "on"
    arch = request.POST.get("architecture", "mobilenetv2")

    ok, errs = 0, 0
    for f in files:
        photo = EmployeePhoto.objects.create(
            employee=emp,
            image=f,
            apply_low_light=apply_ll,
            uploaded_by=request.user,
        )
        try:
            process_employee_photo(photo, architecture=arch)
            ok += 1
        except Exception:
            errs += 1

    if errs:
        messages.warning(request, f"فشلت معالجة {errs} صورة.")
    messages.success(request, f"تم رفع ومعالجة {ok} صورة (نفس pipeline البيانات).")
    return redirect("checkpoint:employee_detail", pk=pk)


@role_required(UserProfile.Role.ADMIN, UserProfile.Role.OPERATOR)
@require_http_methods(["POST"])
def employee_reprocess_photo(request, photo_id: int):
    photo = get_object_or_404(EmployeePhoto, pk=photo_id)
    arch = request.POST.get("architecture", "mobilenetv2")
    process_employee_photo(photo, architecture=arch)
    messages.info(request, "أُعيدت معالجة الصورة.")
    return redirect("checkpoint:employee_detail", pk=photo.employee_id)
