"""Employee enrollment forms."""

from django import forms

from web.checkpoint.models import BankEmployee


class BankEmployeeForm(forms.ModelForm):
    class Meta:
        model = BankEmployee
        fields = (
            "employee_code",
            "full_name_ar",
            "full_name_en",
            "branch",
            "department",
            "is_active",
            "notes",
        )
        labels = {
            "employee_code": "رقم الموظف",
            "full_name_ar": "الاسم (عربي)",
            "full_name_en": "الاسم (إنجليزي)",
            "branch": "الفرع",
            "department": "القسم",
            "is_active": "نشط",
            "notes": "ملاحظات",
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field in self.fields.values():
            field.widget.attrs.setdefault("class", "form-input")
