"""Auth forms — login and user provisioning."""

from django import forms
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm
from django.contrib.auth.models import User

from web.accounts.models import UserProfile


class ArabicAuthenticationForm(AuthenticationForm):
    username = forms.CharField(
        label="اسم المستخدم",
        widget=forms.TextInput(attrs={"class": "form-input", "autocomplete": "username"}),
    )
    password = forms.CharField(
        label="كلمة المرور",
        widget=forms.PasswordInput(
            attrs={"class": "form-input", "autocomplete": "current-password"}
        ),
    )


class UserCreateForm(UserCreationForm):
    full_name_ar = forms.CharField(label="الاسم الكامل", max_length=120, required=False)
    role = forms.ChoiceField(label="الدور", choices=UserProfile.Role.choices)
    employee_id = forms.CharField(label="رقم الموظف", max_length=64, required=False)
    branch = forms.CharField(label="الفرع", max_length=120, required=False)
    email = forms.EmailField(label="البريد", required=False)
    is_active = forms.BooleanField(label="نشط", required=False, initial=True)

    class Meta:
        model = User
        fields = ("username", "email", "is_active")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for name in ("username", "password1", "password2", "email"):
            if name in self.fields:
                self.fields[name].widget.attrs["class"] = "form-input"

    def save(self, commit=True):
        user = super().save(commit=False)
        user.email = self.cleaned_data.get("email", "")
        user.is_active = self.cleaned_data.get("is_active", True)
        if commit:
            user.save()
            UserProfile.objects.update_or_create(
                user=user,
                defaults={
                    "role": self.cleaned_data["role"],
                    "full_name_ar": self.cleaned_data.get("full_name_ar", ""),
                    "employee_id": self.cleaned_data.get("employee_id", ""),
                    "branch": self.cleaned_data.get("branch", ""),
                },
            )
        return user


class UserUpdateForm(forms.ModelForm):
    full_name_ar = forms.CharField(label="الاسم الكامل", max_length=120, required=False)
    role = forms.ChoiceField(label="الدور", choices=UserProfile.Role.choices)
    employee_id = forms.CharField(label="رقم الموظف", max_length=64, required=False)
    branch = forms.CharField(label="الفرع", max_length=120, required=False)
    email = forms.EmailField(label="البريد", required=False)
    is_active = forms.BooleanField(label="نشط", required=False)

    class Meta:
        model = User
        fields = ("username", "email", "is_active")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["is_active"].initial = self.instance.is_active
        profile = getattr(self.instance, "profile", None)
        if profile:
            self.fields["full_name_ar"].initial = profile.full_name_ar
            self.fields["role"].initial = profile.role
            self.fields["employee_id"].initial = profile.employee_id
            self.fields["branch"].initial = profile.branch
        for field in self.fields.values():
            if hasattr(field.widget, "attrs"):
                field.widget.attrs.setdefault("class", "form-input")

    def save(self, commit=True):
        user = super().save(commit=commit)
        UserProfile.objects.update_or_create(
            user=user,
            defaults={
                "role": self.cleaned_data["role"],
                "full_name_ar": self.cleaned_data.get("full_name_ar", ""),
                "employee_id": self.cleaned_data.get("employee_id", ""),
                "branch": self.cleaned_data.get("branch", ""),
            },
        )
        return user
