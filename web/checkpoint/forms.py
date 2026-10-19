"""Django forms for verification checkpoint."""

from django import forms
from django.conf import settings


class VerifyForm(forms.Form):
    image = forms.ImageField(
        label="صورة الوجه",
        required=False,
        help_text="JPG أو PNG — صورة واضحة للوجه",
    )
    architecture = forms.ChoiceField(
        label="نموذج التعرف",
        choices=[],
        widget=forms.Select(attrs={"class": "form-select"}),
    )
    threshold = forms.FloatField(
        widget=forms.HiddenInput(),
        min_value=0.05,
        max_value=0.99,
        initial=settings.BFR_DEFAULT_THRESHOLD,
    )
    brightness = forms.FloatField(
        widget=forms.HiddenInput(),
        min_value=0.1,
        max_value=1.0,
        initial=1.0,
    )
    verify_mode = forms.ChoiceField(
        label="نوع التحقق",
        choices=[
            ("employees_only", "موظفو المصرف فقط (موصى به)"),
            ("employees_first", "موظفون أولاً ثم LFW"),
            ("lfw_only", "تصنيف LFW فقط"),
        ],
        initial="employees_only",
        widget=forms.Select(attrs={"class": "form-select"}),
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        models = settings.BFR_MODELS
        self.fields["architecture"].choices = [(m, m.upper()) for m in models]
        if models:
            self.fields["architecture"].initial = models[0]
