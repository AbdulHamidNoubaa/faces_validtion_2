from django.contrib import admin

from web.accounts.models import UserProfile


@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ("user", "role", "full_name_ar", "branch", "employee_id")
    list_filter = ("role",)
    search_fields = ("user__username", "full_name_ar", "employee_id")
