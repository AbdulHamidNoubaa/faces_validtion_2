from django.core.management.base import BaseCommand
from django.contrib.auth.models import User

from web.accounts.models import UserProfile


class Command(BaseCommand):
    help = "Create default admin user and sample roles if database is empty."

    def add_arguments(self, parser):
        parser.add_argument("--username", default="admin")
        parser.add_argument("--password", default="Admin@12345")
        parser.add_argument("--force", action="store_true", help="Reset admin password if user exists")

    def handle(self, *args, **options):
        username = options["username"]
        password = options["password"]
        user, created = User.objects.get_or_create(
            username=username,
            defaults={"email": "admin@bank.local", "is_staff": True, "is_superuser": True},
        )
        if created or options["force"]:
            user.set_password(password)
            user.is_active = True
            user.is_staff = True
            user.is_superuser = True
            user.save()
            self.stdout.write(self.style.SUCCESS(f"Admin ready: {username} / (password you set)"))
        else:
            self.stdout.write(f"User '{username}' already exists (use --force to reset password).")

        UserProfile.objects.update_or_create(
            user=user,
            defaults={"role": UserProfile.Role.ADMIN, "full_name_ar": "مدير النظام"},
        )

        demo_users = [
            ("operator1", UserProfile.Role.OPERATOR, "مشغّل الصراف"),
            ("analyst1", UserProfile.Role.ANALYST, "محلّل بيانات"),
            ("viewer1", UserProfile.Role.VIEWER, "مراجع"),
        ]
        for uname, role, name_ar in demo_users:
            u, c = User.objects.get_or_create(username=uname, defaults={"is_active": True})
            if c:
                u.set_password("Demo@12345")
                u.save()
            UserProfile.objects.update_or_create(
                user=u, defaults={"role": role, "full_name_ar": name_ar}
            )
        self.stdout.write("Demo users (password Demo@12345): operator1, analyst1, viewer1")
