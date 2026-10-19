"""Django settings — banking face recognition web interface."""

from __future__ import annotations

import os
import sys
from pathlib import Path

WEB_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = WEB_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import load_config, resolve_path  # noqa: E402

_cfg = load_config()
_demo = _cfg.get("demo", {})
_web = _cfg.get("web", {})
_db = _web.get("database", {})

SECRET_KEY = os.environ.get(
    "DJANGO_SECRET_KEY",
    "dev-only-change-in-production-bfr-2026",
)
DEBUG = os.environ.get("DJANGO_DEBUG", "1") == "1"
ALLOWED_HOSTS = [
    h.strip()
    for h in os.environ.get("DJANGO_ALLOWED_HOSTS", "127.0.0.1,localhost").split(",")
    if h.strip()
]

BASE_DIR = WEB_DIR

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "web.accounts",
    "web.checkpoint",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "web.config.urls"
WSGI_APPLICATION = "web.config.wsgi.application"

_db_path = _db.get("path", "web/db.sqlite3")
_enrollment = _cfg.get("enrollment", {})
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": resolve_path(_db_path),
        "OPTIONS": {
            "timeout": int(_db.get("timeout_seconds", 20)),
        },
    }
}

LANGUAGE_CODE = "ar"
TIME_ZONE = "Africa/Tripoli"
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"
STATICFILES_DIRS = [BASE_DIR / "static"]
STATIC_ROOT = BASE_DIR / "staticfiles"

MEDIA_URL = "/media/"
MEDIA_ROOT = resolve_path(_enrollment.get("raw_dir", "data/enrollment/raw"))

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.template.context_processors.static",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
                "web.checkpoint.context_processors.site_settings",
            ],
        },
    },
]

# Max upload ~8MB for camera captures
DATA_UPLOAD_MAX_MEMORY_SIZE = 8 * 1024 * 1024
FILE_UPLOAD_MAX_MEMORY_SIZE = 8 * 1024 * 1024

# Exposed to templates via context processor + views
BFR_DEMO_TITLE_AR = _demo.get("title_ar", "نقطة التحقق المصرفية")
BFR_DEMO_TITLE_EN = _demo.get("title_en", "Banking Checkpoint")
BFR_DEFAULT_THRESHOLD = float(_demo.get("default_threshold", 0.5))
BFR_ARABIC_FONT = _web.get("arabic_font", "tajawal")  # tajawal | cairo
BFR_MODELS = _web.get(
    "models",
    ["mobilenetv2", "mobilenetv1", "resnet50", "vgg16"],
)

LOGIN_URL = "accounts:login"
LOGIN_REDIRECT_URL = "accounts:dashboard"
LOGOUT_REDIRECT_URL = "accounts:login"

SESSION_COOKIE_AGE = 60 * 60 * 8
SESSION_SAVE_EVERY_REQUEST = True
CSRF_COOKIE_HTTPONLY = False  # Required for camera JS to send CSRF token

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator", "OPTIONS": {"min_length": 8}},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

if not DEBUG:
    SESSION_COOKIE_SECURE = True
    CSRF_COOKIE_SECURE = True
    SECURE_BROWSER_XSS_FILTER = True
    SECURE_CONTENT_TYPE_NOSNIFF = True
    X_FRAME_OPTIONS = "DENY"

