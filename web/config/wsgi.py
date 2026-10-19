"""WSGI config for banking face recognition web app."""

import os
import sys
from pathlib import Path

WEB_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = WEB_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "web.config.settings")

from django.core.wsgi import get_wsgi_application

application = get_wsgi_application()
