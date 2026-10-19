#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys
from pathlib import Path


def main():
    web_dir = Path(__file__).resolve().parent
    project_root = web_dir.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "web.config.settings")
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Install it with: pip install Django"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    main()
