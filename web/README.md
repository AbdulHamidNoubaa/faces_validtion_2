# Django Web App / تطبيق الويب المتكامل

## Structure

```
web/
  manage.py
  config/              # settings, urls, wsgi
  accounts/            # auth, roles, user CRUD
  checkpoint/          # verification + dataset browser
  templates/
  static/
```

## SQLite3

Database file (default): `web/db.sqlite3` — configured in `config/config.yaml` → `web.database.path`.

## Authentication & roles

| Role | الصلاحيات |
|------|-----------|
| **admin** | كل شيء + إدارة المستخدمين |
| **operator** | نقطة التحقق + سجل عملياته |
| **analyst** | البيانات + كل سجلات التحقق |
| **viewer** | لوحة التحكم + استعراض البيانات فقط |

## First-time setup

From project root:

```bash
pip install -r requirements.txt
python web/manage.py makemigrations
python web/manage.py migrate
python web/manage.py bootstrap_system --force
python web/manage.py runserver
```

Open `http://127.0.0.1:8000/` → login.

**Default admin** (after bootstrap): `admin` / `Admin@12345` — change in production.

**Demo users** (password `Demo@12345`): `operator1`, `analyst1`, `viewer1`.

Or create users via **المستخدمون → إضافة مستخدم** (admin only).

## Main URLs

| Path | Description |
|------|-------------|
| `/accounts/login/` | تسجيل الدخول |
| `/accounts/dashboard/` | لوحة التحكم |
| `/accounts/users/` | إدارة المستخدمين |
| `/checkpoint/verify/` | نقطة التحقق |
| `/checkpoint/history/` | سجل SQLite |
| `/checkpoint/data/` | نظرة عامة على بيانات LFW المعالجة |
| `/checkpoint/employees/` | **موظفو المصرف** — إضافة صور ومعالجة |
| `/admin/` | Django admin |

## Employee enrollment / تسجيل موظفي المصرف

1. **الموظفون → إضافة موظف** (مدير / مشغّل)
2. افتح صفحة الموظف → **رفع صور** (متعدد)
3. تُطبَّق نفس معالجة LFW: كشف وجه → CLAHE/Gamma → 224×224 (+ اختياري محاكاة إضاءة منخفضة)
4. تُحفظ النتائج في:
   - `data/processed/employees/<رقم_الموظف>/`
   - نسخة mirror في `data/raw/custom/` لإعادة التدريب (`dataset.name: custom`)
5. عند **نقطة التحقق**، إن وُجدت بصمة مطابقة لموظف مسجّل (cosine ≥ `enrollment.match_threshold` في config) يُعرض اسمه بدلاً من فئة LFW.


`config/config.yaml` → `web.arabic_font`: `tajawal` or `cairo`.
