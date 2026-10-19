# Banking Face Recognition under Low Light / تحسين التعرف على الوجوه تحت الإضاءة المنخفضة

**Title (AR):** تحسين التعرف على الوجوه في أنظمة الأمن المصرفي تحت الإضاءة المنخفضة باستخدام MobileNetV2  
**Stack:** Python 3.10+, TensorFlow/Keras, OpenCV, MediaPipe, scikit-learn, **Django**

---

## English

### Overview
End-to-end research pipeline for **closed-set face recognition** in banking security
scenarios (ATM / access control), optimized for **low-light** conditions using
**MobileNetV2 + Transfer Learning**, classical enhancement (**CLAHE + Gamma**), and
optional **Retinex**.

### Design decisions
| Decision | Choice | Rationale |
|----------|--------|-----------|
| Task | Closed-set classification | Matches enrolled customer/staff lists; clear Accuracy/F1/AUC/CM |
| Dataset | LFW subset (sklearn) | Standard academic faces; easy download |
| Low-light | Synthetic + enhancement | Real bank low-light corpora are scarce — documented honestly in `data/README_DATA.md` |
| Detector | MediaPipe (+ Haar fallback) | Lightweight for demo / edge-like inference |
| Baselines | MobileNetV1, ResNet50 | Speed vs accuracy academic comparison |

### Setup

```bash
cd banking-face-recognition-lowlight
python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux/macOS
# source .venv/bin/activate

pip install -r requirements.txt
```

All hyperparameters and paths live in `config/config.yaml` (no hard-coded absolute paths).

### Pipeline commands (one per stage)

```bash
# 1) Download/load LFW + build train/val/test with optional low-light simulation
python -m src.data_preparation.download_or_load_dataset
python -m src.data_preparation.build_dataset_splits

# 2) Train MobileNetV2 (head then fine-tune) — full academic settings
python -m src.training.train --arch mobilenetv2

# Optional quick smoke train (2+2 epochs) to verify the pipeline
python -m src.training.train --config config/config.smoke.yaml --arch mobilenetv2

# 3) Evaluate
python -m src.evaluation.evaluate --arch mobilenetv2
python -m src.evaluation.inference_benchmark --arch mobilenetv2

# 4) Compare vs baselines (trains missing models — can take a long time on CPU)
python -m src.evaluation.compare_models

# 5) Banking checkpoint web app (Django — RTL, Tajawal/Cairo, SQLite3)
python web/manage.py makemigrations checkpoint
python web/manage.py migrate
python web/manage.py runserver

# Unit tests
pytest tests/ -q
```

Change Arabic UI font in `config/config.yaml` → `web.arabic_font`: `tajawal` or `cairo`.

**Legacy (optional):** `streamlit run app/streamlit_demo.py` — deprecated; use Django instead.

**Note:** On Windows + Python 3.13, TensorFlow runs on CPU only (no native GPU). Prefer WSL2 for GPU. Use `config/config.smoke.yaml` only for pipeline checks — thesis tables should use `config/config.yaml` with a full training run.

### Project structure
See the repository tree: `config/`, `data/`, `src/` (data_preparation, preprocessing,
models, training, evaluation, inference, utils), `web/` (Django UI), `outputs/`, `tests/`.

### Outputs
- Models: `outputs/models/*.keras`
- Curves / CM / ROC: `outputs/reports/`
- Final report: `outputs/reports/evaluation_report.md`
- Comparison CSV: `outputs/reports/model_comparison.csv`

### Academic honesty
Do **not** invent metrics. If a full train run was not executed on your machine,
state that as a **limitation**. Synthetic low-light ≠ real ATM footage — cite
`data/README_DATA.md` in the methodology section.

---

## العربية

### نظرة عامة
نظام كامل للتعرف على الوجوه (تصنيف مغلق) لأنظمة الأمن المصرفي تحت الإضاءة المنخفضة،
باستخدام MobileNetV2 مع Transfer Learning، وتحسين الإضاءة بـ CLAHE + Gamma
(وRetinex اختيارياً)، و**واجهة Django** احترافية (RTL، خط Tajawal أو Cairo) تحاكي نقطة تحقق مصرفية.

### التشغيل السريع

```bash
cd banking-face-recognition-lowlight
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

python -m src.data_preparation.download_or_load_dataset
python -m src.data_preparation.build_dataset_splits
python -m src.training.train --arch mobilenetv2
python -m src.evaluation.evaluate --arch mobilenetv2
python -m src.evaluation.compare_models
python web/manage.py migrate
python web/manage.py bootstrap_system --force
python web/manage.py runserver
```

افتح `http://127.0.0.1:8000/` — تسجيل الدخول: **admin** / **Admin@12345** (غيّرها في الإنتاج).

**نظام Auth متكامل:** أدوار (مدير / مشغّل / محلّل / قراءة)، إضافة مستخدمين من `/accounts/users/`، استعراض بيانات LFW من `/checkpoint/data/`.

قاعدة البيانات: **SQLite3** في `web/db.sqlite3`.  
سجل التحقق: `/checkpoint/history/` — لوحة Django: `/admin/`.

لتغيير الخط العربي: في `config/config.yaml` اضبط `web.arabic_font` إلى `tajawal` أو `cairo`.

للتأكد السريع من عمل الأنبوب فقط (بدون نتائج بحث نهائية):

```bash
python -m src.training.train --config config/config.smoke.yaml --arch mobilenetv2
```

### الإعدادات
عدّل كل المعاملات من `config/config.yaml` فقط (حجم الدفعة، عدد الـ epochs،
طريقة تحسين الإضاءة، عتبة القبول في الواجهة، إلخ).

### ملاحظة منهجية
المحاكاة الاصطناعية للإضاءة المنخفضة مخصّصة لأغراض البحث الأكاديمي عند ندرة
البيانات الحقيقية — يجب ذكر ذلك صراحة في تقرير التخرج.
