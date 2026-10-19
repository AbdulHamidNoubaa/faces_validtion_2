# Dataset Guide / دليل البيانات

## English

### Purpose
This project uses face images for **closed-set identity classification** under low-light
conditions, targeting banking security scenarios (ATM / access control).

### Recommended dataset: LFW (Labeled Faces in the Wild)
- **Source:** http://vis-www.cs.umass.edu/lfw/
- **TensorFlow Datasets / sklearn:** The pipeline can download a filtered LFW subset via
  `sklearn.datasets.fetch_lfw_people` (research-friendly, no manual zip required).
- **Filter:** Identities with at least `min_images_per_identity` images (default: 10),
  capped at `max_identities` (default: 50) for feasible university experiments.

### Low-light data (honest methodology note)
Dedicated real low-light face datasets for banking cameras are scarce and often
restricted. For academic research this project supports:

1. **Synthetic low-light simulation** (`src/data_preparation/low_light_simulation.py`):
   brightness reduction, gamma darkening, and mild Gaussian noise applied to a
   configurable fraction of images.
2. **Optional real low-light images:** Place them under `data/raw/custom/` following
   the folder layout below and set `dataset.name: custom` in `config/config.yaml`.

**Always state in the thesis methodology section** whether results used synthetic
low-light, real low-light, or both. Do not present synthetic results as if they were
collected from bank ATMs.

### Folder layout (custom / after processing)

```
data/raw/
  lfw/                    # optional manual LFW tree
  custom/
    Person_A/
      img001.jpg
    Person_B/
      img001.jpg
data/processed/
  images/                 # aligned, enhanced, 224x224 faces
  label_map.json
  splits.json
```

### How to prepare data

```bash
# From project root
python -m src.data_preparation.download_or_load_dataset
python -m src.data_preparation.build_dataset_splits
```

---

## العربية

### الغرض
يستخدم المشروع صور وجوه لمهمة **تصنيف هوية مغلق (Closed-set)** تحت إضاءة منخفضة،
محاكياً سيناريوهات الأمن المصرفي (صراف آلي / أنظمة دخول).

### مجموعة البيانات الموصى بها: LFW
- المصدر: http://vis-www.cs.umass.edu/lfw/
- يمكن التحميل عبر `sklearn.datasets.fetch_lfw_people` تلقائياً.
- تُفلتر الهويات التي لديها عدداً كافياً من الصور (افتراضي ≥ 10)، مع حد أقصى للهويات
  لتسهيل التجارب الجامعية.

### بيانات الإضاءة المنخفضة (ملاحظة منهجية صادقة)
مجموعات بيانات الإضاءة المنخفضة الحقيقية الخاصة بوجوه الكاميرات المصرفية محدودة.
لذلك يدعم المشروع:

1. **محاكاة اصطناعية** لخفض السطوع وGamma والضجيج.
2. **بيانات حقيقية اختيارية** عبر مجلد `data/raw/custom/`.

يجب ذكر مصدر البيانات بدقة في قسم منهجية البحث الأكاديمي، وعدم تقديم نتائج المحاكاة
كأنها قياسات ميدانية من أجهزة مصرفية.
