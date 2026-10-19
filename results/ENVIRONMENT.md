# بيئة التشغيل

القيم أدناه قُرئت من الجهاز أثناء تشغيل `experiments/run_all.py`.

- نظام التشغيل: `Windows-11-10.0.26200-SP0`
- المعالج (platform): `Intel64 Family 6 Model 198 Stepping 2, GenuineIntel`
- عدد المعالجات المنطقية: `24`
- Python: `3.13.9 (tags/v3.13.9:8183fa5, Oct 14 2025, 14:09:13) [MSC v.1944 64 bit (AMD64)]`
- RAM (GiB): `63.46`

## المعالج والذاكرة وبطاقة العرض كما أعادها Windows CIM

```json
{
  "cpu_cim": {
    "Name": "Intel(R) Core(TM) Ultra 9 275HX",
    "NumberOfCores": 24,
    "NumberOfLogicalProcessors": 24,
    "MaxClockSpeed": 2700
  },
  "memory_cim": {
    "TotalPhysicalMemory": 68137205760,
    "Manufacturer": "Alienware",
    "Model": "Alienware 18 Area-51 AA18250"
  },
  "gpu_cim": [
    {
      "Name": "NVIDIA GeForce RTX 5090 Laptop GPU",
      "AdapterRAM": 4293918720,
      "DriverVersion": "32.0.15.9202"
    },
    {
      "Name": "Intel(R) Graphics",
      "AdapterRAM": 2147479552,
      "DriverVersion": "32.0.101.8860"
    }
  ]
}
```

## المكتبات

| الحزمة | الإصدار |
|---|---|
| tensorflow | 2.21.0 |
| opencv-python | 4.13.0.92 |
| Django | 5.2.16 |
| mediapipe | 0.10.35 |
| scikit-learn | 1.9.0 |
| scikit-image | 0.26.0 |
| numpy | 2.5.1 |
| Pillow | 12.3.0 |
| pandas | 3.0.3 |
| matplotlib | 3.11.1 |
| PyYAML | 6.0.3 |
| scipy | 1.18.0 |

## TensorFlow أثناء التجربة

- الإصدار: `2.21.0`
- أجهزة GPU: `[]`
- أجهزة CPU: `['/physical_device:CPU:0']`
- التشغيل المستخدم: `CPU`
- مبني مع CUDA: `False`
