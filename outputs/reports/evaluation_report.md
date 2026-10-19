# Evaluation Report / تقرير التقييم

> **Honesty note / ملاحظة صدق:** لا تُدرج هنا أي أرقام أداء ملفّقة.
> تُملأ المقاييس تلقائياً بعد تشغيل التقييم.

## Methodology summary / ملخص المنهجية

| Item | Choice |
|------|--------|
| Task | Closed-set identity classification |
| Backbone | MobileNetV2 (ImageNet transfer learning) |
| Dataset | LFW subset (30 identities, ≥10 images each) |
| Low-light | Synthetic simulation (~50%) + CLAHE/Gamma enhancement |
| Baselines | MobileNetV1, ResNet50 (via `compare_models`) |

## Limitation / قيد مهم

The metrics currently recorded for MobileNetV2 come from a **smoke training run**
(`config/config.smoke.yaml`: 2 head epochs + 2 fine-tune epochs) used only to verify
the end-to-end pipeline on this machine (Windows CPU, Python 3.13).

**Do not use smoke metrics as final thesis results.** Re-train with:

```bash
python -m src.training.train --config config/config.yaml --arch mobilenetv2
python -m src.evaluation.evaluate --arch mobilenetv2
python -m src.evaluation.compare_models
```

Pipeline verification (smoke) observed roughly: Accuracy ≈ 0.32, F1-macro ≈ 0.12,
AUC-micro ≈ 0.80, inference ≈ 144 ms/image, model ≈ 24.7 MB / ~2.6M params —
these numbers are **pipeline-check only**, not claimed research performance.
