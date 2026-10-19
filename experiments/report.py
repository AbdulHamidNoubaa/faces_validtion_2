"""Write Markdown tables and figures from measured results only."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.common import (
    CACHE,
    FIGURES,
    RESULTS,
    THRESHOLD,
    fmt_float,
    fmt_ratio,
    package_versions,
)

EXP_ORDER = ("E1", "E2", "E3", "E3b")
MASK_ORDER = ("M_NL", "M_LL", "M_LL_ENH")


def _setup_font() -> None:
    font_path = Path(r"C:\Windows\Fonts\tahoma.ttf")
    if font_path.exists():
        from matplotlib import font_manager

        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = "Tahoma"
    plt.rcParams["axes.unicode_minus"] = False


def _metric_row(name: str, metrics: dict) -> str:
    return (
        f"| {name} | {metrics['n']} | {metrics['n_genuine']} | {metrics['n_impostor']} | "
        f"{metrics['tp']} | {metrics['tn']} | {metrics['fp']} | {metrics['fn']} | {metrics['n_misidentified']} | "
        f"{fmt_ratio(metrics['accuracy'])} | {fmt_ratio(metrics['precision'])} | {fmt_ratio(metrics['recall'])} | "
        f"{fmt_ratio(metrics['f1'])} | {fmt_ratio(metrics['far'])} | {fmt_ratio(metrics['frr'])} | {fmt_float(metrics['auc'], 6)} |"
    )


def _header() -> str:
    return (
        "| التجربة | N | Genuine | Impostor | TP | TN | FP | FN | خطأ تعرّف | "
        "Accuracy | Precision | Recall | F1 | FAR | FRR | AUC |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )


def _source(info: dict, key: str) -> str:
    src = (info.get("sources") or {}).get(key)
    if not src:
        return "غير موجود في الملف"
    return f"`{src['file']}` سطر {src['line']}: `{src['text']}`"


def write_environment(payload: dict) -> None:
    env = payload.get("environment", {})
    devices = payload.get("devices", {})
    lines = [
        "# بيئة التشغيل",
        "",
        "القيم أدناه قُرئت من الجهاز أثناء تشغيل `experiments/run_all.py`.",
        "",
        f"- نظام التشغيل: `{env.get('platform')}`",
        f"- المعالج (platform): `{env.get('processor')}`",
        f"- عدد المعالجات المنطقية: `{env.get('cpu_count')}`",
        f"- Python: `{env.get('python')}`",
        f"- RAM (GiB): `{fmt_float(env.get('ram_gib'), 2)}`",
        "",
        "## المعالج والذاكرة وبطاقة العرض كما أعادها Windows CIM",
        "",
        "```json",
    ]
    import json

    lines.append(json.dumps(
        {"cpu_cim": env.get("cpu_cim"), "memory_cim": env.get("memory_cim"), "gpu_cim": env.get("gpu_cim")},
        ensure_ascii=False,
        indent=2,
    ))
    lines += [
        "```",
        "",
        "## المكتبات",
        "",
        "| الحزمة | الإصدار |",
        "|---|---|",
    ]
    for name, version in (env.get("packages") or {}).items():
        lines.append(f"| {name} | {version} |")
    lines += [
        "",
        "## TensorFlow أثناء التجربة",
        "",
        f"- الإصدار: `{devices.get('tensorflow')}`",
        f"- أجهزة GPU: `{devices.get('gpus')}`",
        f"- أجهزة CPU: `{devices.get('cpus')}`",
        f"- التشغيل المستخدم: `{devices.get('runtime')}`",
        f"- مبني مع CUDA: `{devices.get('built_with_cuda')}`",
        "",
    ]
    (RESULTS / "ENVIRONMENT.md").write_text("\n".join(lines), encoding="utf-8")


def write_figures(payload: dict) -> list[str]:
    _setup_font()
    written = []
    sample_path = CACHE / "samples_enhancement.npz"
    if sample_path.exists():
        data = np.load(sample_path, allow_pickle=True)
        images = data["images"]
        fig, axes = plt.subplots(images.shape[0], 3, figsize=(9, 14), dpi=300)
        titles = ["NL original", "LL before CLAHE+Gamma", "LL after CLAHE+Gamma"]
        for row in range(images.shape[0]):
            for col in range(3):
                ax = axes[row, col]
                ax.imshow(cv2.cvtColor(images[row, col], cv2.COLOR_BGR2RGB))
                ax.axis("off")
                if row == 0:
                    ax.set_title(titles[col], fontsize=9)
        fig.suptitle("Registered face crops: original, low light, CLAHE+Gamma", fontsize=12)
        fig.tight_layout()
        dest = FIGURES / "samples_before_after.png"
        fig.savefig(dest, dpi=300, bbox_inches="tight")
        plt.close(fig)
        written.append(dest.name)

    roc_path = CACHE / "roc_curves.npz"
    experiments = payload.get("experiments") or {}
    if roc_path.exists() and experiments:
        data = np.load(roc_path)
        fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
        ax.plot([0, 1], [0, 1], linestyle="--", color="0.5", linewidth=1)
        for name in ("E1", "E2", "E3"):
            if f"{name}_fpr" not in data:
                continue
            auc = experiments[name]["auc"]
            label = name if auc is None else f"{name}  AUC={auc:.4f}"
            ax.plot(data[f"{name}_fpr"], data[f"{name}_tpr"], linewidth=2, label=label)
        ax.set_xlabel("False accept rate")
        ax.set_ylabel("True positive rate (genuine score >= threshold)")
        ax.set_title("ROC of max cosine, genuine vs impostor")
        ax.legend(loc="lower right")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        fig.tight_layout()
        dest = FIGURES / "roc_e1_e2_e3.png"
        fig.savefig(dest, dpi=300, bbox_inches="tight")
        plt.close(fig)
        written.append(dest.name)

    if experiments:
        fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
        names = [name for name in EXP_ORDER if name in experiments and experiments[name].get("accuracy") is not None]
        values = [experiments[name]["accuracy"] * 100.0 for name in names]
        bars = ax.bar(names, values, color=["#4C78A8", "#F58518", "#54A24B", "#B279A2"][: len(names)])
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 100)
        ax.set_title("Verification accuracy at threshold 0.52")
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
        fig.tight_layout()
        dest = FIGURES / "accuracy_bars.png"
        fig.savefig(dest, dpi=300, bbox_inches="tight")
        plt.close(fig)
        written.append(dest.name)

    mask_path = CACHE / "samples_masked.npz"
    if mask_path.exists():
        data = np.load(mask_path, allow_pickle=True)
        images = data["images"]
        fig, axes = plt.subplots(2, images.shape[0], figsize=(12, 6), dpi=300)
        for col in range(images.shape[0]):
            axes[0, col].imshow(cv2.cvtColor(images[col, 0], cv2.COLOR_BGR2RGB))
            axes[1, col].imshow(cv2.cvtColor(images[col, 1], cv2.COLOR_BGR2RGB))
            axes[0, col].axis("off")
            axes[1, col].axis("off")
            if col == 0:
                axes[0, col].set_title("aligned")
                axes[1, col].set_title("masked")
        fig.suptitle("Masked samples (top: aligned probe, bottom: mask applied)", fontsize=12)
        fig.tight_layout()
        dest = FIGURES / "masked_samples.png"
        fig.savefig(dest, dpi=300, bbox_inches="tight")
        plt.close(fig)
        written.append(dest.name)
    return written


def score_overview() -> dict:
    path = RESULTS / "scores.csv"
    if not path.exists():
        return {}
    import csv

    grouped: dict[str, list[dict]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            grouped.setdefault(row["experiment"], []).append(row)
    overview = {}
    table_rows = []
    for name, rows in grouped.items():
        genuine = [float(row["score"]) for row in rows if row["genuine"] == "True"]
        impostor = [float(row["score"]) for row in rows if row["genuine"] == "False"]
        correct = [
            float(row["score"])
            for row in rows
            if row["genuine"] == "True" and row["pred_identity"] == row["true_identity"]
        ]
        block = {
            "n": len(rows),
            "genuine_mean": float(np.mean(genuine)) if genuine else None,
            "genuine_median": float(np.median(genuine)) if genuine else None,
            "impostor_mean": float(np.mean(impostor)) if impostor else None,
            "impostor_median": float(np.median(impostor)) if impostor else None,
            "n_genuine_correct_identity": len(correct),
            "n_genuine_score_ge_0_52": int(sum(value >= THRESHOLD for value in genuine)),
            "n_impostor_score_ge_0_52": int(sum(value >= THRESHOLD for value in impostor)),
        }
        overview[name] = block
        table_rows.append({"experiment": name, **block})
    from experiments.common import write_csv

    write_csv(RESULTS / "score_distribution.csv", table_rows)
    return overview


def _delta_words(points: float) -> str:
    magnitude = abs(points)
    if magnitude <= 1:
        size = "صغير جدًا (ضمن نقطة مئوية واحدة)"
    elif magnitude <= 5:
        size = "صغير (بين 1 و 5 نقاط مئوية)"
    else:
        size = "واضح (أكثر من 5 نقاط مئوية)"
    if points > 1:
        direction = "ارتفعت الدقة"
    elif points < -1:
        direction = "انخفضت الدقة"
    else:
        direction = "لم يتغير مستوى الدقة بشكل يُذكر"
    return f"{direction}، والفرق {size}"


def ten_lines(payload: dict) -> list[str]:
    experiments = payload.get("experiments") or {}
    if not all(name in experiments for name in ("E1", "E2", "E3", "E3b")):
        return ["التجارب الأربع لم تكتمل، لذلك لا يوجد ملخص عددي. راجع قسم الخطأ إن وُجد."]
    e1, e2, e3, e3b = (experiments[name] for name in EXP_ORDER)
    d_ll = (e3["accuracy"] - e2["accuracy"]) * 100.0
    d_nl = (e3b["accuracy"] - e1["accuracy"]) * 100.0
    auc_ll = None
    if e2.get("auc") is not None and e3.get("auc") is not None:
        auc_ll = e3["auc"] - e2["auc"]
    scores = payload.get("score_overview") or {}
    e2s, e3s = scores.get("E2") or {}, scores.get("E3") or {}
    quality = payload.get("quality") or {}
    psnr_gain = (quality.get("psnr_improvement") or {}).get("mean")
    ssim_gain = (quality.get("ssim_improvement") or {}).get("mean")
    leakage = payload.get("leakage") or {}
    held = ((payload.get("replay") or {}).get("held_out") or {}).get("held_out", [])
    masked = payload.get("masked") or {}
    latency = payload.get("latency") or {}
    total_with = ((latency.get("with_enhancement") or {}).get("total_ms") or {}).get("mean_ms")
    total_without = ((latency.get("without_enhancement") or {}).get("total_ms") or {}).get("mean_ms")
    dataset = payload.get("dataset") or {}
    recounted = dataset.get("recounted_counts") or {}
    lines = [
        f"البذرة 42، والعتبة 0.52، ومجموعة الاختبار هي splits.test وفيها {recounted.get('test')} صورة (train={recounted.get('train')}، val={recounted.get('val')}). الهويات المستبعدة من المعرض: {', '.join(held)}.",
        f"E1 (NL بدون CLAHE+Gamma): Accuracy {fmt_ratio(e1['accuracy'])}، F1 {fmt_ratio(e1['f1'])}، FAR {fmt_ratio(e1['far'])}، FRR {fmt_ratio(e1['frr'])}، AUC {fmt_float(e1['auc'], 6)} على N={e1['n']} (genuine={e1['n_genuine']}، impostor={e1['n_impostor']}).",
        f"E2 (LL بدون معالجة): Accuracy {fmt_ratio(e2['accuracy'])}، F1 {fmt_ratio(e2['f1'])}، FAR {fmt_ratio(e2['far'])}، FRR {fmt_ratio(e2['frr'])}، AUC {fmt_float(e2['auc'], 6)}.",
        f"E3 (نفس صور E2 بعد CLAHE+Gamma): Accuracy {fmt_ratio(e3['accuracy'])}، F1 {fmt_ratio(e3['f1'])}، FAR {fmt_ratio(e3['far'])}، FRR {fmt_ratio(e3['frr'])}، AUC {fmt_float(e3['auc'], 6)}.",
        f"الفرق E3 − E2 = {d_ll:.2f} نقطة مئوية في Accuracy، وفرق AUC = {fmt_float(auc_ll, 6)}. {_delta_words(d_ll)}. عند 0.52 كان TP في E2 وE3 = {e2['tp']} و{e3['tp']}، أي أن التساوي ليس نجاحًا متساويًا بل بقاء الرفض شبه التام. وسيط درجة genuine هو {fmt_float(e2s.get('genuine_median'), 4)} قبل المعالجة و{fmt_float(e3s.get('genuine_median'), 4)} بعدها، وكلاهما بعيد عن 0.52. قاعدة حجم الفرق معلنة وليست اختبارًا إحصائيًا: ±1 نقطة صغير جدًا، ومن 1 إلى 5 صغير، وأكثر من 5 واضح.",
        f"على الصور السليمة، E3b − E1 = {d_nl:.2f} نقطة مئوية ({fmt_ratio(e1['accuracy'])} ثم {fmt_ratio(e3b['accuracy'])}). {_delta_words(d_nl)}.",
        (
            f"جودة الصورة على {quality.get('n_ll_images')} صورة LL مسجّلة: متوسط تحسن PSNR = {fmt_float(psnr_gain, 4)} dB، ومتوسط تحسن SSIM = {fmt_float(ssim_gain, 4)}. "
            + (
                "المقياسان ارتفعا، أي أن القصّة بعد CLAHE+Gamma أقرب إلى القصّة الأصلية من القصّة المظلمة."
                if (psnr_gain or 0) > 0 and (ssim_gain or 0) > 0
                else "انخفاض المقياس يعني أن الصورة بعد المعالجة أبعد عن الأصلية."
            )
        ),
        f"فحص التسرب: مصادر موزعة على أكثر من تقسيم = {leakage.get('n_same_source_across_splits')}، نفس المصدر NL وLL معًا = {leakage.get('n_sources_with_both_nl_and_ll')}، تكرار MD5 للصور المعالجة = {leakage.get('n_duplicate_processed_md5')}، تكرار MD5 للصور الخام = {leakage.get('n_duplicate_raw_md5')}.",
        f"الوجوه المقنعة: NL بدون معالجة {fmt_ratio((masked.get('M_NL') or {}).get('accuracy'))}، LL بدون معالجة {fmt_ratio((masked.get('M_LL') or {}).get('accuracy'))}، LL مع CLAHE+Gamma {fmt_ratio((masked.get('M_LL_ENH') or {}).get('accuracy'))}.",
        f"متوسط الزمن الكلي {fmt_float(total_with, 2)} ms مع CLAHE+Gamma و {fmt_float(total_without, 2)} ms بدونها، على {latency.get('runtime')} بعد 10 تشغيلات تسخين. النموذج صُمّم كمصنّف softmax لا كمطابقة cosine، وعدد الحالات السلبية في كل تجربة هو {e2['n_impostor']} لصورة LL و {e1['n_impostor']} لصورة NL، لذلك فارق النقاط المئوية يُقرأ مع هذا الحجم.",
    ]
    return lines


def write_summary(payload: dict) -> None:
    payload["score_overview"] = score_overview()
    dataset = payload.get("dataset") or {}
    per_split = dataset.get("per_split") or {}
    training = payload.get("training") or {}
    facts = payload.get("code_facts") or {}
    quality = payload.get("quality") or {}
    experiments = payload.get("experiments") or {}
    masked = payload.get("masked") or {}
    replay = payload.get("replay") or {}
    leakage = payload.get("leakage") or {}
    latency = payload.get("latency") or {}
    lines: list[str] = []
    lines.append("# نتائج التجارب المقاسة")
    lines.append("")
    lines.append("كل رقم في هذا الملف مأخوذ من `results/all_results.json` الذي كتبه التشغيل نفسه. لا توجد أرقام منسوخة يدويًا.")
    lines.append("")
    lines.append("## ما الذي يقيسه هذا التشغيل")
    lines.append("")
    lines.append("- النموذج: MobileNetV2 المحفوظ، وطبقة `embedding` ثم تطبيع L2 ثم cosine similarity.")
    lines.append("- المعالجة الموجودة في الكود هي CLAHE ثم Gamma، وليست فلتر إزالة ضوضاء. تجارب «مع المعالجة» تستخدم هذه الدالة كما هي.")
    hits = facts.get("token_0_52_hits_in_system_code") or []
    lines.append(f"- البحث عن النص `0.52` داخل `src/` و`web/` و`config/` و`app/` و`tests/` و`README.md` أعاد {len(hits)} نتيجة. عتبة الواجهة المقروءة من الإعداد هي `{facts.get('demo_default_threshold')}`، وعتبة التقييم `{facts.get('evaluation_confidence_threshold')}`، وعتبة معرض الموظفين `{facts.get('enrollment_match_threshold')}`. تجارب المطابقة هنا تستخدم 0.52 كعتبة بروتوكول البحث.")
    lines.append("- ترتيب المعالجة في الكود: محاكاة الإضاءة المنخفضة على الإطار كاملًا، ثم اكتشاف الوجه وتغيير الحجم، ثم CLAHE+Gamma.")
    lines.append("- تعريف TP/TN/FP/FN مكتوب في `experiments/README.md` ومطبّق في `experiments/evaluate.py`.")
    lines.append("")

    lines.append("## جدول 3.1 — حجم البيانات")
    lines.append("")
    lines.append("| التقسيم | NL | LL | المجموع |")
    lines.append("|---|---:|---:|---:|")
    for split in ("train", "val", "test"):
        block = per_split.get(split) or {}
        lines.append(f"| {split} | {block.get('nl')} | {block.get('ll')} | {block.get('total')} |")
    if per_split:
        lines.append(
            "| المجموع | "
            f"{sum(per_split[s]['nl'] for s in per_split)} | "
            f"{sum(per_split[s]['ll'] for s in per_split)} | "
            f"{sum(per_split[s]['total'] for s in per_split)} |"
        )
    lines.append("")
    overall = dataset.get("images_per_identity_overall") or {}
    lines.append("| النطاق | عدد الهويات | الأدنى | الأعلى | المتوسط |")
    lines.append("|---|---:|---:|---:|---:|")
    lines.append(f"| كل الصور | {overall.get('n_identities')} | {overall.get('min')} | {overall.get('max')} | {fmt_float(overall.get('mean'), 4)} |")
    for split, stats in (dataset.get("images_per_identity_by_split") or {}).items():
        lines.append(f"| {split} | {stats.get('n_identities')} | {stats.get('min')} | {stats.get('max')} | {fmt_float(stats.get('mean'), 4)} |")
    lines.append("")
    lines.append(f"حقل `counts` داخل ملف التقسيم يطابق إعادة العد: `{dataset.get('counts_match_file_field')}`.")
    lines.append(f"تعارض لاحقة اسم الملف مع علم low_light: `{dataset.get('suffix_flag_mismatches')}`.")
    lines.append("التجارب التالية تستخدم تقسيم `test` فقط، كما هو محفوظ، ولا تبدّل أعداد التدريب والاختبار.")
    lines.append("")

    lines.append("## جدول 3.2 — إعدادات التدريب كما هي في الكود والسجل")
    lines.append("")
    lines.append("| البند | القيمة | المصدر |")
    lines.append("|---|---|---|")
    rows = [
        ("Optimizer", training.get("optimizer"), _source(training, "config_optimizer")),
        ("Learning rate (head)", training.get("learning_rate_head"), _source(training, "config_lr")),
        ("Learning rate (fine-tune)", training.get("learning_rate_finetune"), _source(training, "config_ft_lr")),
        ("Batch size", training.get("batch_size"), _source(training, "config_batch")),
        ("Epochs head في الإعداد", training.get("epochs_head_config"), _source(training, "epochs_fit_head")),
        ("Epochs fine-tune في الإعداد", training.get("epochs_finetune_config"), _source(training, "epochs_fit_ft")),
        ("Epochs المسجّلة في history", training.get("epochs_recorded_in_history"), training.get("history_file")),
        ("Epochs head المستنتجة من هبوط معدل التعلم", training.get("inferred_head_epochs_ran"), training.get("history_file")),
        ("Epochs fine-tune المستنتجة", training.get("inferred_finetune_epochs_ran"), training.get("history_file")),
        ("Loss", training.get("loss"), _source(training, "loss")),
        ("طول embedding في الإعداد", training.get("embedding_units_config"), _source(training, "config_dense")),
        ("طول embedding المقاس من النموذج", payload.get("embedding_dim_runtime"), payload.get("model_path")),
        ("Dropout", training.get("dropout"), "config.yaml model.dropout"),
        ("Early stopping patience", training.get("early_stopping_patience"), _source(training, "early_stopping")),
        ("عدد معاملات النموذج", payload.get("model_param_count"), payload.get("model_path")),
    ]
    for label, value, source in rows:
        lines.append(f"| {label} | `{value}` | {source} |")
    lines.append("")
    lines.append("الاستنتاج الخاص بعدد epochs الفعلية يعتمد على أول معدل تعلم في السجل أقل أو يساوي معدل الضبط الدقيق. السجل نفسه محفوظ في ملف history المذكور.")
    lines.append("")

    lines.append("## جدول 3.3 — تعريف التجارب")
    lines.append("")
    lines.append("| الرمز | صور الاختبار | المعالجة | المعرض | العتبة |")
    lines.append("|---|---|---|---|---:|")
    design = {
        "E1": "NL من test",
        "E2": "LL من test",
        "E3": "LL من test، نفس القصّة المستخدمة في E2",
        "E3b": "NL من test، نفس القصّة المستخدمة في E1",
    }
    treatment = {
        "E1": "بدون CLAHE+Gamma",
        "E2": "بدون CLAHE+Gamma",
        "E3": "CLAHE ثم Gamma",
        "E3b": "CLAHE ثم Gamma",
    }
    for name in EXP_ORDER:
        lines.append(f"| {name} | {design[name]} | {treatment[name]} | نفس المعرض | {THRESHOLD:.2f} |")
    lines.append("")
    if experiments:
        lines.append(_header())
        for name in EXP_ORDER:
            if name in experiments:
                lines.append(_metric_row(name, experiments[name]))
        lines.append("")

    lines.append("## جدول 3.4 — مصفوفة الالتباس عند 0.52")
    lines.append("")
    lines.append("الصف هو الحقيقة (genuine / impostor) والعمود هو قرار التقييم. القبول الصحيح للمسجّل يشترط تطابق الهوية مع العتبة. قبول غير المسجّل هو FP حتى لو كانت الهوية المتنبأ بها شخصًا موجودًا في المعرض.")
    lines.append("")
    for name in EXP_ORDER:
        metrics = experiments.get(name)
        if not metrics:
            continue
        lines.append(f"### {name}")
        lines.append("")
        lines.append("|  | قبول | رفض |")
        lines.append("|---|---:|---:|")
        lines.append(f"| Genuine | {metrics['tp']} | {metrics['fn']} |")
        lines.append(f"| Impostor | {metrics['fp']} | {metrics['tn']} |")
        lines.append("")
        lines.append(f"من حالات FN، عدد من تجاوز العتبة بهوية خاطئة: {metrics['n_misidentified']}.")
        lines.append("")

    lines.append("## جدول 4.2 — PSNR و SSIM")
    lines.append("")
    lines.append(quality.get("psnr_definition", ""))
    lines.append("")
    lines.append(quality.get("ssim_definition", ""))
    lines.append("")
    lines.append(quality.get("registration", ""))
    lines.append("")
    lines.append("| المرحلة | n | PSNR mean | PSNR std | SSIM mean | SSIM std |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for label, key in (
        ("قبل CLAHE+Gamma", "psnr_before"),
        ("بعد CLAHE+Gamma", "psnr_after"),
    ):
        psnr = quality.get(key) or {}
        ssim_key = "ssim_before" if "before" in key else "ssim_after"
        ssim = quality.get(ssim_key) or {}
        lines.append(
            f"| {label} | {psnr.get('n')} | {fmt_float(psnr.get('mean'), 4)} | {fmt_float(psnr.get('std'), 4)} | "
            f"{fmt_float(ssim.get('mean'), 4)} | {fmt_float(ssim.get('std'), 4)} |"
        )
    psnr_i = quality.get("psnr_improvement") or {}
    ssim_i = quality.get("ssim_improvement") or {}
    lines.append(
        f"| بعد − قبل | {psnr_i.get('n')} | {fmt_float(psnr_i.get('mean'), 4)} | {fmt_float(psnr_i.get('std'), 4)} | "
        f"{fmt_float(ssim_i.get('mean'), 4)} | {fmt_float(ssim_i.get('std'), 4)} |"
    )
    lines.append("")
    jpeg = quality.get("jpeg_mae_replay_enhanced_vs_stored") or {}
    lines.append(
        f"متوسط الفرق المطلق بين صورة التعرف بعد CLAHE+Gamma والصورة المحفوظة في التقسيم: "
        f"{fmt_float(jpeg.get('mean'), 4)} (std {fmt_float(jpeg.get('std'), 4)}، n={jpeg.get('n')}). "
        "هذا يقيس قرب إعادة التشغيل من ملفات JPEG، والانحراف المعياري هنا عينة (ddof=1)."
    )
    lines.append(f"عدد صور LL التي وُجد لها صندوق وجه على الصورة الأصلية: {quality.get('n_face_box_found')} من {quality.get('n_ll_images')}.")
    lines.append("")

    lines.append("## جدول 4.3 — نتائج التحقق")
    lines.append("")
    if experiments:
        lines.append(_header())
        for name in EXP_ORDER:
            if name in experiments:
                lines.append(_metric_row(name, experiments[name]))
        lines.append("")
        lines.append("AUC هنا مساحة ROC لدرجة أعلى cosine بين genuine وimpostor. FRR في الجدول هو FRR الخاص بقرار البحث (يشمل خطأ الهوية). `frr_score_only` محفوظ في JSON لمن يريد رفض الدرجة وحدها.")
        lines.append("")

    lines.append("## جدول 4.4 — الوجوه المقنعة")
    lines.append("")
    lines.append(payload.get("mask_rule", ""))
    lines.append("")
    lines.append(f"لون القناع BGR: `{payload.get('mask_color_bgr')}`.")
    lines.append(f"عدد الصور حسب طريقة القناع: `{payload.get('mask_method_counts')}`.")
    if payload.get("mask_landmarker_error"):
        lines.append(f"ملاحظة المعالم: `{payload.get('mask_landmarker_error')}`.")
    lines.append("")
    if masked:
        lines.append("| التجربة | N | Accuracy | Precision | Recall | F1 |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for name in MASK_ORDER:
            metrics = masked[name]
            lines.append(
                f"| {name} | {metrics['n']} | {fmt_ratio(metrics['accuracy'])} | {fmt_ratio(metrics['precision'])} | "
                f"{fmt_ratio(metrics['recall'])} | {fmt_ratio(metrics['f1'])} |"
            )
        lines.append("")
        lines.append("TP وFAR وFRR لنفس التشغيل موجودة في `results/table_4_4_masked.csv`.")
        lines.append("")

    lines.append("## جدول 4.5 — زمن الاستجابة")
    lines.append("")
    lines.append(
        f"n={latency.get('n')}، تسخين غير محسوب={latency.get('warmup_runs_not_counted')}، "
        f"الجهاز=`{latency.get('runtime')}`، الاختيار: {latency.get('selection')}."
    )
    lines.append("")
    lines.append("| الشرط | المرحلة | المتوسط (ms) | الانحراف المعياري (ms) |")
    lines.append("|---|---|---:|---:|")
    labels = {
        "resize_ms": "اكتشاف الوجه وتغيير الحجم",
        "clahe_gamma_ms": "CLAHE+Gamma",
        "normalize_ms": "تحويل RGB وتطبيع النوع إلى float32",
        "resize_normalize_ms": "تغيير الحجم + التحويل",
        "embedding_ms": "استخراج embedding",
        "matching_ms": "المطابقة",
        "total_ms": "الزمن الكلي",
    }
    for condition, title in (("with_enhancement", "مع CLAHE+Gamma"), ("without_enhancement", "بدون CLAHE+Gamma")):
        block = latency.get(condition) or {}
        for key, label in labels.items():
            if key not in block:
                continue
            stats = block[key]
            lines.append(f"| {title} | {label} | {fmt_float(stats.get('mean_ms'), 2)} | {fmt_float(stats.get('std_ms'), 2)} |")
    lines.append("")
    for note in latency.get("notes") or []:
        lines.append(f"- {note}")
    lines.append("")

    lines.append("## حساسية العتبة — LL قبل المعالجة وبعدها")
    lines.append("")
    lines.append("FAR وFRR هنا يتبعان تعريف البحث (FRR يشمل خطأ الهوية). EER التقليدي على الدرجة وحدها مذكور بعده.")
    lines.append("")
    sweep = payload.get("threshold_sweep") or {}
    lines.append("| العتبة | E2 Accuracy | E2 FAR | E2 FRR | E3 Accuracy | E3 FAR | E3 FRR |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    e2_rows = (sweep.get("E2") or {}).get("rows") or []
    e3_rows = {row["threshold"]: row for row in (sweep.get("E3") or {}).get("rows") or []}
    for row in e2_rows:
        other = e3_rows.get(row["threshold"], {})
        lines.append(
            f"| {row['threshold']:.2f} | {fmt_ratio(row['accuracy'])} | {fmt_ratio(row['far'])} | {fmt_ratio(row['frr'])} | "
            f"{fmt_ratio(other.get('accuracy'))} | {fmt_ratio(other.get('far'))} | {fmt_ratio(other.get('frr'))} |"
        )
    lines.append("")
    for name in ("E2", "E3"):
        block = sweep.get(name) or {}
        grid = (block.get("eer_identity_aware") or {}).get("grid_min_abs_gap") or {}
        interp = (block.get("eer_identity_aware") or {}).get("interpolated")
        score = block.get("eer_score_only") or {}
        lines.append(
            f"- {name}: أصغر |FAR−FRR| على الشبكة عند العتبة `{grid.get('threshold')}` "
            f"(FAR {fmt_ratio(grid.get('far'))}، FRR {fmt_ratio(grid.get('frr'))}). "
            f"الاستيفاء الخطي: `{interp}`. "
            f"EER على الدرجة وحدها: العتبة `{score.get('threshold')}` والقيمة `{fmt_float(score.get('eer'), 6)}`."
        )
    lines.append(f"- العتبة 0.52 تبقى عتبة البروتوكول المطلوبة. المقارنة معها هي الفرق بين العتبات المقاسة أعلاه و0.52.")
    lines.append("")

    lines.append("## المعرض والهويات المستبعدة")
    lines.append("")
    held = (replay.get("held_out") or {})
    lines.append(f"الهويات المستبعدة: `{held.get('held_out')}`.")
    lines.append(f"عدد الهويات المؤهلة (لديها NL وLL في الاختبار): `{held.get('eligible_count')}`.")
    lines.append(f"عدد صور المعرض لكل هوية: `{replay.get('gallery_per_identity')}`.")
    lines.append(f"هويات بأقل من 3 صور: `{replay.get('gallery_identities_with_fewer_than_3')}`.")
    lines.append(f"بصمة مسارات الاختبار SHA-256: `{replay.get('test_paths_sha256')}`.")
    lines.append("")

    lines.append("## تسرب البيانات")
    lines.append("")
    lines.append(f"- نفس الصورة المصدرية في أكثر من تقسيم: {leakage.get('n_same_source_across_splits')}")
    lines.append(f"- نفس المصدر له NL وLL معًا: {leakage.get('n_sources_with_both_nl_and_ll')}")
    lines.append(f"- تكرار محتوى (MD5) بين الصور المعالجة: {leakage.get('n_duplicate_processed_md5')}")
    lines.append(f"- تكرار محتوى (MD5) بين الصور الخام: {leakage.get('n_duplicate_raw_md5')}")
    lines.append(f"- مصدر ظهر أكثر من مرة داخل البيانات: {leakage.get('n_duplicated_sources')}")
    lines.append(f"- صور معالجة مفقودة: {leakage.get('n_processed_missing')}")
    if leakage.get("leakage_found"):
        lines.append("- النتيجة: الفحص وجد حالة من الحالات أعلاه. التفاصيل في `results/leakage.json`.")
    else:
        lines.append("- النتيجة: الفحص لم يجد صورة مصدرية مشتركة بين التقسيمات، ولا نسخة NL/LL لنفس الملف، ولا تكرارًا بايتيًا.")
    lines.append("")

    softmax = payload.get("softmax_sanity") or {}
    if softmax:
        lines.append("## فحص إضافي: دقة softmax المغلقة")
        lines.append("")
        lines.append("هذا ليس بروتوكول cosine. هو فحص أن النموذج المحمّل ما زال يعطي تصنيفًا على صور الاختبار المحفوظة.")
        lines.append("")
        for key in ("stored_jpeg_tf_loader", "replay_enhanced_in_memory"):
            block = softmax.get(key) or {}
            lines.append(f"- {key}: {block.get('correct')} / {block.get('n')} = {fmt_float(block.get('accuracy'), 6)}")
        previous = softmax.get("previously_saved_file") or {}
        if previous:
            lines.append(
                f"- ملف تقييم سابق `{softmax.get('previously_saved_file_path')}` يسجل accuracy = {previous.get('accuracy')} "
                f"على {previous.get('num_test_samples')} صورة. هذا رقم محفوظ من تشغيل أقدم، وأُعيد قياسه في السطر السابق."
            )
        lines.append("")

    if payload.get("error"):
        lines.append("## توقف التشغيل")
        lines.append("")
        lines.append("```")
        lines.append(str(payload["error"])[:4000])
        lines.append("```")
        lines.append("")

    figures = payload.get("figures") or []
    lines.append("## الأشكال")
    lines.append("")
    for name in figures:
        lines.append(f"- `results/figures/{name}`")
    lines.append("")

    overview = payload.get("score_overview") or {}
    if overview:
        lines.append("## توزيع درجات cosine")
        lines.append("")
        lines.append("هذه أرقام محسوبة من `results/scores.csv` لنفس التشغيل. وسيط الدرجة يفسر لماذا عتبة 0.52 تقبل كل صور NL تقريبًا وترفض صور LL.")
        lines.append("")
        lines.append("| التجربة | وسيط genuine | وسيط impostor | هوية صحيحة قبل العتبة | genuine ≥ 0.52 | impostor ≥ 0.52 |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for name in ("E1", "E2", "E3", "E3b", "M_NL", "M_LL", "M_LL_ENH"):
            block = overview.get(name)
            if not block:
                continue
            lines.append(
                f"| {name} | {fmt_float(block['genuine_median'], 4)} | {fmt_float(block['impostor_median'], 4)} | "
                f"{block['n_genuine_correct_identity']} | {block['n_genuine_score_ge_0_52']} | {block['n_impostor_score_ge_0_52']} |"
            )
        lines.append("")

    lines.append("## الملخص")
    lines.append("")
    for index, line in enumerate(ten_lines(payload), start=1):
        lines.append(f"{index}. {line}")
    lines.append("")
    (RESULTS / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def write_table_csvs(payload: dict) -> None:
    from experiments.common import write_csv

    training = payload.get("training") or {}
    write_csv(
        RESULTS / "table_3_2_training.csv",
        [
            {"item": "optimizer", "value": training.get("optimizer")},
            {"item": "learning_rate_head", "value": training.get("learning_rate_head")},
            {"item": "learning_rate_finetune", "value": training.get("learning_rate_finetune")},
            {"item": "batch_size", "value": training.get("batch_size")},
            {"item": "epochs_head_config", "value": training.get("epochs_head_config")},
            {"item": "epochs_finetune_config", "value": training.get("epochs_finetune_config")},
            {"item": "epochs_recorded_in_history", "value": training.get("epochs_recorded_in_history")},
            {"item": "inferred_head_epochs_ran", "value": training.get("inferred_head_epochs_ran")},
            {"item": "inferred_finetune_epochs_ran", "value": training.get("inferred_finetune_epochs_ran")},
            {"item": "loss", "value": training.get("loss")},
            {"item": "embedding_units_config", "value": training.get("embedding_units_config")},
            {"item": "embedding_dim_runtime", "value": payload.get("embedding_dim_runtime")},
            {"item": "dropout", "value": training.get("dropout")},
            {"item": "early_stopping_patience", "value": training.get("early_stopping_patience")},
            {"item": "model_param_count", "value": payload.get("model_param_count")},
        ],
    )
    design_rows = []
    for name, metrics in (payload.get("experiments") or {}).items():
        design_rows.append(
            {
                "experiment": name,
                "description": metrics.get("description"),
                "n": metrics.get("n"),
                "accuracy": metrics.get("accuracy"),
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "f1": metrics.get("f1"),
                "far": metrics.get("far"),
                "frr": metrics.get("frr"),
                "auc": metrics.get("auc"),
            }
        )
    write_csv(RESULTS / "table_3_3_experiments.csv", design_rows)
    write_csv(
        RESULTS / "table_3_4_confusion.csv",
        [
            {
                "experiment": name,
                "tp": metrics.get("tp"),
                "tn": metrics.get("tn"),
                "fp": metrics.get("fp"),
                "fn": metrics.get("fn"),
                "n_misidentified": metrics.get("n_misidentified"),
            }
            for name, metrics in (payload.get("experiments") or {}).items()
        ],
    )


def write_report(payload: dict) -> None:
    write_table_csvs(payload)
    payload["figures"] = write_figures(payload)
    write_environment(payload)
    write_summary(payload)
