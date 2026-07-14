"""Comprehensive dataset analysis for the 3D TEE segmentation data.

Answers the five questions every segmentation dataset review should:

1. **Profile** — voxel-level *and* volume-level class distribution (kept
   separate: a class can be 0.5 % of voxels yet appear in 60 % of volumes),
   connected-component (object) size distribution, class co-occurrence.
2. **Quality** — out-of-range label values, image/label shape mismatch, empty
   masks, and tiny fragments (floating noise / pinholes) via component sizes.
3. **Split & leakage** — per-split class distribution using the patient-level
   ``splits.json``; flags rare classes with too few val/test cases to be
   statistically meaningful.
4. **Baseline** — the trivial "predict all background" pixel accuracy, so a
   later "94 %" has context and pixel-accuracy is shown to be a useless metric.
5. **Reporting** — writes an issue-driven ``analysis_report.md`` (finding →
   impact → action), a ``DATA_CARD.md``, log-scale distribution plots, object-
   size histograms, a co-occurrence heatmap, and mid-slice GT overlays.

Run:
    python -m mvseg.data.analyze --data-dir data/raw --splits-file data/splits/splits.json

Everything except SimpleITK-based loading works on plain numpy arrays, so the
statistics are unit-tested without touching disk.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from mvseg import CLASS_NAMES, FG_CLASS_NAMES, NUM_CLASSES
from mvseg.data.splits import Splits, extract_patient_id

try:
    import SimpleITK as sitk  # noqa: N813
except ImportError:  # pragma: no cover
    sitk = None

_CONN_3D = np.ones((3, 3, 3), dtype=int)  # 26-connectivity


# --------------------------------------------------------------------------- #
# Per-case statistics
# --------------------------------------------------------------------------- #


@dataclass
class CaseStats:
    case_id: str
    patient_id: str
    shape: tuple[int, ...]
    spacing: tuple[float, ...]
    voxel_counts: dict[int, int]  # label value -> voxel count
    present: set[int]  # foreground classes present (>0)
    component_sizes: dict[int, list[int]]  # fg class -> connected-component sizes
    n_tiny: dict[int, int]  # fg class -> #components smaller than tiny_threshold
    intensity: dict[str, float]  # volume min/max/mean/std
    empty_mask: bool
    issues: list[str] = field(default_factory=list)


def case_stats_from_arrays(
    case_id: str,
    gt: np.ndarray,
    volume: np.ndarray | None = None,
    spacing: tuple[float, ...] = (1.0, 1.0, 1.0),
    num_classes: int = NUM_CLASSES,
    tiny_threshold: int = 10,
) -> CaseStats:
    """Compute all per-case statistics from in-memory arrays (no file I/O)."""
    from scipy import ndimage

    gt = np.asarray(gt)
    issues: list[str] = []

    uniq, counts = np.unique(gt, return_counts=True)
    voxel_counts = {int(u): int(c) for u, c in zip(uniq.tolist(), counts.tolist(), strict=False)}
    unexpected = [u for u in voxel_counts if u < 0 or u >= num_classes]
    if unexpected:
        issues.append(f"out-of-range label values {sorted(unexpected)}")

    if volume is not None and volume.shape != gt.shape:
        issues.append(f"image {volume.shape} != label {gt.shape}")

    present = {u for u in voxel_counts if u != 0 and 0 < u < num_classes}
    empty_mask = len(present) == 0
    if empty_mask:
        issues.append("empty mask (no foreground)")

    component_sizes: dict[int, list[int]] = {}
    n_tiny: dict[int, int] = {}
    for c in range(1, num_classes):
        if c not in present:
            continue
        lbl, n = ndimage.label(gt == c, structure=_CONN_3D)
        sizes = np.bincount(lbl.ravel())[1:] if n else np.array([], dtype=int)
        component_sizes[c] = sizes.tolist()
        n_tiny[c] = int((sizes < tiny_threshold).sum())

    intensity: dict[str, float] = {}
    if volume is not None:
        v = np.asarray(volume, dtype=np.float32)
        intensity = {
            "min": float(v.min()),
            "max": float(v.max()),
            "mean": float(v.mean()),
            "std": float(v.std()),
        }

    return CaseStats(
        case_id=case_id,
        patient_id=extract_patient_id(case_id),
        shape=tuple(int(s) for s in gt.shape),
        spacing=tuple(round(float(s), 4) for s in spacing),
        voxel_counts=voxel_counts,
        present=present,
        component_sizes=component_sizes,
        n_tiny=n_tiny,
        intensity=intensity,
        empty_mask=empty_mask,
        issues=issues,
    )


def _read(path: Path):
    img = sitk.ReadImage(str(path))
    return sitk.GetArrayFromImage(img), img.GetSpacing()


def analyze_case(
    case_id: str,
    volume_path: Path,
    gt_path: Path,
    num_classes: int = NUM_CLASSES,
    tiny_threshold: int = 10,
) -> CaseStats:
    """Load a volume/GT pair from disk and compute per-case statistics."""
    if sitk is None:  # pragma: no cover
        raise RuntimeError("SimpleITK is required: uv sync --extra dev")
    gt, spacing = _read(gt_path)
    volume = None
    if volume_path.exists():
        volume, _ = _read(volume_path)
    else:
        return CaseStats(
            case_id=case_id,
            patient_id=extract_patient_id(case_id),
            shape=tuple(int(s) for s in gt.shape),
            spacing=tuple(round(float(s), 4) for s in spacing),
            voxel_counts={},
            present=set(),
            component_sizes={},
            n_tiny={},
            intensity={},
            empty_mask=True,
            issues=["missing volume file"],
        )
    return case_stats_from_arrays(
        case_id, gt, volume, spacing, num_classes=num_classes, tiny_threshold=tiny_threshold
    )


# --------------------------------------------------------------------------- #
# Dataset aggregation
# --------------------------------------------------------------------------- #


@dataclass
class DatasetAnalysis:
    cases: list[CaseStats]
    num_classes: int = NUM_CLASSES
    tiny_threshold: int = 10

    @property
    def n_cases(self) -> int:
        return len(self.cases)

    @property
    def patients(self) -> set[str]:
        return {c.patient_id for c in self.cases}

    def voxel_distribution(self) -> dict[int, tuple[int, float]]:
        """class -> (total voxels, fraction of all voxels). The pixel-level view."""
        totals: dict[int, int] = defaultdict(int)
        for case in self.cases:
            for label, cnt in case.voxel_counts.items():
                totals[label] += cnt
        grand = sum(totals.values()) or 1
        return {c: (totals.get(c, 0), totals.get(c, 0) / grand) for c in range(self.num_classes)}

    def image_frequency(self) -> dict[int, tuple[int, float]]:
        """fg class -> (#volumes containing it, fraction of volumes). The image-level view."""
        counts: dict[int, int] = defaultdict(int)
        for case in self.cases:
            for c in case.present:
                counts[c] += 1
        n = self.n_cases or 1
        return {c: (counts.get(c, 0), counts.get(c, 0) / n) for c in range(1, self.num_classes)}

    def component_size_summary(self) -> dict[int, dict[str, float]]:
        """fg class -> object-size stats over all connected components in the dataset."""
        pooled: dict[int, list[int]] = defaultdict(list)
        tiny: dict[int, int] = defaultdict(int)
        for case in self.cases:
            for c, sizes in case.component_sizes.items():
                pooled[c].extend(sizes)
            for c, nt in case.n_tiny.items():
                tiny[c] += nt
        out: dict[int, dict[str, float]] = {}
        for c in range(1, self.num_classes):
            sizes = np.array(pooled.get(c, []), dtype=float)
            if sizes.size == 0:
                continue
            out[c] = {
                "n_objects": int(sizes.size),
                "min": float(sizes.min()),
                "median": float(np.median(sizes)),
                "mean": float(sizes.mean()),
                "p90": float(np.percentile(sizes, 90)),
                "max": float(sizes.max()),
                "n_tiny": int(tiny.get(c, 0)),
            }
        return out

    def cooccurrence(self) -> np.ndarray:
        """(C, C) matrix: #volumes where classes i and j both appear (diagonal = frequency)."""
        m = np.zeros((self.num_classes, self.num_classes), dtype=int)
        for case in self.cases:
            present = sorted(case.present)
            for i in present:
                for j in present:
                    m[i, j] += 1
        return m

    def baseline(self) -> dict[str, float]:
        """Trivial 'predict all background' baseline — shows why pixel accuracy lies."""
        dist = self.voxel_distribution()
        bg_frac = dist.get(0, (0, 0.0))[1]
        return {
            "all_background_pixel_accuracy": bg_frac,
            "all_background_foreground_dice": 0.0,  # every fg class scores 0
        }

    def intensity_summary(self) -> dict[str, float]:
        means = [c.intensity["mean"] for c in self.cases if c.intensity]
        stds = [c.intensity["std"] for c in self.cases if c.intensity]
        mins = [c.intensity["min"] for c in self.cases if c.intensity]
        maxs = [c.intensity["max"] for c in self.cases if c.intensity]
        if not means:
            return {}
        return {
            "mean_of_means": float(np.mean(means)),
            "mean_of_stds": float(np.mean(stds)),
            "global_min": float(min(mins)),
            "global_max": float(max(maxs)),
        }

    def shape_counts(self) -> dict[tuple[int, ...], int]:
        counts: dict[tuple[int, ...], int] = defaultdict(int)
        for case in self.cases:
            counts[case.shape] += 1
        return dict(counts)

    def issues(self) -> list[tuple[str, str]]:
        return [(c.case_id, msg) for c in self.cases for msg in c.issues]


# --------------------------------------------------------------------------- #
# Split-aware analysis
# --------------------------------------------------------------------------- #


@dataclass
class SplitAnalysis:
    per_split: dict[str, DatasetAnalysis]
    rare_warnings: list[str]
    leakage: list[str] = field(default_factory=list)


def analyze_splits(
    by_case: dict[str, CaseStats], splits: Splits, num_classes: int, min_test_cases: int = 5
) -> SplitAnalysis:
    per_split: dict[str, DatasetAnalysis] = {}
    for name in ("train", "val", "test"):
        ids = getattr(splits, name)
        cases = [by_case[c] for c in ids if c in by_case]
        per_split[name] = DatasetAnalysis(cases, num_classes=num_classes)

    # Patient-level leakage: no patient may appear in more than one split. This is
    # the single most important integrity check — if it fails, every downstream
    # metric is inflated (a patient's correlated frames straddle train and test).
    leakage: list[str] = []
    pat = {name: per_split[name].patients for name in ("train", "val", "test")}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        shared = sorted(pat[a] & pat[b])
        if shared:
            leakage.append(f"{a}∩{b}: {shared}")

    warnings: list[str] = []
    for name in ("val", "test"):
        freq = per_split[name].image_frequency()
        for c in range(1, num_classes):
            n_present = freq.get(c, (0, 0.0))[0]
            if n_present < min_test_cases:
                warnings.append(
                    f"class '{CLASS_NAMES[c]}' appears in only {n_present} {name} case(s) "
                    f"(< {min_test_cases}) — its {name} score is statistically unreliable"
                )
    return SplitAnalysis(per_split=per_split, rare_warnings=warnings, leakage=leakage)


# --------------------------------------------------------------------------- #
# Findings (finding -> impact -> action)
# --------------------------------------------------------------------------- #


def build_findings(
    analysis: DatasetAnalysis, split_analysis: SplitAnalysis | None
) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    dist = analysis.voxel_distribution()
    freq = analysis.image_frequency()
    bg_frac = dist.get(0, (0, 0.0))[1]

    if bg_frac > 0.8:
        findings.append(
            {
                "severity": "high",
                "finding": f"Background occupies {bg_frac:.1%} of all voxels — severe class imbalance.",
                "impact": "Cross-entropy alone lets the thin foreground classes be ignored; pixel accuracy is meaningless (see baseline).",
                "action": "Use a region-overlap loss (DiceCE / DiceFocal, include_background=false) and report per-class Dice, not a single mean.",
            }
        )

    for c in range(1, analysis.num_classes):
        px = dist.get(c, (0, 0.0))[1]
        img = freq.get(c, (0, 0.0))[1]
        name = CLASS_NAMES[c]
        if px < 0.01 and img > 0.5:
            findings.append(
                {
                    "severity": "medium",
                    "finding": f"'{name}' is only {px:.2%} of voxels but appears in {img:.0%} of volumes — a small, frequent object.",
                    "impact": "Downsampling can erase it; this is a resolution/crop problem, not a sampling one.",
                    "action": "Avoid aggressive resize; train on patches at native resolution + sliding-window inference. Verify crop size vs. object-size histogram.",
                }
            )
        elif img < 0.1 and px >= 0.01:
            findings.append(
                {
                    "severity": "medium",
                    "finding": f"'{name}' is {px:.2%} of voxels but appears in only {img:.0%} of volumes — a rare object.",
                    "impact": "The model sees it too seldom; a sampling problem, not a resolution one.",
                    "action": "Oversample volumes containing this class (weighted sampler) and confirm enough val/test coverage.",
                }
            )

    sizes = analysis.component_size_summary()
    total_tiny = sum(s.get("n_tiny", 0) for s in sizes.values())
    if total_tiny > 0:
        per_class = ", ".join(
            f"{CLASS_NAMES[c]}={s['n_tiny']}" for c, s in sizes.items() if s.get("n_tiny", 0)
        )
        findings.append(
            {
                "severity": "medium",
                "finding": f"{total_tiny} tiny connected components (< {analysis.tiny_threshold} voxels) across the dataset ({per_class}).",
                "impact": "Likely annotation noise / pinholes; they add label noise and inflate the object count.",
                "action": "Review a few, and consider a morphological cleanup (remove_small_objects / fill_holes) in a new data version.",
            }
        )

    issues = analysis.issues()
    if issues:
        kinds = defaultdict(int)
        for _, msg in issues:
            kinds[msg.split(" ")[0]] += 1
        findings.append(
            {
                "severity": "high",
                "finding": f"{len(issues)} automated quality issue(s): {dict(kinds)}.",
                "impact": "Shape mismatches / out-of-range labels / empty masks corrupt training and metrics.",
                "action": "Fix or exclude the listed cases before the next data version; see the issue table below.",
            }
        )

    if split_analysis and split_analysis.leakage:
        findings.append(
            {
                "severity": "high",
                "finding": "PATIENT LEAKAGE across splits — " + "; ".join(split_analysis.leakage),
                "impact": "A patient's correlated frames straddle train and test; every reported metric is inflated and invalid.",
                "action": "Regenerate splits with `scripts/prepare_splits.py` (patient-level) before any training. Do not trust prior results.",
            }
        )

    if split_analysis and split_analysis.rare_warnings:
        findings.append(
            {
                "severity": "high",
                "finding": "Rare classes are under-represented in val/test: "
                + "; ".join(split_analysis.rare_warnings),
                "impact": "Their reported IoU/Dice is dominated by 1–2 cases and is statistically meaningless.",
                "action": "Collect more of these classes or pin a frozen test set that guarantees coverage.",
            }
        )

    return findings


# --------------------------------------------------------------------------- #
# Report / data card writers
# --------------------------------------------------------------------------- #


def _fmt_distribution_table(analysis: DatasetAnalysis) -> str:
    dist = analysis.voxel_distribution()
    freq = analysis.image_frequency()
    lines = [
        "| idx | class | voxel share | # volumes | volume freq |",
        "|-----|-------|-------------|-----------|-------------|",
    ]
    for c in range(analysis.num_classes):
        px_cnt, px_frac = dist.get(c, (0, 0.0))
        if c == 0:
            lines.append(f"| 0 | {CLASS_NAMES[0]} | {px_frac:.3%} | — | — |")
        else:
            n_img, img_frac = freq.get(c, (0, 0.0))
            lines.append(
                f"| {c} | {CLASS_NAMES[c]} | {px_frac:.3%} | {n_img}/{analysis.n_cases} | {img_frac:.1%} |"
            )
    return "\n".join(lines)


def _fmt_size_table(analysis: DatasetAnalysis) -> str:
    sizes = analysis.component_size_summary()
    if not sizes:
        return "_(no foreground objects found)_"
    lines = [
        "| class | #objects | min | median | mean | p90 | max | #tiny |",
        "|-------|----------|-----|--------|------|-----|-----|-------|",
    ]
    for c in range(1, analysis.num_classes):
        s = sizes.get(c)
        if not s:
            continue
        lines.append(
            f"| {CLASS_NAMES[c]} | {int(s['n_objects'])} | {int(s['min'])} | {int(s['median'])} "
            f"| {s['mean']:.0f} | {int(s['p90'])} | {int(s['max'])} | {int(s['n_tiny'])} |"
        )
    return "\n".join(lines)


def _fmt_cooccurrence(analysis: DatasetAnalysis) -> str:
    m = analysis.cooccurrence()
    header = "| both-present | " + " | ".join(CLASS_NAMES[1:]) + " |"
    sep = "|" + "---|" * (analysis.num_classes)
    lines = [header, sep]
    for i in range(1, analysis.num_classes):
        row = " | ".join(str(m[i, j]) for j in range(1, analysis.num_classes))
        lines.append(f"| **{CLASS_NAMES[i]}** | {row} |")
    return "\n".join(lines)


def write_report(
    analysis: DatasetAnalysis,
    split_analysis: SplitAnalysis | None,
    findings: list[dict[str, str]],
    out_path: Path,
) -> None:
    base = analysis.baseline()
    high = sum(1 for f in findings if f["severity"] == "high")
    lines = [
        "# Dataset analysis report",
        "",
        "## Summary",
        f"- **Scale**: {analysis.n_cases} labeled frames from {len(analysis.patients)} patients.",
        f"- **Balance**: background = {analysis.voxel_distribution()[0][1]:.1%} of voxels "
        f"→ 'predict all background' scores **{base['all_background_pixel_accuracy']:.1%} pixel "
        "accuracy** yet 0 Dice on every structure. Report per-class Dice.",
        f"- **Risks**: {high} high-severity finding(s); {len(analysis.issues())} automated quality issue(s).",
        "",
        "## Key findings (finding → impact → action)",
    ]
    if not findings:
        lines.append("\n_No automated findings triggered._")
    for i, f in enumerate(findings, 1):
        lines += [
            "",
            f"### {i}. [{f['severity'].upper()}] {f['finding']}",
            f"- **Impact**: {f['impact']}",
            f"- **Action**: {f['action']}",
        ]

    lines += [
        "",
        "## Class distribution (pixel-level vs. image-level)",
        "",
        _fmt_distribution_table(analysis),
        "",
        "> Pixel share and volume frequency are deliberately separate: a class that is a tiny",
        "> fraction of voxels but appears in most volumes is a *resolution* problem; one that is",
        "> sizeable but appears in few volumes is a *sampling* problem.",
        "",
        "## Object-size distribution (connected components, 26-conn)",
        "",
        _fmt_size_table(analysis),
        "",
        "> If the median object is small relative to the crop/downsampling factor, prefer patch",
        "> training + sliding-window inference over resizing (see README / P100 memory notes).",
        "",
        "## Class co-occurrence (volumes where both appear)",
        "",
        _fmt_cooccurrence(analysis),
    ]

    if split_analysis:
        lines += ["", "## Split distribution (leakage & coverage)", ""]
        lines.append("| split | patients | frames | " + " | ".join(FG_CLASS_NAMES) + " |")
        lines.append("|" + "---|" * (3 + len(FG_CLASS_NAMES)))
        for name in ("train", "val", "test"):
            da = split_analysis.per_split[name]
            dist = da.voxel_distribution()
            fracs = " | ".join(f"{dist.get(c, (0, 0.0))[1]:.2%}" for c in range(1, analysis.num_classes))
            lines.append(f"| {name} | {len(da.patients)} | {da.n_cases} | {fracs} |")
        if split_analysis.leakage:
            lines.append("")
            lines.append(
                "- ❌ **PATIENT LEAKAGE**: " + "; ".join(split_analysis.leakage)
                + " — regenerate splits before training."
            )
        else:
            lines.append("")
            lines.append("- ✅ No patient appears in more than one split (patient-level integrity verified).")
        lines += [
            "",
            "> Splitting is patient-level (see `mvseg.data.splits`), preventing a patient's",
            "> correlated frames from leaking across train/val/test. Compare the per-class",
            "> shares above: rare classes must be present in val/test to be measurable.",
        ]
        if split_analysis.rare_warnings:
            lines.append("")
            for w in split_analysis.rare_warnings:
                lines.append(f"- ⚠️ {w}")

    issues = analysis.issues()
    if issues:
        lines += ["", "## Automated quality issues", "", "| case | issue |", "|------|-------|"]
        for cid, msg in issues[:100]:
            lines.append(f"| {cid} | {msg} |")
        if len(issues) > 100:
            lines.append(f"| … | (+{len(issues) - 100} more) |")

    lines += [
        "",
        "## Baseline",
        f"- **All-background** predictor: pixel accuracy **{base['all_background_pixel_accuracy']:.1%}**, "
        "foreground Dice **0.0**. This is the number every real model must beat, and the reason",
        "  pixel accuracy is not reported as a headline metric.",
        "",
        "_Generated by `python -m mvseg.data.analyze`. Plots and overlays are in the same folder._",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_data_card(
    analysis: DatasetAnalysis,
    split_analysis: SplitAnalysis | None,
    findings: list[dict[str, str]],
    out_path: Path,
    version: str,
) -> None:
    dist = analysis.voxel_distribution()
    split_line = "single pool (no splits.json provided)"
    if split_analysis:
        parts = []
        for name in ("train", "val", "test"):
            da = split_analysis.per_split[name]
            parts.append(f"{name} {da.n_cases} ({len(da.patients)}p)")
        split_line = " / ".join(parts)
    limits = [f["finding"] for f in findings if f["severity"] in ("high", "medium")]
    lines = [
        f"# Dataset card — {version}",
        "",
        f"- **Task**: 3D TEE mitral-valve multi-class segmentation ({analysis.num_classes} classes).",
        "- **Modality / format**: single-channel `.nrrd` volumes + integer label `.nrrd`.",
        f"- **Scale**: {analysis.n_cases} labeled frames from {len(analysis.patients)} patients.",
        f"- **Split (patient-level, hash-stable)**: {split_line}.",
        "- **Class definitions** (label = index):",
    ]
    for c in range(analysis.num_classes):
        lines.append(f"    - `{c}` {CLASS_NAMES[c]} — voxel share {dist.get(c, (0, 0.0))[1]:.3%}")
    lines += [
        "- **Ignore-label policy**: none (labels are exactly 0–"
        f"{analysis.num_classes - 1}; out-of-range values are flagged as issues).",
        f"- **Known limitations**: {'; '.join(limits) if limits else 'none flagged automatically'}.",
        f"- **Automated quality issues**: {len(analysis.issues())} "
        f"(tiny-fragment threshold = {analysis.tiny_threshold} voxels).",
        "- **Inspection**: run `python -m mvseg.data.analyze`; review the mid-slice overlays in "
        "`overlays/` and the loss-ranked samples after training.",
        "",
        "_This card is regenerated per data version. Commit it alongside the version's "
        "`splits.json`; keep the raw volumes and overlays out of git (see `.gitignore`)._",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Plots & overlays
# --------------------------------------------------------------------------- #


def make_plots(analysis: DatasetAnalysis, split_analysis: SplitAnalysis | None, out_dir: Path) -> list[str]:
    written: list[str] = []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        return [f"(plots skipped: {e})"]

    # 1) log-scale voxel distribution
    dist = analysis.voxel_distribution()
    fig, ax = plt.subplots(figsize=(7, 4))
    fracs = [max(dist.get(c, (0, 0.0))[1], 1e-9) for c in range(analysis.num_classes)]
    ax.bar(range(analysis.num_classes), fracs)
    ax.set_yscale("log")
    ax.set_xticks(range(analysis.num_classes))
    ax.set_xticklabels([CLASS_NAMES[c] for c in range(analysis.num_classes)], rotation=30, ha="right")
    ax.set_ylabel("voxel share (log)")
    ax.set_title("Class distribution (pixel-level, log scale)")
    fig.tight_layout()
    p = out_dir / "class_distribution_log.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    written.append(p.name)

    # 2) object-size histograms per fg class
    pooled = {c: [] for c in range(1, analysis.num_classes)}
    for case in analysis.cases:
        for c, sizes in case.component_sizes.items():
            pooled[c].extend(sizes)
    active = [c for c in pooled if pooled[c]]
    if active:
        fig, axes = plt.subplots(1, len(active), figsize=(4 * len(active), 3.2), squeeze=False)
        for ax, c in zip(axes[0], active, strict=False):
            ax.hist(np.log10(np.array(pooled[c]) + 1), bins=30)
            ax.set_title(CLASS_NAMES[c], fontsize=9)
            ax.set_xlabel("log10(object size)")
        fig.suptitle("Object-size distribution (connected components)")
        fig.tight_layout()
        p = out_dir / "object_sizes.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        written.append(p.name)

    # 3) split distribution comparison
    if split_analysis:
        fig, ax = plt.subplots(figsize=(8, 4))
        width = 0.25
        x = np.arange(1, analysis.num_classes)
        for i, name in enumerate(("train", "val", "test")):
            da = split_analysis.per_split[name]
            d = da.voxel_distribution()
            vals = [max(d.get(c, (0, 0.0))[1], 1e-9) for c in range(1, analysis.num_classes)]
            ax.bar(x + (i - 1) * width, vals, width, label=name)
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(FG_CLASS_NAMES, rotation=30, ha="right")
        ax.set_ylabel("voxel share (log)")
        ax.set_title("Per-split class distribution")
        ax.legend()
        fig.tight_layout()
        p = out_dir / "split_distribution.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        written.append(p.name)

    # 4) co-occurrence heatmap
    m = analysis.cooccurrence()[1:, 1:]
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(m, cmap="viridis")
    ax.set_xticks(range(len(FG_CLASS_NAMES)))
    ax.set_yticks(range(len(FG_CLASS_NAMES)))
    ax.set_xticklabels(FG_CLASS_NAMES, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(FG_CLASS_NAMES, fontsize=8)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, int(m[i, j]), ha="center", va="center", color="w", fontsize=8)
    ax.set_title("Class co-occurrence (# volumes)")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    p = out_dir / "cooccurrence.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    written.append(p.name)
    return written


def make_overlays(
    cases: list[CaseStats],
    resolve_paths,
    out_dir: Path,
    n: int = 12,
) -> int:
    """Save mid-(best-foreground)-slice image|GT overlays for a sample of cases.

    ``resolve_paths(case_id) -> (volume_path, gt_path)``. Overlays contain real
    imagery, so they are written under the (git-ignored) report folder only.
    """
    if sitk is None:  # pragma: no cover
        return 0
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from mvseg.utils.viz import colorize_label
    except Exception:  # pragma: no cover
        return 0

    overlay_dir = out_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    # Prefer cases with foreground; spread across the dataset.
    picks = [c for c in cases if not c.empty_mask][:: max(1, len(cases) // max(1, n))][:n]
    written = 0
    for case in picks:
        vol_path, gt_path = resolve_paths(case.case_id)
        if not gt_path.exists():
            continue
        gt = sitk.GetArrayFromImage(sitk.ReadImage(str(gt_path)))  # (z, y, x)
        vol = (
            sitk.GetArrayFromImage(sitk.ReadImage(str(vol_path)))
            if vol_path.exists()
            else np.zeros_like(gt)
        )
        fg_per_slice = (gt > 0).reshape(gt.shape[0], -1).sum(axis=1)
        z = int(np.argmax(fg_per_slice))
        img2d, lab2d = vol[z].astype(np.float32), gt[z]
        lo, hi = float(img2d.min()), float(img2d.max())
        gray = (img2d - lo) / (hi - lo) if hi > lo else np.zeros_like(img2d)
        color = colorize_label(lab2d).astype(np.float32) / 255.0
        blend = np.stack([gray] * 3, -1)
        fg = (lab2d > 0)[..., None]
        blend = np.where(fg, 0.5 * blend + 0.5 * color, blend)

        fig, axes = plt.subplots(1, 2, figsize=(7, 3.6))
        axes[0].imshow(gray, cmap="gray")
        axes[0].set_title(f"{case.case_id}  z={z}", fontsize=8)
        axes[1].imshow(np.clip(blend, 0, 1))
        axes[1].set_title("GT overlay", fontsize=8)
        for a in axes:
            a.axis("off")
        fig.tight_layout()
        fig.savefig(overlay_dir / f"{case.case_id}.png", dpi=110)
        plt.close(fig)
        written += 1
    return written


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _discover(base: Path, label_suffix: str, file_ext: str) -> list[str]:
    tail = f"{label_suffix}{file_ext}"
    return sorted(p.name[: -len(tail)] for p in base.glob(f"*{tail}"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Comprehensive MVSeg dataset analysis")
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--subdir", default="")
    ap.add_argument("--image-suffix", default="_volume")
    ap.add_argument("--label-suffix", default="_gt")
    ap.add_argument("--file-ext", default=".nrrd")
    ap.add_argument("--splits-file", default="data/splits/splits.json")
    ap.add_argument("--out-dir", default="reports/data_analysis")
    ap.add_argument("--num-classes", type=int, default=NUM_CLASSES)
    ap.add_argument("--tiny-threshold", type=int, default=10, help="components smaller than this = fragments")
    ap.add_argument("--n-overlays", type=int, default=12)
    ap.add_argument("--max-cases", type=int, default=0, help="0 = all (use to sample large sets)")
    ap.add_argument("--version", default="unversioned", help="label for the data card")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--no-overlays", action="store_true")
    args = ap.parse_args()

    if sitk is None:
        raise SystemExit("SimpleITK is required: uv sync --extra dev")

    base = Path(args.data_dir) / args.subdir if args.subdir else Path(args.data_dir)
    case_ids = _discover(base, args.label_suffix, args.file_ext)
    if args.max_cases:
        case_ids = case_ids[: args.max_cases]
    if not case_ids:
        raise SystemExit(f"No '*{args.label_suffix}{args.file_ext}' files under {base}")

    def resolve(cid: str) -> tuple[Path, Path]:
        return (
            base / f"{cid}{args.image_suffix}{args.file_ext}",
            base / f"{cid}{args.label_suffix}{args.file_ext}",
        )

    print(f"Analyzing {len(case_ids)} cases under {base} ...")
    cases: list[CaseStats] = []
    for i, cid in enumerate(case_ids, 1):
        vol_path, gt_path = resolve(cid)
        cases.append(
            analyze_case(cid, vol_path, gt_path, args.num_classes, args.tiny_threshold)
        )
        if i % 25 == 0 or i == len(case_ids):
            print(f"  {i}/{len(case_ids)}")

    analysis = DatasetAnalysis(cases, num_classes=args.num_classes, tiny_threshold=args.tiny_threshold)
    by_case = {c.case_id: c for c in cases}

    split_analysis = None
    splits_path = Path(args.splits_file)
    if splits_path.exists():
        split_analysis = analyze_splits(by_case, Splits.from_json(splits_path), args.num_classes)
    else:
        print(f"[info] no splits file at {splits_path}; skipping split analysis")

    findings = build_findings(analysis, split_analysis)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_report(analysis, split_analysis, findings, out_dir / "analysis_report.md")
    write_data_card(analysis, split_analysis, findings, out_dir / "DATA_CARD.md", args.version)
    # machine-readable dump for downstream tooling / diffing across data versions
    (out_dir / "stats.json").write_text(
        json.dumps(
            {
                "n_cases": analysis.n_cases,
                "n_patients": len(analysis.patients),
                "voxel_distribution": {CLASS_NAMES[c]: analysis.voxel_distribution()[c][1] for c in range(args.num_classes)},
                "image_frequency": {CLASS_NAMES[c]: analysis.image_frequency()[c][1] for c in range(1, args.num_classes)},
                "component_sizes": {CLASS_NAMES[c]: s for c, s in analysis.component_size_summary().items()},
                "baseline": analysis.baseline(),
                "intensity": analysis.intensity_summary(),
                "n_issues": len(analysis.issues()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    plots: list[str] = []
    if not args.no_plots:
        plots = make_plots(analysis, split_analysis, out_dir)
    n_overlays = 0
    if not args.no_overlays:
        n_overlays = make_overlays(cases, resolve, out_dir, args.n_overlays)

    print(f"\n{len(findings)} finding(s), {len(analysis.issues())} quality issue(s).")
    print(f"  report:    {out_dir / 'analysis_report.md'}")
    print(f"  data card: {out_dir / 'DATA_CARD.md'}")
    print(f"  plots:     {', '.join(plots) if plots else '(none)'}")
    print(f"  overlays:  {n_overlays} written to {out_dir / 'overlays'}")


if __name__ == "__main__":
    main()
