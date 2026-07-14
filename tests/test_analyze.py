"""Tests for the dataset analysis module (synthetic arrays + on-disk nrrd)."""

from __future__ import annotations

import numpy as np
import pytest

from mvseg.data.analyze import (
    DatasetAnalysis,
    analyze_case,
    analyze_splits,
    build_findings,
    case_stats_from_arrays,
)
from mvseg.data.splits import Splits

sitk = pytest.importorskip("SimpleITK")


def _gt_with(labels_at: dict[int, list[tuple]], shape=(16, 16, 16)) -> np.ndarray:
    """Build an int GT volume; labels_at maps class -> list of (z,y,x) voxels."""
    gt = np.zeros(shape, dtype=np.int16)
    for cls, coords in labels_at.items():
        for z, y, x in coords:
            gt[z, y, x] = cls
    return gt


def test_case_stats_counts_and_components():
    # class 1: one 2-voxel blob; class 4: two separate single-voxel blobs.
    gt = _gt_with(
        {
            1: [(2, 2, 2), (2, 2, 3)],  # one connected component (size 2)
            4: [(8, 8, 8), (12, 12, 12)],  # two components (size 1 each)
        }
    )
    vol = np.random.rand(*gt.shape).astype(np.float32)
    s = case_stats_from_arrays("P001_IMG0_00", gt, vol, num_classes=5, tiny_threshold=10)

    assert s.voxel_counts[1] == 2
    assert s.voxel_counts[4] == 2
    assert s.present == {1, 4}
    assert not s.empty_mask
    assert len(s.component_sizes[1]) == 1 and s.component_sizes[1][0] == 2
    assert len(s.component_sizes[4]) == 2  # two disconnected pieces
    assert s.n_tiny[4] == 2  # both < threshold 10
    assert s.issues == []


def test_case_stats_flags_quality_problems():
    gt = _gt_with({9: [(1, 1, 1)]})  # out-of-range label, no valid foreground
    s = case_stats_from_arrays("P002_IMG0_00", gt, volume=None, num_classes=5)
    assert any("out-of-range" in msg for msg in s.issues)
    assert s.empty_mask  # 9 is not a valid foreground class

    gt2 = _gt_with({1: [(1, 1, 1)]}, shape=(16, 16, 16))
    bad_vol = np.zeros((16, 16, 8), dtype=np.float32)  # mismatched shape
    s2 = case_stats_from_arrays("P003_IMG0_00", gt2, bad_vol, num_classes=5)
    assert any("!=" in msg for msg in s2.issues)


def test_dataset_aggregation_pixel_vs_image_level():
    a = case_stats_from_arrays("P001_IMG0_00", _gt_with({1: [(1, 1, 1)], 4: [(2, 2, 2)]}))
    b = case_stats_from_arrays("P002_IMG0_00", _gt_with({1: [(3, 3, 3)]}))
    ds = DatasetAnalysis([a, b], num_classes=5)

    # class 1 appears in both volumes; class 4 in one.
    freq = ds.image_frequency()
    assert freq[1] == (2, 1.0)
    assert freq[4] == (1, 0.5)

    # co-occurrence: class 1 & 4 together only in case a.
    m = ds.cooccurrence()
    assert m[1, 4] == 1
    assert m[1, 1] == 2  # diagonal == frequency

    # baseline: background dominates -> high pixel accuracy, zero fg dice.
    base = ds.baseline()
    assert base["all_background_pixel_accuracy"] > 0.99
    assert base["all_background_foreground_dice"] == 0.0


def test_findings_flag_imbalance_and_small_frequent():
    # class 1 present in every volume but tiny -> "small, frequent" finding.
    cases = [
        case_stats_from_arrays(f"P{p:03d}_IMG0_00", _gt_with({1: [(p % 16, 1, 1)]}))
        for p in range(10)
    ]
    ds = DatasetAnalysis(cases, num_classes=5)
    findings = build_findings(ds, None)
    text = " ".join(f["finding"] for f in findings)
    assert "Background occupies" in text  # imbalance
    assert "small, frequent" in text  # resolution-not-sampling diagnosis


def test_split_analysis_flags_rare_class(tmp_path):
    # Write a tiny on-disk dataset and split it, exercising the file + split path.
    data_dir = tmp_path / "raw"
    data_dir.mkdir()
    case_ids = []
    for p in range(4):
        cid = f"P{p:03d}_IMG0_00"
        gt = _gt_with({1: [(1, 1, 1)], 2: [(2, 2, 2)]})
        vol = np.random.rand(*gt.shape).astype(np.float32)
        sitk.WriteImage(sitk.GetImageFromArray(gt), str(data_dir / f"{cid}_gt.nrrd"))
        sitk.WriteImage(sitk.GetImageFromArray(vol), str(data_dir / f"{cid}_volume.nrrd"))
        case_ids.append(cid)

    cases = [
        analyze_case(cid, data_dir / f"{cid}_volume.nrrd", data_dir / f"{cid}_gt.nrrd")
        for cid in case_ids
    ]
    assert all(not c.issues for c in cases)
    by_case = {c.case_id: c for c in cases}

    splits = Splits(train=case_ids[:2], val=[case_ids[2]], test=[case_ids[3]])
    sa = analyze_splits(by_case, splits, num_classes=5, min_test_cases=5)
    # only 1 test case -> every present class is under the 5-case threshold
    assert any("statistically unreliable" in w for w in sa.rare_warnings)
    assert sa.per_split["train"].n_cases == 2
