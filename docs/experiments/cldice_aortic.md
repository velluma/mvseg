# Experiment — clDice for aortic-valve annulus topology

## Motivation

The aortic-valve annulus (label `4`) is a thin, **ring-shaped** structure. Under a
pure voxel-overlap loss (Dice/DiceCE), a prediction that is 95 % correct by volume
can still **break the ring into disconnected arcs** — clinically the connectivity
of the annulus is what matters, and Dice/HD95 are blind to it (a 1-voxel gap
barely moves Dice but destroys the topology).

**clDice** (centerline Dice, [Shit et al., CVPR 2021]) compares the *soft
skeletons* of prediction and ground truth, so it directly rewards keeping the
centerline connected. We add it as an auxiliary term on the aortic annulus only,
on top of the existing DiceCE loss.

## Hypothesis

Adding a clDice term on class 4 improves the **topological correctness** of the
aortic annulus (fewer broken components, higher clDice) **without degrading**
voxel Dice / HD95, compared with the DiceCE baseline.

## Loss

```
total = DiceCE(logits, y)  +  λ_cldice · (1 − clDice(p_4, y_4))
```

- `p_4` = softmax probability of channel 4; `y_4` = binary GT mask of class 4.
- `clDice = 2·Tprec·Tsens / (Tprec + Tsens)`, where `Tprec`, `Tsens` use soft
  skeletons from iterative soft morphology (`losses.soft_skeletonize`).
- Implemented in `mvseg.losses.ClDiceAugmentedLoss`; selected via loss name
  `dice_ce_cldice` with `cldice_classes`, `lambda_cldice`, `cldice_iters`.

## Design (single-factor ablation)

Everything is held fixed except the loss. Same `splits.json`, same `seed=42`,
same network / optimizer / schedule / augmentation / epochs.

| Arm | Loss | `topo_classes` | Command |
|-----|------|----------------|---------|
| **A — control** | `dice_ce` | `[4]` | `python -m mvseg.train +experiment=resunet_baseline model.topo_classes=[4]` |
| **B — treatment** | `dice_ce_cldice` (λ=0.5, iters=10) | `[4]` | `python -m mvseg.train +experiment=resunet_cldice` |

Both arms compute the topology metrics so they are directly comparable.

### Secondary sweeps (after A vs B shows signal)

1. **Weight** `λ_cldice ∈ {0.25, 0.5, 1.0}` — trade-off between overlap and topology.
   ```bash
   python -m mvseg.train --multirun +experiment=resunet_cldice \
       model.loss.lambda_cldice=0.25,0.5,1.0
   ```
2. **Skeleton depth** `cldice_iters ∈ {5, 10, 15}` — must exceed the annulus'
   thickest half-radius (in voxels) for the skeleton to form; too small → weak
   term, too large → wasted compute.
3. **Both annuli** `cldice_classes=[3,4]` and `topo_classes=[3,4]` — the mitral
   annulus (3) is also ring-like; test whether the benefit generalizes.
4. **Loss base** `dice_focal_cldice` — pair clDice with a focal base for the
   class-imbalanced thin structures.

## Metrics (reported per arm on val during training, and on the frozen test set)

Voxel-overlap (existing):
- `dice/aortic_valve_annulus`, `hd95/aortic_valve_annulus`

Topology (new, `mvseg.metrics`, enabled by `topo_classes`):
- `topo/aortic_valve_annulus/betti0_err` — |#components(pred) − #components(GT)|,
  26-connectivity. **Primary topology metric** (0 = same number of pieces as GT).
- `topo/aortic_valve_annulus/n_comp_pred` — mean predicted component count (ideal 1).
- `topo/aortic_valve_annulus/connected_rate` — fraction of cases matching GT
  connectivity (**higher = fewer broken rings**).
- `cldice/aortic_valve_annulus` — hard skeleton-based clDice (**higher better**).

## Success criteria

Arm B beats Arm A on **`betti0_err` ↓ and `connected_rate` ↑ and `cldice` ↑**,
while `dice`/`hd95` on class 4 are **within noise** (not worse by more than a
small margin). Prefer the smallest `λ_cldice` that achieves this.

## Evaluation

```bash
# Frozen test-set report for a trained checkpoint (topo metrics via model.topo_classes)
python -m mvseg.evaluate ckpt_path=outputs/resunet_cldice/<run>/checkpoints/best.ckpt \
    model.topo_classes=[4]
```

## Practical notes / pitfalls

- **Warm-up.** The soft skeleton of noisy early predictions is meaningless. If
  training is unstable, start with `lambda_cldice` small (0.25) or warm up (train
  a few epochs on DiceCE, then resume with the clDice variant — the term is
  purely additive so a mid-training switch is safe).
- **Patch sampling.** With `patch_based=true`, `RandCropByPosNegLabeld` samples
  around *any* foreground, so many patches may lack class 4 → the clDice term is
  skipped for those (guarded by `smooth`). If AV coverage is too sparse, consider
  whole-volume training (256³ inference fits on a 16 GB P100; see README) or a
  class-4-biased crop.
- **Determinism.** `max_pool3d` backward has no deterministic CUDA kernel; the
  repo runs `use_deterministic_algorithms(warn_only=True)`, so runs will emit a
  warning but not crash. clDice runs are reproducible up to that op.
- **Cost.** clDice adds ~`iters`×4 pooling passes on a single channel per step;
  negligible vs. the UNet. Topology metrics run only at val/test.
