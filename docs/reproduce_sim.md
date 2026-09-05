# Reproducing the simulated bottles-in-bin results

This document describes how to reproduce the paper's simulated bottles-in-bin
results: 512 paired scenes x 6 bottles, 60 s horizon, every curation arm
holding 31.5% of the data.

| Method | Data kept | Bottles/scene | Thrpt (/hr) | All 6 |
|---|---|---|---|---|
| Vanilla BC | 100% | 3.885 | 237 | 9.4% |
| Random | 31.5% | 3.770 | 230 | 10.9% |
| ReWiND | 31.5% | 3.781 | 231 | 9.4% |
| SARM (oracle) | 31.5% | 4.191 | 265 | 20.5% |
| WARP-BC (IID) | 31.5% | 4.285 | 269 | 18.9% |
| SCIZOR | 31.5% | 4.299 | 270 | 19.1% |
| DemInf | 31.5% | 4.332 | 271 | 18.8% |
| **WARP-BC** | 31.5% | **4.533** | **290** | **25.0%** |

SARM uses *oracle* stage boundaries from simulator state. All curation arms
share demonstrations, policy architecture and train-step budget; they differ
**only in which action chunks they train on**.

```
 (1) train WARP-RM     (2) score + inject      (3) curate + train pi0     (4) rollout + score
     this repo      ->     this repo        ->        openpi          ->     abc-rabc
```

## What each arm reproduces

Six arms run end to end from public artifacts. Two do not, and this document
says so rather than implying otherwise.

| arm | tier | what you can do |
|---|---|---|
| Vanilla BC | **full** | no curation; train directly |
| WARP-BC | **full** | train the RM, score, gate, train pi0 |
| WARP-BC (IID) | **full** | same, with `--sampler iid` |
| SCIZOR | **full** | sidecar parquet is published |
| Random | **full** † | trivially regenerable; the original RNG seed was not recorded, so you reproduce it *in distribution*, not bit-exactly |
| SARM (oracle) | **full** † | oracle boundaries are derivable from simulator state; the extraction is described below but was not shipped as a script |
| **ReWiND** | **verify-only** | policy checkpoint and traces are published, so you can verify the *policy*. The curation is **not** regenerable: our code loads a ReWiND reward model, it cannot train one, and the checkpoint used was a third party's. |
| **DemInf** | **verify-only** | policy checkpoint and traces are published. The MI/KSG scorer that produced the episode keep-list is not part of this release, so the keep-list is not regenerable here. |

For the two verify-only arms, the criterion and threshold are documented so the
selection is auditable, and the trained policy is published so the reported
number is checkable. Only the curation step is closed.

## Artifacts

| artifact | where |
|---|---|
| episodes | `uynitsuj/sim-bottles-mjwarp-v1` (HF dataset) |
| WARP-RM checkpoint | `uynitsuj/warp-rm-sim-bottles-sss15` |
| policy checkpoints | `uynitsuj/paper-sim-policy-checkpoints` — vanilla and WARP-BC published; the other six arms are being uploaded |
| rollout traces, n=512 | `uynitsuj/paper-sim-n512-traces` — vanilla + WARP-BC, 512 paired scenes each |
| rollout traces, n=128 | `uynitsuj/paper-sim-n128-traces` — same two arms, the deterministic replay set |
| dataset metadata | ships with the checkpoint release — **contains `object_counts.json`, which is not in this git repo** and which training needs |

| repo | branch | pin | job |
|---|---|---|---|
| [`uynitsuj/openpi`](https://github.com/uynitsuj/openpi) | `public/table1-repro` | `9c8e7b75483edd5c2dd5f403109d3b0f16549d60` | **trains** the six arms |
| [`uynitsuj/openpi`](https://github.com/uynitsuj/openpi) | `release-candidate` | `204eb92dd2af37c4d1189b587d5fbff978383930` | **serves** a checkpoint for rollout |
| [`uynitsuj/abc-rabc`](https://github.com/uynitsuj/abc-rabc) | `public/table1-eval` | `59db543d` | **batched** rollout + scoring |

## 1. Train the reward model

```bash
python scripts/train.py \
  --lerobot-repo <path-to>/sim-bottles-mjwarp-v1 \
  --feature-stride 1 --shortest-frac 0.25 --crop-mode squash \
  --object-counts-json object_counts.json \
  --source-standard-stride 15 \
  --tag warp_sss15
```

**`--source-standard-stride 15` sets the label calibration.** The sampler's
path budget stays at the fixed paper-experiment defaults: centre 1.5 s, half-range
1.0 s — a `[0.5, 2.5]` s band — and the C51 support stays fixed at ±3.0.
These ARE the values the reward model used for the reported results trained with; do not override
them when reproducing the table.

An earlier revision of this repo (2026-08-15, PR #2) derived the band from
SSS (`centre = SSS/fps`, `half = 2/3 × centre`) and auto-sized the C51
support, as defaults. Measured downstream at n=512 those defaults cost
−0.486 bottles/scene against the fixed values above (p=2e-08, paired, same
seed/recipe/eval), so PR #3 pinned the defaults back. Both derivations remain
available as explicit opt-ins (`--ar-center-stride-sec -1`,
`--ar-half-range-sec -1`, `--auto-bin-range`) — do not use them to reproduce the reported results.

Every run prints two audits. Check them before trusting a head:

```
[path-budget] centre=1.5000s (explicit) half=1.0000s (explicit)
              -> band [465, 2325] feat frames | standard window = 465
              | train n_feat p50=796 p90=1117
[label-audit] |label|: p50=0.516 p90=1.372 p99=2.099 p99.5=2.217 max=2.551
[label-audit] clamped by the +-3.0 support: 0.00% of labels
```

For the IID ablation, add `--sampler iid`. That swaps the AR(1) log-speed
process for an i.i.d. draw with the same marginal; reversal sampling and the
path budget are unchanged, so only temporal correlation differs.

> [!NOTE]
> **Frame geometry.** The published dataset is 640x480, not square. With
> `--crop-mode squash` (the default) frames are resized straight to 224x224,
> compressing the horizontal axis by 1.33x; `--crop-mode center` center-crops
> first. The released results use `squash`. Which is *better* was never measured
> end to end, so treat the default as a convention, not a finding — but do use
> the same mode for scoring as for training, since the gate thresholds the raw
> injected velocity and the crop therefore decides which chunks train.

## 2. Score and inject

```bash
python scripts/data/write_warp_rm_annotations.py \
  --checkpoint checkpoints/best_model_warp_sss15_no_abs.pt \
  --lerobot-repo <dataset-copy>
```

This writes `warp_rm_signed_magnitude` (and `warp_rm_progress`) into the
dataset copy in place. Copy `meta/` and `data/` and symlink `videos/` — the
annotation only touches parquet columns, and the feature cache keys off the
video path, so a symlinked copy reuses the existing cache.

Budget roughly **3.5 h per corpus pass** on one H100.

## 3. Curation is one mechanism, not eight configs

Every arm except vanilla selects a subset at a per-arm threshold. Three config
shapes cover all of them:

| arm | config | parameters |
|---|---|---|
| WARP-BC, IID | `LeRobotYamRormDataConfig` | `rabc_threshold`, `rabc_use_final_action_condition=True` |
| SCIZOR, Random, ReWiND | `LeRobotScizorSidecarDataConfig` | `scizor_sidecar_path`, `scizor_eps_s`, `scizor_weight_mode="binary"`, `scizor_score_column` |
| DemInf | `deminf_keep_episodes_path` | episode-level keep list |

### Thresholds

**Retention must be matched to 31.5%, and the denominator is not obvious.**
openpi treats *every frame* as a candidate anchor, so the denominator is all
**2,228,979 frames**, not the 2,158,277 full-window anchors. Tail anchors where
`offset+H > L` have their window padded with the episode's last velocity, so
their decision value is `vel[L-1]`. Calibrating against the truncated anchor
set sends thresholds the wrong direction.

| arm | criterion | threshold | kept |
|---|---|---|---|
| WARP-BC | chunk-final velocity | > 1.0 | 31.5% |
| WARP-BC (IID) | chunk-final velocity | > 1.047659 | 31.500% |
| SCIZOR | anchor suboptimality | <= 0.1245 | 31.500% (702,129) |
| DemInf | per-episode MI (KSG) | >= -7.1791 | 31.502% (850/2438 eps) |
| Random | U[0,1] per chunk | <= 0.315 | 31.497% |

Thresholds are **per-arm calibrated** because each scorer's distribution
differs. Applying WARP's 1.0 everywhere silently changes the data budget — the
IID head keeps 37.03% at thr=1.0, not 31.5%. Any arm whose
`[rabc_precompute] ... kept (NN.N%)` line does not read ~31.5% is not a matched
comparison and must be retrained.

**SARM (oracle)** selects chunks inside oracle stage boundaries taken from
simulator state rather than from a learned model. It is an upper-bound
reference, not a method you would have at deployment.

## 4. Train the policies

```bash
cd openpi          # public/table1-repro
HF_LEROBOT_HOME=<datasets> uv run scripts/train.py <config> \
  --exp-name <arm> --seed 0 --fsdp-devices 2 \
  --save-interval 30000 --keep-period 30000 \
  --checkpoint-base-dir <ckpts> --no-wandb-enabled
```

All arms: pi0, `action_horizon=30`, batch 32, 30,000 steps, cosine decay,
initialised from `gs://openpi-assets/checkpoints/pi0_base/params`. Equal
retention equalises effective epochs across arms.

> [!IMPORTANT]
> **Norm stats are keyed by `repo_id`**, resolved at
> `assets/<config_name>/<repo_id>/norm_stats.json`. A new `repo_id` has no
> stats and training dies with `Normalization stats not found`. Either run
> `scripts/compute_norm_stats.py`, or copy the existing stats to the new asset
> path — the statistics are computed from actions and state, which are
> identical across curation copies.

## 5. Rollout and score

```bash
# serve one arm (openpi @ release-candidate)
python scripts/serve_paper_sim_policy.py --checkpoint-dir <ckpts>/<arm>

# roll out (abc-rabc @ public/table1-eval — batched; release-candidate has
# only the serial eval_policy.py and cannot run this)
python batched_runner.py --seeds <512-scene set> --steps 1800
```

Control stack, fixed for every arm:

| parameter | value |
|---|---|
| action horizon predicted | 30 |
| actions executed per chunk | **30 — all of them** |
| replan | on chunk exhaustion only |
| sim timestep | 0.002 s |
| control decimation | 17 |
| control rate | 29.4 Hz |
| replan period | 1.02 s |
| rollout | 1800 steps = 61.2 s |
| bottles per scene | 6 |

Execution is fully open-loop in ~1 s blocks: no temporal ensembling and no
receding-horizon truncation.

Scenes are **paired across arms** — the same 512 world seeds for every arm — so
contrasts are paired t-tests, not independent samples.

## Limits of fresh repro

| step | deterministic? |
|---|---|
| RM training | no — weight init and window sampling are unseeded, so replicates differ |
| annotation | yes, given a fixed checkpoint |
| threshold calibration | yes |
| pi0 training | no — data order and JAX nondeterminism; no seed recorded |
| rollouts | no — policy sampling and physics are stochastic |

Compare with the paired n=512 protocol, not trace by trace. The published
n=128 traces plus `--self-test` are the deterministic artifact.

Measured seed spread, for calibration: RM composite scores replicate to about
±0.002 across seeds, so offline differences smaller than ~0.01 are noise. On
the policy side, three RM heads spanning a large offline range (fine-stride
velocity Spearman 0.60 to 0.84) produced downstream results that were
statistically indistinguishable at n=128 (p >= 0.68). **A better offline
reward model did not produce a better policy in that range** — worth knowing
before attributing a downstream difference to RM quality.

## Traps

1. `object_counts.json` is not in this repo. It ships in `ds_meta/`.
2. DINOv3 is a gated HF repo; export a token before RM training.
3. Norm stats are keyed by `repo_id` (see §4).
4. Use the same `--crop-mode` for scoring as for training.
5. Retention that is not ~31.5% means the arm is not budget-matched.
