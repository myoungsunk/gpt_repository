# RSS-DA Architecture Review

## Alignment with stated plan
- **Two-pass + LUPI implementation**: `Stage1Trainer.train_step` performs Pass-0 using `z5d` only, reconstructs privileged `r4_hat` via `DecoderD`, re-encodes with `E4`, fuses with `E5`, and feeds the detached adapter output `phi` into a second pass (`mu1`, `kappa1`), matching the proposed two-pass LUPI strategy.【F:rss_da/training/stage1.py†L263-L368】
- **LUPI scope respected at runtime**: Stage 2.5 dataloading drops `four_rss` when building samples so the student operates purely on `[z5d, φ]` without privileged inputs during adaptation.【F:rss_da/train.py†L26-L42】【F:rss_da/data/dataset.py†L55-L146】
- **Teacher-Student KD with EMA**: The Stage 2.5 trainer constructs an EMA teacher, applies temperature-scaled KD on both outputs and fused features, and uses kappa-based gating, consistent with the design notes.【F:rss_da/training/stage25.py†L443-L530】

## Strengths
- **Data standardization & fallbacks**: Training automatically loads standardized CSVs when present, caches scalers, and gracefully backs off to synthetic generation when missing—useful for iterative experimentation.【F:rss_da/train.py†L45-L136】
- **Robust stabilization toolkit**: Stage 1 and Stage 2.5 include stop-grad for `φ`, residual penalties, gate entropy regularization, warm-up schedules, and optional GradNorm/uncertainty weighting, providing multiple levers to prevent collapse and balance losses.【F:rss_da/training/stage1.py†L319-L368】【F:rss_da/training/stage25.py†L480-L505】

## Risks / Suggestions
1. **Domain adversarial alignment is under-specified**: When `use_dann`/`use_cdan` is enabled, `apply_alignment` only receives the student features with a single domain label (all zeros), so the discriminator never sees a contrasting source distribution. Consider concatenating teacher/source features with an alternate label or sampling cached LoS batches to provide the adversary with meaningful domain supervision.【F:rss_da/training/stage25.py†L457-L520】【F:rss_da/losses/align.py†L31-L88】
2. **Privileged mask handling in synthetic Stage-2.5 data**: Synthetic fallback data for Stage 2.5 zeros out `four_rss` and mask flags, but downstream code still passes the empty tensor through `_assemble_m3_features`. Double-check that this case is exercised in tests to avoid silent shape mismatches when no privileged data exist in real NLoS logs.【F:rss_da/train.py†L26-L42】【F:rss_da/training/stage25.py†L337-L370】
3. **Logging granularity for mix variance**: Reconstruction loss normalizes by the observed variance of `c_meas_rel`, yet Stage logs only the weighted term. Surfacing the unscaled `mix_raw` and variance in validation summaries would make diagnosing instabilities in the mix schedule easier, especially when switching between relative/absolute dB modes.【F:rss_da/losses/recon.py†L16-L53】【F:rss_da/train.py†L181-L215】

