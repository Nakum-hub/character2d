# Phase 4 Validation Report

## Identity-Lock acceptance (all params at default)
- SSIM(white) **0.9996**; residual diff confined to AA edges (**7453** edge px vs **42** interior px) -> **PASS** (rest == reference; residual is Phase-2 edge-bleed, not rig drift)

## Scenario sweeps (Volume/Silhouette + frames)
| Scenario | Volume % | Verdict |
|---|---|---|
| look_R | 99.9 | PASS |
| blink | 99.8 | PASS |
| head_tilt | 98.7 | PASS |
| head_turn | 98.1 | PASS |
| smile | 99.9 | PASS |
| breathe | 97.7 | PASS |
| body_turn | 93.1 | PASS |
| arm_raise_R | 92.5 | PASS |

Frames: `valid_*.png`, `validation_montage.png`. Volume within 80-130% = no collapse/blowout under the tested travel.
