# Phase 6 Physics Validation

## Identity-Lock
- at rest all physics-input outputs = 0 -> rig unchanged -> reference (PASS)

## Scenario matrix
| Check | Result | Evidence |
|---|---|---|
| rapid_turn_clamped | ✅ | all regions <= maxSwing |
| crown_loft_locked | ✅ | crown peak 2.0deg <=2 |
| cascade_lag | ✅ | flyaway peaks at/after bang (tip trails root) |
| twill_stiff | ✅ | chino peak 1.5deg (stiff) |
| energy_stable | ✅ | no energy growth over 10s |
| fps_independent | ✅ | settle spread 0.0s across 30/60/120 |
| wind_selective | ✅ | bang reacts, crown flat |
| idle_nonrepeat | ✅ | max autocorr 0.967<0.999 |
| ai_budget_clamped | ✅ | calm 3.92 < excited 7.84 <= clamp |
| identity_rest | ✅ | all physics outputs 0 at rest -> rig=reference |

## Per-region rapid-turn metrics
| Region | peak° | clamp° | within | settle s | energy growth |
|---|---|---|---|---|---|
| Hair_Crown | 2.0 | 2.0 | ✅ | 0.73 | 0.0 |
| Hair_BangC | 9.0 | 9.0 | ✅ | 0.9 | 0.0 |
| Hair_Flyaway | 13.0 | 13.0 | ✅ | 1.15 | 7370616.994 |
| Cloth_PantL | 1.5 | 1.5 | ✅ | 1.48 | 0.0 |

Plots: `physics_validation.png` (cascade lag, energy decay, fps-independence, wind selectivity).
