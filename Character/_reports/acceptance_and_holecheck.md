# Phase 2 — Acceptance & Hole-Check

## Identity-Lock acceptance (rest recomposite vs FRONT)
- Visible leaf parts: **79** form a disjoint partition (overlap 0) whose union == figure.
- **Figure pixel diff = 0** (exact) · SSIM over white = **0.99976** → **PASS**.
- Artifacts: `accept_recomposite_front.png`, `accept_diff_front.png` (near-black = identical).

## Hole-check (in-scope motions)
| Motion | Revealed px | Gap px | Verdict |
|---|---|---|---|
| arm_raise_R | 20259 | 0 | ✅ pass |
| head_turn | 5647 | 0 | ✅ pass |
| leg_lift_R | 18519 | 0 | ✅ pass |
| neck_tilt | 6711 | 0 | ✅ pass |
| cuff_expose | 3186 | 0 | ✅ pass |
| sleeve_twist_L | 1152 | 0 | ✅ pass |

Gaps are 0: every revealed region is backed by a painted hidden/alt asset or another layer behind it.
