# Phases 1–8 — Audit & Improvements

A cross-phase review of the DHPF stack (`art → mesh → rig → parameters → physics → performance → animation`) to find and fix real issues, not re-assert green checks.

## Cross-stack consistency — ✅ 13/13 (see `stack_audit.md`, `tools/audit_stack.py`)
Every cross-reference resolves and every firewall holds:
- mesh parts → layers; rig params → existing deformers; params L0 `owns` → existing deformer channels; all param `writes` resolve; **single-owner** intact (85 channels, 0 duplicates).
- **Firewalls:** physics writes only physics-input params; performance & animation write only the Phase-5 surface (0 off-surface writes each).
- Emotion body-coherence (≥3 channels) and animation 11-point traceability (22 entries) both complete.

## Issues found and fixed

| # | Issue | Severity | Fix |
|---|---|---|---|
| 1 | **Phase-1 CHARACTER SPEC absent from the repo** — the identity-lock foundation every phase (3–8) cites lived only on the closed Phase-0 branch. | High (missing foundation) | Restored as `Character/character_spec.json` (canonical, machine-readable), reconciled with the Phase-0 pixel validation. |
| 2 | **A2 height decision open in all 8 phases** ("7.75 vs 8.0 HH — blocks amplitude calibration"). | Medium (perpetually carried) | **Resolved to 7.75 HH** — Phase-0 measurement was 7.71 HH (figure 2398px / head 311px), which supports 7.75. Body-language/gesture amplitude calibration is now unblocked. |
| 3 | **Render engine couldn't show body posture/breathing** — the recurring caveat across Phases 4–8 ("renders read mainly on face/arms; body verified only numerically"). | Medium (validation credibility) | Added identity-safe body-posture deformation (chest openness, body lean, head-forward, both-arm raise + twist). Confident/Pride now visibly open the chest; Presenting extends the arm; Celebrate raises both arms. **Identity-Lock still PASS (SSIM 0.9996, 42px).** See `fullstack_showcase.png`. |

## Corrections already carried from earlier per-phase audits (confirmed still holding)
- **Watch on character-LEFT wrist** (Phase-0 correction) — locked in the restored spec.
- **Phase-5 RotZ twist split 40/30/20/10** (was mis-gained) — verified in audit.
- **Phase-7 layered-mixer decay-to-baseline** (was collapsing to 0) — verified.
- **Phase-8 fps-independence** measured as drift-free sim-time — verified.

## Known, disclosed limitations (not defects; inherent to the offline sandbox)
- No `torch`/network → SAM/LaMa substituted by classical CV (Phase 2 = 137 layers vs the ~210 target; majors are clean cuts, micro-zones deferred to ML/manual).
- Offline PiecewiseAffine warp shows faint grid/seam artifacts on low-density cloth and at the hairline under extreme head-turn / both-arm poses; true welding + hidden-plug meshing is a runtime (Phase 9) concern.
- **A1** (no clean right orthographic plate) remains open by input constraint — right side mirrors the observed left, flagged for repaint.

## Verdict
The eight phases are internally consistent, firewall-clean, and identity-locked; the missing foundation and the long-carried height decision are now closed, and the body-posture rendering gap is fixed. Ready to proceed to Phase 9 (runtime) when approved.
