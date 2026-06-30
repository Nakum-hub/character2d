# Phase 6 Failure Analysis (Section 15)

- **Hair explosion** — detect: peak 685.3deg >> clamp without guard · root: uncapped force/low damping · mitigation: velocity+energy+maxSwing clamp -> bounded to 9.0deg → _MITIGATED_
- **Infinite oscillation** — detect: energy grows over time · root: damping<=0 / fps feedback · mitigation: guaranteed positive damping + fixed timestep -> energy ratio 0.0<=1 → _MITIGATED_
- **Collision clipping (bang/eye)** — detect: bang enters eye on fast turn · root: inset too tight · mitigation: bang maxSwing clamp 8-9deg + Phase-3 inset + hard bang_eye priority → _CLAMPED_
- **Fabric collapse (twill)** — detect: folds invert on bend · root: over-compression · mitigation: twill high damping/low elasticity, maxSwing 1.5deg, honor Phase-3 rings → _STIFF/STABLE_
- **Idle jitter** — detect: shimmer at rest · root: no rest-snap · mitigation: rest-snap threshold zeroes sub-0.02deg → _SNAPPED_
- **Feedback loop** — detect: physics drives a driver that drives physics · root: circular input · mitigation: one-way drivers only (Phase-5), correctives last, one-frame buffer in cascade → _ONE-WAY_