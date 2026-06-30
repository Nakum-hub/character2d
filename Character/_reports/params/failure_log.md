# Phase 5 Failure Analysis (Section 15)

- **Parameter conflict (two owners)** — detect: isolation test moves unexpected part → avoid: single-owner rule enforced at build (assert); 85 channels each one L0 owner → _PASS_
- **Double transformation** — detect: composite+atomic both write geometry -> value doubles → avoid: composites/expressions/gestures write L0 ids only; geometry touched only by L0 → _PASS_
- **Circular dependency** — detect: mixer fails to converge → avoid: resolver visited-set prevents cycles; correctives (CounterRot, Balance) are one-way, applied last → _PASS_
- **Overlapping responsibility (Tilt vs RotZ)** — detect: redundant sliders drift → avoid: P_Head_Tilt is an alias that writes only P_Head_RotZ (canonical owner) → _PASS_
- **Unstable blending** — detect: jitter/overshoot on stacked inputs → avoid: clamp + critically-damped smoothing tau per param + priority order idle<phys<expr<gesture<ai<runtime → _PASS_
- **Viseme/emotion clash** — detect: mouth flickers talking+smiling → avoid: emotion writes LipCorners (signed-sum), visemes write JawOpen/Width additively -> coexist → _PASS (smile+viseme check True)_