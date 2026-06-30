# Phase 0 — Blueprint Validation & Lock

**Status:** complete, awaiting art-direction approval before Phase 1.
**Inputs validated against:** the four committed plates in `assets/reference/` (`front`, `back`, `side_left`, `side_right`), each 1536×2730 RGBA.
**Method:** every `[O]` claim was re-checked by *looking at the pixels* (full plates + zoomed wrist/belt crops) and, where the blueprint gave hard ratios, by measuring the alpha silhouette in Python. Confidence legend: `[O]` Observed/canon · `[H]` High inference · `[M]` Medium · `[L]` Low.

---

## 1. Headline correction (canon-affecting)

> **The wristwatch is on the character's LEFT wrist, not the right.** `[O]`

The blueprint locked the watch to the character-**right** wrist (attributes #25/#26, assumption A3). The plates say otherwise, conclusively:

| Plate | Watch appears on | Anatomical side |
| --- | --- | --- |
| Front | viewer-**right** | character-left |
| Back | viewer-**left** | character-left |
| side_right (¾) | **near** arm (visible) | character-left |
| side_left (profile) | **absent** on near arm | (it's on the far/left arm) |

An item that sits viewer-right in front **and** viewer-left in back is, by mirror vs. non-mirror geometry, a **left-side** item. The blueprint's own cross-view method was right; its premise ("viewer-left in both front and back") was simply misread. This propagates to attribute #25, #26, A3, and the §9 counter-rotation note (the watch must counter-rotate with **left**-forearm pronation).

---

## 2. Second critical finding: the plates are not rigidly orthographic

> `side_right.png` shows a **frontal lower torso** (centered belt buckle + chino fly) together with a **profile head** pointing image-right and the **left-wrist watch on the near arm** — a combination a rigid 90° turn cannot produce. `[O]`

These references are stylized/AI-generated, so **pixel-exact bilateral correspondence between plates cannot be assumed.** Practical consequences:

- There is **no clean orthographic right-profile plate.** `side_left` is a left-facing profile; `side_right` is a right-facing ¾ hybrid. This expands risk **R1**.
- Cutout (Phase 1) and rigging (Phase 3) must **reconcile** minor per-plate disagreements rather than trust them blindly. Silhouette wins on conflict (blueprint §7).

---

## 3. Confirmations (blueprint held up)

- **Sleeves pushed up to below the elbow, bare forearms — in *all four* plates.** `[O]` (Verified via wrist crops. The blueprint said "front + ¾"; it's actually universal, and the watch sits on bare skin.)
- **Palette** — warm-brown hair, light warm-beige skin, dark olive-charcoal (not black) top, tan chino, dark belt, cream sneakers, black low-gloss watch, warm-brown linework. `[O]`
- **Posture** — relaxed A-stance, head set slightly forward of shoulders (cervical-forward, clear in both profiles), even weight. `[O]`
- **Lower-body detail** — single ankle cuff, two rear welt pockets, center-back belt loops, front rectangular single-prong buckle. `[O]`
- **Profile anatomy** — posterior calf belly tapering to a slim ankle. `[O]`
- **Hair** — spiky-layered crown loft, longest bangs to brow, high clean nape, rear-crown whorl. `[O]`

---

## 4. Measured proportions (Front plate, pixels)

| Landmark | Measured | Derived |
| --- | --- | --- |
| Figure bbox | y[284–2682] = **2398 px** tall, **694 px** wide | — |
| Head (crown→chin) | crown y≈284, chin y≈595 → **311 px** | — |
| **Total height** | 2398 / 311 | **≈ 7.71 HH** |
| Shoulder span | ≈ **530 px** | ≈ 1.70 HH (≈2.3 HW only if HW≈0.74 HH) |
| Neck (narrowest) | ≈ **126 px** | ≈ 0.45–0.55 HW (blueprint said 0.62 → revise toward measured) |
| Widest span (hands at thigh) | ≈ 690 px | (arms, not shoulders) |

**Takeaways:** the **7.75 HH default is well-supported** (measured 7.71); the 8.0 HH alternative is weakened (resolves **R7** toward 7.75). Neck reads slightly thinner than the blueprint's 0.62 HW. Shoulder-in-HW is highly sensitive to how cheekbone width is defined, so the spec records the firmer **HH** figure alongside it.

---

## 5. Open questions for art direction (please answer before Phase 1)

| # | Question | Priority |
| --- | --- | --- |
| **Q1** | Approve the canon correction: **watch on the character-LEFT wrist**? | High |
| **R1** | No clean right-ortho exists and `side_right` is a non-rigid ¾. OK to build the right side by **bilateral mirror of the left** until a true right plate is supplied? | High |
| **R6** | Confirm exact **iris hue** (reads warm amber/brown) and whether a catchlight is canon. | Medium |
| **R7** | Lock total height at **7.75 HH** (measurement supports it)? | Medium |
| **Q-name** | Character **working name** for asset naming? | Low |
| **R3** | Belt tail length/keeper unseen — default **short tuck** OK? | Low |
| **R9** | Reserve spare **teeth/tongue** layers for future lip-sync? | Low |

---

## 6. Self-QA checklist

- [x] All four plates inspected at full res + zoomed crops.
- [x] Every blueprint `[O]` tag re-checked against pixels; contradictions logged (watch side; orthographic inconsistency; neck width).
- [x] Hard ratios measured, not estimated (height, shoulder, neck, head).
- [x] No locked attribute silently changed — the one correction (watch side) is flagged explicitly for approval.
- [x] `CHARACTER_SPEC v1` emitted as machine-readable source of truth (`docs/character_spec_v1.json`, validates as JSON).
- [ ] **Art-direction sign-off on §5 questions** — *blocking Phase 1.*

**Next:** on approval, Phase 1 = auto-segment each plate into the independent art-mesh parts and inpaint all hidden-geometry/overdraw reserves. **Stopping here for your review.**
