# Milestone 1 — The Living Head (2.5D gaze · breath · blink)

> **TL;DR** We turned the flat front illustration of the character into a **2.5D living head** that follows the visitor's cursor, breathes, and blinks — while staying *pixel-for-pixel on-model* and **never tearing**. It runs in the browser as a dependency-free Canvas engine (`web_runtime/index.html`) and every motion was verified headlessly before shipping.

This is Milestone 1 of the narrator avatar that will host recruiters through the portfolio. It is deliberately scoped to the *head*, because the gaze-follow + breathing + blink is the beating heart of "someone is looking at you," and it is the hardest thing to get right without the character coming apart.

---

## Background

### For the newcomer: what "rigging a 2D character" even means

A flat drawing can't move. To animate one, you slice it into **layers** (hair, face, eyes…), then move each layer with a little matrix of numbers every frame — shift it, rotate it, squash it. Do this well and a 2D drawing appears to turn its head in space; this is the trick behind every VTuber and every Live2D "living illustration."

> **Callout — the cardinal sin: *tearing***
> If you move a layer and there's *nothing real drawn behind it*, a hole opens — a seam, a gap, a floating limb. This is "tearing," and it is the single thing that destroys the illusion of a living character. Avoiding it is the whole game.

### What existed before this milestone

The repository had already gone down two roads, both of which dead-ended:

1. **A per-triangle mesh warp** (Phases 2–8). It sliced the character into 137 auto-generated pieces and warped each with a fine triangle mesh. Its own PR notes admit the result: *"PiecewiseAffine warp shows faint grid/seam artifacts… on the hairline under extreme poses."* It **tore**.
2. **A "clean runtime"** (`web_runtime` on PR #12) that swung the other way — it drew whole, un-warped parts so nothing could tear, but to stay safe it *barely moved*. The bangs were baked into the face so nothing could parallax, and the back hair was a **blurred fake blob**. It was seamless but **lifeless and flat**, a cardboard cut-out being tilted.

The lesson sitting in that history is the design brief for Milestone 1:

> **Per-triangle warp = tears. Whole-stiff-parts = lifeless. We need the third thing: smooth, dimensional, human motion that still never tears.**

---

## Intuition

Two ideas do almost all the work.

### 1. Fidelity-first: cut as little as possible

Decomposing a flat, AI-rendered anime face into clean animatable pieces is genuinely hard — colour-thresholding the eyes just harvests eye-socket shadow and produces dark rings; filling under the hair with classical inpainting smears mud across the forehead. Every attempt to aggressively decompose **degraded the character** (washed skin, halos, raccoon eyes).

So we invert the instinct. **Keep the head whole and perfect**, and lift out *only* the two tiny things that must move independently:

- the **iris** (for gaze), drawn slightly *larger* than the real iris so that when it slides a few pixels it always covers the baked-in iris underneath — no double-iris, no hole;
- a **duplicate of the front hair**, laid back exactly over the identical baked hair, so it can drift for parallax while, at rest, being invisible.

Everything else — eyes, brows, nose, mouth, the hair itself — stays baked into one intact `head` sprite. The neutral pose is therefore *the original drawing*, untouched.

### 2. Depth from parallax, not from warping

You don't need to bend the artwork to make a head look like it's turning in 3D. You just need the layers at different "depths" to slide by *different amounts*.

> **Callout — parallax, with toy numbers**
> When he looks right (`yaw = +1`), the front hair shifts `+30px`, the face shifts `+24px`, the iris shifts `+31px`, and the back hair only `+16px`. Because the near things move more than the far things, your eye reads *rotation in depth* — even though every layer is a rigid, un-warped sprite that physically cannot tear.

Add a gentle horizontal squash on the face during a turn (foreshortening), a breath sine on the body, a spring on the head so it *settles* instead of snapping, eyes that lead the head, and blinks on a natural random timer — and a stack of flat sprites becomes a person.

### The blink and the neck — two traps

- **Blink**: we can't squash a baked-in eye. Instead a small **skin-coloured lid**, sampled from his own eyelid tone, is drawn descending over each eye. A real-skin shutter for ~120 ms reads instantly as a blink and can never tear.
- **The neck**: the head must turn but the neck must not slide. So the `head` sprite is **faded out at the jaw**, and the `body` layer (which contains the very same neck pixels) provides the neck. The head rotates above a neck that stays put — no ledge, no seam.

---

## Code

Everything is reproducible: `python3 tools/m1_build.py` cuts the layers, `python3 tools/m1_verify.py` renders the proof sheets, `python3 tools/m1_gif.py` renders the motion GIF, and `web_runtime/index.html` is the shippable engine.

### 1. Cutting the layers — `tools/m1_build.py`

The head is kept whole; only its neck edge is faded so it blends into the body's neck on a turn:

```python
# HEAD — fully intact original (perfect fidelity); bottom (neck) edge faded so there is NO
# hard line over the body when the head turns (the body holds the same neck pixels beneath)
fa = cv2.GaussianBlur(headSil.astype(np.uint8)*255, (0,0), 1.2).astype(np.float32)
ramp = np.ones((H,1),np.float32); y0r,y1r = 628,688
ys = np.arange(y0r,y1r); ramp[y0r:y1r,0] = 1-(ys-y0r)/(y1r-y0r); ramp[y1r:,0] = 0
fa = (fa*ramp).astype(np.uint8)
```

Hair is segmented by *region*, not by a single colour rule — the trick that finally kept the warm, orange-lit strands as hair instead of mistaking them for skin:

```python
faceOval  = ell((779,548),128,150)                     # protect the face
hair_core  = headSil & (V<128)&(RG<34)&(warmth>=8)      # reliable dark hair
hair_frame = headSil & ~skinBright & ~faceOval & (V<215)# framing hair outside the face (incl. lit tips)
hair_bangs = headSil & faceOval & (V<122)&(RG<40)       # only dark strands allowed across the face
hairM = (hair_core | hair_frame | hair_bangs) & ~eyeExcl
```

The iris is cut *enlarged* so gaze offsets always cover the baked iris:

```python
irisR_c, irisL_c = (722,489),(837,489); iris_r = 19     # enlarged: covers the baked iris on gaze move
```

### 2. The engine — `web_runtime/index.html`

The head transform is a single about-pivot affine (rotate · squash · shear · translate), computed once per frame:

```js
function headParams(){ return {
  rot: 0.05*S.yaw + 0.02*S.pitch,   sx: 1 - 0.12*Math.abs(S.yaw),  // foreshorten on turn
  sy: 1 + 0.05*S.pitch,             shear: 0.05*S.yaw,
  tx: 24*S.yaw,                     ty: 26*S.pitch + 2.4*S.breath }; }
```

Depth is just different translation gains per layer (back hair least, front hair most, iris most-plus-gaze):

```js
drawPart("back_hair", N[0],N[1], 0.05*S.yaw, 1-0.10*Math.abs(S.yaw),1, 0.04*S.yaw, 16*S.yaw, bob*0.7);
drawPart("head",      N[0],N[1], hp.rot,hp.sx,hp.sy,hp.shear, hp.tx, hp.ty);
drawPart("iris",      N[0],N[1], hp.rot,hp.sx,hp.sy,hp.shear, hp.tx+7*S.yaw+4.5*gx, hp.ty+4*S.pitch+3.5*gy);
drawPart("front_hair",N[0],N[1], hp.rot+0.03*S.yaw, hp.sx,hp.sy,hp.shear, hp.tx+6*S.yaw, bob*0.5+...);
```

Life comes from a spring (secondary settle), eyes leading the head, idle drift, saccades, and a randomised blink timer:

```js
const k=90, damp=15;                                   // critically-damped-ish head spring
S.vyaw += (k*(S.tYaw-S.yaw) - damp*S.vyaw)*dt; S.yaw += S.vyaw*dt;
S.gx   += (S.tYaw-S.gx)*0.24;                           // eyes lead the head
nextBlink -= dt; if(nextBlink<=0){ blinkT=0.13; nextBlink=2.2+Math.random()*3.8; }
```

### 3. The verifier — `tools/m1_verify.py`

Crucially, the Python renderer mirrors the JS math 1:1, so we can render extreme poses **before** shipping and prove no seam opens.

---

## Verification

This milestone was verified three ways, and every fix in it was *found* by verification (the eye-ring, the mud band, the neck ledge all showed up in a proof sheet and were driven to zero):

1. **Pose sweep** (`Character/_reports/m1/posesweep.png`) — neutral, look L/R/up/down, blink, gaze-only, breath. Neutral is indistinguishable from the source art.
2. **Torture test** (`Character/_reports/m1/torture.png`) — `yaw ±1`, `pitch ±1`, combined extremes with blink. No holes, no seams, no dark band.
3. **Real headless browser** — the shipped `web_runtime/index.html` was driven with Playwright/Chromium, the cursor scripted across the viewport, and screenshots + a GIF captured (`Character/_reports/m1/preview_web.gif`). This proves the *engine*, not just the Python mirror.

> **How to QA it yourself**
> 1. `cd web_runtime && python3 -m http.server 8000`, open `http://localhost:8000`.
> 2. Move your cursor around — his eyes lead, his head follows and settles.
> 3. Leave the cursor still — he idles, drifts, and glances on his own.
> 4. Watch for a few seconds — he breathes and blinks on a natural rhythm.
> 5. Whip the cursor corner to corner — confirm **nothing tears**, no seam opens at the hairline or neck.
> 6. Reproduce assets from scratch: `python3 tools/m1_build.py && python3 tools/m1_verify.py`.

---

## Alternatives

Two other roads could reach a living head; here's why we chose this one for Milestone 1.

| | **Chosen: fidelity-first parallax (Canvas)** | **Alternative A: per-vertex mesh warp (WebGL/Live2D)** |
|---|---|---|
| **Pros** | On-model by construction; cannot tear; tiny & dependency-free; trivial to embed | True foreshortening; larger turn range; the industry-standard look |
| **Cons** | Limited turn range; foreshortening is faked, not real | Needs hand-drawn keyforms + clean layered art; the exact thing that tore before |

| | **Chosen: fidelity-first parallax (2.5D)** | **Alternative B: toon-shaded 3D (Meshy → three.js)** |
|---|---|---|
| **Pros** | Keeps his precise 2D anime look 100%; light | Cannot tear at all; free head rotation; real human articulation |
| **Cons** | Not "true" 3D depth | Image-to-3D **softens/genericises the anime face** — risks not looking like *him*; heavier asset |

Both remain open as later options; the reference `Frosted_Aurora` biped in the repo is exactly the seed for Alternative B if we ever want full 3D.

---

## Suggested people to talk to

This part of the codebase is unusual: the entire rigging lineage (Phases 0–8, the prior clean runtime, and this milestone) was **authored by the AI agent with Naveen as art director**, so there isn't a bench of human engineers with deep context to consult.

- **Naveen T.S** — the owner, art director, and the one voice that matters on whether he looks/moves *right*. Every fidelity and motion decision here should be checked against his eye.
- For context on the road not to repeat, read the **PR #12 "clean runtime"** notes — they document precisely why the whole-stiff-parts and per-triangle-warp approaches stalled; this milestone is the reply to that.

---

## Quiz

<details>
<summary><strong>1. Why is the front hair shipped as a *duplicate* laid over the head instead of being cut out of it?</strong></summary>

- **A.** To save file size.
- **B. ✓ So it can parallax for depth while, at rest, sitting over the identical baked hair — giving depth with zero fidelity loss and no gap to reveal.**
- **C.** Because Canvas can't draw the same image twice.
- **D.** To make blinking work.

*Cutting the hair out would force us to fill the scalp underneath, which is exactly the mud-smear/halo problem that degraded earlier attempts. A duplicate avoids all of it.*
</details>

<details>
<summary><strong>2. What makes the head read as turning in 3D even though no layer is warped per-vertex?</strong></summary>

- **A.** A WebGL cylinder shader.
- **B.** The image is bent along a mesh.
- **C. ✓ Parallax — near layers (front hair, iris) translate more than far layers (face, back hair) — plus a horizontal squash on the face.**
- **D.** The head is a 3D model.

*Different translation gains per depth are what your visual system reads as rotation. It's an illusion built from rigid sprites, which is why it can't tear.*
</details>

<details>
<summary><strong>3. Why is the iris sprite cut *larger* than the actual iris?</strong></summary>

- **A.** To make the eyes look bigger.
- **B. ✓ So that when it slides for gaze, it always fully covers the baked-in iris beneath — preventing a "double iris" crescent.**
- **C.** For anti-aliasing.
- **D.** It isn't; it's the exact size.

*The baked head still contains the original iris. A slightly oversized moving iris guarantees coverage over the small gaze range, so no artifact appears.*
</details>

<details>
<summary><strong>4. Why is the head layer faded to transparent at the jaw?</strong></summary>

- **A.** To create a vignette.
- **B.** To hide the mouth.
- **C. ✓ So the neck comes entirely from the (static) body layer — the turning head no longer drags a hard neck edge across it, eliminating the ledge/seam.**
- **D.** To make him look down.

*The body layer contains the same neck pixels, so fading the head there is invisible at rest and seamless during a turn.*
</details>

<details>
<summary><strong>5. How do we know the engine — not just the Python preview — actually works and doesn't tear?</strong></summary>

- **A.** The code was reviewed.
- **B.** The reports say PASS.
- **C. ✓ The shipped `web_runtime/index.html` was driven in a real headless Chromium (Playwright), cursor scripted across the viewport, and screenshots + a GIF captured at extreme poses.**
- **D.** It was tested on a phone.

*A recurring failure of the earlier phases was self-grading JSON that "passed" while the visible result was broken. Here the real browser is the judge.*
</details>
