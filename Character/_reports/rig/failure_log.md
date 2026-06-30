# Phase 4 Failure Analysis (Section 9)

- **Shoulder collapse** — detect: deltoid flatten / armpit hole at raise>clamp → reduce: girdle+arm distribution + deltoid bulge warp + armpit plug + seam loop → _mitigated (clamp 130, plug present)_
- **Neck giraffe/pinch** — detect: stretch>3% or candy-wrapper on turn → reduce: clamp stretch<=3%, twist loop + base/top support rings → _clamped_
- **Hair clip vs eye** — detect: bang enters eye on tilt → reduce: collision insets + behind-brow alt bang + physics limit → _alt layer reserved (Phase2)_
- **Clothing penetration** — detect: hem enters belt on bend → reduce: hem slides over belt (separate warp) + twist clamp → _slide warp set_
- **Eye distortion** — detect: iris exits sclera on look → reduce: LookX/Y bounds so iris stays in white → _bounded (look frame vol 99.9%)_
- **Mouth corner tear** — detect: lip-corner thinning on wide smile → reduce: dense corner loops + cheek expansion + jaw coordination → _cheek warp engaged_
- **Face/hair boundary reveal (forehead)** — detect: thin gap at the hairline when bangs shift on head turn/tilt/breath → reduce: composite the hidden scalp/forehead plug + honor Phase-3 sharedBoundaryLoops (FACE_Forehead<->bang, face<->neck) → _RIG CORRECT (scalp plug + boundary loops defined); VALIDATION-RENDER limitation: hidden plugs not composited because Phase-2 recorded x=y=0 for rgba-emitted parts. Fix: record real offsets for hidden/overlay parts in layers.json, then plugs fill the reveal._