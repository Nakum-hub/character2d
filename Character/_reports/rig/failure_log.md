# Phase 4 Failure Analysis (Section 9)

- **Shoulder collapse** — detect: deltoid flatten / armpit hole at raise>clamp → reduce: girdle+arm distribution + deltoid bulge warp + armpit plug + seam loop → _mitigated (clamp 130, plug present)_
- **Neck giraffe/pinch** — detect: stretch>3% or candy-wrapper on turn → reduce: clamp stretch<=3%, twist loop + base/top support rings → _clamped_
- **Hair clip vs eye** — detect: bang enters eye on tilt → reduce: collision insets + behind-brow alt bang + physics limit → _alt layer reserved (Phase2)_
- **Clothing penetration** — detect: hem enters belt on bend → reduce: hem slides over belt (separate warp) + twist clamp → _slide warp set_
- **Eye distortion** — detect: iris exits sclera on look → reduce: LookX/Y bounds so iris stays in white → _bounded (look frame vol 100.0%)_
- **Mouth corner tear** — detect: lip-corner thinning on wide smile → reduce: dense corner loops + cheek expansion + jaw coordination → _cheek warp engaged_
- **Face/hair boundary reveal (forehead)** — detect: gap at the hairline when bangs shift on head turn/tilt → reduce: composite the scalp/forehead plug behind the head (now positioned via the Phase-2 offset fix); honor sharedBoundaryLoops → _RESOLVED in render — scalp plug fills the reveal with skin; identity stays PASS (plug off at rest). Coarse body/neck plugs are defined in the rig but deferred to the runtime (meshed shapes + boundary welding) as they are rectangular in this offline build._