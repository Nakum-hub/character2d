# Phase 7 Failure Analysis (Section 15)

- **Robotic expressions** — cause: discrete pose swap, no easing · detect: snaps between states · mitigation: transition engine eased crossfade (max delta 0.052/frame) + smoothing + micros → _SMOOTH_
- **Emotion mismatch** — cause: channels disagree · detect: smile + tense body · mitigation: every emotion writes >=3 body channels coherently (min 3) → _COHERENT_
- **Dead eyes** — cause: no saccade/blink variation · detect: static stare · mitigation: always-on micro_saccade + context blink scheduler → _ALIVE_
- **Overactive blinking** — cause: rate too high · detect: flutter · mitigation: context-driven rate + cooldown (micro scheduler rate-limit) → _RATE-LIMITED_
- **Frozen posture** — cause: emotion writes face only · detect: body still · mitigation: >=3 body channels per emotion enforced → _BODY-DRIVEN_
- **Abrupt emotional change** — cause: direct opposite morph · detect: visible pop · mitigation: opposite pairs routed via Neutral (Fear->Joy min smile 0.0) → _NEUTRAL-ROUTED_
- **Repetitive idle** — cause: looping micros · detect: visible pattern · mitigation: de-synced Phase-6 idle + micro cooldowns + randomization → _NON-REPEAT_