# Phase 7 Performance Validation

## Identity-Lock (Neutral)
- SSIM 0.9996, interior residual 42px -> **PASS**

## Matrix
| Check | Result | Evidence |
|---|---|---|
| identity_neutral | ✅ | SSIM 0.9996, interior 42px (Neutral=reference) |
| all_within_clamp | ✅ | every emotion resolves within Identity-Lock clamps |
| body_coherence | ✅ | each emotion writes >=3 body channels |
| emotional_clarity | ✅ | min pairwise emotion distance 0.171>0.05 (distinguishable) |
| eye_lead_gaze | ✅ | eyes reach target before head; head partial-follows |
| transition_smooth | ✅ | max smile delta 0.052/frame <0.06 (no snap) |
| opposite_neutral_routed | ✅ | Fear->Joy passes through Neutral (min 0.0) |
| firewall_surface_only | ✅ | all writes on Phase-5 surface (173 params), asserted |

## Emotion library coherence (body channels per emotion)
Joy:6, Confidence:4, Curiosity:3, Focus:4, Calmness:3, Excitement:3, Surprise:4, Embarrassment:3, Confusion:4, Determination:5, Pride:4, Concern:5, Disappointment:4, Fear:5, Thinking:4, Interest:4, Playfulness:3, Professional:4, Relaxed:4, Agreement:3

Renders: `emotion_montage.png` (Neutral/Joy/Confidence/Surprise/Thinking/Excitement). Curves: `performance_curves.png`.
