# Phase 5 Validation Report

## Identity-Lock (all params at default)
- SSIM(white) **0.9996**, interior residual **42px** -> **PASS**

## Must-pass cases (Section 14)
| Check | Result | Evidence |
|---|---|---|
| single_owner_no_dup | ✅ | 74 owned channels, 0 duplicates |
| rotZ_split_4seg | ✅ | chest/waist/neck/shoulder = {'chest': 0.4, 'waist': 0.3, 'neck': 0.2, 'shoulder': 0.1} sum 0.9999999999999999 |
| smile_plus_viseme | ✅ | LipCorners=0.8 JawOpen=0.42 coexist |
| blink_wide_maxwins | ✅ | lidUpper=0.3 (max-magnitude wins, not summed) |
| gaze_in_sclera | ✅ | irisX=0.8 <= 0.8 bound |
| breath_gesture_additive | ✅ | armRaise=0.7 while breathing (independent) |
| isolation_single | ✅ | P_Head_RotX moved only: ['P_Head_RotX'] |

## Family demos (resolved intent -> L0 -> rig -> render)
See `params_demo_montage.png` (Happy, Confident, BodyTurn, SmileTalk, Wave, Surprised).
