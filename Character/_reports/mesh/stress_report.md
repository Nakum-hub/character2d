# Phase 3 Stress-Test Report

Total verts **10031** (target 9k-12k), tris **15918**, parts **129**.

Budget by region: {'misc': 1664, 'field': 2676, 'face': 565, 'feature': 790, 'hairedge': 786, 'rigid': 28, 'hand': 857, 'joint': 2665}

## Clamp angles (engineered range before first ✗)
- CHR_LEG_LEG_Upper_L: range 90°, clamp 30°
- CHR_LEG_LEG_Upper_R: range 90°, clamp 45°
- CHR_MID_NECK_Skin: range 30°, clamp 30°
- CHR_HEAD_FACE_Cheek_R: range 30°, clamp 30°
- CHR_HEAD_FACE_Cheek_L: range 30°, clamp 30°
- CHR_LEG_LEG_Lower_L: range 120°, clamp 40°
- CHR_ARM_SLEEVE_Upper_L: range 130°, clamp 60°
- CHR_ARM_SLEEVE_Lower_L: range 130°, clamp 60°
- CHR_ARM_FOREARM_L: range 90°, clamp 30°
- CHR_LEG_LEG_Lower_R: range 120°, clamp 40°
- CHR_ARM_SLEEVE_Upper_R: range 130°, clamp 15°
- CHR_ARM_SLEEVE_Lower_R: range 130°, clamp 90°
- CHR_ARM_FOREARM_R: range 90°, clamp 90°
- CHR_MID_NECK_Hidden_Column_hidden: range 30°, clamp 20°

Metric thresholds: integrity ✓ when 0 inverted triangles, volume retention 82-120%, and texture-stretch < 45%. Inversion/volume are the density-sensitive signals; stretch is largely geometric (material-dependent).
