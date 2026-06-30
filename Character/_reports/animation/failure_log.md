# Phase 8 Failure Analysis (Section 16)

- **Robotic timing** → ease-in/out + anticipation windup + follow-through (Wave shoulder dip before raise) → _EASED_
- **Abrupt transitions** → pose-match eased blend (max delta 0.0075/frame) → _POSE-MATCHED_
- **Animation conflicts** → channel ownership + priority arbitration (talking mouth overrides, body keeps idle) → _ARBITRATED_
- **Pose popping** → read current resolved pose, ease toward target start → _NO-POP_
- **Excessive looping** → variation curves + noise selection (conversation sets) → _VARIED_
- **Physics fighting animation** → procedural/physics own channels exclusively; animation never writes them → _OWNED_
- **Gesture repetition** → gesture variants + cooldown + randomized timing → _NON-REPEAT_
- **Dead idle** → always-on breathing + saccades + micros → _ALIVE_