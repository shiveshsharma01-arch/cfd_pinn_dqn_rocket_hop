# Blender rocket-hop animation

Open `rocket_hop_v04.blend` in Blender 5.2 and press Play. The 794-frame,
30 fps animation follows the nominal 26.414 s simulator trajectory. The
`ROCKET_RIG` vertical position and tracking-camera height come from
`../../flight_model/results/landing/final/dqn_ascent_guided_descent_nominal_trace.csv`.
The camera stays at the same horizontal offset and follows the rocket up and
down; it does not use a dramatic orbit or handheld motion.

The scene has a procedural rocket, launch pad, ground, sky, neutral daylight,
an actuator-linked exhaust cone and 18 small animated nozzle particles. The
exhaust is **illustrative**: the CFD dataset is 300 K air and does not model
fuel combustion or plume chemistry. The vehicle geometry is also illustrative.

Version 3 adds two procedural mountain ranges as visual altitude references
and a camera-fixed top-left panel. The panel shows engine phase, altitude in
metres, commanded inlet pressure in kPa, actuator opening as a percentage,
and fuel remaining in kilograms. Actuator opening is **not** measured thrust
percentage. The mountains are scenery, not terrain in the flight model.

Version 4 adds packed 1K PBR textures for ground, concrete, and mountains,
subtle roughness variation and seams on the rocket, and a daylight compositor
with restrained exhaust glow. Texture sources are [Poly Haven Dirt](https://polyhaven.com/a/dirt),
[Concrete Floor](https://polyhaven.com/a/concrete_floor), and [Rock 01](https://polyhaven.com/a/rock_01),
all [CC0](https://polyhaven.com/license). Source URLs and file sizes are in
`textures/asset_manifest.json`. Powered by Poly Haven. The `.blend` packs these
textures so it can be opened without the source image folder.

`rocket_hop_v04_draft.mp4` is a half-resolution, 10 fps review render of the full
flight. The final project remains 30 fps and 1200 × 900 pixels. Three PNG
previews show launch, ascent and touchdown. Camera framing and lighting are
draft choices for visual review, not physical validation. The vehicle and terrain
remain illustrative procedural geometry, so this is a textured simulation
visualization rather than documentary rocket footage.

The separate `rocket_hop_failed_step000000.blend` uses the saved **zero-step
DQN** simulator trace. That attempt reaches 621.20 m (24.24% above the 500 m
target) and contacts the ground at 75.11 m/s (1502.12% of the 5 m/s limit).
The tip-over and dust after contact are illustrative effects; the simulator
only records the flight up to impact. The failure scene is intended for a
side-by-side learning-progress presentation. The successful v04 scene uses
**DQN ascent plus guided descent**, so a comparison must not be described as
proof that the pure DQN learned to land. The failure trace is
`../../flight_model/results/landing/seed0_v3/step000000_trace.csv`.

To rebuild from the saved static pad and frozen flight trace:

```powershell
& 'C:\Program Files\Blender Foundation\Blender 5.2\blender.exe' --background rocket_pad_v01.blend --python animate_rocket_pad.py
& 'C:\Program Files\Blender Foundation\Blender 5.2\blender.exe' --background rocket_hop_v02.blend --python enhance_rocket_hop.py
& 'C:\Program Files\Blender Foundation\Blender 5.2\blender.exe' --background rocket_hop_v03.blend --python texture_and_composite.py
& 'C:\Program Files\Blender Foundation\Blender 5.2\blender.exe' --background rocket_hop_v04.blend --python render_draft.py
python make_draft_video.py
& 'C:\Program Files\Blender Foundation\Blender 5.2\blender.exe' --background rocket_hop_v04.blend --python build_failed_early_dqn.py
& 'C:\Program Files\Blender Foundation\Blender 5.2\blender.exe' --background rocket_hop_failed_step000000.blend --python render_failed_draft.py
python make_failed_video.py
& 'C:\Program Files\Blender Foundation\Blender 5.2\blender.exe' --background rocket_hop_v02.blend --python verify_rocket_scene.py
```

The `draft_frames` and `render_frames` folders are intermediate render output.
