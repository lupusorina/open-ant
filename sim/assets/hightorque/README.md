# High Torque Robotics Mini Pi+ Pro

23-DoF humanoid, 14.72 kg, 0.367 m standing base height.
Product page: https://www.hightorquerobotics.com/mini-pi

## Provenance

Supplied by High Torque as `PiPlus_S_12L8A0G2H1W_LSE_260611`
(`PiPlus_S_12L8A0G2H1W_LSE_260611_with_armature.xml` plus `meshes/`). The code
in the name is the joint census: 12 leg, 8 arm, 0 gripper, 2 head, 1 waist.
No licence file accompanied the drop.

## What changed from the vendor file

`mini_pi.xml` is the vendor MJCF with five edits, all mechanical:

1. **The scene came out.** The vendor file carries its own floor plane, light,
   skybox and `<visual>`/`<statistic>` block. `ArenaBuilder` supplies those, and
   a second ground plane z-fights with the arena's. They now live in
   `scene.xml`, which `<include>`s the robot — the layout `booster_t1/` uses.
2. **The six floating-base motors came out.** The vendor block drove
   `floating_base_joint` with an `fx fy fz tx ty tz` wrench. That is a test
   harness: a free-floating robot in an arena has no such actuator, and the six
   would show up as extra `ctrl` slots.
3. **`meshdir` is `meshes`,** not `../meshes/` — the meshes sit beside the file
   here, not one level up as they did in the drop.
4. **Motor `ctrlrange` was aligned to the joint's own `actuatorfrcrange`,** per
   joint, taking the smaller of the two. The vendor authored them
   inconsistently: legs ±20 N·m under a ±21 joint rating, the waist ±30 under
   ±16, the arms ±20 under ±10. MuJoCo enforces the joint rating regardless, so
   the pair disagreeing only hides the real budget. `assets/manifests/mini_pi.yaml`
   carries the same numbers as `torque_limits`.
5. **A `home` keyframe was added.** The vendor file has none; see below.
6. **The visual meshes were decimated** from 1.23 M triangles to 178 k. See
   below.

Quote characters in `class='leg_motor'` attributes were normalised to double
quotes, and the model was renamed from the part number to `mini_pi_plus_pro`.

## The `home` stance

A 0.25 rad symmetric crouch (hip pitch −0.25, knee +0.50, ankle pitch −0.25
keeps the sole flat) with the arms hanging but abducted 0.20 rad at the
shoulder roll, base at z = 0.367 so the soles rest on z = 0.

Vendor joint names carry the segment, not the axis: `*_thigh_joint` is hip yaw
(axis z) and `*_calf_joint` is the knee (axis y).

The abduction is not cosmetic. At the vendor's all-zero pose the 0.03 m hand
spheres on `*_wrist_link` sit ~0.1 mm inside the thigh cylinders; 0.05 rad of
shoulder roll clears them and 0.20 leaves margin. Nothing resets to the all-zero
pose in normal use — `MdpEnv` writes the manifest `default_pose` — so this is
recorded rather than worked around.

## Where the PD gains came from

High Torque ships no trained controller with the model, so unlike
`assets/manifests/t1.yaml` there is no vendor RL config to copy. The manifest
gains are the smallest round values that hold `home` for 10 s under gravity,
measured on this MJCF with `install_pd_servos` at the training sim settings
(`sim_dt` 0.02/3, 30 iterations, `implicitfast`):

| hip/knee kp | kd  | 10 s base z | knee sag | peak τ |
|------------:|----:|------------:|---------:|-------:|
| 60          | 3.0 | 0.086 (down at 1.7 s) | — | 100% |
| 80          | 3.5 | 0.366       | 0.031 rad | 14% |
| **100**     | **3.5** | **0.366** | **0.022 rad** | **12%** |
| 150         | 4.0 | 0.367       | 0.013 rad | 11% |
| 200         | 5.0 | 0.367       | 0.010 rad | 10% |

Below kp 80 an open-loop PD hold topples backwards over the heels — a balance
failure, not a numerical one: the CoM starts 9 mm ahead of the ankles with
67 mm of rear margin and walks out of the support polygon as the hips and
ankles sag. kp 100 was taken because its 0.022 rad knee sag is already under
the 0.030 the T1 manifest accepted, and ±10% (the `randomize_pd_gains` band)
still stands.

The measurement agrees with the vendor precedent: T1's hips are 200 kp at a
45 N·m rating, i.e. 4.4 kp per N·m, which at this robot's 20 N·m would give 89.
Ankles use T1's softer ratio (2.5 kp per N·m → 50), and arms and head scale the
same way off their own ratings.

## Known gaps

- `head_yaw_joint` and `head_pitch_joint` carry no `class`, so they alone have
  `armature="0"` — the vendor gave every other joint a rotor inertia. Left as
  authored: at the manifest's kp 10 the head tracks to 0.008 rad and never
  approaches its 3.6 N·m limit, so nothing yet depends on filling it in.
- The 28 mesh geoms are visual only (`contype=0 conaffinity=0 density=0`); all
  collision is primitive boxes, cylinders and spheres.
- There is no separate hand body — the forearm ends in a fixed `*_wrist_link` —
  so the wrist link is the hand for contact purposes and its 0.03 m sphere is
  the fist geom.

## Why the meshes were decimated

As supplied, the 28 visual meshes were 1.23 M triangles / 59 MB, compiling to a
**226 MB `MjModel`** against the Booster T1's 30 MB — seven times the weight of
any other robot here, for detail only a renderer sees.

They are now 178 k triangles / 8.7 MB, compiling to 36.9 MB — the T1's weight
class. Total mass is bit-identical at 14.71826 kg, because the mesh geoms carry
`density="0"` and every body has an explicit `<inertial>`, so triangle count
cannot touch the physics. The face budget is split across meshes by surface
area rather than capped per mesh: a flat cap starved the torso and visibly
flattened its curvature.

This is about repository weight and render cost. What made the size urgent was
training — a job randomizing model fields keeps one `MjModel` per environment,
so 512 envs wanted 113 GB and never reached iteration 1 — but that is now fixed
properly, by not building the visual assets into a scene nothing draws
(`ArenaBuilder(visual=False)` + `strip_visual_assets()`, see #319). Training
this robot at 512 envs holds 1.4 GB whatever the meshes weigh.

Reproduce from the vendor drop with any quadric decimator; the one used was
`fast-simplification`, installed into the worktree venv for the one-off pass and
deliberately NOT added to `pyproject.toml` — nothing at runtime reads it.
