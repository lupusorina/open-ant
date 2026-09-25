"""Render close-up, high-quality still images of a MuJoCo scene (ant or crane).

Usage:
    python sim/snapshot.py                                  # ant, 5 angles
    python sim/snapshot.py --scene crane --azimuths 125 60
    python sim/snapshot.py --view sim/snapshots/view_xxx.json   # pose + camera saved by view.py

Quality tricks used:
  * large offscreen buffer (4K by default) + 2x supersampling, downscaled with Lanczos
  * MSAA (offsamples), high-res shadow map, reflections
  * depth / shadow ranges pinned to the scene size (no z-fighting, sharp shadows)
  * clutter hidden (axis markers etc.; --keep_clutter to show)
"""

import argparse
import json
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from PIL import Image

from scenes import HERE, SCENES, apply_quality


def make_parser(extra=None):
    """Two-stage parse: --scene first, then that scene's own options."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--scene", choices=SCENES, default="ant")
    pre.add_argument("--view", default=None)
    known, _ = pre.parse_known_args()
    if known.view is not None:  # a saved view knows its own scene
        with open(known.view) as f:
            known.scene = json.load(f).get("scene", known.scene)

    p = argparse.ArgumentParser(parents=[pre])
    p.add_argument("--out_dir", default=os.path.join(HERE, "snapshots"))
    p.add_argument("--width", type=int, default=3840)
    p.add_argument("--height", type=int, default=2160)
    p.add_argument("--supersample", type=int, default=2, help="render at Nx then downscale")
    p.add_argument("--fovy", type=float, default=30.0, help="smaller = more telephoto, less distortion")
    p.add_argument("--distance", type=float, default=None, help="camera distance (m); default per scene")
    p.add_argument("--elevation", type=float, default=None, help="degrees, negative = looking down")
    p.add_argument("--lookat", type=float, nargs=3, default=None, help="camera target x y z (m); default per scene")
    p.add_argument("--keep_clutter", action="store_true", help="show axis markers / debug geoms")
    if extra:
        extra(p)
    scene = SCENES[known.scene]()
    scene.add_args(p)
    return p, scene


def default_camera(scene, model, data, args):
    lookat, dist, az, el = scene.camera(model, data, args)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = args.lookat if args.lookat is not None else lookat
    cam.distance = args.distance if args.distance is not None else dist
    cam.azimuth = az
    cam.elevation = args.elevation if args.elevation is not None else el
    return cam


def scene_option(scene):
    opt = mujoco.MjvOption()
    for g in scene.geomgroups:
        opt.geomgroup[g] = 1
    return opt


def make_renderer(model, width, height):
    renderer = mujoco.Renderer(model, height=height, width=width, max_geom=5000)
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 1
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 1
    return renderer


def main():
    p, scene = make_parser(lambda p: p.add_argument(
        "--azimuths", type=float, nargs="+", default=None,
        help="one image per azimuth (degrees); default: scene's default angle"))
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    view = None
    if args.view is not None:
        with open(args.view) as f:
            view = json.load(f)
        args.fovy = view.get("fovy", args.fovy)

    rw, rh = args.width * args.supersample, args.height * args.supersample
    model = scene.build(args)
    apply_quality(model, scene, args, offscreen_size=(rw, rh))
    data = mujoco.MjData(model)

    cam = default_camera(scene, model, data, args)
    if view is None:
        scene.reset(model, data, args)
        cam = default_camera(scene, model, data, args)  # lookat may depend on the settled pose
        azimuths = args.azimuths or [cam.azimuth]
        shots = [(az, f"{args.scene}_az{int(az):03d}_d{cam.distance:.2f}.png") for az in azimuths]
    else:
        data.qpos[:] = view["qpos"]
        data.ctrl[:] = view["ctrl"]
        if "act" in view:
            data.act[:] = view["act"]
        mujoco.mj_forward(model, data)
        cam.lookat[:] = view["lookat"]
        cam.distance = view["distance"]
        cam.elevation = view["elevation"]
        shots = [(view["azimuth"], os.path.splitext(os.path.basename(args.view))[0] + ".png")]

    renderer = make_renderer(model, rw, rh)
    opt = scene_option(scene)
    for az, fname in shots:
        cam.azimuth = az
        renderer.update_scene(data, camera=cam, scene_option=opt)
        img = Image.fromarray(renderer.render())
        if args.supersample > 1:
            img = img.resize((args.width, args.height), Image.LANCZOS)
        path = os.path.join(args.out_dir, fname)
        img.save(path)
        print("saved", path)
    renderer.close()


if __name__ == "__main__":
    main()
