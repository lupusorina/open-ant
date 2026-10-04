"""Render close-up, high-quality still images of a MuJoCo scene (ant or crane).

Usage:
    python sim/snapshot.py                                  # ant, 5 angles
    python sim/snapshot.py --scene crane --azimuths 125 60
    python sim/snapshot.py --view sim/snapshots/view_xxx.json   # pose + camera saved by view.py
    python sim/snapshot.py --style technical                # line-drawing look for papers

Quality tricks used:
  * large offscreen buffer (4K by default) + 2x supersampling, downscaled with Lanczos
  * MSAA (offsamples), high-res shadow map, reflections
  * depth / shadow ranges pinned to the scene size (no z-fighting, sharp shadows)
  * clutter hidden (axis markers etc.; --keep_clutter to show)
"""

import argparse
import copy
import json
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from PIL import Image
from scipy import ndimage

from scenes import HERE, SCENES, XRAY_GROUP, XRAY_INSIDE, apply_quality, technical_look


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
    p.add_argument("--style", choices=["render", "technical"], default="render",
                   help="technical: grey parts with ink outlines on white, for paper figures")
    p.add_argument("--palette", choices=["steel", "grey", "blue", "yellow"], default="steel",
                   help="technical style part colours: cool machined metal, or plain grey")
    p.add_argument("--shading", choices=["smooth", "toon", "none"], default="smooth",
                   help="technical style fill: soft shading, 3 cel-shaded tones, or plain white")
    p.add_argument("--line_px", type=float, default=4.0,
                   help="technical style outline width in output pixels at 3840 wide (scales with --width)")
    p.add_argument("--shadow", type=float, default=0.25,
                   help="technical style ground-shadow strength (0 = none, 1 = black)")
    p.add_argument("--checker_floor", action="store_true",
                   help="technical style: light checkered floor (MuJoCo's tile size) instead of white paper")
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


def build_model(scene, args, offscreen_size=None):
    model = scene.build(args)
    if args.style == "technical":
        technical_look(model, args.palette, getattr(scene, "roles", None), args.checker_floor)
    apply_quality(model, scene, args, offscreen_size=offscreen_size)
    return model


def _disk(radius):
    r = max(int(np.ceil(radius)), 0)
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= radius * radius + 0.25


CHECKER = (0.95, 0.85)  # --checker_floor tile brightness (paper white is 1.0)


def _floor_xy(renderer, model, shape):
    """World (x, y) where each pixel's view ray meets the floor plane, and the floor's checker
    tile size (from its material, as MuJoCo would draw it). None if there's no floor."""
    planes = np.flatnonzero(model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE)
    if len(planes) == 0:
        return None
    g = planes[0]
    tile = 0.5
    m = model.geom_matid[g]
    if m >= 0 and model.mat_texrepeat[m, 0] > 0:  # 2x2 checks per texture repeat
        per_repeat = 1.0 if model.mat_texuniform[m] else 2 * model.geom_size[g, 0]
        tile = per_repeat / model.mat_texrepeat[m, 0] / 2
    cam = renderer.scene.camera[0]
    pos, fwd, up = np.array(cam.pos), np.array(cam.forward), np.array(cam.up)
    right = np.cross(fwd, up)
    h, w = shape
    # frustum_width 0 = "from the viewport aspect", as MuJoCo does at render time
    half_w = cam.frustum_width or (cam.frustum_top - cam.frustum_bottom) / 2 * w / h
    u = cam.frustum_center + (2 * (np.arange(w) + 0.5) / w - 1) * half_w
    v = cam.frustum_bottom + (1 - (np.arange(h) + 0.5) / h) * (cam.frustum_top - cam.frustum_bottom)
    d = fwd * cam.frustum_near + u[None, :, None] * right + v[:, None, None] * up
    t = (model.geom_pos[g, 2] - pos[2]) / np.where(np.abs(d[..., 2]) < 1e-9, -1e-9, d[..., 2])
    return (pos[:2] + t[..., None] * d[..., :2]), tile


def _part_ids(model):
    """Outline part id per geom: geoms named "<part>#<k>" share one (e.g. a target built from
    many boxes), every other geom is its own part."""
    part = np.arange(model.ngeom)
    first = {}
    for g in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if "#" in name:
            part[g] = first.setdefault(name.split("#")[0], g)
    return part


def _smoothstep(lo, hi, x):
    t = np.clip((x - lo) / (hi - lo), 0, 1)
    return t * t * (3 - 2 * t)


def _drop_specks(mask, min_px):
    """Remove connected edge fragments smaller than min_px pixels (depth-test noise)."""
    lab, n = ndimage.label(mask, np.ones((3, 3)))
    if n == 0:
        return mask
    sizes = np.bincount(lab.ravel())
    keep = sizes >= min_px
    keep[0] = False
    return keep[lab]


def technical_composite(model, rgb, seg, depth, line_px, shading="smooth", shadow=0.25,
                        return_masks=False, rgb_noshadow=None, floor_xy=None):
    """Technical-illustration look from the colour, segmentation and depth buffers.

    Each part keeps its base colour (model.geom_rgba) and is tone-mapped from the rendered
    lighting: soft gradients or cel-shaded bands, lifted shadows, white specular highlights.
    Outlines go where the geom id or the depth jumps: a heavy line around each part's
    silhouette against the background, a lighter one where parts meet or overlap.
    Returns a uint8 RGB image, plus (geom ids, outline mask) if return_masks.
    """
    gid = np.where(seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM, seg[..., 0], -1)
    floor = np.isin(gid, np.flatnonzero(model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE))
    gid[floor] = -1
    part = _part_ids(model)
    pid = np.where(gid >= 0, part[np.maximum(gid, 0)], -1)  # "<name>#<k>" geoms = one part
    body = gid >= 0
    lum = rgb.astype(np.float32).mean(-1) / 255

    out = np.ones(rgb.shape, np.float32)  # white paper
    if body.any() and shading != "none":
        base = model.geom_rgba[gid[body], :3].astype(np.float32)
        # Light intensity relative to the part's own colour; ~1 on fully lit faces, >1 in
        # specular highlights.
        s = lum[body] / np.maximum(base.mean(-1), 1e-3)
        s /= max(np.percentile(s, 90), 1e-3)
        if shading == "toon":
            # 3 bands with a hairline smoothstep between them, so band edges anti-alias
            tone = 0.62 + 0.18 * _smoothstep(0.47, 0.53, s) + 0.2 * _smoothstep(0.77, 0.83, s)
            hl = np.zeros_like(s)  # point highlights read as dust specks in flat bands
        else:
            tone = 0.5 + 0.5 * np.clip(s, 0, 1)
            hl = 0.7 * np.clip((s - 1.05) / 0.5, 0, 1)
        col = base * tone[:, None]
        out[body] = col + hl[:, None] * (1 - col)
    if floor.any() and shadow > 0:
        if rgb_noshadow is not None:  # true cast shadows only, not light falloff across the floor
            f = np.clip(lum / np.maximum(rgb_noshadow.astype(np.float32).mean(-1) / 255, 1e-3), 0, 1)
        else:
            f = np.clip(lum / max(np.percentile(lum[floor], 99), 1e-3), 0, 1)
        out[floor] = (1 - shadow * (1 - f[floor]))[:, None]
    if floor.any() and floor_xy is not None:  # light checker tiles, times the shadows
        xy, tile = floor_xy
        odd = (np.floor(xy[floor] / tile).sum(-1) % 2).astype(bool)
        out[floor] *= np.where(odd, CHECKER[1], CHECKER[0])[:, None]

    # Edges between horizontally/vertically adjacent pixels.
    sil = np.zeros_like(body)
    inner = np.zeros_like(body)
    for ax in (0, 1):
        a = [slice(None)] * 2
        b = [slice(None)] * 2
        a[ax], b[ax] = slice(None, -1), slice(1, None)
        a, b = tuple(a), tuple(b)
        ga, gb = pid[a], pid[b]
        s = (ga >= 0) != (gb >= 0)
        both = (ga >= 0) & (gb >= 0)
        za, zb = depth[a], depth[b]
        jump = both & (np.abs(za - zb) > 0.03 * np.minimum(za, zb))  # self-occlusion creases
        i = both & ((ga != gb) | jump)
        for m, e in ((sil, s), (inner, i)):
            m[a] |= e
            m[b] |= e
    inner = _drop_specks(inner, 4 * line_px)
    sil = ndimage.binary_dilation(sil, _disk(line_px / 2))
    inner = ndimage.binary_dilation(inner, _disk(0.3 * line_px))
    ink = np.array([0.11, 0.13, 0.16], np.float32)  # dark slate rather than pure black
    out[inner] = 0.35 * out[inner] + 0.65 * ink
    out[sil] = ink
    out[seg[..., 1] == mujoco.mjtObj.mjOBJ_TENDON] = ink  # cables (not geoms): solid ink lines
    img = (np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8)
    return (img, gid, sil | inner) if return_masks else img


def _render_pass(renderer, model, data, cam, opt, args, line_scale, buffers=False):
    """One pass in the requested --style; with buffers, also (geom ids, depth, outline mask)."""
    renderer.update_scene(data, camera=cam, scene_option=opt)
    rgb = renderer.render()
    if args.style != "technical" and not buffers:
        return rgb
    renderer.enable_segmentation_rendering()
    seg = renderer.render()
    renderer.disable_segmentation_rendering()
    renderer.enable_depth_rendering()
    depth = renderer.render()
    renderer.disable_depth_rendering()
    if args.style == "technical":
        flags = renderer.scene.flags
        flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        rgb_ns = renderer.render()
        flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1
        img, gid, ink = technical_composite(model, rgb, seg, depth, args.line_px * line_scale,
                                            args.shading, args.shadow, return_masks=True,
                                            rgb_noshadow=rgb_ns,
                                            floor_xy=_floor_xy(renderer, model, rgb.shape[:2])
                                            if getattr(args, "checker_floor", False) else None)
    else:
        img, ink = rgb, np.zeros(rgb.shape[:2], bool)
        gid = np.where(seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM, seg[..., 0], -1)
    return (img, gid, depth, ink) if buffers else img


def render_frame(renderer, model, data, cam, opt, args, line_scale=1.0):
    """One frame as uint8 RGB, in the requested --style. line_scale = render width / 3840."""
    xray = np.flatnonzero(model.geom_group == XRAY_GROUP)
    if len(xray) == 0 or not opt.geomgroup[XRAY_GROUP]:
        return _render_pass(renderer, model, data, cam, opt, args, line_scale)
    # X-ray shells: render with and without them; where a shell covers a part that sits inside
    # it (depth within the shell's thickness), blend the shell over that part. Elsewhere the
    # shell stays opaque, and its own outlines are kept at full strength.
    full, gid, depth, ink = _render_pass(renderer, model, data, cam, opt, args, line_scale, True)
    inner_opt = copy.copy(opt)
    inner_opt.geomgroup[XRAY_GROUP] = 0
    behind, bgid, bdepth, _ = _render_pass(renderer, model, data, cam, inner_opt, args,
                                           line_scale, True)
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "" for g in range(model.ngeom)]
    inside_ids = [g for g, n in enumerate(names) if n.startswith(XRAY_INSIDE)]
    # a shell's thickness: capsule/cylinder/sphere diameter, or a box's smallest side
    size = model.geom_size
    thick = 2 * np.where(model.geom_type == mujoco.mjtGeom.mjGEOM_BOX, size.min(axis=1), size[:, 0])
    inside = np.isin(bgid, inside_ids) & (bdepth - depth < thick[np.maximum(gid, 0)])
    # grow by the outline width so the inner parts' own outlines aren't clipped at the edge
    inside = ndimage.binary_dilation(inside, _disk(args.line_px * line_scale))
    see = inside & np.isin(gid, xray) & ~ink
    a = getattr(args, "leg_alpha", None) or getattr(args, "shell_alpha", 0.35)
    out = full.copy()
    out[see] = (a * full[see] + (1 - a) * behind[see] + 0.5).astype(np.uint8)
    return out


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
    model = build_model(scene, args, offscreen_size=(rw, rh))
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
        img = Image.fromarray(render_frame(renderer, model, data, cam, opt, args, rw / 3840))
        if args.supersample > 1:
            img = img.resize((args.width, args.height), Image.LANCZOS)
        path = os.path.join(args.out_dir, fname)
        img.save(path)
        print("saved", path)
    renderer.close()


if __name__ == "__main__":
    main()
