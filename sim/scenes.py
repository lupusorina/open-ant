"""Scenes for snapshot.py / view.py: how to build, pose and frame each MuJoCo model.

A scene provides:
    add_args(parser)                  scene-specific CLI options
    build(args) -> MjModel            load the model and apply visual tweaks
    reset(model, data, args)          put it in the default pose (and let physics settle)
    sliders(model, data)              [{name, min, max, value}] for the viewer's pose sliders
    apply(model, data, values)        pose the model from slider values
    camera(model, data, args)         default (lookat, distance, azimuth, elevation)
    znear, zfar, shadowclip           render ranges in metres, sized to the scene
    geomgroups                        extra geom groups to draw (MuJoCo shows 0-2 by default)

technical_look(model) restyles any scene for --style technical (line-drawing figures).
"""

import importlib.util
import os
import sys

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def apply_quality(model, scene, args, offscreen_size=None):
    """Visual-quality settings shared by all scenes (set before creating a Renderer)."""
    if offscreen_size is not None:
        rw, rh = offscreen_size
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, rw)
        model.vis.global_.offheight = max(model.vis.global_.offheight, rh)
    model.vis.global_.fovy = args.fovy
    model.vis.quality.offsamples = 8        # MSAA
    model.vis.quality.shadowsize = 8192     # crisp shadows
    # znear/zfar/shadowclip are relative to model extent; pin them to absolute metres
    # sized for the scene to avoid z-fighting and blocky shadows.
    ext = model.stat.extent
    model.vis.map.znear = scene.znear / ext
    model.vis.map.zfar = scene.zfar / ext
    model.vis.map.shadowclip = scene.shadowclip / ext
    model.vis.map.shadowscale = 0.6


HIDDEN_GROUP = 5  # a geom/site group that is never drawn
XRAY_GROUP = 4    # shells (--actuators legs) that snapshot.render_frame makes see-through where
                  # a geom named XRAY_INSIDE* sits inside them; opaque everywhere else
XRAY_INSIDE = "servo_"

# Base colours per part role for --style technical. "steel" is a cool machined-metal look
# (light links, darker motor housings) in the spirit of textbook robot-arm illustrations.
# "blue" is light-blue links/body with dark-grey joint regions (hip, knee, foot).
_BLUE = [0x80 / 255, 0xca / 255, 0xf6 / 255]  # #80caf6
PALETTES = {
    "grey":  {"link": [0.80, 0.80, 0.80], "housing": [0.80, 0.80, 0.80], "body": [0.80, 0.80, 0.80],
              "joint": [0.35, 0.35, 0.35],
              "servo": [0.58, 0.58, 0.58], "horn": [0.90, 0.90, 0.90], "screw": [0.45, 0.45, 0.45]},
    "steel": {"link": [0.66, 0.69, 0.73], "housing": [0.36, 0.39, 0.43], "body": [0.72, 0.75, 0.79],
              "joint": [0.30, 0.32, 0.36],
              "servo": [0.30, 0.32, 0.36], "horn": [0.82, 0.84, 0.87], "screw": [0.42, 0.44, 0.48]},
    "blue":  {"link": _BLUE, "housing": _BLUE, "body": _BLUE, "joint": [0.33, 0.34, 0.36],
              "servo": [0.33, 0.34, 0.36], "horn": [0.90, 0.90, 0.90], "screw": [0.45, 0.45, 0.45]},
}
# Crane parts: base platform, slewing turret, brackets/plates, payload, target, winch rope/drum.
PALETTES["grey"].update(plate=[0.50, 0.50, 0.50], base=[0.90, 0.90, 0.90], turret=[0.66, 0.66, 0.66], bracket=[0.50, 0.50, 0.50],
                        payload=[0.72, 0.72, 0.72], target=[0.84, 0.84, 0.84],
                        rope=[0.28, 0.28, 0.28], drum=[0.76, 0.76, 0.76])
PALETTES["steel"].update(plate=[0.36, 0.39, 0.43], base=[0.82, 0.84, 0.87], turret=[0.50, 0.53, 0.57],
                         bracket=[0.36, 0.39, 0.43], payload=[0.62, 0.66, 0.72],
                         target=[0.76, 0.78, 0.81], rope=[0.22, 0.23, 0.25], drum=[0.72, 0.74, 0.77])
# "yellow": construction-crane look, several dusty yellows/ochres plus concrete and steel greys.
PALETTES["yellow"] = {
    "link": [0.86, 0.72, 0.36], "body": [0.80, 0.66, 0.32], "housing": [0.70, 0.56, 0.24],
    "joint": [0.30, 0.30, 0.31], "turret": [0.74, 0.59, 0.27], "bracket": [0.62, 0.49, 0.21],
    "base": [0.80, 0.79, 0.76], "payload": [0.58, 0.60, 0.62], "target": [0.86, 0.85, 0.82],
    "rope": [0.20, 0.20, 0.21], "drum": [0.66, 0.67, 0.69],
    "servo": [0.24, 0.24, 0.25], "horn": [0.84, 0.84, 0.85], "screw": [0.40, 0.40, 0.41]}
PALETTES["yellow"]["plate"] = [0.36, 0.37, 0.39]
PALETTES["blue"].update(plate=[0.33, 0.34, 0.36], base=[0.88, 0.89, 0.90], turret=[0.33, 0.34, 0.36], bracket=[0.33, 0.34, 0.36],
                        payload=_BLUE, target=[0.84, 0.85, 0.86], rope=[0.25, 0.25, 0.27],
                        drum=[0.76, 0.77, 0.78])
# Crane reference trajectory (--traj): one accent colour that reads against every palette.
for _p in PALETTES.values():
    _p["traj"] = [0.16, 0.44, 0.86]  # blue


def technical_look(model, palette="grey", roles=None, checker_floor=False):
    """Plain materials for --style technical: matte parts coloured by role, white floor (or the
    scene's checker with checker_floor), no sites.

    roles(geom_name) -> a PALETTES role ("link", "housing", "body", ...) picks each part's
    colour (default / unknown: "link"). Geoms named "<part>#<k>" are outlined as one part.
    snapshot.technical_composite() does the final look (tone mapping, ink outlines from the
    segmentation/depth buffers, white paper background).
    """
    colours = PALETTES[palette]
    for g in range(model.ngeom):
        if model.geom_rgba[g, 3] == 0:  # hidden clutter: move it out of the drawn groups so it
            model.geom_group[g] = HIDDEN_GROUP  # doesn't show up in the segmentation buffer
            continue
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
            # white paper; with checker_floor keep the material (its tile size is read later,
            # and snapshot draws light checks), but untextured so it lights like paper
            if checker_floor and model.geom_matid[g] >= 0:
                model.mat_texid[model.geom_matid[g]] = -1
                model.mat_rgba[model.geom_matid[g]] = [1, 1, 1, 1]
            else:
                model.geom_matid[g] = -1
            model.geom_rgba[g] = [1, 1, 1, 1]
            continue
        model.geom_matid[g] = -1
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        model.geom_rgba[g, :3] = colours.get(roles(name) if roles else "link", colours["link"])
        model.geom_rgba[g, 3] = 1
    model.site_group[:] = HIDDEN_GROUP  # eyes, IMU markers, ...
    model.mat_reflectance[:] = 0
    # Dim lights so lit surfaces stay below white; the composite re-normalises exposure, and
    # the headroom keeps specular highlights (metal sheen) from clipping into the base colour.
    model.light_diffuse[:] *= 0.5
    model.light_specular[:] = 0.3 if palette != "grey" else 0
    model.vis.headlight.ambient[:] = 0.2
    model.vis.headlight.diffuse[:] = 0.3
    model.vis.headlight.specular[:] = 0.2 if palette != "grey" else 0
    model.vis.quality.numslices = 128  # finer tessellation -> smooth intersection lines
    model.vis.quality.numstacks = 64


def _hide_geoms(model, names):
    for name in names:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid >= 0:
            model.geom_rgba[gid, 3] = 0.0


def _hide_sites(model, names):
    for name in names:
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        if sid >= 0:
            model.site_rgba[sid, 3] = 0.0


# ---------------------------------------------------------------------------------------------
# Servo models (visual only). The physical ant uses Dynamixel XM430-W350 / XC430-W240 servos,
# which share one case: 28.5 (W) x 46.5 (H) x 34 (D) mm, output shaft along D, horn centre
# 11.25 mm from one end, case split into front / middle / back shells along the shaft.
# Servo frame: z = shaft (horn on +z), x = long axis with the horn end at +x, origin on the shaft.
XM430 = dict(w=28.5e-3, h=46.5e-3, d=34e-3, horn_off=11.25e-3, corner=3.0e-3, bevel=0.8e-3,
             front=5.0e-3, back=5.0e-3, seam=0.25e-3,
             horn_r=10.0e-3, horn_t=2.5e-3, idler_t=1.5e-3, bolt_circle=7.0e-3,
             screw_r=1.9e-3, screw_t=1.4e-3, hub_r=2.4e-3)


def _shell_verts(x0, x1, hw, z0, z1, corner, bevel, nseg=8):
    """Rounded-rectangle prism (edges along z rounded, z faces bevelled); convex, so MuJoCo's
    hull of these points is the solid itself."""
    verts = []
    for z, inset in ((z0, bevel), (z0 + bevel, 0.0), (z1 - bevel, 0.0), (z1, bevel)):
        r = corner - inset
        for cx, cy, a0 in ((x1 - corner, hw - corner, 0.0), (x0 + corner, hw - corner, 90.0),
                           (x0 + corner, -hw + corner, 180.0), (x1 - corner, -hw + corner, 270.0)):
            for a in np.radians(np.linspace(a0, a0 + 90.0, nseg)):
                verts.append([cx + r * np.cos(a), cy + r * np.sin(a), z])
    return np.asarray(verts).ravel().tolist()


def _frame_quat(x, z):
    """Quaternion of the frame with axes x, y = z cross x, z (given in the parent frame)."""
    x, z = x / np.linalg.norm(x), z / np.linalg.norm(z)
    mat = np.column_stack([x, np.cross(z, x), z])
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, mat.ravel())
    return quat


def add_servo(spec, parent, child, name, axis, out, origin, scale=1.0, colours=None):
    """One XM430-style servo: case fixed to `parent`, horn + idler + screws fixed to `child` (so
    they turn with the joint; pass child=parent for a static one). `axis` is the output shaft
    (horn on +axis), `out` the case's long axis (the case runs back along -out from the shaft),
    `origin` the point on the shaft at the case's mid-depth, in the parent frame; the child frame
    must be unrotated w.r.t. the parent at qpos0. Geoms are massless and collision-free, so
    dynamics are unchanged; names start with "servo_<name>_" (see servo_role)."""
    s = {k: v * scale for k, v in XM430.items()}
    x1 = s["horn_off"]
    x0 = x1 - s["h"]
    hw, hd = s["w"] / 2, s["d"] / 2
    colours = colours or {}

    # front / middle / back shells; the end shells are a hair smaller so the seams read as a
    # small step in the shaded render (the technical style outlines them anyway)
    zs = [(hd - s["front"], hd, "front"), (-hd + s["back"], hd - s["front"], "middle"),
          (-hd, -hd + s["back"], "back")]
    for z0, z1, part in zs:
        inset = s["seam"] if part != "middle" else 0.0
        spec.add_mesh(name=f"servo_{name}_{part}", uservert=_shell_verts(
            x0 + inset, x1 - inset, hw - inset, z0, z1, s["corner"], s["bevel"]))

    def geom(body, gname, role, **kw):
        g = body.add_geom(name=gname, contype=0, conaffinity=0, mass=0, density=0, **kw)
        g.rgba = colours.get(role, [0.5, 0.5, 0.5, 1])
        return g

    axis = np.asarray(axis, float)
    axis /= np.linalg.norm(axis)
    out = np.asarray(out, float)
    out -= axis * (out @ axis)
    quat = _frame_quat(out, axis)
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, quat)
    rot = rot.reshape(3, 3)
    at = np.asarray(origin, float)
    jp = at - (np.asarray(child.pos, float) if child is not parent else 0.0)  # in child frame

    for _, _, part in zs:
        geom(parent, f"servo_{name}_{part}", "servo", type=mujoco.mjtGeom.mjGEOM_MESH,
             meshname=f"servo_{name}_{part}", pos=at, quat=quat)
    # output horn on the front face, idler on the back
    for side, t, part in ((1, s["horn_t"], "horn"), (-1, s["idler_t"], "idler")):
        zc = side * (hd + t / 2)
        geom(child, f"servo_{name}_{part}", "horn", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
             size=[s["horn_r"], t / 2, 0], pos=jp + rot @ [0, 0, zc], quat=quat)
        # 4 M2 socket-head screws on the bolt circle
        zs_ = side * (hd + t + s["screw_t"] / 2)
        for k in range(4):
            a = np.pi / 4 + k * np.pi / 2
            p = [s["bolt_circle"] * np.cos(a), s["bolt_circle"] * np.sin(a), zs_]
            geom(child, f"servo_{name}_{part}_screw{k}", "screw",
                 type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[s["screw_r"], s["screw_t"] / 2, 0],
                 pos=jp + rot @ p, quat=quat)
    # horn retaining bolt in the centre
    geom(child, f"servo_{name}_hub", "screw", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
         size=[s["hub_r"], s["screw_t"] / 2, 0],
         pos=jp + rot @ [0, 0, hd + s["horn_t"] + s["screw_t"] / 2], quat=quat)


def servo_role(name):
    """Palette role of a servo_* geom."""
    return "screw" if ("screw" in name or name.endswith("_hub")) else \
        "horn" if name.endswith(("_horn", "_idler")) else "servo"


def add_servos(spec, joints, scale=1.0, colours=None):
    """add_servo at each named hinge, case running back along the link toward the parent."""
    child_of = {}
    stack = [spec.worldbody]
    while stack:
        b = stack.pop()
        for j in b.joints:
            child_of[j.name] = (b, j)
        stack.extend(b.bodies)
    for jname in joints:
        child, joint = child_of[jname]
        out = np.asarray(child.pos, float)  # link direction, parent joint -> this joint
        add_servo(spec, child.parent, child, jname, joint.axis, out,
                  np.asarray(child.pos, float) + np.asarray(joint.pos, float), scale, colours)


# ---------------------------------------------------------------------------------------------
class AntScene:
    znear, zfar, shadowclip = 0.02, 30.0, 1.0
    geomgroups = ()
    CLUTTER_GEOMS = ["x_axis", "y_axis", "z_axis", "boundary",
                     "north_wall", "south_wall", "east_wall", "west_wall"]
    JOINTS = ["hip_rr", "knee_rr", "hip_fr", "knee_fr", "hip_fl", "knee_fl", "hip_rl", "knee_rl"]
    STAND = np.array([0.0, -np.radians(50)] * 4)
    # --walking: mid-stride of a crawl gait (ant faces +x). Front-left is the swing leg, lifted
    # and reaching forward; the stance feet are shifted so the torso sits inside their triangle.
    # Positive hip angle swings fr/rr forward but fl/rl backward, hence the signs.
    #                  hip_rr knee_rr hip_fr knee_fr hip_fl knee_fl hip_rl knee_rl
    WALK = np.array([-0.30, -0.87,  0.25, -0.87, -0.45, -0.10, -0.35, -0.87])

    @staticmethod
    def add_args(p):
        p.add_argument("--xml", default=os.path.join(HERE, "assets/ant_with_camera_after_sys_id.xml"))
        p.add_argument("--settle_steps", type=int, default=1500, help="physics steps before capture")
        p.add_argument("--random_pose", action="store_true", help="drive legs with random targets for a mid-gait look")
        p.add_argument("--walking", action="store_true",
                       help="start mid-stride: front-left leg lifted and swinging forward")
        p.add_argument("--light_floor", action="store_true", help="light grey floor instead of dark checker")
        p.add_argument("--leg_scale", type=float, default=1.0,
                       help="upper/lower leg length factor (e.g. 0.8 = more compact ant); the hip "
                            "segment is kept so the hips stay outside the torso. Changes the "
                            "rendered model's geometry, so the settled pose shifts with it")
        p.add_argument("--joint_regions", action="store_true",
                       help="dark hip / knee / foot regions (always on with --palette blue); visual only")
        p.add_argument("--actuators", action="store_true",
                       help="draw the hip/knee servos (Dynamixel XM430 case + horn); visual only")
        p.add_argument("--actuator_legs", nargs="+", choices=["fl", "fr", "rl", "rr"],
                       default=["fl", "rr"],
                       help="with --actuators: legs that get servos (default: one diagonal pair); "
                            "the others are drawn as plain legs")
        p.add_argument("--actuator_scale", type=float, default=1.25,
                       help="servo size vs the real XM430 (the sim's legs are ~1.7x thicker than "
                            "the real links, so this keeps the servos just inside them)")
        p.add_argument("--leg_alpha", type=float, default=0.35,
                       help="with --actuators: leg opacity over the servos (the rest stays opaque)")

    # Render-style colours for --actuators: black servo case, bare aluminium horn, dark screws.
    SERVO_RGBA = {"servo": [0.10, 0.10, 0.11, 1], "horn": [0.74, 0.75, 0.77, 1],
                  "screw": [0.22, 0.22, 0.24, 1]}
    actuators = False

    def roles(self, name):
        """--style technical colouring: torso, hip segments as motor housings (or the servos
        themselves with --actuators), legs as links."""
        if name == "torso_geom":
            return "body"
        if name.startswith("region_"):
            return "joint"
        if name.startswith("servo_"):
            return servo_role(name)
        if name.endswith("_leg_geom") and not name.endswith(("upper_leg_geom", "lower_leg_geom")):
            return "link" if self.actuators else "housing"  # the servo is the housing now
        return "link"

    FOOT_LEN = 0.035  # length of the dark foot region at the end of each lower leg (m)

    @staticmethod
    def _scale_legs(spec, f):
        """Shorten/lengthen the upper and lower legs (knee position + both capsules) by f."""
        stack = [spec.worldbody]
        while stack:
            b = stack.pop()
            if any(j.name.startswith("hip_") for j in b.joints):  # hip body: upper leg + knee body
                for g in b.geoms:
                    if g.name.endswith("upper_leg_geom"):
                        g.fromto = np.asarray(g.fromto) * f
                for knee in b.bodies:
                    knee.pos = np.asarray(knee.pos) * f
                    for g in knee.geoms:
                        if g.name.endswith("lower_leg_geom"):
                            g.fromto = np.asarray(g.fromto) * f
            stack.extend(b.bodies)

    def _add_joint_regions(self, spec):
        """Massless, collision-free parts marking each hip, knee and foot: a ball at each joint
        and a sleeve over the foot end. Radii are fitted to the legs in _fit_joint_regions."""
        bodies = {}
        stack = [spec.worldbody]
        while stack:
            b = stack.pop()
            for j in b.joints:
                bodies[j.name] = b
            stack.extend(b.bodies)
        for leg in ("rr", "fr", "fl", "rl"):
            for kind in ("hip", "knee"):
                b = bodies[f"{kind}_{leg}"]
                b.add_geom(name=f"region_{kind}_{leg}", type=mujoco.mjtGeom.mjGEOM_SPHERE,
                           size=[0.02, 0, 0], pos=b.joints[0].pos, contype=0, conaffinity=0,
                           mass=0, density=0, rgba=[0.12, 0.12, 0.13, 1])
            knee = bodies[f"knee_{leg}"]
            lower = next(g for g in knee.geoms if g.name.endswith("lower_leg_geom"))
            a, b = np.asarray(lower.fromto[:3]), np.asarray(lower.fromto[3:])
            start = b + (a - b) * self.FOOT_LEN / np.linalg.norm(b - a)
            knee.add_geom(name=f"region_foot_{leg}", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                          fromto=np.concatenate([start, b]), size=[0.02, 0, 0], contype=0,
                          conaffinity=0, mass=0, density=0, rgba=[0.12, 0.12, 0.13, 1])

    @staticmethod
    def _fit_joint_regions(model):
        """Size the region parts just proud of the (already restyled) leg capsules they cover."""
        def gid(name):
            return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        leg_r = {}  # body id -> radius of the leg capsule attached to it
        for g in range(model.ngeom):
            if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "").endswith("_leg_geom"):
                leg_r[model.geom_bodyid[g]] = model.geom_size[g, 0]
        for leg in ("rr", "fr", "fl", "rl"):
            for kind in ("hip", "knee"):
                g = gid(f"region_{kind}_{leg}")
                b = model.geom_bodyid[g]
                model.geom_size[g, 0] = 1.1 * max(leg_r[b], leg_r[model.body_parentid[b]])
            g = gid(f"region_foot_{leg}")
            model.geom_size[g, 0] = 1.06 * leg_r[model.geom_bodyid[g]]

    def build(self, args):
        self.actuators = args.actuators
        regions = args.joint_regions or getattr(args, "palette", None) == "blue"
        if args.actuators or regions or args.leg_scale != 1.0:
            spec = mujoco.MjSpec.from_file(args.xml)
            if args.leg_scale != 1.0:  # first, so servos / regions are placed on the new legs
                self._scale_legs(spec, args.leg_scale)
            if args.actuators:
                joints = [j for j in self.JOINTS if j.split("_")[1] in args.actuator_legs]
                add_servos(spec, joints, args.actuator_scale, self.SERVO_RGBA)
            if regions:
                self._add_joint_regions(spec)
            model = spec.compile()
        else:
            model = mujoco.MjModel.from_xml_path(args.xml)
        model.vis.headlight.ambient[:] = 0.35
        model.vis.headlight.diffuse[:] = 0.5
        model.vis.headlight.specular[:] = 0.1

        if not args.keep_clutter:
            _hide_geoms(model, self.CLUTTER_GEOMS)
            _hide_sites(model, ["imu_location"])

        if args.light_floor:
            mid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, "MatPlane")
            model.mat_texid[mid] = -1  # drop checker texture
            fid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
            model.geom_rgba[fid] = [0.72, 0.72, 0.74, 1]
            model.mat_reflectance[mid] = 0.1

        # Hip/upper/lower leg capsules share end-caps at each joint -> z-fighting speckle.
        # Shrink the lower legs slightly (visually negligible; ~no effect on physics). Outlines
        # make any speckle obvious, so the technical style tapers the whole chain instead.
        shrink = {"upper_leg_geom": 0.92, "lower_leg_geom": 0.85} if args.style == "technical" \
            else {"lower_leg_geom": 0.97}
        for gid in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
            for suffix, f in shrink.items():
                if name.endswith(suffix):
                    model.geom_size[gid, 0] *= f
        if regions:
            self._fit_joint_regions(model)

        if args.actuators:  # servo legs see-through over the servos only; size unchanged
            self.geomgroups = (XRAY_GROUP,)
            for leg in args.actuator_legs:
                hip = model.jnt_bodyid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"hip_{leg}")]
                knee = model.jnt_bodyid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"knee_{leg}")]
                bodies = [model.body_parentid[hip], hip, knee]  # hip segment, upper, lower leg
                for gid in range(model.ngeom):
                    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
                    if model.geom_bodyid[gid] in bodies and (name.endswith("_leg_geom")
                                                             or name.startswith("region_")):
                        model.geom_group[gid] = XRAY_GROUP
        return model

    def reset(self, model, data, args):
        """Same as AntEnv.reset (standing, knees at -50 deg), then let it settle."""
        pose = self.WALK if getattr(args, "walking", False) else self.STAND
        data.qpos[:] = 0
        data.qpos[2] = 0.2
        data.qpos[3] = 1.0
        data.qpos[7:15] = pose
        data.ctrl[:] = pose
        mujoco.mj_forward(model, data)

        rng = np.random.default_rng(0)
        for t in range(args.settle_steps):
            if args.random_pose and t % 200 == 0:
                data.ctrl[:] = pose + rng.uniform(-0.4, 0.4, size=8)
            mujoco.mj_step(model, data)

    def sliders(self, model, data):
        return [{"name": j, "min": -1.3, "max": 1.3, "value": float(data.ctrl[i])}
                for i, j in enumerate(self.JOINTS)]

    def apply(self, model, data, values):
        # New leg targets: let the physics settle into the new pose (~0.6 s sim time).
        data.ctrl[:] = values
        for _ in range(600):
            mujoco.mj_step(model, data)

    def camera(self, model, data, args):
        torso = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        return data.xpos[torso].copy(), 0.8, 180.0, -35.0


# ---------------------------------------------------------------------------------------------
CRANE_DIR = "/home/serenaliu/caltech_linc_home/caltech_linc_ws/src/simulation/simulation_mujoco"


class CraneScene:
    """Electric crane from caltech_linc_ws (built with dm_control mjcf), plus floor/sky/lights."""
    znear, zfar, shadowclip = 0.05, 60.0, 8.0
    geomgroups = (3,)  # decorative boom-offset plates; the crane sim enables this group too

    @staticmethod
    def add_args(p):
        p.add_argument("--crane_dir", default=CRANE_DIR, help="simulation_mujoco package root")
        p.add_argument("--crane_params", default=None,
                       help="crane_params.yaml (default: <crane_dir>/models/electric_crane/crane_params.yaml)")
        p.add_argument("--no_floor", action="store_true", help="don't add a floor (original look)")
        p.add_argument("--no_target", action="store_true",
                       help="leave out the target disc/hole (not built at all, so the payload "
                            "can't rest on an invisible target)")
        p.add_argument("--actuators", action="store_true",
                       help="draw the slew / luff servos and the hoist winch; visual only")
        p.add_argument("--shell_alpha", type=float, default=0.35,
                       help="--style technical with --actuators: platform/turret opacity over the "
                            "motors inside them (the rest stays opaque)")
        p.add_argument("--traj", choices=["figure8", "circle"], default=None,
                       help="draw the payload reference trajectory (same shapes as the RL "
                            "controller's desired_traj_pub / ReferenceMotion); visual only; hides the target like --no_target")
        p.add_argument("--traj_radius", type=float, default=0.6, help="trajectory radius (m)")
        p.add_argument("--traj_phase", type=float, default=0.0,
                       help="phase (deg) where the payload sits on the curve, i.e. where it starts")
        p.add_argument("--traj_center", type=float, nargs=3, default=None,
                       help="trajectory centre x y z (m); default: placed so the payload (at its "
                            "reset pose) sits on the curve at --traj_phase")
        p.add_argument("--traj_dz", type=float, default=0.0,
                       help="shift the trajectory vertically (m, negative = down); the crane "
                            "and payload stay put")

    def build(self, args):
        import yaml

        if args.traj:  # the reference path replaces the fixed target
            args.no_target = True

        sys.path.insert(0, args.crane_dir)  # for `simulation_mujoco.common_models`
        model_dir = os.path.join(args.crane_dir, "models/electric_crane")
        spec = importlib.util.spec_from_file_location("electric_crane", os.path.join(model_dir, "crane.py"))
        crane = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(crane)

        with open(args.crane_params or os.path.join(model_dir, "crane_params.yaml")) as f:
            params = yaml.safe_load(f)
        if not args.keep_clutter:
            params["viz"]["show_world_frame"] = False
            params["viz"]["show_base_link_frame"] = False
        if args.no_target:
            params["scene_config"]["generate_target_geom"] = False
        self.payload_length = params["payload"]["length"]

        spec = mujoco.MjSpec.from_string(crane.create_model(params))
        self._name_parts(spec)
        if args.actuators:
            self._add_motors(spec, params["crane"])

        # Backdrop: sky gradient + checker floor (visual only, so the dynamics are unchanged).
        spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                         builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                         rgb1=[0.55, 0.7, 0.85], rgb2=[0.05, 0.08, 0.12], width=512, height=512)
        spec.visual.rgba.haze = [0.55, 0.7, 0.85, 1]
        if not args.no_floor:
            spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D,
                             builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                             rgb1=[0.32, 0.34, 0.36], rgb2=[0.26, 0.28, 0.30], width=512, height=512)
            mat = spec.add_material(name="floor", texrepeat=[1, 1], texuniform=True, reflectance=0.08)
            mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
            spec.worldbody.add_geom(name="render_floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                                    size=[0, 0, 0.05], pos=[0, 0, -0.102], material="floor",
                                    contype=0, conaffinity=0)
        # Key light with shadows, from above and to the side.
        spec.worldbody.add_light(name="key", pos=[3, -3, 6], dir=[-0.4, 0.4, -1],
                                 diffuse=[0.5, 0.5, 0.5], specular=[0.2, 0.2, 0.2], castshadow=True)

        if args.style == "technical":
            # Even, directional key light, so the white floor stays white away from the crane.
            next(l for l in spec.worldbody.lights if l.name == "key").type = \
                mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
        if args.traj:
            self._add_traj(spec, args)
        model = spec.compile()

        # The target disc/hole is ~36 overlapping 5 mm boxes with coplanar faces -> z-fighting
        # "cracks". Stagger them by 0.2 mm each (render-only size; invisible, no effect on the look).
        red = [g for g in range(model.ngeom)
               if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX
               and np.allclose(model.geom_rgba[g], [0.8, 0.2, 0.2, 1])]
        for k, g in enumerate(red):
            model.geom_pos[g, 2] += (k % 12) * 2e-4

        model.vis.headlight.ambient[:] = 0.35
        model.vis.headlight.diffuse[:] = 0.35
        model.vis.headlight.specular[:] = 0.1

        if args.no_target:
            _hide_sites(model, ["target"])  # crane.py adds the target marker site regardless

        if args.actuators and args.style == "technical":
            # See-through over the motors only (the shaded style's platform/turret are already
            # translucent in the original model).
            self.geomgroups = (3, XRAY_GROUP)
            for name in ("crane_platform_geom", "turret"):
                model.geom_group[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)] = XRAY_GROUP
        return model

    ROLES = {"crane_platform_geom": "base", "base_plate": "plate", "turret": "turret",
             "boom": "link", "luff_bracket": "bracket", "payload_cylinder": "payload",
             "target": "target", "winch_rope": "rope", "winch_drum": "drum",
             "winch_flange": "drum", "sheave": "drum", "winch_cheek": "bracket",
             "winch_axle": "screw", "sheave_axle": "screw", "traj": "traj"}

    TRAJ_RGBA = [0.12, 0.40, 0.90, 1]  # render style; --style technical uses PALETTES[..]["traj"]

    @staticmethod
    def traj_offset(kind, radius, phase):
        """ReferenceMotion.offset in controller_rl_node/crane.py (horizontal plane)."""
        if kind == "circle":
            return radius * np.array([np.cos(phase), np.sin(phase), 0.0])
        return radius * np.array([np.sin(phase), np.sin(phase) * np.cos(phase), 0.0])

    def _add_traj(self, spec, args):
        """Reference path as a thin tube with direction chevrons, the payload sitting on it at
        --traj_phase about to set off. Massless and collision-free."""
        p0 = np.radians(args.traj_phase)
        if args.traj_center is not None:
            center = np.asarray(args.traj_center, float)
        else:
            # Like the RL env, freeze the centre at reset: settle the crane once (sans path)
            # and put the payload centre on the curve at the start phase.
            model = spec.compile()
            data = mujoco.MjData(model)
            self.reset(model, data, args)
            center = data.body("payload").xpos - self.traj_offset(args.traj, args.traj_radius, p0)
        center = center + [0, 0, args.traj_dz]

        def point(p):
            return center + self.traj_offset(args.traj, args.traj_radius, p)

        def capsule(name, a, b, r):
            g = spec.worldbody.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                                        size=[r, 0, 0], fromto=np.concatenate([a, b]),
                                        contype=0, conaffinity=0, mass=0, density=0)
            g.rgba = self.TRAJ_RGBA

        # The path is one closed tube mesh, not a chain of capsules: segmentation is rendered with
        # MSAA, which blends geom ids into garbage once more than 255 geoms are in view.
        r_tube, n, m = 0.008, 400, 12
        ps = p0 + 2 * np.pi * np.arange(n) / n
        pts = np.array([point(p) for p in ps])
        tang = np.roll(pts, -1, 0) - np.roll(pts, 1, 0)
        tang /= np.linalg.norm(tang, axis=1, keepdims=True)
        side = np.cross([0, 0, 1], tang)
        side /= np.linalg.norm(side, axis=1, keepdims=True)
        up = np.cross(tang, side)
        a = 2 * np.pi * np.arange(m) / m
        verts = (pts[:, None] + r_tube * (np.cos(a)[None, :, None] * side[:, None]
                                          + np.sin(a)[None, :, None] * up[:, None])) - center
        faces = []
        for i in range(n):
            for j in range(m):
                v00, v01 = i * m + j, i * m + (j + 1) % m
                v10, v11 = ((i + 1) % n) * m + j, ((i + 1) % n) * m + (j + 1) % m
                faces += [v00, v11, v10, v00, v01, v11]  # outward normals
        spec.add_mesh(name="traj_tube", uservert=verts.ravel().tolist(), userface=faces)
        g = spec.worldbody.add_geom(name="traj#tube", type=mujoco.mjtGeom.mjGEOM_MESH,
                                    meshname="traj_tube", pos=center,
                                    contype=0, conaffinity=0, mass=0, density=0)
        g.rgba = self.TRAJ_RGBA
        # Chevrons pointing along the direction of travel, the first one just ahead of the payload.
        size = 0.07
        for i, frac in enumerate((0.06, 0.25, 0.5, 0.75)):
            p = p0 + 2 * np.pi * frac
            tip = point(p)
            t = point(p + 1e-4) - point(p - 1e-4)
            t /= np.linalg.norm(t)
            side = np.cross([0, 0, 1], t)
            for j, s in enumerate((-1, 1)):
                tail = tip - size * (np.cos(np.radians(35)) * t + s * np.sin(np.radians(35)) * side)
                capsule(f"traj#arrow{i}_{j}", tip, tail, 1.6 * r_tube)

    def roles(self, name):
        """--style technical colouring (see PALETTES)."""
        if name.startswith("servo_"):
            return servo_role(name)
        return self.ROLES.get(name.split("#")[0], "link")

    @staticmethod
    def _name_parts(spec):
        """crane.py leaves most geoms unnamed; name them for roles / outlines. The target's ~36
        boxes become target#k so the technical outlines treat them as one part."""
        platform = spec.body("crane_platform")
        k = 0
        for g in platform.geoms:
            unnamed = not g.name or g.name.startswith("//unnamed")  # dm_control's auto names
            if unnamed and np.allclose(g.rgba, [0.8, 0.2, 0.2, 1]):
                g.name = f"target#{k}"
                k += 1
        spec.body("base_link").geoms[0].name = "base_plate"
        spec.body("slewed").geoms[0].name = "turret"
        for i, g in enumerate(g for g in spec.body("luffed").geoms if g.name != "boom"):
            g.name = f"luff_bracket#{i}"

    def _add_motors(self, spec, crane):
        """Slew servo in the platform driving the turret, luff servo in the turret between the
        boom brackets, and a hoist winch on the boom with the rope running over a tip sheave to
        where the (simulated) cable leaves. All massless and collision-free."""
        c = self.MOTOR_RGBA
        base, slewed, luffed = spec.body("base_link"), spec.body("slewed"), spec.body("luffed")
        L, off, bw = crane["boom_length"], crane["boom_offset"], 0.1  # crane.BOOM_WIDTH
        h = crane["base_link_height"]

        # Slew: shaft vertical on the slew axis, output flange just above the base plate (inside
        # the turret), case down in the platform.
        s = self.SLEW_SCALE
        hd = XM430["d"] * s / 2
        add_servo(spec, base, slewed, "slew", [0, 0, 1], [-1, 0, 0],
                  [0, 0, -h + 0.001 + hd], s, c)
        # Luff: shaft on the luff axis, case hanging down into the turret between the brackets.
        add_servo(spec, slewed, luffed, "luff", [0, 1, 0], [0, 0, 1], [0, 0, 0], self.LUFF_SCALE, c)

        # Hoist winch on top of the boom near its root.
        def geom(name, role, **kw):
            g = luffed.add_geom(name=name, contype=0, conaffinity=0, mass=0, density=0, **kw)
            g.rgba = c[role]
        y_axis = [np.sqrt(0.5), np.sqrt(0.5), 0, 0]  # cylinder axis along y
        top = off + bw / 2
        xd, r_drum, r_fl, half_w = 0.3, 0.03, 0.05, 0.04
        zc = top + r_fl + 0.01
        geom("winch_drum", "drum", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[r_drum, half_w, 0],
             pos=[xd, 0, zc], quat=y_axis)
        geom("winch_rope#coil", "rope", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
             size=[r_drum + 0.007, half_w - 0.006, 0], pos=[xd, 0, zc], quat=y_axis)
        for i, y in enumerate((-1, 1)):
            geom(f"winch_flange#{i}", "drum", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                 size=[r_fl, 0.003, 0], pos=[xd, y * (half_w + 0.003), zc], quat=y_axis)
            geom(f"winch_cheek#{i}", "bracket", type=mujoco.mjtGeom.mjGEOM_BOX,
                 size=[0.045, 0.005, (zc + 0.04 - top) / 2],
                 pos=[xd, y * (half_w + 0.015), (zc + 0.04 + top) / 2])
        geom("winch_axle", "screw", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
             size=[0.008, half_w + 0.025, 0], pos=[xd, 0, zc], quat=y_axis)
        s = self.WINCH_SCALE
        cheek_out = half_w + 0.02
        add_servo(spec, luffed, luffed, "hoist", [0, -1, 0], [1, 0, 0],
                  [xd, cheek_out + XM430["horn_t"] * s + XM430["d"] * s / 2, zc], s, c)
        # Tip sheave: its bottom is the "tip" site, where the simulated cable starts.
        r_sh = bw / 2
        geom("sheave", "drum", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[r_sh, 0.02, 0],
             pos=[L, 0, off], quat=y_axis)
        geom("sheave_axle", "screw", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.012, 0.03, 0],
             pos=[L, 0, off], quat=y_axis)
        geom("winch_rope#run", "rope", type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[0.005, 0, 0],
             fromto=[xd, 0, zc + r_drum + 0.007, L, 0, off + r_sh])

    SLEW_SCALE, LUFF_SCALE, WINCH_SCALE = 4.0, 2.6, 2.2  # x the XM430 case
    # Render-style colours: black servo cases, bare aluminium horns / drum, dark steel brackets.
    MOTOR_RGBA = {"servo": [0.10, 0.10, 0.11, 1], "horn": [0.74, 0.75, 0.77, 1],
                  "screw": [0.22, 0.22, 0.24, 1], "drum": [0.70, 0.71, 0.73, 1],
                  "bracket": [0.30, 0.31, 0.33, 1], "rope": [0.15, 0.15, 0.16, 1]}

    # The three intvelocity actuators integrate to position targets, stored in data.act.
    SLIDERS = [("slew", -90, 90), ("luff", -80, 60), ("hoist", 0.2, 2.5)]

    def _ids(self, model):
        return {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n, _, _ in self.SLIDERS}

    def reset(self, model, data, args):
        # Same initial state as the simulator (sim_config_electric.yaml): slew 0, luff -45 deg, hoist 1 m.
        self.apply(model, data, [0.0, -45.0, 1.0])

    def sliders(self, model, data):
        ids = self._ids(model)
        vals = [np.degrees(data.act[model.actuator_actadr[ids["slew"]]]),
                np.degrees(data.act[model.actuator_actadr[ids["luff"]]]),
                data.act[model.actuator_actadr[ids["hoist"]]]]
        return [{"name": n + (" (m)" if n == "hoist" else " (deg)"), "min": lo, "max": hi,
                 "value": float(v)} for (n, lo, hi), v in zip(self.SLIDERS, vals)]

    def apply(self, model, data, values):
        slew, luff, hoist = np.radians(values[0]), np.radians(values[1]), values[2]
        ids = self._ids(model)
        for name, v in [("slew", slew), ("luff", luff), ("hoist", hoist)]:
            data.act[model.actuator_actadr[ids[name]]] = v
        data.ctrl[:] = 0.0

        # Place joints directly and hang the payload on a short cable straight below the boom tip,
        # then pay the cable out to the requested length so the payload lands on anything below
        # it instead of being teleported into it (which makes the contact solver fling it away).
        for name, v in [("slew", slew), ("luff", luff)]:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            data.qpos[model.jnt_qposadr[jid]] = v
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        tip = data.site("tip").xpos.copy()
        start = min(hoist, 0.2)
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "payload_free_joint")
        adr = model.jnt_qposadr[jid]
        data.qpos[adr:adr + 3] = tip - [0, 0, start + self.payload_length / 2]
        data.qpos[adr + 3:adr + 7] = [1, 0, 0, 0]
        mujoco.mj_forward(model, data)

        hoist_act = model.actuator_actadr[ids["hoist"]]
        n_lower = 1500
        for k in range(1, n_lower + 1):
            data.act[hoist_act] = start + (hoist - start) * k / n_lower
            mujoco.mj_step(model, data)
        for _ in range(500):  # settle, then freeze any leftover swing
            mujoco.mj_step(model, data)
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)

    def camera(self, model, data, args):
        return np.array([0.9, 0.4, 0.8]), 6.0, 125.0, -18.0


SCENES = {"ant": AntScene, "crane": CraneScene}
