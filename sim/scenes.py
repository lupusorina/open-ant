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
class AntScene:
    znear, zfar, shadowclip = 0.02, 30.0, 1.0
    geomgroups = ()
    CLUTTER_GEOMS = ["x_axis", "y_axis", "z_axis", "boundary",
                     "north_wall", "south_wall", "east_wall", "west_wall"]
    JOINTS = ["hip_rr", "knee_rr", "hip_fr", "knee_fr", "hip_fl", "knee_fl", "hip_rl", "knee_rl"]
    STAND = np.array([0.0, -np.radians(50)] * 4)

    @staticmethod
    def add_args(p):
        p.add_argument("--xml", default=os.path.join(HERE, "assets/ant_with_camera_after_sys_id.xml"))
        p.add_argument("--settle_steps", type=int, default=1500, help="physics steps before capture")
        p.add_argument("--random_pose", action="store_true", help="drive legs with random targets for a mid-gait look")
        p.add_argument("--light_floor", action="store_true", help="light grey floor instead of dark checker")

    def build(self, args):
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

        # Upper/lower leg capsules share an end-cap at the knee -> z-fighting speckle.
        # Shrink the lower legs slightly (visually negligible; ~no effect on physics).
        for gid in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
            if name.endswith("lower_leg_geom"):
                model.geom_size[gid, 0] *= 0.97
        return model

    def reset(self, model, data, args):
        """Same as AntEnv.reset (standing, knees at -50 deg), then let it settle."""
        data.qpos[:] = 0
        data.qpos[2] = 0.2
        data.qpos[3] = 1.0
        data.qpos[7:15] = self.STAND
        data.ctrl[:] = self.STAND
        mujoco.mj_forward(model, data)

        rng = np.random.default_rng(0)
        for t in range(args.settle_steps):
            if args.random_pose and t % 200 == 0:
                data.ctrl[:] = self.STAND + rng.uniform(-0.4, 0.4, size=8)
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
CRANE_DIR = "/home/seliu/caltech_linc_ws/src/simulation/simulation_mujoco"


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

    def build(self, args):
        import yaml

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
        self.payload_length = params["payload"]["length"]

        spec = mujoco.MjSpec.from_string(crane.create_model(params))

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
        return model

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
