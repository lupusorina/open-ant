"""Standalone Gymnasium environment for the Microduck robot (robot_walk.xml).

Structured like open-ant/sim/ant_mujoco.py: a self-contained gym.Env that
loads an MJCF directly with `mujoco` (no mjlab / RL-framework dependency) and
drives it with `gymnasium.envs.mujoco.mujoco_rendering.MujocoRenderer`.

Reward terms are a faithful, single-environment port of the velocity-tracking
task actually trained in this repo — see
``mjlab_microduck/tasks/microduck_velocity_env_cfg.py`` (weights/params) and
``mjlab_microduck/tasks/mdp.py`` + mjlab's own
``mjlab/tasks/velocity/mdp/rewards.py`` (formulas). Every reward term, weight,
and the four reward-weight/command-range curricula are reproduced exactly.

Deliberately NOT reproduced (training-only infrastructure, not "reward"):
  - Domain randomization (mass/CoM/friction/armature/push events, IMU/encoder
    bias, etc.) and the BAM voltage-controlled actuator model. This env drives
    the robot with the native MuJoCo `<position>` actuators already baked
    into robot_walk.xml (class "chosen_actuator": kp=0.55, forcerange
    ±0.96 N·m) instead of BAM.
  - Rough-terrain generation. Ground truth is a single flat plane, so
    foot-height terms use the site's raw world z instead of a ray-cast scan.
  - Observation noise/delay (Unoise / delay_min_lag on the trained obs) — the
    61D actor observation layout is preserved for parity with the runtime
    contract described in microduck_rl/AGENTS.md, but values are noiseless.

Reward math is computed from ground-truth kinematics (`mj_objectVelocity`,
qpos/qvel), matching mjlab, which also computes rewards from the noiseless
physics state (noise is only ever injected into the policy's observations).
"""

from __future__ import annotations

import math
import os

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer

DEFAULT_XML_PATH = os.path.join(os.path.dirname(__file__), "assets", "robot_walk.xml")

# Canonical 14-servo order == the XML's <actuator> order == the runtime's
# ctrl-index-equals-joint-index layout (see microduck_rl/AGENTS.md).
JOINT_NAMES = [
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
]
NECK_JOINT_NAMES = ["neck_pitch", "head_pitch", "head_yaw", "head_roll"]
LEG_JOINT_NAMES = [j for j in JOINT_NAMES if j not in NECK_JOINT_NAMES]
FOOT_SITES = ["left_foot", "right_foot"]
FOOT_GEOMS = ["left_foot_collision", "right_foot_collision"]
NUM_STEPS_PER_ENV = 24  # rsl_rl rollout length; curriculum steps are iteration*24.

# HOME / "STAND2" pose, copied from HOME_FRAME in
# mjlab_microduck/robot/microduck_constants.py (trunk shifted ~5mm forward so
# the CoM sits over the ankle axis).
HOME_JOINT_POS = {
    "left_hip_yaw": 0.0, "right_hip_yaw": 0.0,
    "left_hip_roll": -0.0873, "right_hip_roll": 0.0873,
    "left_hip_pitch": -0.4579, "right_hip_pitch": 0.4579,
    "left_knee": -0.0049, "right_knee": 0.0049,
    "left_ankle": 0.4530, "right_ankle": -0.4530,
    "neck_pitch": 0.3491, "head_pitch": 0.3491,
    "head_yaw": 0.0, "head_roll": 0.0,
}

# variable_posture per-joint std tables (mjlab_microduck/tasks/
# microduck_velocity_env_cfg.py). std_running == std_walking for this task.
STD_STANDING = {"hip_yaw": 0.1, "hip_roll": 0.05, "hip_pitch": 0.15, "knee": 0.15, "ankle": 0.1}
STD_WALKING = {"hip_yaw": 0.3, "hip_roll": 0.05, "hip_pitch": 0.4, "knee": 0.4, "ankle": 0.25}

SOFT_JOINT_POS_LIMIT_FACTOR = 0.9  # EntityArticulationInfoCfg on MICRODUCK_WALK_ROBOT_CFG


def _std_for(joint_name: str, table: dict[str, float]) -> float:
    for key, std in table.items():
        if key in joint_name:
            return std
    raise KeyError(f"No std entry matches joint {joint_name!r}")


def _staged_value(total_steps: int, stages: list[tuple[int, object]]):
    value = stages[0][1]
    for step, v in stages:
        if total_steps >= step:
            value = v
    return value


# ---------------------------------------------------------------------------
# Reward-weight / command-range curricula (mjlab CurriculumTermCfg schedules
# in microduck_velocity_env_cfg.py, steps expressed in raw env steps).
# ---------------------------------------------------------------------------

def action_rate_weight_curriculum(total_steps: int) -> float:
    return _staged_value(total_steps, [
        (0, -0.1),
        (500 * NUM_STEPS_PER_ENV, -0.2),
        (750 * NUM_STEPS_PER_ENV, -0.4),
        (1000 * NUM_STEPS_PER_ENV, -0.6),
        (1250 * NUM_STEPS_PER_ENV, -0.8),
        (1500 * NUM_STEPS_PER_ENV, -1.0),
    ])


def head_pose_bias_weight_curriculum(total_steps: int) -> float:
    return _staged_value(total_steps, [
        (0, 0.0),
        (600 * NUM_STEPS_PER_ENV, 1.0),
        (1000 * NUM_STEPS_PER_ENV, 2.0),
        (1500 * NUM_STEPS_PER_ENV, 3.0),
    ])


def standing_envs_curriculum(total_steps: int) -> float:
    return _staged_value(total_steps, [
        (0, 0.02),
        (500 * NUM_STEPS_PER_ENV, 0.05),
        (750 * NUM_STEPS_PER_ENV, 0.1),
        (1000 * NUM_STEPS_PER_ENV, 0.15),
        (1500 * NUM_STEPS_PER_ENV, 0.2),
        (2000 * NUM_STEPS_PER_ENV, 0.25),
    ])


def head_pose_range_curriculum(total_steps: int):
    return _staged_value(total_steps, [
        (0, ((-0.05, 0.05), (-0.05, 0.05), (-0.07, 0.07), (-0.015, 0.015))),
        (500 * NUM_STEPS_PER_ENV, ((-0.17, 0.17), (-0.17, 0.17), (-0.21, 0.21), (-0.047, 0.047))),
        (1000 * NUM_STEPS_PER_ENV, ((-0.39, 0.39), (-0.39, 0.39), (-0.49, 0.49), (-0.11, 0.11))),
        (1500 * NUM_STEPS_PER_ENV, ((-0.72, 0.72), (-0.72, 0.72), (-0.91, 0.91), (-0.20, 0.20))),
        (2000 * NUM_STEPS_PER_ENV, ((-1.10, 1.10), (-1.10, 1.10), (-1.40, 1.40), (-0.31, 0.31))),
    ])


# ---------------------------------------------------------------------------
# Reward terms. Each takes the env instance (mirrors mjlab's `func(env, ...)`
# reward-term signature) and returns a scalar; the env weights and sums them.
# ---------------------------------------------------------------------------

def track_linear_velocity(env: "MicroduckEnv", std: float) -> float:
    cmd = env.commands["twist"]
    ang_b, lin_b = env._root_vel_body()
    xy_error = float(np.sum((cmd[:2] - lin_b[:2]) ** 2))
    z_error = float(lin_b[2] ** 2)
    return math.exp(-(xy_error + z_error) / std ** 2)


def track_angular_velocity(env: "MicroduckEnv", std: float) -> float:
    cmd = env.commands["twist"]
    ang_b, _ = env._root_vel_body()
    z_error = float((cmd[2] - ang_b[2]) ** 2)
    xy_error = float(np.sum(ang_b[:2] ** 2))
    return math.exp(-(z_error + xy_error) / std ** 2)


def upright(env: "MicroduckEnv", std: float) -> float:
    pg = env._projected_gravity_b()
    xy_sq = float(pg[0] ** 2 + pg[1] ** 2)
    return math.exp(-xy_sq / std ** 2)


def variable_posture(env: "MicroduckEnv", walking_threshold: float, running_threshold: float) -> float:
    cmd = env.commands["twist"]
    total_speed = float(np.linalg.norm(cmd[:2]) + abs(cmd[2]))
    table = STD_STANDING if total_speed < walking_threshold else STD_WALKING
    errs = []
    for j in LEG_JOINT_NAMES:
        pos = env.data.qpos[env.joint_qposadr[j]]
        std = _std_for(j, table)
        errs.append(((pos - HOME_JOINT_POS[j]) / std) ** 2)
    return math.exp(-float(np.mean(errs)))


def body_angular_velocity_penalty(env: "MicroduckEnv") -> float:
    ang_b, _ = env._root_vel_body()
    return float(ang_b[0] ** 2 + ang_b[1] ** 2)


def angular_momentum_penalty(env: "MicroduckEnv") -> float:
    angmom = env._sensor("root_angmom")
    return float(np.sum(angmom ** 2))


def joint_pos_limits(env: "MicroduckEnv") -> float:
    cost = 0.0
    for j in JOINT_NAMES:
        lo, hi = env.soft_joint_limits[j]
        pos = env.data.qpos[env.joint_qposadr[j]]
        cost += max(lo - pos, 0.0) + max(pos - hi, 0.0)
    return cost


def action_rate_l2(env: "MicroduckEnv") -> float:
    return float(np.sum((env.last_action - env.prev_action) ** 2))


def _command_active(env: "MicroduckEnv", command_threshold: float) -> bool:
    cmd = env.commands["twist"]
    total = float(np.linalg.norm(cmd[:2]) + abs(cmd[2]))
    return total > command_threshold


def feet_air_time(env: "MicroduckEnv", threshold_min: float, threshold_max: float, command_threshold: float) -> float:
    if not _command_active(env, command_threshold):
        return 0.0
    return float(sum(1 for t in env.foot_air_time if threshold_min < t < threshold_max))


def feet_clearance(env: "MicroduckEnv", target_height: float, command_threshold: float) -> float:
    if not _command_active(env, command_threshold):
        return 0.0
    cost = 0.0
    for site in FOOT_SITES:
        h = env._foot_height(site)
        vel_xy = env._site_vel_world(site)[:2]
        cost += abs(h - target_height) * float(np.linalg.norm(vel_xy))
    return cost


def feet_swing_height(env: "MicroduckEnv", target_height: float, command_threshold: float) -> float:
    """Evaluated at landing: penalize peak swing height vs target. Has side
    effects (peak tracking) — call exactly once per env.step()."""
    active = _command_active(env, command_threshold)
    cost = 0.0
    for i, site in enumerate(FOOT_SITES):
        h = env._foot_height(site)
        if not env.foot_contact[i]:
            env.foot_peak_height[i] = max(env.foot_peak_height[i], h)
        first_contact = env.foot_contact[i] and not env.prev_foot_contact[i]
        if first_contact:
            if active:
                err = env.foot_peak_height[i] / target_height - 1.0
                cost += err * err
            env.foot_peak_height[i] = 0.0
    return cost


def feet_slip(env: "MicroduckEnv", command_threshold: float) -> float:
    if not _command_active(env, command_threshold):
        return 0.0
    cost = 0.0
    for i, site in enumerate(FOOT_SITES):
        if env.foot_contact[i]:
            v = env._site_vel_world(site)[:2]
            cost += float(np.dot(v, v))
    return cost


def self_collision_cost(env: "MicroduckEnv") -> float:
    return float(env.self_collision_count)


def head_pose_tracking(env: "MicroduckEnv", std: float) -> float:
    cmd = env.commands["head_pose"]
    per_joint = []
    for i, j in enumerate(NECK_JOINT_NAMES):
        pos = env.data.qpos[env.joint_qposadr[j]]
        actual = pos - HOME_JOINT_POS[j]
        err = actual - cmd[i]
        per_joint.append(math.exp(-(err / std) ** 2))
    return float(np.mean(per_joint))


def head_pose_bias_penalty(env: "MicroduckEnv", tau_s: float) -> float:
    """L1 penalty on an EMA of the head-tracking error. Has side effects
    (EMA update) — call exactly once per env.step()."""
    cmd = env.commands["head_pose"]
    err = np.zeros(4, dtype=np.float64)
    for i, j in enumerate(NECK_JOINT_NAMES):
        pos = env.data.qpos[env.joint_qposadr[j]]
        err[i] = (pos - HOME_JOINT_POS[j]) - cmd[i]
    alpha = min(1.0, env.control_dt / max(tau_s, 1e-6))
    env.head_bias_ema = (1.0 - alpha) * env.head_bias_ema + alpha * err
    return float(-np.mean(np.abs(env.head_bias_ema)))


def _quat_from_yaw(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)])


class MicroduckEnv(gym.Env):
    """Velocity-command walking task for the Microduck robot (robot_walk.xml)."""

    # render_fps = 1 / control_dt (sim_dt=0.005 * decimation=4, fixed below).
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        xml_file: str = DEFAULT_XML_PATH,
        render_mode: str | None = None,
        use_curriculum: bool = True,
        episode_length_s: float = 20.0,
    ):
        # `xml_file` (not `model_path`) matches the kwarg Gymnasium's built-in
        # MuJoCo envs accept, so agents/mpo/envs.py's
        # `gym.make(env_id, xml_file=args.model_path, ...)` call works unchanged.
        super().__init__()
        self.use_curriculum = use_curriculum
        self.episode_length_s = episode_length_s

        self.model = self._build_model(xml_file)
        self.sim_dt = 0.005  # mjlab velocity task's MujocoCfg(timestep=0.005)
        self.decimation = 4  # -> 50 Hz control (matches AGENTS.md)
        self.model.opt.timestep = self.sim_dt
        self.control_dt = self.sim_dt * self.decimation
        self.dt = self.control_dt  # alias expected by agents/mpo/envs.py's dt-sync
        self.data = mujoco.MjData(self.model)

        self.joint_qposadr = {j: int(self.model.joint(j).qposadr[0]) for j in JOINT_NAMES}
        self.joint_dofadr = {j: int(self.model.joint(j).dofadr[0]) for j in JOINT_NAMES}
        self.joint_range = {j: tuple(self.model.joint(j).range) for j in JOINT_NAMES}
        self.actuator_id = np.array([self.model.actuator(j).id for j in JOINT_NAMES])

        self.soft_joint_limits = {}
        for j in JOINT_NAMES:
            lo, hi = self.joint_range[j]
            mid, half = (lo + hi) / 2.0, (hi - lo) / 2.0 * SOFT_JOINT_POS_LIMIT_FACTOR
            self.soft_joint_limits[j] = (mid - half, mid + half)

        self.foot_site_id = {s: self.model.site(s).id for s in FOOT_SITES}
        self.foot_geom_id = [self.model.geom(g).id for g in FOOT_GEOMS]
        self.terrain_geom_id = self.model.geom("terrain").id
        self.self_collision_geom_ids = set(np.where(self.model.geom_contype == 2)[0].tolist())
        self.trunk_body_id = self.model.body("trunk_base").id

        low = np.array([self.joint_range[j][0] - HOME_JOINT_POS[j] for j in JOINT_NAMES], dtype=np.float32)
        high = np.array([self.joint_range[j][1] - HOME_JOINT_POS[j] for j in JOINT_NAMES], dtype=np.float32)
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # 61D actor layout: base_ang_vel(3) + projected_gravity(3) +
        # joint_pos_rel(14) + joint_vel(14) + last_action(14) + twist(3) +
        # head_pose(4) + body_pose(6, zero-padded — see microduck_rl/AGENTS.md).
        obs_dim = 3 + 3 + 14 + 14 + 14 + 3 + 4 + 6
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self.render_mode = render_mode
        self.mujoco_renderer = MujocoRenderer(
            self.model,
            self.data,
            width=640,
            height=480,
            max_geom=1000,
            visual_options={},
        )

        self.max_episode_steps = int(round(self.episode_length_s / self.control_dt))

        # Command sampling config (mjlab_microduck/tasks/microduck_velocity_env_cfg.py).
        self.twist_resample_range_s = (3.0, 8.0)
        self.head_pose_resample_range_s = (2.0, 5.0)
        self.lin_vel_x_range = (-0.4, 0.4)
        self.lin_vel_y_range = (-0.3, 0.3)
        self.ang_vel_z_range = (-1.0, 1.0)
        self.rel_turn_in_place_envs = 0.15

        self.total_steps = 0  # persists across resets (== mjlab common_step_counter)
        self.episode_steps = 0
        self.commands: dict[str, np.ndarray] = {}
        self.last_action = np.zeros(14, dtype=np.float32)
        self.prev_action = np.zeros(14, dtype=np.float32)
        self.foot_air_time = np.zeros(2)
        self.foot_peak_height = np.zeros(2)
        self.foot_contact = [False, False]
        self.prev_foot_contact = [False, False]
        self.self_collision_count = 0
        self.head_bias_ema = np.zeros(4)
        self._twist_elapsed_s = 0.0
        self._twist_interval_s = 0.0
        self._head_elapsed_s = 0.0
        self._head_interval_s = 0.0

    @staticmethod
    def _build_model(xml_path: str) -> mujoco.MjModel:
        """robot_walk.xml has no ground plane (mjlab manages terrain
        separately) — add a flat plane here so the feet have something to
        stand on."""
        spec = mujoco.MjSpec.from_file(xml_path)
        terrain = spec.worldbody.add_body(name="terrain")
        terrain.add_geom(
            name="terrain",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[25.0, 25.0, 0.1],
            rgba=[0.55, 0.55, 0.55, 1.0],
        )
        return spec.compile()

    # -- physics/state helpers -------------------------------------------------

    def _sensor(self, name: str) -> np.ndarray:
        sid = self.model.sensor(name).id
        adr = self.model.sensor_adr[sid]
        dim = self.model.sensor_dim[sid]
        return self.data.sensordata[adr: adr + dim]

    def _root_vel_body(self) -> tuple[np.ndarray, np.ndarray]:
        """Trunk angular + linear velocity in the trunk's local (body) frame."""
        res = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.trunk_body_id, res, 1)
        return res[:3], res[3:]

    def _site_vel_world(self, site_name: str) -> np.ndarray:
        """Site linear velocity, world-aligned axes (mjlab's site_lin_vel_w)."""
        sid = self.foot_site_id[site_name]
        res = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_SITE, sid, res, 0)
        return res[3:]

    def _foot_height(self, site_name: str) -> float:
        """Foot height above terrain. Flat ground only (z_terrain == 0)."""
        return float(self.data.site_xpos[self.foot_site_id[site_name]][2])

    def _projected_gravity_b(self) -> np.ndarray:
        quat = self.data.qpos[3:7]
        mat = np.zeros(9)
        mujoco.mju_quat2Mat(mat, quat)
        rot = mat.reshape(3, 3)
        return rot.T @ np.array([0.0, 0.0, -1.0])

    def _update_contacts(self) -> None:
        new_contact = [False, False]
        self_collisions = 0
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if g1 in self.self_collision_geom_ids and g2 in self.self_collision_geom_ids:
                self_collisions += 1
            for k, fg in enumerate(self.foot_geom_id):
                if (g1 == fg and g2 == self.terrain_geom_id) or (g2 == fg and g1 == self.terrain_geom_id):
                    new_contact[k] = True
        self.foot_contact = new_contact
        self.self_collision_count = self_collisions

    def _update_air_time(self) -> None:
        for i in range(2):
            if self.foot_contact[i]:
                self.foot_air_time[i] = 0.0
            else:
                self.foot_air_time[i] += self.control_dt

    # -- command sampling -------------------------------------------------------

    def _standing_env_prob(self) -> float:
        return standing_envs_curriculum(self.total_steps) if self.use_curriculum else 0.25

    def _resample_twist_command(self) -> None:
        self._twist_elapsed_s = 0.0
        self._twist_interval_s = float(self.np_random.uniform(*self.twist_resample_range_s))

        lin_x = float(self.np_random.uniform(*self.lin_vel_x_range))
        lin_y = float(self.np_random.uniform(*self.lin_vel_y_range))
        ang_z = float(self.np_random.uniform(*self.ang_vel_z_range))

        is_turn_in_place = self.np_random.uniform() < self.rel_turn_in_place_envs
        if is_turn_in_place:
            maxr = max(abs(self.ang_vel_z_range[0]), abs(self.ang_vel_z_range[1]))
            sign = -1.0 if self.np_random.uniform() < 0.5 else 1.0
            mag = float(self.np_random.uniform(0.4 * maxr, maxr))
            lin_x, lin_y, ang_z = 0.0, 0.0, sign * mag
        elif self.np_random.uniform() < self._standing_env_prob():
            lin_x = lin_y = ang_z = 0.0

        self.commands["twist"] = np.array([lin_x, lin_y, ang_z], dtype=np.float32)

    def _resample_head_pose_command(self) -> None:
        self._head_elapsed_s = 0.0
        self._head_interval_s = float(self.np_random.uniform(*self.head_pose_resample_range_s))
        ranges = head_pose_range_curriculum(self.total_steps) if self.use_curriculum else (
            (-1.10, 1.10), (-1.10, 1.10), (-1.40, 1.40), (-0.31, 0.31)
        )
        cmd = np.array([self.np_random.uniform(lo, hi) for lo, hi in ranges], dtype=np.float32)
        self.commands["head_pose"] = cmd

    def _update_commands(self) -> None:
        self._twist_elapsed_s += self.control_dt
        if self._twist_elapsed_s >= self._twist_interval_s:
            self._resample_twist_command()
        self._head_elapsed_s += self.control_dt
        if self._head_elapsed_s >= self._head_interval_s:
            self._resample_head_pose_command()

    # -- reward -------------------------------------------------------------

    def _reward_terms(self):
        if self.use_curriculum:
            action_rate_w = action_rate_weight_curriculum(self.total_steps)
            head_bias_w = head_pose_bias_weight_curriculum(self.total_steps)
        else:
            action_rate_w = -1.0  # fully-ramped (post iter-1500) value
            head_bias_w = 3.0

        # (name, func, params, weight) — mirrors the RewardTermCfg table in
        # microduck_velocity_env_cfg.py exactly (soft_landing removed there;
        # body_pose_tracking_6d kept at weight 0.0 there, so omitted here).
        return [
            ("track_linear_velocity", track_linear_velocity, {"std": math.sqrt(0.1)}, 2.0),
            ("track_angular_velocity", track_angular_velocity, {"std": math.sqrt(0.5)}, 2.0),
            ("upright", upright, {"std": math.sqrt(0.05)}, 2.0),
            ("pose", variable_posture, {"walking_threshold": 0.01, "running_threshold": 1.5}, 1.0),
            ("body_ang_vel", body_angular_velocity_penalty, {}, -0.05),
            ("angular_momentum", angular_momentum_penalty, {}, -0.02),
            ("dof_pos_limits", joint_pos_limits, {}, -1.0),
            ("action_rate_l2", action_rate_l2, {}, action_rate_w),
            ("air_time", feet_air_time,
             {"threshold_min": 0.125, "threshold_max": 0.300, "command_threshold": 0.01}, 3.0),
            ("foot_clearance", feet_clearance, {"target_height": 0.02, "command_threshold": 0.01}, -2.0),
            ("foot_swing_height", feet_swing_height, {"target_height": 0.02, "command_threshold": 0.01}, -0.25),
            ("foot_slip", feet_slip, {"command_threshold": 0.01}, -0.1),
            ("self_collisions", self_collision_cost, {}, -1.0),
            ("head_pose_tracking", head_pose_tracking, {"std": 0.5}, 2.0),
            ("head_pose_bias", head_pose_bias_penalty, {"tau_s": 1.0}, head_bias_w),
        ]

    def _compute_reward(self) -> tuple[float, dict[str, float]]:
        terms = {}
        for name, func, params, weight in self._reward_terms():
            terms[name] = weight * func(self, **params)
        return float(sum(terms.values())), terms

    def _build_info(self, reward_terms: dict[str, float]) -> dict[str, float]:
        """Flat (scalar-valued) info dict.

        agents/mpo/mpo_acme.py's CSV logger (`log_step`) assumes every info
        key maps to a plain per-env scalar/array (`infos[k][0]`) — the shape
        gym.vector.SyncVectorEnv produces when merging N per-env leaf values.
        A nested dict value (e.g. {"twist": ..., "head_pose": ...}) instead
        merges into a nested dict of per-key arrays, so `infos[k][0]` becomes
        a dict lookup on key 0 and raises KeyError. Every reward term and
        command dim is therefore logged as its own flat scalar key instead of
        under "reward_terms" / "commands" dicts.
        """
        info: dict[str, float] = {f"reward/{name}": float(value) for name, value in reward_terms.items()}
        twist = self.commands["twist"]
        info["cmd_lin_vel_x"] = float(twist[0])
        info["cmd_lin_vel_y"] = float(twist[1])
        info["cmd_ang_vel_z"] = float(twist[2])
        head_pose = self.commands["head_pose"]
        for name, value in zip(NECK_JOINT_NAMES, head_pose):
            info[f"cmd_head_pose_{name}"] = float(value)
        return info

    def _check_terminated(self) -> bool:
        if not (np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all()):
            return True
        pg = self._projected_gravity_b()
        tilt = math.acos(float(np.clip(-pg[2], -1.0, 1.0)))
        return tilt > math.radians(70.0)

    def _get_obs(self) -> np.ndarray:
        ang_b, _ = self._root_vel_body()
        pg = self._projected_gravity_b()
        joint_pos_rel = np.array(
            [self.data.qpos[self.joint_qposadr[j]] - HOME_JOINT_POS[j] for j in JOINT_NAMES]
        )
        joint_vel = np.array([self.data.qvel[self.joint_dofadr[j]] for j in JOINT_NAMES])
        return np.concatenate([
            ang_b, pg, joint_pos_rel, joint_vel, self.last_action,
            self.commands["twist"], self.commands["head_pose"], self.commands["body_pose"],
        ]).astype(np.float32)

    # -- gym API --------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        for j in JOINT_NAMES:
            self.data.qpos[self.joint_qposadr[j]] = HOME_JOINT_POS[j]

        x = float(self.np_random.uniform(-0.5, 0.5))
        y = float(self.np_random.uniform(-0.5, 0.5))
        z = float(self.np_random.uniform(0.12, 0.13))
        yaw = float(self.np_random.uniform(-math.pi, math.pi))
        self.data.qpos[0:3] = [x, y, z]
        self.data.qpos[3:7] = _quat_from_yaw(yaw)
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        self.last_action = np.zeros(14, dtype=np.float32)
        self.prev_action = np.zeros(14, dtype=np.float32)
        self.foot_air_time = np.zeros(2)
        self.foot_peak_height = np.zeros(2)
        self.foot_contact = [False, False]
        self.prev_foot_contact = [False, False]
        self.self_collision_count = 0
        self.head_bias_ema = np.zeros(4)
        self.episode_steps = 0

        self.commands = {"body_pose": np.zeros(6, dtype=np.float32)}
        self._resample_twist_command()
        self._resample_head_pose_command()
        self._update_contacts()

        # Compute (but discard) a reward breakdown at the initial state purely
        # so every "reward/<term>" key already exists in the info dict that
        # RunLogger.initialize_logging() uses to fix its CSV columns (see
        # _build_info) — otherwise columns first appearing at step() would
        # silently never be logged.
        _, reward_terms = self._compute_reward()

        obs = self._get_obs()
        info = self._build_info(reward_terms)
        return obs, info

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float32), self.action_space.low, self.action_space.high)
        ctrl_target = np.array([HOME_JOINT_POS[j] for j in JOINT_NAMES], dtype=np.float32) + action
        self.data.ctrl[self.actuator_id] = ctrl_target

        for _ in range(self.decimation):
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_rnePostConstraint(self.model, self.data)

        self.prev_action = self.last_action
        self.last_action = action.copy()

        self._update_contacts()
        self._update_air_time()
        self._update_commands()

        reward, reward_terms = self._compute_reward()

        self.total_steps += 1
        self.episode_steps += 1

        terminated = self._check_terminated()
        truncated = self.episode_steps >= self.max_episode_steps

        self.prev_foot_contact = list(self.foot_contact)

        obs = self._get_obs()
        info = self._build_info(reward_terms)
        if self.render_mode == "human":
            self.render()
        return obs, reward, terminated, truncated, info

    def render(self):
        return self.mujoco_renderer.render(self.render_mode)

    def close(self):
        if self.mujoco_renderer is not None:
            self.mujoco_renderer.close()


def main():
    env = MicroduckEnv(render_mode=None)
    obs, info = env.reset(seed=0)
    print("obs shape:", obs.shape, "action shape:", env.action_space.shape)
    total_reward = 0.0
    for i in range(200):
        action = env.action_space.sample() * 0.0  # hold HOME pose
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            print(f"episode ended at step {i} (terminated={terminated}, truncated={truncated})")
            obs, info = env.reset()
    print("mean reward while holding HOME:", total_reward / 200)
    env.close()


if __name__ == "__main__":
    main()
