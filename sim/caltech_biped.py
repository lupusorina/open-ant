"""NumPy MuJoCo port of the Caltech biped walking task.

Mirrors ``robot_learning`` ``BipedSim`` (policy observation, rewards, gait
phase, and reset noise) without JAX or MJX. The policy sees a short history
of the actor observation. SAC trains on that vector; the privileged critic
features from the JAX env are not part of this observation.
"""

from __future__ import annotations

import os

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_XML = os.path.join(_HERE, "assets", "caltech_biped", "biped_RL.xml")

FEET_SITES = ("l_foot", "r_foot")
FEET_GEOMS = ("L_FOOT", "R_FOOT")
SIDES = ("L", "R")
HIP_JOINT_NAMES = ("HAA", "HFE")
KNEE_JOINT_NAMES = ("KFE",)

GRAVITY_SENSOR = "upvector"
GLOBAL_LINVEL_SENSOR = "global_linvel"
GLOBAL_ANGVEL_SENSOR = "global_angvel"
LOCAL_LINVEL_SENSOR = "local_linvel"
GYRO_SENSOR = "gyro"

DESIRED_HEIGHT = 0.56
DESIRED_FOOT_HEIGHT = 0.15
FEET_PHASE_STD = 0.06
GAIT_PERIOD = 0.70

RANDOMIZE_ARMATURE = True
ARMATURE_MIN = 0.0001
ARMATURE_MAX = 0.005

# (negative_action_offset, positive_action_offset) around the home pose.
ACTION_TARGET_OFFSETS = {
    "L_HAA": (-0.20, 0.20),
    "L_HFE": (-0.70, 0.90),
    "L_KFE": (-0.90, 1.10),
    "L_ANKLE": (-0.50, 0.50),
    "R_HAA": (-0.20, 0.20),
    "R_HFE": (-0.70, 0.90),
    "R_KFE": (-0.90, 1.10),
    "R_ANKLE": (-0.50, 0.50),
}

CTRL_DT = 0.01
SIM_DT = 0.002
HISTORY_LEN = 3
LIMIT_OBSERVATIONS = 10.0
ADD_OBSERVATION_NOISE = True

OBS_NOISE_BASE_LIN_VEL = 0.3
OBS_NOISE_BASE_ANG_VEL = 0.3
OBS_NOISE_BASE_ROT = 0.05
OBS_NOISE_JOINT_POS = 0.01
OBS_NOISE_JOINT_VEL = 0.5

OBS_SCALE = {
    "base_lin_vel": 1.0,
    "base_ang_vel": 0.25,
    "upvector": 1.0,
    "command": 1.0,
    "joint_pos_err": 0.5,
    "joint_vel": 0.05,
    "last_action": 1.0,
    "phase": 1.0,
}

REWARD_SCALES = {
    "tracking_lin_vel": 2.0,
    "tracking_ang_vel": 1.0,
    "lin_vel_z": -2.0,
    "ang_vel_xy": -0.15,
    "orientation": -1.0,
    "base_height": 0.0,
    "torques": -2.5e-4,
    "action_rate": -1e-3,
    "feet_air_time": 0.0,
    "feet_slip": -1.0,
    "feet_height": 0.0,
    "feet_phase": 0.5,
    "contact_schedule": 2.0,
    "termination": -1.0,
    "joint_deviation_knee": -0.0,
    "joint_deviation_hip": -0.0,
    "dof_pos_limits": -1.0,
    "pose": -1.0,
}


def _foot_height(phase: np.ndarray, swing_height: float = DESIRED_FOOT_HEIGHT) -> np.ndarray:
    swing_z = swing_height * np.sin(phase)
    return np.where(phase > 0.0, swing_z, 0.0)


class _ObsHistory:
    """Fixed-length ring buffer. Flattened output is oldest frame first."""

    def __init__(self, example: np.ndarray, length: int):
        self.length = length
        self.buf = np.broadcast_to(example, (length,) + example.shape).copy()
        self.idx = 0

    def push(self, x: np.ndarray) -> None:
        self.buf[self.idx] = x
        self.idx = (self.idx + 1) % self.length

    def flattened(self) -> np.ndarray:
        ks = np.arange(self.length)
        index = (self.idx - 1 - ks) % self.length
        newest_first = self.buf[index].reshape(-1)
        return newest_first[::-1].astype(np.float32, copy=False)


class CaltechBipedEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": int(1.0 / CTRL_DT)}

    def __init__(
        self,
        xml_file: str | None = None,
        render_mode: str | None = None,
        add_observation_noise: bool = ADD_OBSERVATION_NOISE,
        randomize_armature: bool = RANDOMIZE_ARMATURE,
    ):
        super().__init__()
        if xml_file is None:
            xml_file = DEFAULT_XML
        xml_file = os.path.abspath(os.path.expanduser(xml_file))

        self.model = mujoco.MjModel.from_xml_path(xml_file)
        self.model.opt.timestep = SIM_DT
        self.data = mujoco.MjData(self.model)
        self.ctrl_dt = CTRL_DT
        self.n_substeps = int(round(self.ctrl_dt / self.model.opt.timestep))
        self.dt = self.ctrl_dt
        self.add_observation_noise = add_observation_noise
        self.randomize_armature = randomize_armature
        self.render_mode = render_mode
        self._renderer = None

        self._init_q = np.array(self.model.keyframe("home").qpos, dtype=np.float64)
        self._default_q_joints = self._init_q[7:].copy()
        self._nominal_armature = self.model.dof_armature.copy()

        q_j_min, q_j_max = self.model.jnt_range[1:].T
        center = (q_j_min + q_j_max) / 2
        radius = q_j_max - q_j_min
        soft = 0.95
        self._q_j_min = q_j_min
        self._q_j_max = q_j_max
        self._soft_q_j_min = center - 0.5 * radius * soft
        self._soft_q_j_max = center + 0.5 * radius * soft

        self._action_target_offsets = np.array(
            [
                ACTION_TARGET_OFFSETS[mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)]
                for i in range(self.model.nu)
            ],
            dtype=np.float64,
        )
        self._feet_site_id = np.array([self.model.site(name).id for name in FEET_SITES])
        self._feet_geom_id = np.array([self.model.geom(name).id for name in FEET_GEOMS])
        self._floor_geom_id = self.model.geom("floor").id

        def joint_indices(names):
            return np.array(
                [
                    self.model.joint(f"{side}_{name}").qposadr[0] - 7
                    for side in SIDES
                    for name in names
                ],
                dtype=np.int32,
            )

        self._hip_indices = joint_indices(HIP_JOINT_NAMES)
        self._knee_indices = joint_indices(KNEE_JOINT_NAMES)

        self._sensor_adr = {}
        for name in (
            GRAVITY_SENSOR,
            LOCAL_LINVEL_SENSOR,
            GYRO_SENSOR,
            GLOBAL_LINVEL_SENSOR,
            GLOBAL_ANGVEL_SENSOR,
        ):
            sensor = self.model.sensor(name)
            adr = int(self.model.sensor_adr[sensor.id])
            dim = int(self.model.sensor_dim[sensor.id])
            self._sensor_adr[name] = (adr, dim)

        foot_adr = []
        for site in FEET_SITES:
            sensor = self.model.sensor(f"{site}_global_linvel")
            adr = int(self.model.sensor_adr[sensor.id])
            dim = int(self.model.sensor_dim[sensor.id])
            foot_adr.append(list(range(adr, adr + dim)))
        self._foot_linvel_adr = np.array(foot_adr, dtype=np.int32)

        n_joint = int(self._default_q_joints.shape[0])
        nu = int(self.model.nu)
        n_feet = len(FEET_SITES)

        def tile(key: str, n: int) -> np.ndarray:
            return np.full(n, OBS_SCALE[key], dtype=np.float32)

        self._actor_obs_scale = np.concatenate(
            [
                tile("base_lin_vel", 3),
                tile("base_ang_vel", 3),
                tile("upvector", 3),
                tile("command", 3),
                tile("joint_pos_err", n_joint),
                tile("joint_vel", n_joint),
                tile("last_action", nu),
                tile("phase", 2 * n_feet),
            ]
        )
        self._obs_dim = int(self._actor_obs_scale.shape[0])

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(nu,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-LIMIT_OBSERVATIONS,
            high=LIMIT_OBSERVATIONS,
            shape=(HISTORY_LEN * self._obs_dim,),
            dtype=np.float32,
        )

        self._command = np.zeros(3, dtype=np.float64)
        self._last_act = np.zeros(nu, dtype=np.float64)
        self._phase = np.zeros(n_feet, dtype=np.float64)
        self._phase_dt = np.array([2 * np.pi * self.ctrl_dt / GAIT_PERIOD])
        self._feet_air_time = np.zeros(n_feet, dtype=np.float64)
        self._last_contact = np.zeros(n_feet, dtype=bool)
        self._contact_match_steps = 0.0
        self._swing_peak = np.zeros(n_feet, dtype=np.float64)
        self._obs_history = _ObsHistory(np.zeros(self._obs_dim, dtype=np.float32), HISTORY_LEN)

    def _sensor(self, name: str) -> np.ndarray:
        adr, dim = self._sensor_adr[name]
        return self.data.sensordata[adr : adr + dim]

    def _geoms_colliding(self, geom1: int, geom2: int) -> bool:
        for i in range(self.data.ncon):
            g1, g2 = self.data.contact[i].geom
            if (g1 == geom1 and g2 == geom2) or (g1 == geom2 and g2 == geom1):
                if self.data.contact[i].dist < 0:
                    return True
        return False

    def _sample_command(self) -> np.ndarray:
        if self.np_random.random() < 0.1:
            return np.zeros(3, dtype=np.float64)
        return np.array(
            [
                self.np_random.uniform(-0.2, 0.2),
                self.np_random.uniform(-0.2, 0.2),
                self.np_random.uniform(-0.2, 0.2),
            ],
            dtype=np.float64,
        )

    def _randomize_armature(self) -> None:
        self.model.dof_armature[:] = self._nominal_armature
        if not self.randomize_armature:
            return
        self.model.dof_armature[6:] = self.np_random.uniform(
            ARMATURE_MIN, ARMATURE_MAX, size=self.model.nv - 6
        )

    def _current_obs(self, last_act: np.ndarray, phase: np.ndarray) -> np.ndarray:
        q_joints = self.data.qpos[7:]
        q_vel = self.data.qvel[6:]
        phase_feat = np.concatenate([np.cos(phase), np.sin(phase)])
        obs = np.concatenate(
            [
                self._sensor(LOCAL_LINVEL_SENSOR),
                self._sensor(GYRO_SENSOR),
                self._sensor(GRAVITY_SENSOR),
                self._command,
                q_joints - self._default_q_joints,
                q_vel,
                last_act,
                phase_feat,
            ]
        )
        if self.add_observation_noise:
            n_joints = self.model.nv - 6
            noise = np.concatenate(
                [
                    self.np_random.uniform(-OBS_NOISE_BASE_LIN_VEL, OBS_NOISE_BASE_LIN_VEL, size=3),
                    self.np_random.uniform(-OBS_NOISE_BASE_ANG_VEL, OBS_NOISE_BASE_ANG_VEL, size=3),
                    self.np_random.uniform(-OBS_NOISE_BASE_ROT, OBS_NOISE_BASE_ROT, size=3),
                    np.zeros(3),
                    self.np_random.uniform(-OBS_NOISE_JOINT_POS, OBS_NOISE_JOINT_POS, size=n_joints),
                    self.np_random.uniform(-OBS_NOISE_JOINT_VEL, OBS_NOISE_JOINT_VEL, size=n_joints),
                    np.zeros(self.model.nu),
                    np.zeros(2 * len(FEET_SITES)),
                ]
            )
            obs = obs + noise
        obs = obs * self._actor_obs_scale
        return np.clip(obs, -LIMIT_OBSERVATIONS, LIMIT_OBSERVATIONS).astype(np.float32)

    def _policy_obs(self) -> np.ndarray:
        return self._obs_history.flattened()

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._randomize_armature()

        qpos = self._init_q.copy()
        qvel = np.zeros(self.model.nv, dtype=np.float64)
        qpos[0:2] += self.np_random.uniform(-0.5, 0.5, size=2)
        yaw = float(self.np_random.uniform(-np.pi, np.pi))
        yaw_quat = np.zeros(4)
        mujoco.mju_axisAngle2Quat(yaw_quat, np.array([0.0, 0.0, 1.0]), yaw)
        quat = np.zeros(4)
        mujoco.mju_mulQuat(quat, qpos[3:7], yaw_quat)
        qpos[3:7] = quat
        qpos[7:] *= 1.0 + self.np_random.uniform(-0.1, 0.1, size=self.model.nu)
        qvel[0:6] = self.np_random.uniform(-1.0, 1.0, size=6)

        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        self.data.ctrl[:] = qpos[7:]
        mujoco.mj_forward(self.model, self.data)

        self._command = self._sample_command()
        base_phase = float(self.np_random.uniform(-np.pi, np.pi))
        self._phase = np.fmod(np.array([base_phase, base_phase + np.pi]) + np.pi, 2 * np.pi) - np.pi
        nu = self.action_space.shape[0]
        self._last_act = np.zeros(nu, dtype=np.float64)
        self._feet_air_time = np.zeros(len(FEET_SITES), dtype=np.float64)
        self._last_contact = np.zeros(len(FEET_SITES), dtype=bool)
        self._contact_match_steps = 0.0
        self._swing_peak = np.zeros(len(FEET_SITES), dtype=np.float64)

        current = self._current_obs(self._last_act, self._phase)
        self._obs_history = _ObsHistory(current, HISTORY_LEN)
        return self._policy_obs(), {"command": self._command.copy()}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        offsets = np.where(
            action >= 0.0,
            action * self._action_target_offsets[:, 1],
            -action * self._action_target_offsets[:, 0],
        )
        motor_targets = np.clip(self._default_q_joints + offsets, self._q_j_min, self._q_j_max)
        self.data.ctrl[:] = motor_targets
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

        contact = np.array(
            [self._geoms_colliding(g, self._floor_geom_id) for g in self._feet_geom_id],
            dtype=bool,
        )
        contact_filt = contact | self._last_contact
        contact_match = bool(np.all(contact == (self._phase < 0.0)))
        # Reward uses the gait counters from the previous step, matching BipedSim.
        first_contact = (self._feet_air_time > 0.0) & contact_filt
        terminated = self._terminated()
        reward, reward_terms = self._reward(action, first_contact, contact, terminated)

        if contact_match:
            self._contact_match_steps += 1.0
        else:
            self._contact_match_steps = 0.0
        feet_z = self.data.site_xpos[self._feet_site_id, 2]
        self._swing_peak = np.maximum(self._swing_peak, feet_z)
        self._feet_air_time = (self._feet_air_time + self.ctrl_dt) * (~contact)
        self._swing_peak = self._swing_peak * (~contact)
        self._phase = np.fmod(self._phase + self._phase_dt + np.pi, 2 * np.pi) - np.pi
        self._last_contact = contact
        self._last_act = action

        current = self._current_obs(action, self._phase)
        self._obs_history.push(current)
        info = {
            "command": self._command.copy(),
            "reward_terms": reward_terms,
            "contact": contact.copy(),
        }
        return self._policy_obs(), float(reward), bool(terminated), False, info

    def _terminated(self) -> bool:
        gravity = self._sensor(GRAVITY_SENSOR)
        return bool(
            gravity[-1] < 0.0
            or np.isnan(self.data.qpos).any()
            or np.isnan(self.data.qvel).any()
        )

    def _reward(self, action, first_contact, contact, terminated) -> tuple[float, dict]:
        cmd = self._command
        lin = self._sensor(LOCAL_LINVEL_SENSOR)
        gyro = self._sensor(GYRO_SENSOR)
        global_linvel = self._sensor(GLOBAL_LINVEL_SENSOR)
        global_angvel = self._sensor(GLOBAL_ANGVEL_SENSOR)
        grav = self._sensor(GRAVITY_SENSOR)

        tracking_lin = np.exp(-np.sum(np.square(cmd[:2] - lin[:2])))
        tracking_ang = np.exp(-np.square(cmd[2] - gyro[2]))
        lin_vel_z = np.square(global_linvel[2])
        ang_vel_xy = np.sum(np.square(global_angvel[:2]))
        orientation = np.sum(np.square(grav[:2]))
        base_height = np.square(self.data.qpos[2] - DESIRED_HEIGHT)

        torques = np.sum(np.abs(self.data.actuator_force))
        action_rate = np.sum(np.square(action - self._last_act) / self.ctrl_dt)

        feet_vel_xy = self.data.sensordata[self._foot_linvel_adr][..., :2]
        feet_slip = np.sum(np.linalg.norm(feet_vel_xy, axis=-1) * contact)

        cmd_norm = np.linalg.norm(cmd)
        air_time = (self._feet_air_time - 0.2) * first_contact
        air_time = np.clip(air_time, a_min=None, a_max=0.3)
        feet_air_time = np.sum(air_time) * (cmd_norm > 0.1)

        feet_height_error = self._swing_peak / DESIRED_FOOT_HEIGHT - 1.0
        feet_height = np.sum(np.square(feet_height_error) * first_contact)

        foot_z = self.data.site_xpos[self._feet_site_id, 2]
        rz = _foot_height(self._phase)
        feet_phase = np.sum(np.exp(-np.square((foot_z - rz) / FEET_PHASE_STD)))

        alpha = np.clip(self._contact_match_steps / 10.0, 0.0, 1.0)
        schedule = 0.2 * (1.0 - alpha) + 1.0 * alpha
        contact_schedule = schedule if bool(np.all(contact == (self._phase < 0.0))) else 0.0

        q = self.data.qpos[7:]
        joint_deviation_hip = np.sum(np.abs(q[self._hip_indices] - self._default_q_joints[self._hip_indices])) * (
            np.abs(cmd[1]) > 0.1
        )
        joint_deviation_knee = np.sum(np.abs(q[self._knee_indices] - self._default_q_joints[self._knee_indices]))
        out_of_limits = -np.clip(q - self._soft_q_j_min, a_min=None, a_max=0.0)
        out_of_limits += np.clip(q - self._soft_q_j_max, a_min=0.0, a_max=None)
        dof_pos_limits = np.sum(out_of_limits)
        pose = np.sum(np.square(q - self._default_q_joints))

        terms = {
            "tracking_lin_vel": tracking_lin,
            "tracking_ang_vel": tracking_ang,
            "lin_vel_z": lin_vel_z,
            "ang_vel_xy": ang_vel_xy,
            "orientation": orientation,
            "base_height": base_height,
            "torques": torques,
            "action_rate": action_rate,
            "feet_air_time": feet_air_time,
            "feet_slip": feet_slip,
            "feet_height": feet_height,
            "feet_phase": feet_phase,
            "contact_schedule": contact_schedule,
            "termination": float(terminated),
            "joint_deviation_knee": joint_deviation_knee,
            "joint_deviation_hip": joint_deviation_hip,
            "dof_pos_limits": dof_pos_limits,
            "pose": pose,
        }
        scaled = {k: float(REWARD_SCALES[k] * v) for k, v in terms.items()}
        total = float(np.clip(sum(scaled.values()) * self.ctrl_dt, 0.0, 10000.0))
        return total, scaled

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
        self._renderer.update_scene(self.data, camera="front")
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        super().close()


def register():
    if "CaltechBiped-v0" not in gym.envs.registry:
        gym.register(
            id="CaltechBiped-v0",
            entry_point=f"{__name__}:CaltechBipedEnv",
            max_episode_steps=1000,
        )


register()
