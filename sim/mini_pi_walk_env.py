"""Walking task for the High Torque Mini Pi+ Pro humanoid.

Modelled on Gymnasium's ``Humanoid-v5`` (gymnasium/envs/mujoco/humanoid_v5.py):
same reward terms (forward velocity of the centre of mass + healthy bonus
- control cost - contact cost), same observation layout, same reset noise,
same info keys, registered with the same 1000-step time limit.

What differs from Humanoid-v5, and why:

* **PD position control instead of raw torque.** The Mini Pi's motors in
  ``mini_pi.xml`` are torque motors. The README measures that holding the
  ``home`` crouch needs kp ~100 on hips/knees -- below kp 80 the robot topples
  backwards -- so a raw-torque policy would first have to learn a stiff PD
  loop just to stand. Here the policy outputs joint-angle offsets from the
  ``home`` pose, ``q_target = q_home + action_scale * action``, and a PD law
  runs at every physics substep (500 Hz) to produce the torques. Gains come
  from the README table.
* **Only the 12 leg joints are policy-controlled by default.** Waist, arms and
  head are PD-held at ``home`` (``control_upper_body=True`` exposes all 23).
* **Control cost is on normalised torque** ``(tau / tau_max)^2`` because the
  torques here span +-20 N*m rather than Humanoid's +-0.4 ctrl range.
* **Action-rate cost and previous action in the observation.** Standard for PD
  policies; keeps the targets from chattering. Set ``action_rate_cost_weight=0``
  to drop it.
* **Health also checks tilt**, not just base height: a 0.37 m robot that falls
  onto its knees can stay above a pure height threshold.
* **Orientation and roll/pitch-rate costs**, taken from the Caltech biped
  reward (``orientation``, ``ang_vel_xy``). The healthy check is only a hard
  cutoff near 57°, so a leaned gait that still shifts the centre of mass
  scores the same as an upright one. ``orientation`` is ``||g_xy||^2`` of
  gravity in the base frame (0 when upright); ``ang_vel_xy`` damps rocking
  into that lean. Both are heading-independent, so they apply to
  ``back_and_forth`` as well.
* **Reward weights are rescaled** for a robot ~1/4 the height of Humanoid; see
  the comment on ``forward_reward_weight``.

Tasks (``task=``):

* ``"forward"`` (``MiniPiWalk-v0``): progress = centre-of-mass velocity along +x.
* ``"back_and_forth"`` (``MiniPiBackAndForth-v0``): the rule of
  ``embodied_ant_env.BackAndForthTask``. Each episode draws a random reward
  direction; once the robot is outside a circle of ``task_radius`` around
  ``task_origin`` and still heading outward, the direction flips to point back
  at the origin. Progress = centre-of-mass velocity along that direction, and
  the direction in the robot's body frame is appended to the observation (as
  the Ant task does). The base also starts at a random yaw, so the direction is
  not always "straight ahead". Everything else -- healthy bonus, fall
  termination, control/contact/action-rate/orientation/ang-vel costs -- is the
  same as "forward":
  a biped, unlike the Ant, can fall, so those terms stay.
"""

from __future__ import annotations

import os

import mujoco
import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.envs.mujoco.humanoid_v5 import mass_center
from gymnasium.spaces import Box

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_XML = os.path.join(_HERE, "assets", "hightorque", "scene.xml")

# Humanoid-v5's camera, scaled to a 0.37 m robot. Only used if the XML has no
# camera named "track": Gymnasium renders through that camera when it exists
# (scene.xml defines one that follows the robot).
DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance": 1.5,
    "lookat": np.array((0.0, 0.0, 0.3)),
    "elevation": -20.0,
}

LEG_JOINTS = (
    "l_hip_pitch_joint", "l_hip_roll_joint", "l_thigh_joint",
    "l_calf_joint", "l_ankle_pitch_joint", "l_ankle_roll_joint",
    "r_hip_pitch_joint", "r_hip_roll_joint", "r_thigh_joint",
    "r_calf_joint", "r_ankle_pitch_joint", "r_ankle_roll_joint",
)

# (kp, kd) per joint-name suffix. Hips/knees and ankle kp are from the README
# ("Where the PD gains came from"); the README gives arm/head/waist kp by
# scaling the ankle ratio (2.5 kp per N*m) off each joint's rating, and head
# kp 10 explicitly. kd values other than hip/knee are not in the README; they
# were chosen here and checked by the zero-action stand test in __main__.
PD_GAINS = {
    "hip_pitch_joint": (100.0, 3.5),
    "hip_roll_joint": (100.0, 3.5),
    "thigh_joint": (100.0, 3.5),      # hip yaw
    "calf_joint": (100.0, 3.5),       # knee
    "ankle_pitch_joint": (50.0, 2.0),
    "ankle_roll_joint": (50.0, 2.0),
    "waist_yaw_joint": (40.0, 2.0),   # 2.5 * 16 N*m
    "shoulder_pitch_joint": (25.0, 1.0),  # 2.5 * 10 N*m
    "shoulder_roll_joint": (25.0, 1.0),
    "upper_arm_joint": (25.0, 1.0),
    "elbow_joint": (25.0, 1.0),
    "head_yaw_joint": (10.0, 0.5),
    "head_pitch_joint": (10.0, 0.5),
}


def _gains_for(joint_name: str) -> tuple[float, float]:
    for suffix, gains in PD_GAINS.items():
        if joint_name.endswith(suffix):
            return gains
    raise KeyError(f"No PD gains for joint {joint_name!r}")


class MiniPiWalkEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array", "rgbd_tuple"],
    }

    def __init__(
        self,
        xml_file: str | None = None,
        frame_skip: int = 10,  # 0.002 s * 10 = 0.02 s -> 50 Hz policy
        default_camera_config: dict[str, float | int] = DEFAULT_CAMERA_CONFIG,
        # Humanoid-v5 uses forward 1.25 / healthy 5.0, which balances a ~1.4 m
        # robot running several m/s. A 0.37 m robot walks well under 1 m/s, so
        # with those weights standing still out-earns walking. These keep
        # walking at ~0.5 m/s worth more than the survival bonus alone.
        forward_reward_weight: float = 2.5,
        healthy_reward: float = 1.0,
        ctrl_cost_weight: float = 0.1,
        contact_cost_weight: float = 5e-7,
        contact_cost_range: tuple[float, float] = (-np.inf, 10.0),
        action_rate_cost_weight: float = 0.01,
        # Caltech uses -1 * ||g_xy||^2 and -0.15 * ||w_xy||^2, then multiplies
        # the whole reward by dt and sets it against a tracking term of order
        # 2. This env does not scale by dt, and a walk is worth ~1 per step
        # (forward_reward_weight 2.5 at 0.4 m/s). A 20° lean has ||g_xy||^2 ≈
        # 0.12, so weight 4 costs ~0.5 -- about 0.2 m/s of forward reward --
        # while a 10° walking pitch stays cheap (~0.12).
        orientation_cost_weight: float = 4.0,
        ang_vel_xy_cost_weight: float = 0.1,
        terminate_when_unhealthy: bool = True,
        # Humanoid-v5: (1.0, 2.0) around a 1.4 m start, i.e. down to ~0.7x.
        healthy_z_range: tuple[float, float] = (0.25, 0.6),
        # Base z-axis must stay within this angle of vertical (radians).
        healthy_max_tilt: float = 1.0,
        reset_noise_scale: float = 1e-2,
        action_scale: float = 0.25,
        control_upper_body: bool = False,
        exclude_current_positions_from_observation: bool = True,
        include_cinert_in_observation: bool = True,
        include_cvel_in_observation: bool = True,
        include_qfrc_actuator_in_observation: bool = True,
        include_cfrc_ext_in_observation: bool = True,
        task: str = "forward",
        task_radius: float = 1.0,
        task_origin: tuple[float, float] = (0.0, 0.0),
        **kwargs,
    ):
        if task not in ("forward", "back_and_forth"):
            raise ValueError(f"task must be 'forward' or 'back_and_forth', got {task!r}")
        if xml_file is None:
            xml_file = DEFAULT_XML
        # MujocoEnv resolves bare relative paths against Gymnasium's own assets
        # folder; resolve against the cwd instead.
        xml_file = os.path.abspath(os.path.expanduser(xml_file))

        utils.EzPickle.__init__(
            self,
            xml_file,
            frame_skip,
            default_camera_config,
            forward_reward_weight,
            healthy_reward,
            ctrl_cost_weight,
            contact_cost_weight,
            contact_cost_range,
            action_rate_cost_weight,
            orientation_cost_weight,
            ang_vel_xy_cost_weight,
            terminate_when_unhealthy,
            healthy_z_range,
            healthy_max_tilt,
            reset_noise_scale,
            action_scale,
            control_upper_body,
            exclude_current_positions_from_observation,
            include_cinert_in_observation,
            include_cvel_in_observation,
            include_qfrc_actuator_in_observation,
            include_cfrc_ext_in_observation,
            task,
            task_radius,
            task_origin,
            **kwargs,
        )

        self._forward_reward_weight = forward_reward_weight
        self._healthy_reward = healthy_reward
        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight
        self._contact_cost_range = contact_cost_range
        self._action_rate_cost_weight = action_rate_cost_weight
        self._orientation_cost_weight = orientation_cost_weight
        self._ang_vel_xy_cost_weight = ang_vel_xy_cost_weight
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._healthy_min_up = np.cos(healthy_max_tilt)
        self._reset_noise_scale = reset_noise_scale
        self._action_scale = action_scale
        self._control_upper_body = control_upper_body
        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )
        self._include_cinert_in_observation = include_cinert_in_observation
        self._include_cvel_in_observation = include_cvel_in_observation
        self._include_qfrc_actuator_in_observation = include_qfrc_actuator_in_observation
        self._include_cfrc_ext_in_observation = include_cfrc_ext_in_observation
        self._back_and_forth = task == "back_and_forth"
        self._task_radius = task_radius
        self._task_origin = np.asarray(task_origin, dtype=np.float64)
        # Unit reward direction in the world xy-plane; fixed at +x for "forward".
        self._direction = np.array([1.0, 0.0])

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=None,
            default_camera_config=default_camera_config,
            **kwargs,
        )

        self.metadata = {
            "render_modes": ["human", "rgb_array", "depth_array", "rgbd_tuple"],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

        self._setup_pd()

        # Start from the `home` keyframe rather than the all-zero pose
        # MujocoEnv captured at load time.
        home = self.model.key("home")
        self.init_qpos = home.qpos.copy()
        self.init_qvel = np.zeros(self.model.nv)

        n_act = len(self._policy_act_ids)
        self.action_space = Box(low=-1.0, high=1.0, shape=(n_act,), dtype=np.float32)
        self._prev_action = np.zeros(n_act)

        obs_size = self.data.qpos.size + self.data.qvel.size
        obs_size -= 2 * exclude_current_positions_from_observation
        obs_size += self.data.cinert[1:].size * include_cinert_in_observation
        obs_size += self.data.cvel[1:].size * include_cvel_in_observation
        obs_size += (self.data.qvel.size - 6) * include_qfrc_actuator_in_observation
        obs_size += self.data.cfrc_ext[1:].size * include_cfrc_ext_in_observation
        obs_size += n_act  # previous action
        obs_size += 2 * self._back_and_forth  # reward direction in body frame
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

    # ------------------------------------------------------------------ PD

    def _setup_pd(self):
        m = self.model
        joint_ids = m.actuator_trnid[:, 0]
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in joint_ids]
        # Actuator order (l leg, r leg, waist, l arm, r arm, head) differs from
        # qpos order (r leg, l leg, ...), so index through the joint tables.
        self._act_qpos_adr = m.jnt_qposadr[joint_ids]
        self._act_dof_adr = m.jnt_dofadr[joint_ids]
        self._kp = np.array([_gains_for(n)[0] for n in names])
        self._kd = np.array([_gains_for(n)[1] for n in names])
        self._tau_max = m.actuator_ctrlrange[:, 1].copy()
        self._q_lo = m.jnt_range[joint_ids, 0]
        self._q_hi = m.jnt_range[joint_ids, 1]
        self._q_home = m.key("home").qpos[self._act_qpos_adr].copy()

        if self._control_upper_body:
            self._policy_act_ids = np.arange(m.nu)
        else:
            self._policy_act_ids = np.array([names.index(j) for j in LEG_JOINTS])

    def _targets(self, action):
        q_target = self._q_home.copy()
        q_target[self._policy_act_ids] += self._action_scale * action
        return np.clip(q_target, self._q_lo, self._q_hi)

    def _pd_simulate(self, q_target, n_frames):
        for _ in range(n_frames):
            q = self.data.qpos[self._act_qpos_adr]
            qd = self.data.qvel[self._act_dof_adr]
            tau = self._kp * (q_target - q) - self._kd * qd
            self.data.ctrl[:] = np.clip(tau, -self._tau_max, self._tau_max)
            mujoco.mj_step(self.model, self.data)
        # Same as MujocoEnv._step_mujoco_simulation: fill cfrc_ext etc.
        mujoco.mj_rnePostConstraint(self.model, self.data)

    # ------------------------------------------------------------- reward

    @property
    def is_healthy(self):
        min_z, max_z = self._healthy_z_range
        height_ok = min_z < self.data.qpos[2] < max_z
        # z-component of the base's z-axis in world frame (1 = upright).
        up = self.data.xmat[1].reshape(3, 3)[2, 2]
        return bool(height_ok and up > self._healthy_min_up)

    @property
    def healthy_reward(self):
        return self.is_healthy * self._healthy_reward

    def control_cost(self):
        return self._ctrl_cost_weight * np.sum(np.square(self.data.ctrl / self._tau_max))

    @property
    def contact_cost(self):
        contact_cost = self._contact_cost_weight * np.sum(np.square(self.data.cfrc_ext))
        min_cost, max_cost = self._contact_cost_range
        return np.clip(contact_cost, min_cost, max_cost)

    def action_rate_cost(self, action):
        return self._action_rate_cost_weight * np.sum(np.square(action - self._prev_action))

    def orientation_cost(self):
        # Gravity in the base frame. xy is 0 upright, and ||g_xy||^2 = sin^2
        # of the tilt angle. Same quantity as Caltech's `orientation` term.
        grav_body = -self.data.xmat[1].reshape(3, 3)[2]
        return self._orientation_cost_weight * float(np.sum(np.square(grav_body[:2])))

    def ang_vel_xy_cost(self):
        # Free-joint angular velocity is in the base frame; qvel[3:5] is the
        # roll/pitch rate. Yaw (qvel[5]) is left alone so the robot can still
        # turn toward the reward direction.
        return self._ang_vel_xy_cost_weight * float(np.sum(np.square(self.data.qvel[3:5])))

    def _get_rew(self, progress_velocity, action):
        forward_reward = self._forward_reward_weight * progress_velocity
        healthy_reward = self.healthy_reward
        ctrl_cost = self.control_cost()
        contact_cost = self.contact_cost
        action_rate_cost = self.action_rate_cost(action)
        orientation_cost = self.orientation_cost()
        ang_vel_xy_cost = self.ang_vel_xy_cost()

        reward = (
            forward_reward
            + healthy_reward
            - ctrl_cost
            - contact_cost
            - action_rate_cost
            - orientation_cost
            - ang_vel_xy_cost
        )
        reward_info = {
            "reward_survive": healthy_reward,
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_contact": -contact_cost,
            "reward_action_rate": -action_rate_cost,
            "reward_orientation": -orientation_cost,
            "reward_ang_vel_xy": -ang_vel_xy_cost,
        }
        return reward, reward_info

    # ---------------------------------------------------------------- obs

    def _get_obs(self):
        position = self.data.qpos.flatten()
        velocity = self.data.qvel.flatten()
        parts = [position[2:] if self._exclude_current_positions_from_observation else position,
                 velocity]
        if self._include_cinert_in_observation:
            parts.append(self.data.cinert[1:].flatten())
        if self._include_cvel_in_observation:
            parts.append(self.data.cvel[1:].flatten())
        if self._include_qfrc_actuator_in_observation:
            parts.append(self.data.qfrc_actuator[6:].flatten())
        if self._include_cfrc_ext_in_observation:
            parts.append(self.data.cfrc_ext[1:].flatten())
        parts.append(self._prev_action)
        if self._back_and_forth:
            parts.append(self._direction_body())
        return np.concatenate(parts)

    # --------------------------------------------------------- back & forth

    def _direction_body(self):
        """Reward direction in the base's yaw frame: (ahead, left) components."""
        heading = self.data.xmat[1].reshape(3, 3)[:2, 0]
        heading = heading / max(np.linalg.norm(heading), 1e-8)
        left = np.array([-heading[1], heading[0]])
        return np.array([self._direction @ heading, self._direction @ left])

    def _update_direction(self, xy_position):
        """BackAndForthTask's bounce: past the radius and still heading outward
        -> point the direction back at the origin."""
        offset = xy_position - self._task_origin
        if offset @ self._direction > 0 and np.linalg.norm(offset) > self._task_radius:
            self._direction = -offset / np.linalg.norm(offset)

    def _direction_info(self):
        if not self._back_and_forth:
            return {}
        direction_body = self._direction_body()
        return {
            "reward_direction_I_x": self._direction[0],
            "reward_direction_I_y": self._direction[1],
            "reward_direction_B_x": direction_body[0],
            "reward_direction_B_y": direction_body[1],
        }

    def render(self):
        # Draw the reward direction as an arrow above the robot, like AntEnv.
        if self._back_and_forth and self.render_mode in ("human", "rgb_array"):
            x_axis = np.array([self._direction[0], self._direction[1], 0.0])
            z_axis = np.array([0.0, 0.0, 1.0])
            frame = np.array([x_axis, np.cross(z_axis, x_axis), z_axis]).T
            # MuJoCo arrows point along their local z; rotate z onto x_axis.
            z_to_x = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
            self.mujoco_renderer._get_viewer(self.render_mode).add_marker(
                type=mujoco.mjtGeom.mjGEOM_ARROW,
                size=np.array([0.02, 0.02, 0.4]),
                pos=np.array([self.data.qpos[0], self.data.qpos[1], 0.03]),  # on the floor
                mat=(frame @ z_to_x).flatten(),
                rgba=np.array([1.0, 0.0, 0.0, 1.0]),
                label="",
            )
        return super().render()

    # --------------------------------------------------------------- step

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)

        xy_position_before = mass_center(self.model, self.data)
        self._pd_simulate(self._targets(action), self.frame_skip)
        xy_position_after = mass_center(self.model, self.data)

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        # Bounce first, then score progress along the (possibly new) direction,
        # in the same order as BackAndForthTask. For "forward" this is x_velocity.
        if self._back_and_forth:
            self._update_direction(xy_position_after)
        progress_velocity = xy_velocity @ self._direction

        reward, reward_info = self._get_rew(progress_velocity, action)
        self._prev_action = action
        observation = self._get_obs()
        terminated = (not self.is_healthy) and self._terminate_when_unhealthy
        info = {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            **reward_info,
            **self._direction_info(),
        }

        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, False, info

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )
        if self._back_and_forth:
            # Random starting yaw, composed onto the (noisy) home orientation,
            # and a random reward direction, as BackAndForthTask.reset draws.
            yaw = self.np_random.uniform(-np.pi, np.pi)
            yaw_quat = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
            quat = np.zeros(4)
            mujoco.mju_mulQuat(quat, yaw_quat, qpos[3:7])
            qpos[3:7] = quat / np.linalg.norm(quat)
            th = self.np_random.uniform(-np.pi, np.pi)
            self._direction = np.array([np.cos(th), np.sin(th)])
        self.set_state(qpos, qvel)
        self._prev_action = np.zeros(len(self._policy_act_ids))
        return self._get_obs()

    def _get_reset_info(self):
        return {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2),
            **self._direction_info(),
        }


def register():
    """Register ``MiniPiWalk-v0`` and ``MiniPiBackAndForth-v0`` (idempotent)."""
    import gymnasium as gym

    if "MiniPiWalk-v0" not in gym.envs.registry:
        gym.register(
            id="MiniPiWalk-v0",
            entry_point=f"{__name__}:MiniPiWalkEnv",
            max_episode_steps=1000,  # same as Humanoid-v5; 20 s at 50 Hz
        )
    if "MiniPiBackAndForth-v0" not in gym.envs.registry:
        gym.register(
            id="MiniPiBackAndForth-v0",
            entry_point=f"{__name__}:MiniPiWalkEnv",
            max_episode_steps=1000,
            kwargs={"task": "back_and_forth"},
        )


register()


if __name__ == "__main__":
    # Sanity check: with zero action the PD loop should hold `home`.
    env = MiniPiWalkEnv()
    obs, _ = env.reset(seed=0)
    print("obs", obs.shape, "act", env.action_space.shape, "dt", env.dt)
    total = 0.0
    for t in range(1000):
        obs, r, term, trunc, info = env.step(np.zeros(env.action_space.shape))
        total += r
        if term:
            print("fell at step", t)
            break
    print(f"z={env.data.qpos[2]:.3f}  x={info['x_position']:.3f}  return={total:.1f}")
    print("peak |tau|/tau_max:", np.max(np.abs(env.data.ctrl) / env._tau_max).round(3))
