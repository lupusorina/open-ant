import gymnasium as gym
import mujoco
import numpy as np

from ant_mujoco import AntEnv


class AntEnvDR(AntEnv):
    """AntEnv with domain randomization of torso mass, floor friction, and actuator time constants."""

    def __init__(
        self,
        *args,
        mass_scale_range: tuple[float, float] = (0.7, 2.2),
        friction_range: tuple[float, float] = (0.45, 0.9),
        timeconst_range: tuple[float, float] = (0.08, 0.45),
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.mass_scale_range = mass_scale_range
        self.friction_range = friction_range
        self.timeconst_range = timeconst_range

        self._torso_id = self.model.body("torso").id
        self._floor_id = self.model.geom("floor").id
        self.nominal_torso_mass = float(self.model.body(self._torso_id).mass[0])
        self.nominal_floor_friction = float(self.model.geom_friction[self._floor_id, 0])
        self.nominal_actuator_timeconst = self.model.actuator_dynprm[:, 0].copy()

    def reset(self, seed=None, options=None):
        # Seed (if requested) before resampling, so DR draws are reproducible.
        gym.Env.reset(self, seed=seed)
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        mass_scale = self.np_random.uniform(*self.mass_scale_range)
        self.model.body_mass[self._torso_id] = self.nominal_torso_mass * mass_scale
        mujoco.mj_setConst(self.model, self.data)

        self.model.geom_friction[self._floor_id, 0] = self.np_random.uniform(*self.friction_range)

        timeconst = self.np_random.uniform(*self.timeconst_range)
        self.model.actuator_dynprm[:, 0] = timeconst

        # Produce the initial observation using the already-randomized model; avoid re-seeding.
        return super().reset(seed=None, options=options)
