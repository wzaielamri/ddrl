from gym.envs.mujoco.half_cheetah_v3 import HalfCheetahEnv
from mujoco_py import functions
import numpy as np

"""
    Extending the Cheetah environment to work with Mujoco 2.
    In Mujoco_2 the contact forces are not directly calculated,
    but calculation must be explicitly invoked.
"""


class CheetahEnvMujoco2(HalfCheetahEnv):
    @property
    def contact_forces(self):
        # Added required call in Mujoco 2 to explicitly compute contact forces
        functions.mj_rnePostConstraint(self.model, self.data)
        raw_contact_forces = self.sim.data.cfrc_ext
        min_value, max_value = self._contact_force_range
        contact_forces = np.clip(raw_contact_forces, min_value, max_value)
        return contact_forces
