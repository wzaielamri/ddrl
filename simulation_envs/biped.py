from gym.envs.mujoco.half_cheetah_v3 import HalfCheetahEnv
import numpy as np
import os
from scipy import ndimage
from scipy.signal import convolve2d

DEFAULT_CAMERA_CONFIG = {
    'distance': 15.0,
    'type': 1,  # 1 = Tracking camera, 2 = Fixed
    'trackbodyid': 1,
    'elevation': -20.0,
}


def create_new_hfield(mj_model, smoothness=0.15, bump_scale=2.):
    # Generation of the shape of the height field is taken from the dm_control suite,
    # see dm_control/suite/quadruped.py in the escape task (but we don't use the bowl shape).
    # Their parameters are TERRAIN_SMOOTHNESS = 0.15  # 0.0: maximally bumpy; 1.0: completely smooth.
    # and TERRAIN_BUMP_SCALE = 2  # Spatial scale of terrain bumps (in meters).
    res = mj_model.hfield_ncol[0]
    row_grid, col_grid = np.ogrid[-1:1:res*1j, -1:1:res*1j]
    # Random smooth bumps.
    terrain_size = 2 * mj_model.hfield_size[0, 0]
    bump_res = int(terrain_size / bump_scale)
    bumps = np.random.uniform(smoothness, 1, (bump_res, bump_res))
    smooth_bumps = ndimage.zoom(bumps, res / float(bump_res))
    # Terrain is elementwise product.
    hfield = (smooth_bumps - np.min(smooth_bumps)
              )[0:mj_model.hfield_nrow[0], 0:mj_model.hfield_ncol[0]]
    # Clears a patch shaped like box, assuming robot is placed in center of hfield.
    # Function was implemented in an old rllab version.
    h_center = int(0.5 * hfield.shape[0])
    w_center = int(0.5 * hfield.shape[1])
    patch_size = 8
    fromrow, torow = h_center - \
        int(0.5*patch_size), h_center + int(0.5*patch_size)
    fromcol, tocol = w_center - \
        int(0.5*patch_size), w_center + int(0.5*patch_size)
    # convolve to smoothen edges somewhat, in case hills were cut off
    K = np.ones((patch_size, patch_size)) / patch_size**2
    s = convolve2d(hfield[fromrow-(patch_size-1):torow+(patch_size-1), fromcol-(
        patch_size-1):tocol+(patch_size-1)], K, mode='same', boundary='symm')
    hfield[fromrow-(patch_size-1):torow+(patch_size-1),
           fromcol-(patch_size-1):tocol+(patch_size-1)] = s
    # Last, we lower the hfield so that the centre aligns at zero height
    # (importantly, we use a constant offset of -0.5 for rendering purposes)
    #print(np.min(hfield), np.max(hfield))
    hfield = hfield - np.max(hfield[fromrow:torow, fromcol:tocol])
    mj_model.hfield_data[:] = hfield.ravel()
    #print("Smoothness set to: ", smoothness)


class BipedEnv(HalfCheetahEnv):
    """ Environment with a biped walker - derived from the half_cheetah environment

        Uses a different observation space compared to the half_cheetah environment (less inputs).
        Per default, healthy reward is turned of (unnecessary).

        The environment introduces a heightfield which allows to test or train
        the system in uneven terrain (generating new heightfields has to be explicitly
        called, ideally before a reset of the system).
    """

    def __init__(self, ctrl_cost_weight=0.5, contact_cost_weight=5e-4, healthy_reward=0., hf_smoothness=1.):

        self.ctrl_cost_weight = ctrl_cost_weight
        self.contact_cost_weight = contact_cost_weight
        super().__init__(xml_file=os.path.join(os.path.dirname(__file__), 'assets',
                                               'half_cheetah_hfield.xml'), ctrl_cost_weight=ctrl_cost_weight)

        # Heightfield
        self.hf_smoothness = hf_smoothness
        self.hf_bump_scale = 2.
        create_new_hfield(self.model, self.hf_smoothness, self.hf_bump_scale)

        # Otherwise when learning from scratch might abort
        # This allows for more collisions.
        self.model.nconmax = 500
        self.model.njmax = 2000

    @property
    def contact_forces(self):
        raw_contact_forces = self.sim.data.cfrc_ext
        min_value, max_value = (-1.0, 1.0)
        contact_forces = np.clip(raw_contact_forces, min_value, max_value)
        return contact_forces

    @property
    def contact_cost(self):
        contact_cost = self.contact_cost_weight * np.sum(
            np.square(self.contact_forces))
        return contact_cost

    def create_new_random_hfield(self):
        create_new_hfield(self.model, self.hf_smoothness, self.hf_bump_scale)

    def set_hf_parameter(self, smoothness, bump_scale=None):
        # Setting the parameters for the height field.
        self.hf_smoothness = smoothness
        if bump_scale:
            self.hf_bump_scale = bump_scale

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)

    def step(self, action):
        x_position_before = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.sim.data.qpos[0]
        x_velocity = ((x_position_after - x_position_before)
                      / self.dt)

        ctrl_cost = self.control_cost(action)

        forward_reward = self._forward_reward_weight * x_velocity

        observation = self._get_obs()
        reward = forward_reward - ctrl_cost
        done = False
        info = {
            'x_position': x_position_after,
            'x_velocity': x_velocity,

            'reward_run': forward_reward,
            'reward_ctrl': -ctrl_cost,
            'reward_contact': -self.contact_cost

        }

        return observation, reward, done, info
