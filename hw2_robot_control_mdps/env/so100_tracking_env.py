from pathlib import Path
import numpy as np
import mujoco
import mujoco.viewer
import gymnasium as gym
from gymnasium import spaces

from exercises.ex3 import *

class SO100TrackEnv(gym.Env):
    xml_path: Path
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, xml_path: Path, render_mode=None):
        self.xml_path = xml_path
        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)

        # Define Observation and Action Spaces
        obs = self._get_obs()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float64)
        self.action_space = spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)
        
        # Rendering
        self.render_mode = render_mode
        self.viewer = None

        # Timestep & Episode
        self.sim_timestep = self.model.opt.timestep # 0.002s (500 Hz)
        self.ctrl_decimation = 50 # makes control frequency 10 Hz
        self.ctrl_timestep = self.sim_timestep * self.ctrl_decimation # 0.1
        self.max_episode_length_s = 10
        self.max_episode_length = int(self.max_episode_length_s / self.ctrl_timestep) # 100 steps per episode
        self.current_step = 0

        # Deafult robot home position
        self.default_qpos = np.array([0.0, -1.57, 1.0, 1.0, 0.0, 0.02239])

        # Evaluation metrics
        self.ee_tracking_error = 0.0
        self.prev_ee_tracking_error = None
        self.prev_ctrl_target = None

        # Keep reward terms explicit so the shaping is easy to read and tune.
        self.progress_weight = 20.0
        self.error_weight = 3.0
        self.jerk_weight = 0.01
        self.velocity_weight = 0.001
        self.success_bonus = 1.0
        self.success_threshold = 5e-3

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        mujoco.mj_resetData(self.model, self.data)

        # Reset Robot to home position
        self.data.qpos[:] = reset_robot(self.default_qpos)
        mujoco.mj_forward(self.model, self.data)

        # Reset target position around robot base
        base_pos = self.data.body("Base").xpos.copy()
        self.data.mocap_pos[0] = reset_target_position(base_pos)
        mujoco.mj_forward(self.model, self.data)

        self.ee_tracking_error = np.linalg.norm(self.data.site("ee_site").xpos - self.data.mocap_pos[0])
        # Store previous-step values so reward can compare motion over time.
        self.prev_ee_tracking_error = self.ee_tracking_error
        self.prev_ctrl_target = None

        self.current_step = 0
        return self._get_obs(), {}

    def _process_action(self, action):
        return process_action(action, self.model.jnt_range)

    def compute_reward(self, ctrl_target):
        curr_error = self.ee_tracking_error
        prev_error = self.prev_ee_tracking_error if self.prev_ee_tracking_error is not None else curr_error

        # Reward progress toward the target instead of only rewarding the final distance.
        progress_reward = self.progress_weight * (prev_error - curr_error)
        error_penalty = -self.error_weight * curr_error

        # Penalize abrupt control target changes to discourage jerky motion.
        jerk_penalty = 0.0
        if self.prev_ctrl_target is not None:
            jerk_penalty = -self.jerk_weight * np.sum((ctrl_target - self.prev_ctrl_target) ** 2)

        velocity_penalty = -self.velocity_weight * np.sum(self.data.qvel[:] ** 2)
        success_reward = self.success_bonus if curr_error < self.success_threshold else 0.0
        return progress_reward + error_penalty + jerk_penalty + velocity_penalty + success_reward

    def step(self, action):
        ctrl_target = self._process_action(action)
        self.data.ctrl[:] = ctrl_target
        for _ in range(self.ctrl_decimation): 
            mujoco.mj_step(self.model, self.data)
        self.ee_tracking_error = np.linalg.norm(self.data.site("ee_site").xpos - self.data.mocap_pos[0])
        reward = self.compute_reward(ctrl_target)
        self.prev_ee_tracking_error = self.ee_tracking_error
        self.prev_ctrl_target = ctrl_target.copy()

        terminated = False
        truncated = False
        self.current_step += 1
        if self.current_step >= self.max_episode_length:
            truncated = True
        obs = self._get_obs()
        
        if self.render_mode == "human":
            self.render()

        # Extra info as metrics for evaluation
        info = {"ee_tracking_error": self.ee_tracking_error.item()}
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        qpos = self.data.qpos.flat[:].copy()
        ee_pos_w = self.data.site("ee_site").xpos.copy()
        ee_rot_w = self.data.site("ee_site").xmat.reshape(3, 3)
        base_pos_w = self.data.body("Base").xpos.copy()
        base_rot_w = self.data.body("Base").xmat.reshape(3, 3)
        target_pos_w = self.data.mocap_pos[0].copy()        
        return get_obs(qpos, ee_pos_w, ee_rot_w, base_pos_w, base_rot_w, target_pos_w)

    def render(self):
        if self.render_mode != "human":
            return
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        # Update the viewer's copy of the data
        self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
