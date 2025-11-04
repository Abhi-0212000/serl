from pathlib import Path
from typing import Any, Dict, Literal, Tuple

import gym
import mujoco
import numpy as np
from gym import spaces

from trossen_sim.controllers import opspace
from trossen_sim.mujoco_gym_env import GymRenderingSpec, MujocoGymEnv

_HERE = Path(__file__).parent
_XML_PATH = _HERE / "xmls" / "arena.xml"

# Home configuration for WidowX AI (6 DOF): [joint_0, joint_1, joint_2, joint_3, joint_4, joint_5]
_TROSSEN_HOME = np.asarray([0.0, 1.57, 1.18, 0.0, 0.0, 0.0])  # Neutral pose with arm extended

# Cartesian workspace bounds [x_min, y_min, z_min], [x_max, y_max, z_max]
_CARTESIAN_BOUNDS = np.asarray([[0.15, -0.25, 0.0], [0.5, 0.25, 0.4]])

# Sampling bounds for block placement [x_min, y_min], [x_max, y_max]
_SAMPLING_BOUNDS = np.asarray([[0.2, -0.2], [0.45, 0.2]])


class TrossenPickCubeGymEnv(MujocoGymEnv):
    """Gym environment for Trossen WidowX AI pick-and-place task."""

    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(
        self,
        action_scale: np.ndarray = np.asarray([0.1, 1]),
        seed: int = 0,
        control_dt: float = 0.02,
        physics_dt: float = 0.002,
        time_limit: float = 10.0,
        render_spec: GymRenderingSpec = GymRenderingSpec(),
        render_mode: Literal["rgb_array", "human"] = "rgb_array",
        image_obs: bool = False,
    ):
        self._action_scale = action_scale

        super().__init__(
            xml_path=_XML_PATH,
            seed=seed,
            control_dt=control_dt,
            physics_dt=physics_dt,
            time_limit=time_limit,
            render_spec=render_spec,
        )

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
            ],
            "render_fps": int(np.round(1.0 / self.control_dt)),
        }

        self.render_mode = render_mode
        self.camera_id = (0,)  # Only front camera for now
        self.image_obs = image_obs

        # Caching: WidowX AI has 6 DOF joints (joint_0 to joint_5)
        self._trossen_dof_ids = np.asarray(
            [self._model.joint(f"joint_{i}").id for i in range(6)]
        )
        self._trossen_ctrl_ids = np.asarray(
            [self._model.actuator(f"actuator_{i}").id for i in range(6)]
        )
        self._gripper_ctrl_id = self._model.actuator("gripper_actuator").id
        self._ee_site_id = self._model.site("ee_gripper").id
        self._block_z = self._model.geom("block").size[2]

        # Define observation space
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "trossen/tcp_pos": spaces.Box(
                            -np.inf, np.inf, shape=(3,), dtype=np.float32
                        ),
                        "trossen/tcp_vel": spaces.Box(
                            -np.inf, np.inf, shape=(3,), dtype=np.float32
                        ),
                        "trossen/gripper_pos": spaces.Box(
                            -np.inf, np.inf, shape=(1,), dtype=np.float32
                        ),
                        "block_pos": spaces.Box(
                            -np.inf, np.inf, shape=(3,), dtype=np.float32
                        ),
                    }
                ),
            }
        )

        if self.image_obs:
            self.observation_space = gym.spaces.Dict(
                {
                    "state": gym.spaces.Dict(
                        {
                            "trossen/tcp_pos": spaces.Box(
                                -np.inf, np.inf, shape=(3,), dtype=np.float32
                            ),
                            "trossen/tcp_vel": spaces.Box(
                                -np.inf, np.inf, shape=(3,), dtype=np.float32
                            ),
                            "trossen/gripper_pos": spaces.Box(
                                -np.inf, np.inf, shape=(1,), dtype=np.float32
                            ),
                        }
                    ),
                    "images": gym.spaces.Dict(
                        {
                            "front": gym.spaces.Box(
                                low=0,
                                high=255,
                                shape=(render_spec.height, render_spec.width, 3),
                                dtype=np.uint8,
                            ),
                        }
                    ),
                }
            )

        # Action space: [dx, dy, dz, gripper_action]
        self.action_space = gym.spaces.Box(
            low=np.asarray([-1.0, -1.0, -1.0, -1.0]),
            high=np.asarray([1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32,
        )

        # Create separate renderers for image observations
        self._renderers = {}
        if self.image_obs:
            for cam_id in self.camera_id:
                try:
                    renderer = mujoco.Renderer(
                        model=self._model,
                        height=render_spec.height,
                        width=render_spec.width,
                    )
                except Exception as e:
                    print(f"Warning: Failed to create renderer for camera {cam_id}: {e}")
                    renderer = None
                self._renderers[cam_id] = renderer

        # For human rendering mode
        self._viewer = None
        if self.render_mode == "human":
            from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer

            self._viewer = MujocoRenderer(
                self.model,
                self.data,
            )

    def reset(
        self, seed=None, **kwargs
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset the environment."""
        mujoco.mj_resetData(self._model, self._data)

        # Reset arm to home position
        self._data.qpos[self._trossen_dof_ids] = _TROSSEN_HOME
        mujoco.mj_forward(self._model, self._data)

        # Reset mocap body to home position
        tcp_pos = self._data.sensor("trossen/ee_pos").data
        self._data.mocap_pos[0] = tcp_pos

        # Sample a new block position
        block_xy = np.random.uniform(*_SAMPLING_BOUNDS)
        self._data.jnt("block").qpos[:3] = (*block_xy, self._block_z)
        mujoco.mj_forward(self._model, self._data)

        # Cache the initial block height
        self._z_init = self._data.sensor("block_pos").data[2]
        self._z_success = self._z_init + 0.15  # Success threshold: lift 15cm

        obs = self._compute_observation()
        return obs, {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment.

        Args:
            action: [dx, dy, dz, gripper_action]

        Returns:
            observation, reward, terminated, truncated, info
        """
        x, y, z, grasp = action

        # Update mocap position
        pos = self._data.mocap_pos[0].copy()
        dpos = np.asarray([x, y, z]) * self._action_scale[0]
        npos = np.clip(pos + dpos, *_CARTESIAN_BOUNDS)
        self._data.mocap_pos[0] = npos

        # Update gripper
        g = self._data.ctrl[self._gripper_ctrl_id] / 0.044  # Normalize by max gripper range
        dg = grasp * self._action_scale[1]
        ng = np.clip(g + dg, 0.0, 1.0)
        self._data.ctrl[self._gripper_ctrl_id] = ng * 0.044

        # Apply operational space control
        for _ in range(self._n_substeps):
            tau = opspace(
                model=self._model,
                data=self._data,
                site_id=self._ee_site_id,
                dof_ids=self._trossen_dof_ids,
                pos=self._data.mocap_pos[0],
                ori=self._data.mocap_quat[0],
                joint=_TROSSEN_HOME,
                gravity_comp=True,
            )
            self._data.ctrl[self._trossen_ctrl_ids] = tau
            mujoco.mj_step(self._model, self._data)

        obs = self._compute_observation()
        rew = self._compute_reward()
        terminated = self.time_limit_exceeded()

        return obs, rew, terminated, False, {}

    def render(self):
        """Render the environment."""
        # For image observations: render offscreen
        if self.image_obs and self._renderers:
            rendered_frames = []
            for cam_id in self.camera_id:
                renderer = self._renderers[cam_id]
                renderer.update_scene(self._data, camera=cam_id)
                rendered_frames.append(renderer.render())
            return rendered_frames[0] if len(rendered_frames) == 1 else rendered_frames

        # For human visualization
        if self.render_mode == "human" and self._viewer is not None:
            return self._viewer.render("human")

        return None

    # Helper methods

    def _compute_observation(self) -> Dict[str, np.ndarray]:
        """Compute the observation dictionary."""
        obs = {}
        obs["state"] = {}

        # End-effector position
        tcp_pos = self._data.sensor("trossen/ee_pos").data
        obs["state"]["trossen/tcp_pos"] = tcp_pos.astype(np.float32)

        # End-effector velocity
        tcp_vel = self._data.sensor("trossen/ee_vel").data
        obs["state"]["trossen/tcp_vel"] = tcp_vel.astype(np.float32)

        # Gripper position (normalized)
        gripper_pos = np.array(
            self._data.ctrl[self._gripper_ctrl_id] / 0.044, dtype=np.float32
        )
        obs["state"]["trossen/gripper_pos"] = gripper_pos

        if self.image_obs:
            obs["images"] = {}
            rendered = self.render()
            obs["images"]["front"] = rendered
        else:
            block_pos = self._data.sensor("block_pos").data.astype(np.float32)
            obs["state"]["block_pos"] = block_pos

        return obs

    def _compute_reward(self) -> float:
        """Compute the reward for the current state."""
        block_pos = self._data.sensor("block_pos").data
        tcp_pos = self._data.sensor("trossen/ee_pos").data
        dist = np.linalg.norm(block_pos - tcp_pos)

        # Reward for getting close to the block
        r_close = np.exp(-20 * dist)

        # Reward for lifting the block
        r_lift = (block_pos[2] - self._z_init) / (self._z_success - self._z_init)
        r_lift = np.clip(r_lift, 0.0, 1.0)

        # Combined reward
        rew = 0.3 * r_close + 0.7 * r_lift
        return rew


if __name__ == "__main__":
    env = TrossenPickCubeGymEnv(render_mode="human")
    env.reset()
    for i in range(100):
        env.step(np.random.uniform(-1, 1, 4))
        env.render()
    env.close()
