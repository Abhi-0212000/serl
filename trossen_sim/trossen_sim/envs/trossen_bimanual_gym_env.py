"""Trossen bimanual pick-and-place gym environment."""

from pathlib import Path
from typing import Any, Dict, Literal, Tuple

import gym
import mujoco
import numpy as np
from gym import spaces

from trossen_sim.mujoco_gym_env import GymRenderingSpec, MujocoGymEnv
from trossen_sim.envs.utils import compute_distance_reward, denormalize_action, sample_box_pose

_HERE = Path(__file__).parent
_XML_PATH = _HERE / "xmls" / "trossen_ai_scene_joint.xml"

# Start pose for both arms (from trossen_arm_mujoco constants)
# [left_j0, left_j1, left_j2, left_j3, left_j4, left_j5, left_gripper_l, left_gripper_r,
#  right_j0, right_j1, right_j2, right_j3, right_j4, right_j5, right_gripper_l, right_gripper_r]
_START_ARM_POSE = np.array([
    0.0, np.pi / 12, np.pi / 12, 0.0, 0.0, 0.0, 0.044, 0.044,  # Left arm
    0.0, np.pi / 12, np.pi / 12, 0.0, 0.0, 0.0, 0.044, 0.044,  # Right arm
])

# Joint limits for WidowX 250 6DOF arms (approximate, from URDF)
# Format: [min, max] for each joint
_JOINT_LIMITS = np.array([
    [-3.14, 3.14],   # joint_0 (base)
    [-1.88, 1.98],   # joint_1 (shoulder)
    [-2.15, 1.60],   # joint_2 (elbow)
    [-1.75, 2.05],   # joint_3 (wrist_angle)
    [-3.14, 3.14],   # joint_4 (wrist_rotate)
    [-3.14, 3.14],   # joint_5 (ee_rotate)
])

# Gripper limits (in meters, for prismatic joint)
_GRIPPER_LIMITS = np.array([0.0, 0.044])  # 0 = closed, 0.044 = fully open

# Cartesian workspace bounds for sampling
_SAMPLING_BOUNDS = np.array([[-0.1, -0.25], [0.2, 0.25]])  # [x_min, y_min], [x_max, y_max]


class TrossenBimanualPickPlaceGymEnv(MujocoGymEnv):
    """
    Trossen bimanual robot environment for pick-and-place tasks.
    
    Task: Left arm picks up a red cube, hands it over to right arm, right arm places it.
    
    Action Space: 14D continuous
        - left_arm (6D): joint positions for 6DOF arm
        - left_gripper (1D): gripper position (0=closed, 1=open)
        - right_arm (6D): joint positions for 6DOF arm  
        - right_gripper (1D): gripper position (0=closed, 1=open)
    
    Observation Space: Dict with "state" containing:
        - left/joint_pos (6D): left arm joint positions
        - left/tcp_pos (3D): left end-effector position
        - left/gripper_pos (1D): left gripper position
        - right/joint_pos (6D): right arm joint positions
        - right/tcp_pos (3D): right end-effector position
        - right/gripper_pos (1D): right gripper position
        - cube_pos (3D): cube position
    """
    
    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(
        self,
        action_scale: float = 1.0,
        seed: int = 0,
        control_dt: float = 0.02,
        physics_dt: float = 0.002,
        time_limit: float = 10.0,
        render_spec: GymRenderingSpec = GymRenderingSpec(),
        render_mode: Literal["rgb_array", "human"] = "rgb_array",
        image_obs: bool = False,
    ):
        """
        Initialize Trossen bimanual environment.
        
        Args:
            action_scale: Scaling factor for actions (not used in joint control, kept for compatibility)
            seed: Random seed
            control_dt: Control timestep (seconds)
            physics_dt: Physics simulation timestep (seconds)
            time_limit: Episode time limit (seconds)
            render_spec: Rendering specifications
            render_mode: Rendering mode ("rgb_array" or "human")
            image_obs: Whether to include image observations
        """
        self._action_scale = action_scale
        self.image_obs = image_obs

        super().__init__(
            xml_path=_XML_PATH,
            seed=seed,
            control_dt=control_dt,
            physics_dt=physics_dt,
            time_limit=time_limit,
            render_spec=render_spec,
        )

        self.metadata = {
            "render_modes": ["human", "rgb_array"],
            "render_fps": int(np.round(1.0 / self.control_dt)),
        }

        self.render_mode = render_mode
        self.camera_id = (0, 1)  # cam_high, cam_low

        # Cache MuJoCo IDs for faster access
        self._cache_mujoco_ids()

        # Define observation space
        self.observation_space = self._create_observation_space(render_spec)

        # Define action space: 14D [left_arm(6) + left_gripper(1) + right_arm(6) + right_gripper(1)]
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(14,),
            dtype=np.float32,
        )

        # Setup renderers for image observations if needed
        self._renderers = {}
        if self.image_obs:
            for cam_id in self.camera_id:
                try:
                    renderer = mujoco.Renderer(
                        model=self._model,
                        height=render_spec.height,
                        width=render_spec.width,
                    )
                    self._renderers[cam_id] = renderer
                except Exception as e:
                    print(f"Warning: Failed to create renderer for camera {cam_id}: {e}")
                    self._renderers[cam_id] = None

        # For human rendering mode
        self._viewer = None
        if self.render_mode == "human":
            try:
                from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
                self._viewer = MujocoRenderer(self.model, self.data)
            except ImportError:
                print("Warning: gymnasium not installed, human rendering unavailable")

    def _cache_mujoco_ids(self):
        """Cache MuJoCo object IDs for efficient access."""
        # Left arm joint IDs
        self._left_joint_ids = np.array([
            self._model.joint(f"left/joint_{i}").id for i in range(6)
        ])
        
        # Right arm joint IDs
        self._right_joint_ids = np.array([
            self._model.joint(f"right/joint_{i}").id for i in range(6)
        ])
        
        # Left arm actuator IDs
        self._left_actuator_ids = np.array([
            self._model.actuator(f"left/joint_{i}").id for i in range(6)
        ])
        
        # Right arm actuator IDs
        self._right_actuator_ids = np.array([
            self._model.actuator(f"right/joint_{i}").id for i in range(6)
        ])
        
        # Gripper actuator IDs (both left and right fingers per arm)
        self._left_gripper_actuator_ids = np.array([
            self._model.actuator("left/joint_gl").id,   # left finger
            self._model.actuator("left/joint_gr").id,   # right finger
        ])
        
        self._right_gripper_actuator_ids = np.array([
            self._model.actuator("right/joint_gl").id,  # left finger
            self._model.actuator("right/joint_gr").id,  # right finger
        ])
        
        # Cube information
        try:
            self._cube_joint_id = self._model.joint("red_box_joint").id
            self._cube_body_id = self._model.body("box").id
        except KeyError:
            print("Warning: Cube joint/body not found in model")
            self._cube_joint_id = None
            self._cube_body_id = None

    def _create_observation_space(self, render_spec: GymRenderingSpec) -> gym.spaces.Dict:
        """Create observation space based on configuration."""
        state_space = gym.spaces.Dict({
            "left/joint_pos": spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
            "left/tcp_pos": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
            "left/gripper_pos": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
            "right/joint_pos": spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
            "right/tcp_pos": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
            "right/gripper_pos": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
            "cube_pos": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
        })

        if self.image_obs:
            return gym.spaces.Dict({
                "state": state_space,
                "images": gym.spaces.Dict({
                    "cam_high": gym.spaces.Box(
                        low=0, high=255,
                        shape=(render_spec.height, render_spec.width, 3),
                        dtype=np.uint8,
                    ),
                    "cam_low": gym.spaces.Box(
                        low=0, high=255,
                        shape=(render_spec.height, render_spec.width, 3),
                        dtype=np.uint8,
                    ),
                }),
            })
        else:
            return gym.spaces.Dict({"state": state_space})

    def reset(self, seed=None, **kwargs) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Reset the environment.
        
        Returns:
            observation: Dict containing state (and optionally images)
            info: Additional information dict
        """
        if seed is not None:
            self._random = np.random.RandomState(seed)
            
        # Reset MuJoCo simulation
        mujoco.mj_resetData(self._model, self._data)

        # Reset both arms to start pose
        self._data.qpos[self._left_joint_ids] = _START_ARM_POSE[0:6]
        self._data.qpos[self._right_joint_ids] = _START_ARM_POSE[8:14]
        
        # Reset grippers to open position
        left_gripper_joint_id = self._model.joint("left/left_carriage_joint").id
        right_gripper_joint_id = self._model.joint("left/right_carriage_joint").id
        self._data.qpos[left_gripper_joint_id] = 0.044
        self._data.qpos[right_gripper_joint_id] = 0.044
        
        left_gripper_joint_id = self._model.joint("right/left_carriage_joint").id
        right_gripper_joint_id = self._model.joint("right/right_carriage_joint").id
        self._data.qpos[left_gripper_joint_id] = 0.044
        self._data.qpos[right_gripper_joint_id] = 0.044

        # Forward kinematics to update state
        mujoco.mj_forward(self._model, self._data)

        # Sample cube position (start near left arm)
        if self._cube_joint_id is not None:
            cube_pose = sample_box_pose()
            self._data.joint(self._cube_joint_id).qpos[:7] = cube_pose
            mujoco.mj_forward(self._model, self._data)

        # Cache initial cube height for reward computation
        self._z_init = self._get_cube_pos()[2]
        self._z_lift_threshold = self._z_init + 0.15  # 15cm lift
        
        obs = self._compute_observation()
        info = {}
        
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: 14D array [left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]
                   All values in range [-1, 1]
        
        Returns:
            observation: Dict containing state (and optionally images)
            reward: Scalar reward
            terminated: Whether episode ended
            truncated: Whether episode was truncated
            info: Additional information dict
        """
        # Split action into left and right
        left_arm_action = action[0:6]
        left_gripper_action = action[6]
        right_arm_action = action[7:13]
        right_gripper_action = action[13]

        # Denormalize joint positions from [-1, 1] to actual limits
        left_joint_targets = denormalize_action(left_arm_action, _JOINT_LIMITS)
        right_joint_targets = denormalize_action(right_arm_action, _JOINT_LIMITS)
        
        # Denormalize gripper from [-1, 1] to [0, 0.044]
        left_gripper_target = (left_gripper_action + 1.0) / 2.0 * 0.044
        right_gripper_target = (right_gripper_action + 1.0) / 2.0 * 0.044

        # Set actuator targets (MuJoCo's position actuators handle PD control)
        self._data.ctrl[self._left_actuator_ids] = left_joint_targets
        self._data.ctrl[self._right_actuator_ids] = right_joint_targets
        
        # Set gripper targets (both fingers to same position)
        self._data.ctrl[self._left_gripper_actuator_ids] = left_gripper_target
        self._data.ctrl[self._right_gripper_actuator_ids] = right_gripper_target

        # Step physics simulation
        for _ in range(self._n_substeps):
            mujoco.mj_step(self._model, self._data)

        # Compute observation and reward
        obs = self._compute_observation()
        reward = self._compute_reward()
        terminated = self.time_limit_exceeded()
        truncated = False
        info = {}

        return obs, reward, terminated, truncated, info

    def _compute_observation(self) -> Dict[str, Any]:
        """Compute current observation."""
        obs = {"state": {}}

        # Left arm joint positions
        left_joint_pos = self._data.qpos[self._left_joint_ids].copy()
        obs["state"]["left/joint_pos"] = left_joint_pos.astype(np.float32)

        # Right arm joint positions
        right_joint_pos = self._data.qpos[self._right_joint_ids].copy()
        obs["state"]["right/joint_pos"] = right_joint_pos.astype(np.float32)

        # Left TCP (end-effector) position - get from link_6 body
        left_ee_body_id = self._model.body("left/link_6").id
        left_tcp_pos = self._data.body(left_ee_body_id).xpos.copy()
        obs["state"]["left/tcp_pos"] = left_tcp_pos.astype(np.float32)

        # Right TCP position
        right_ee_body_id = self._model.body("right/link_6").id
        right_tcp_pos = self._data.body(right_ee_body_id).xpos.copy()
        obs["state"]["right/tcp_pos"] = right_tcp_pos.astype(np.float32)

        # Left gripper position (average of both fingers)
        left_gripper_l = self._data.joint("left/left_carriage_joint").qpos[0]
        left_gripper_r = self._data.joint("left/right_carriage_joint").qpos[0]
        left_gripper_pos = np.array([(left_gripper_l + left_gripper_r) / 2.0], dtype=np.float32)
        obs["state"]["left/gripper_pos"] = left_gripper_pos

        # Right gripper position
        right_gripper_l = self._data.joint("right/left_carriage_joint").qpos[0]
        right_gripper_r = self._data.joint("right/right_carriage_joint").qpos[0]
        right_gripper_pos = np.array([(right_gripper_l + right_gripper_r) / 2.0], dtype=np.float32)
        obs["state"]["right/gripper_pos"] = right_gripper_pos

        # Cube position
        cube_pos = self._get_cube_pos()
        obs["state"]["cube_pos"] = cube_pos.astype(np.float32)

        # Add images if requested
        if self.image_obs:
            obs["images"] = {}
            rendered_frames = self.render()
            if rendered_frames is not None and len(rendered_frames) == 2:
                obs["images"]["cam_high"] = rendered_frames[0]
                obs["images"]["cam_low"] = rendered_frames[1]

        return obs

    def _get_cube_pos(self) -> np.ndarray:
        """Get cube position."""
        if self._cube_body_id is not None:
            return self._data.body(self._cube_body_id).xpos.copy()
        else:
            return np.zeros(3)

    # def _compute_reward(self) -> float:
    #     """
    #     Compute reward for pick-and-place task.
        
    #     Reward components:
    #     1. Left gripper approaching cube
    #     2. Cube lifted by left arm
    #     3. Cube near right gripper (handover)
    #     4. Right gripper grasping cube
    #     5. Cube placed by right arm
    #     """
    #     cube_pos = self._get_cube_pos()
        
    #     # Get end-effector positions
    #     left_ee_body_id = self._model.body("left/link_6").id
    #     left_tcp_pos = self._data.body(left_ee_body_id).xpos
        
    #     right_ee_body_id = self._model.body("right/link_6").id
    #     right_tcp_pos = self._data.body(right_ee_body_id).xpos

    #     # Phase 1: Left gripper approaching cube
    #     dist_left_to_cube = np.linalg.norm(left_tcp_pos - cube_pos)
    #     r_approach_left = compute_distance_reward(left_tcp_pos, cube_pos, temperature=20.0)

    #     # Phase 2: Cube lifted by left
    #     cube_height = cube_pos[2]
    #     r_lift_left = max(0.0, min(1.0, (cube_height - self._z_init) / (self._z_lift_threshold - self._z_init)))

    #     # Phase 3: Cube near right gripper (handover)
    #     dist_right_to_cube = np.linalg.norm(right_tcp_pos - cube_pos)
    #     r_approach_right = compute_distance_reward(right_tcp_pos, cube_pos, temperature=20.0)

    #     # Phase 4: Cube placed (lower height after handover)
    #     # This is a placeholder - you can add more sophisticated placement reward
        
    #     # Weighted combination (tune these weights based on task difficulty)
    #     reward = (
    #         0.3 * r_approach_left +
    #         0.3 * r_lift_left +
    #         0.4 * r_approach_right
    #     )

    #     return float(reward)

    def _compute_reward(self) -> float:
        """
        Compute reward for bimanual handover task.
        
        Task: Right robot picks cube → handover to left robot → left robot places
        
        Sequential phases:
        1. Right gripper approaches cube
        2. Right gripper grasps and lifts cube
        3. Both robots meet at handover position (mid-point)
        4. Left gripper grasps cube (right releases)
        5. Left robot moves to placement position
        6. Left robot places cube at target
        """
        cube_pos = self._get_cube_pos()
        
        # Get end-effector positions
        left_ee_body_id = self._model.body("left/link_6").id
        left_tcp_pos = self._data.body(left_ee_body_id).xpos
        
        right_ee_body_id = self._model.body("right/link_6").id
        right_tcp_pos = self._data.body(right_ee_body_id).xpos
        
        # Get gripper states (closed = 0, open = 0.044)
        left_gripper = self._data.qpos[self._left_gripper_id]
        right_gripper = self._data.qpos[self._right_gripper_id]
        
        # Define handover position (mid-point between robots)
        handover_pos = np.array([0.0, 0.0, 0.15])  # Center, 15cm height
        
        # Define placement target (left side)
        placement_target = np.array([-0.15, 0.0, 0.05])  # Left side, low height
        
        cube_height = cube_pos[2]
        
        # Check task progress stages
        cube_lifted = cube_height > (self._z_init + 0.05)  # Cube 5cm above table
        cube_at_handover = np.linalg.norm(cube_pos[:2] - handover_pos[:2]) < 0.05  # Within 5cm of handover XY
        right_gripper_closed = right_gripper < 0.01  # Right gripper closed
        left_gripper_closed = left_gripper < 0.01   # Left gripper closed
        
        # Phase 1: Right robot picks cube (25% of reward)
        dist_right_to_cube = np.linalg.norm(right_tcp_pos - cube_pos)
        r_right_approach = np.exp(-20.0 * dist_right_to_cube)
        
        # Phase 2: Right robot lifts cube (20% of reward)
        r_right_lift = 0.0
        if right_gripper_closed:
            r_right_lift = max(0.0, min(1.0, (cube_height - self._z_init) / 0.10))
        
        # Phase 3: Move to handover position (20% of reward)
        r_handover_position = 0.0
        if cube_lifted:
            dist_to_handover = np.linalg.norm(cube_pos - handover_pos)
            r_handover_position = np.exp(-10.0 * dist_to_handover)
        
        # Phase 4: Left robot approaches handover (15% of reward)
        r_left_approach = 0.0
        if cube_at_handover:
            dist_left_to_cube = np.linalg.norm(left_tcp_pos - cube_pos)
            r_left_approach = np.exp(-20.0 * dist_left_to_cube)
        
        # Phase 5: Left robot takes cube and moves to placement (15% of reward)
        r_left_transfer = 0.0
        if left_gripper_closed and cube_lifted:
            dist_to_placement = np.linalg.norm(cube_pos - placement_target)
            r_left_transfer = np.exp(-10.0 * dist_to_placement)
        
        # Phase 6: Successful placement (5% of reward - bonus)
        r_placement = 0.0
        dist_to_target = np.linalg.norm(cube_pos - placement_target)
        if dist_to_target < 0.03 and cube_pos[2] < 0.08:  # Within 3cm and lowered
            r_placement = 1.0
        
        # Weighted combination
        reward = (
            0.25 * r_right_approach +      # Right approaches cube
            0.20 * r_right_lift +          # Right lifts cube
            0.20 * r_handover_position +   # Move to handover zone
            0.15 * r_left_approach +       # Left approaches handover
            0.15 * r_left_transfer +       # Left takes and moves to placement
            0.05 * r_placement             # Successful placement bonus
        )

    def render(self):
        """Render the environment."""
        if self.image_obs and self._renderers:
            rendered_frames = []
            for cam_id in self.camera_id:
                renderer = self._renderers[cam_id]
                if renderer is not None:
                    renderer.update_scene(self._data, camera=cam_id)
                    rendered_frames.append(renderer.render())
            return rendered_frames if rendered_frames else None

        if self.render_mode == "human" and self._viewer is not None:
            return self._viewer.render("human")

        return None

    def close(self) -> None:
        """Clean up resources."""
        super().close()
        for renderer in self._renderers.values():
            if renderer is not None:
                renderer.close()
        if self._viewer is not None:
            self._viewer.close()


if __name__ == "__main__":
    # Test the environment
    env = TrossenBimanualPickPlaceGymEnv(render_mode="human")
    obs, info = env.reset()
    
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)
    print("Initial observation keys:", obs.keys())
    print("State keys:", obs["state"].keys())
    
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i}: reward={reward:.3f}, cube_pos={obs['state']['cube_pos']}")
        env.render()
        
        if terminated or truncated:
            obs, info = env.reset()
    
    env.close()
