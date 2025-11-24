import os
import numpy as np
from scipy.spatial.transform import Rotation as R
import mujoco
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from ml_collections import config_dict

FRAME_SKIP = 5
TIMESTEP = 0.002  # Defined in XML
RENDER_FPS = int(np.round(1.0 / (TIMESTEP*FRAME_SKIP)))

def get_physics_ranges() -> config_dict.ConfigDict:
    return config_dict.create(
        ramp_size_x = {"low": 1.0, "high": 3.0},
        ramp_pos_x = {"low": 0.5, "high": 1.5},
        slope = {"low": -50, "high": -20},
        object_mass = {"low": 0.5, "high": 3.0},
        tool_mass = {"low": 0.5, "high": 3.0},
        friction = {"low": 0.3, "high": 0.8},
        gravity = {"low": 1.5*-9.81, "high": 0.8*-9.81},
    )

def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        xml="ramp_push.xml",
        vision=False,
        sparse=False,
        reward_weight=1.0,
        ctrl_cost_weight=0.1,
        default_camera_config={"distance": 6.0,},
        img_h=240,
        img_w=320,
        # configurable physical properties of env
        ramp_size=np.array([2.0, 1.0, 0.05]),   # length, width and thickness of ramp
        ramp_pos=np.array([0.5, 0.0, 0.0]),     # position of ramp center
        gravity=np.array([0.0, 0.0, -9.81]),    # gravity
        friction=np.array([0.3, 0.3, 0.3]),     # coeff of friction
        object_mass=0.5,                        # mass of object pushed up the ramp
        tool_mass=0.5,                          # mass of tool pushed up the ramp
        slope=-20,                              # slope of the ramp in degrees
    )

class RampPushEnv(MujocoEnv, utils.EzPickle):
    """
    RampPushEnv: Push a cube up a ramp and stop at the top. 
    The model XML file ramp_push.xml should be located in assets/.

    Expected model components:
      - Static "geoms" named "ground" and "ramp".
      - "ramp" is inclined at an angle named (slope) relative to the ground.
      - Two bodies named "object" and "tool" (in addition to "world").
      - "object" has "free joints" - all translations and rotations are enabled.
      - "tool" has 3 sliding joints "tool_x|y|z", actuated by  "act_tool_x|y|z"

    Observations:
      - (x,y,z) position of "object"
      - (qw,qx,qy,qz) orientation of "object"
      - (vx,vy,vz) velocity of "object"
      - (x,y,z) position of "tool"
      - (vx,vy,vz) velocity of "tool"

    Task objective: Use the "tool" to push the "object" to top of the "ramp" 
                    and stop before overshooting the ramp. 
      
    Reward: There are 2 components:
      - 1. Goal cost
        - a. Negative distance between "tool" and "object".
        - b. Negative z-distance between "object" and top-right edge of "ramp".
        - c. Negative x-distance between "object" and top-right edge of "ramp".
      - 2. Control cost
        - a. Squared sum of control actions
    """

    metadata = {
            "render_modes": ["human", "rgb_array"],
            "render_fps": RENDER_FPS
            }
    
    #render_mode = "rgb_array"

    def __init__(self,
                 config: config_dict.ConfigDict=default_config(),
                 render_mode=None,
                 **kwargs):
        
        # Record parameters for pickling.
        utils.EzPickle.__init__(**locals())
        self._config = config

        self.vision = config.vision
        if self.vision:
            raise NotImplementedError("Vision mode is not implemented for RampPushEnv.")

        self._sparse = config.sparse
        self._reward_weight = config.reward_weight
        self._ctrl_cost_weight = config.ctrl_cost_weight
        self._default_camera_config = config.default_camera_config

        # Physical properties of the environment
        self._ramp_size = config.ramp_size
        self._ramp_pos = config.ramp_pos
        self._gravity = config.gravity
        self._friction = config.friction
        self._object_mass = config.object_mass
        self._tool_mass = config.tool_mass
        self._slope = config.slope

        # Properties of rendered image
        self._render_mode = render_mode
        self._img_h = config.img_h
        self._img_w = config.img_w

        self._steps = 0

        ## Define observation space
        obs_dim = 16
        self.observation_space = Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(obs_dim,), 
            dtype=np.float32
            )
        
        action_dim = 3
        self.action_space = Box(
            low=-5.0,
            high=5.0,
            shape=(action_dim,), 
            dtype=np.float32
            )

        self.xml_path = os.path.join(
            os.path.dirname(__file__),
            "assets",
            config.xml
            )
        
        if not os.path.exists(self.xml_path):
            print(f"XML {self.xml_path} not found!")

        MujocoEnv.__init__(
            self,
            model_path=self.xml_path,
            frame_skip=FRAME_SKIP,
            observation_space=self.observation_space,
            default_camera_config=self._default_camera_config,
            render_mode=self._render_mode,
            **kwargs)

        # Get IDs of geoms and bodies
        # These are used later in _set_physics()
        self._geom_ids = {
            "ramp": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "ramp"),
            "ground": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "ground"),
            "object": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "object"),            
            "tool": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "tool"),            
        }

        self._body_ids = {
            "object": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "object"),            
            "tool": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "tool"),            
        }

        # Check ranges of physical parameters
        self._verify_physics()

        # Set physical properties
        self._set_physics()

        # Set goals that are used to compute rewards
        self._set_goals()          

    def _verify_physics(self):
        # Check that physics parameters are within bounds
        r = get_physics_ranges()

        # Verify the ramp size
        assert self._ramp_size.shape == (3,)
        assert r.ramp_size_x["low"] <= self._ramp_size[0] <= r.ramp_size_x["high"]
        assert self._ramp_size[1] == 1.0 and self._ramp_size[2] == 0.05 # Fixed values

        # Verify the ramp pos
        assert self._ramp_pos.shape == (3,)
        assert r.ramp_pos_x["low"] <= self._ramp_pos[0] <= r.ramp_pos_x["high"]
        assert self._ramp_pos[1] == 0.0 and self._ramp_pos[2] == 0.0 # Fixed values

        # Verify ramp slope
        assert r.slope["low"] <= self._slope <= r.slope["high"]

        # Verify object and tool mass
        assert r.object_mass["low"] <= self._object_mass <= r.object_mass["high"]
        assert r.tool_mass["low"] <= self._tool_mass <= r.tool_mass["high"]

        # Verify friction
        assert self._friction.shape == (3,)
        for f in self._friction:
            assert r.friction["low"] <= f <= r.friction["high"]

        # Verify gravity
        assert self._gravity.shape == (3,)
        assert r.gravity["low"] <= self._gravity[2] <= r.gravity["high"]
        assert self._gravity[0] == 0.0 and self._gravity[1] == 0.0

    def _set_physics(self):

        # Set ramp position
        self.model.geom_pos[self._geom_ids["ramp"]] = self._ramp_pos

        # Set ramp size
        self.model.geom_size[self._geom_ids["ramp"]] = self._ramp_size

        # Set slope of ramp
        euler_deg = [0, self._slope, 0]
        r = R.from_euler('xyz', euler_deg, degrees=True)
        quat_xyzw = r.as_quat()
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        self.model.geom_quat[self._geom_ids["ramp"]] = quat_wxyz

        # Set gravity
        self.model.opt.gravity[:] = self._gravity

        # Set friction
        for geom_id in self._geom_ids.values():
            self.model.geom_friction[geom_id] = self._friction

        # Set mass of object
        self.model.body_mass[self._body_ids["object"]] = self._object_mass

        # Set mass of tool
        self.model.body_mass[self._body_ids["tool"]] = self._tool_mass

    def _set_goals(self):
        self._goal_z = np.abs((self._ramp_size[0]/2.0) * np.sin(np.deg2rad(self._slope)))
        self._goal_x = self._ramp_pos[0] + np.abs((self._ramp_size[0]/2.0) * np.cos(np.deg2rad(self._slope)))

    def _get_obs(self):
        """
            Observations:
            - (x,y,z) position of "object"
            - (qw,qx,qy,qz) orientation of "object"
            - (vx,vy,vz) velocity of "object"
            - (x,y,z) position of "tool"
            - (vx,vy,vz) velocity of "tool"
        """

        id_obj = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "object")
        id_tool = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "tool")

        self.pos_obj = self.data.xpos[id_obj]
        self.quat_obj = self.data.xquat[id_obj]
        self.pos_tool = self.data.xpos[id_tool]

        # TODO Check if this way of getting velocities is correct
        # Gymnasium does it differently
        self.vel_obj = np.zeros(6,)
        mujoco.mj_objectVelocity(
            m=self.model, 
            d=self.data, 
            objtype=mujoco.mjtObj.mjOBJ_BODY, 
            objid=id_obj,
            flg_local=0, # world frame
            res=self.vel_obj
            )

        self.vel_tool = np.zeros(6,)
        mujoco.mj_objectVelocity(
            m=self.model, 
            d=self.data, 
            objtype=mujoco.mjtObj.mjOBJ_BODY, 
            objid=id_tool,
            flg_local=0, # world frame
            res=self.vel_tool
            )
        
        self._d_tool2obj = np.linalg.norm(self.pos_tool - self.pos_obj)
        self._z_obj2goal = np.abs(self._goal_z- self.pos_obj[2])
        self._x_obj2goal = np.abs(self._goal_x- self.pos_obj[0])

        self.obs_dict = dict(
            pos_obj=self.pos_obj,
            quat_obj=self.quat_obj,
            pos_tool=self.pos_tool,
            lin_vel_obj=self.vel_obj[3:],
            lin_vel_tool=self.vel_tool[3:],
        )

        # Observation
        return np.concatenate([
            self.pos_obj, 
            self.quat_obj, 
            self.pos_tool, 
            self.vel_obj[3:],  # mj_objectVelocity returns 6D velocity (rot:lin)
            self.vel_tool[3:], # mj_objectVelocity returns 6D velocity (rot:lin)
            ])

    def reset_model(self):

        # Always initialize the tool and object in the same place
        self.set_state(self.init_qpos.copy(), self.init_qvel.copy())
        self._steps = 0

        return self._get_obs()

    def step(self, action: np.ndarray):
        
        # Step the simulation with provided control
        self.do_simulation(action, FRAME_SKIP)

        # Get the current observation.
        obs = self._get_obs()

        # Goal cost
        # Negative distance between "tool" and "object"
        # Negative z-distance between "object" and "goal"
        # Negative x-distance between "object" and "goal"
        reward_goal = -1.0 * self._reward_weight * (1.0*self._d_tool2obj + 5.0*self._z_obj2goal + 4.0*self._x_obj2goal)
        #reward_goal = -1.0 * self._reward_weight * self._z_obj2goal

        # Control cost
        reward_ctrl = -1.0 * self._ctrl_cost_weight * np.sum(action**2)

        # Compute reward
        reward = reward_ctrl + reward_goal

        # truncated is handled by the time limit wrapper
        truncated = False

        # terminated is always False, as we want to stop at the goal
        terminated = False

        info = dict(
            obs_dict=self.obs_dict,
            d_tool2obj=self._d_tool2obj,
            z_obj2goal=self._z_obj2goal,
            x_obj2goal=self._x_obj2goal,
            goal_x=self._goal_x,
            goal_z=self._goal_z,
            reward_goal=reward_goal, 
            reward_ctrl=reward_ctrl,
        )

        self._steps += 1

        return obs, reward, terminated, truncated, info
    
    def render(self):

        if self.render_mode == 'rgb_array':
            with mujoco.Renderer(self.model, height=self._img_h, width=self._img_w) as renderer:
                mujoco.mj_forward(self.model, self.data)
                renderer.update_scene(self.data)
                return renderer.render()
        elif self.render_mode == 'human':
            raise NotImplementedError('Unknown render_mode: {render_mode}')
        else:
            raise NotImplementedError('Unknown render_mode: {render_mode}')


if __name__ == "__main__":

    env = RampPushEnv(render_mode="rgb_array")
    print(env._ramp_size, env._goal_z)
    for i in range(10):
        env.reset()
        action = np.random.uniform(-5, 5, (3,))
        obs, reward, terminated, truncated, info = env.step(action)
        print(reward)

    env.close()

