from os.path import dirname, join, abspath
from pyrep import PyRep
from pyrep.robots.arms.sawyer import Sawyer
from pyrep.robots.end_effectors.baxter_gripper import BaxterGripper
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.objects.shape import Shape
from pyrep.objects.dummy import Dummy
from pyrep.const import JointType, JointMode
import numpy as np
import matplotlib.pyplot as plt
import math

POS_MIN, POS_MAX = [0.1, -0.3, 1.], [0.45, 0.3, 1.] # valid position range of target object

class GraspEnv(object):
    """Sawyer robot grasping a cuboid """ 
    def __init__(self, headless, control_mode='joint_velocity'):
        """
        Initiliaze the environment, setting the public variables, launching and setting up the scene, setting 
        the porxy variables for the counterparts in the scene

        Args:
            headless (bool): if True, no visualization, else with visualization
            control_mode (str, optional): 'end_position' or 'joint_velocity'. Defaults to 'joint_velocity'.
        """
        # set public variables
        self.headless = headless
        self.reward_offset = 10.0 # reward value of grasping the object
        self.reward_range = self.reward_offset # reward range
        self.penalty_offset = 1. # penalty value for undesired cases
        self.fall_down_offset = 0.1 # distance for judging the target object fall off the table
        self.metadata = [] # gym env argument
        self.control_mode = control_mode

        # launch and setup scene
        self.pr = PyRep() # call the PyRep
        if control_mode == 'end_position':
            # the controle mode with all joints in inverse kinematics mode
            SCENE_FILE = join(dirname, (abspath(__file__)), './simulations/sawyer_reacher_rl_new_ik.ttt') # scene with joints controlled by IK
        elif control_mode == 'joint_velocity':
            SCENE_FILE = join(dirname, (abspath(__file__)), './simulations/sawyer_reacher_rl_new.ttt') # scene with joints controlled by FK
        self.pr.launch(SCENE_FILE, headless = self.headless) # launch the scene
        self.pr.start() # start the scene

        # set proxy variables for the counterparts in the scene
        self.agent = Sawyer() # get robot arm in the scene
        self.gripper = BaxterGripper # get the gripper in the scene
        self.gripper_left_pad = Shape('BaxterGripper_leftPad') # name of the left pad of the gripper finger
        self.proximity_sensor = ProximitySensor('BaxterGripper_attachProxSensor') # name of the proximity sensor
        self.vision_sensor = VisionSensor('Vision_sensor') # name of the vision sensor
        self.table = Shape('diningTable') # name of the table in the scene, for checking collision

        # set action and observation space
        if control_mode == 'end_position':
            # control the robot arm by the position of its end using IK
            self.agent.set_control_loop_enabled(True) # if False, IK won't work
            self.action_space = np.zeros(4) # 3 DOF end position control + 1 DOF rotation of gripper
        elif control_mode == 'joint_velocity':
            # control the robot arm by directly setting velocity values on each joint, using FK
            self.agent.set_control_loop_enabled(False)
            self.action_space = np.zeros(7) # 7 DOF velocity control, no need for extra control of the end rotation, 7th joint controls it
        else:
            raise NotImplementedError
        self.observation_space = np.zeros(17) # scalar positions and scalar velocities of 7 joints + 3-dimensional position of the target
        self.agent.set_motor_locked_at_zero_velocity(True) # joints locked in place when velocity is zero

        # set agent tip and target position for IK chain
        self.target = Shape('target') # get target object
        self.agent_ee_tip = self.agent.get_tip() # a part of robot as the end of IK chain for controlling
        self.tip_target = Dummy('Sawyer_target') # the target point of the tip (end of the robot arm) to move towards
        self.tip_pos = self.agent_ee_tip.get_position() # tip x, y, z position

        # set proper initial robot gesture or tip position
        if control_mode == 'end_position':
            initial_pos = [0.3, 0.1, 0.9]
            self.tip_target.set_position(initial_pos) # set target position
            # one big step for rotation setting is enough, with reset_dynamics = True, set the rotation instantaneously
            self.tip_target.set_orientation([0, np.pi, np.pi/2], reset_dynamics=True) # first two dimensions along x, y axis make gripper face downwards
        elif control_mode == 'joint_velocity':
            self.initial_joint_positions = [0.0, -1.4, 0.7, 2.5, 3.0, -0.5, 4.1] # a proper initial gesture
            self.agent.set_joint_positions(self.initial_joint_positions)
        
        self.pr.step() # Step the physics simulation
        self.initial_tip_positions = self.agent_ee_tip.get_position()
        self.initial_target_positions = self.target.get_position()

    def _get_state(self):
        """
        Return state containing arm joint angles/velocities & target position
        """
        return np.array(self.agent.get_joint_positions() + # list, dim=7
                        self.agent.get_joint_velocities() + # list, dim=7
                        list(self.target.get_position())) # list, dim=3


    def _is_holding(self):
        """
        Return the state of holding the target or not, return bool
        """
        # Note that the collision check is not always accurate all the time, 
        # for continuous collision frames, maybe only the first 4-5 frames of collision can be detected.
        pad_collide_object = self.gripper_left_pad.check_collision(self.target)
        if  pad_collide_object and self.proximity_sensor.is_detected(self.target)==True:
            return True 
        else:
            return False

    def _move(self, action, bounding_offset = 0.15, step_factor = 0.2, max_iter = 20, max_error = 0.05, rotation_norm = 5.):
        """Move the tip according to the action with the IK for 'end_position' controle mode. IK mode control is archieved through
        setting the tip target insead of using .solve_ik(), because sometimes the .solve_ik() does not function correctly.
        Mode: close-loop proportional control, using IK

        Args:
            action ([type]): [description]
            bounding_offset (float, optional): offset of bounding box outside the valid target position range, as valid and safe range of action. Defaults to 0.15.
            step_factor (float, optional): small step factor multiplied on the difference of current and desired positions, i.e. proportional factor. Defaults to 0.2.
            max_iter (int, optional): maximum moving iterations. Defaults to 20.
            max_error (float, optional): upper bound of distance error for movements at each call. Defaults to 0.05.
            rotation_norm ([type], optional): factor of normalization of rotation values, since the actions are of the same scale for each dimension. Defaults to 5..
        """
        pass

    def reinit(self):
        """
        Reinitialize the environment, e.g. when the gripper is broken during exploration
        """
        pass

    def reset(self, random_target = False):
        """
        Reset the gripper position and the target position.
        """
        pass

    def step(self, action):
        """
        Move the robot arm according to the action.
        If control_mode == 'joint_velocity', action is 7 dim of joint velocity values + 1 dim rotation of gripper
        If control_mode == 'end_position', action is 3 dim of tip (end of robot arm) position values + 1 dim rotation of gripper

        Args:
            action ([type]): [description]
        """     
        pass

    def shutdown(self):
        """
        Close the simulator
        """
        pass

