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
        """[summary]

        Args:
            headless (bool): if True, no visualization, else with visualization
            control_mode (str, optional): 'end_position' or 'joint_velocity'. Defaults to 'joint_velocity'.
        """
        pass

    def _get_state(self):
        """
        Return state containing arm joint angles/velocities & target position
        """
        pass

    def _is_holding(self):
        """
        Return the state of holding the target or not, return bool
        """
        pass

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

