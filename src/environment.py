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
    def __init__(self, headless):
        """
        Initiliaze the environment, setting the public variables, launching and setting up the scene, setting 
        the porxy variables for the counterparts in the scene

        Args:
            headless (bool): if True, no visualization, else with visualization
        """
        # set public variables
        self.headless = headless
        self.reward_offset = 10.0 # reward value of grasping the object
        self.reward_range = self.reward_offset # reward range
        self.penalty_offset = 1. # penalty value for undesired cases
        self.fall_down_offset = 0.1 # distance for judging the target object fall off the table
        self.metadata = [] # gym env argument

        # launch and setup scene
        SCENE_FILE = join(dirname, (abspath(__file__)), './simulations/sawyer_reacher.ttt') # scene with joints controlled by FK
        self.pr = PyRep() # call the PyRep
        self.pr.launch(SCENE_FILE, headless = self.headless) # launch the scene
        self.pr.start() # start the scene

        # set proxy variables for the counterparts in the scene
        self.agent = Sawyer() # get robot arm in the scene
        self.gripper = BaxterGripper() # get the gripper in the scene
        self.gripper_left_pad = Shape('BaxterGripper_leftPad') # name of the left pad of the gripper finger
        self.proximity_sensor = ProximitySensor('BaxterGripper_attachProxSensor') # name of the proximity sensor
        self.vision_sensor = VisionSensor('Vision_sensor') # name of the vision sensor
        self.table = Shape('diningTable') # name of the table in the scene, for checking collision

        # set action and observation space
        self.agent.set_control_loop_enabled(False)
        self.action_space = np.zeros(7) # 7 DOF velocity control, no need for extra control of the end rotation, 7th joint controls it
        self.observation_space = np.zeros(17) # scalar positions and scalar velocities of 7 joints + 3-dimensional position of the target
        self.agent.set_motor_locked_at_zero_velocity(True) # joints locked in place when velocity is zero

        # set agent tip and target position for IK chain
        self.target = Shape('target') # get target object
        self.agent_ee_tip = self.agent.get_tip() # a part of robot as the end of IK chain for controlling
        self.tip_target = Dummy('Sawyer_target') # the target point of the tip (end of the robot arm) to move towards /
        self.tip_pos = self.agent_ee_tip.get_position() # tip x, y, z position

        # set proper initial robot gesture
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

    def reinit(self):
        """
        Reinitialize the environment, e.g. when the gripper is broken during exploration
        """
        self.shutdown() # shutdown the original env first
        self.__init__(self.headless)

    def reset(self, random_target = False):
        """
        Reset the gripper position and set the target position.
        """
        # set target object
        if random_target: # randomizd
            pos = list(np.random.uniform(POS_MIN, POS_MAX)) # sample from uniform is valid range
            self.target.set_position(pos) # random positio
        else: # non-renadomized 
            self.target.set_position(self.initial_target_positions) # fixed position
        self.target.set_orientation([0, 0, 0])
        self.pr.step()

        # set end position to be initiliazed
        self.agent.set_joint_positions(self.initial_joint_positions)
        self.pr.step()
        
        # set collidable, for collision detection
        self.gripper_left_pad.set_collidable(True) # set the pad of the gripper to be collidable, so as to check collision
        self.target.set_collidable(True) # set the target to be collidable, so as to check collision
        
        # open the gripper if it's not fully open
        if np.sum(self.gripper.get_open_amount()) < 1.5:
            self.gripper.actuate(1.0, velocity=0.5)
            self.pr.step()

        # return current state of the env
        return self._get_state()


    def step(self, action):
        """
        Move the robot arm according to the action.

        Args:
            action (np.array): Action is 7 dim of joint velocity values + 1 dim rotation of gripper
        """     
        # initialization
        done = False # episode finishes
        reward = 0
        hold_flag = False # holding the object or not

        if action is None or action.shape[0] != 7: # check if action is valid
            print('No actions or wrong action dimensions')
            action = list(np.random.uniform(-0.1, 0.1, 7)) # random action
        self.agent.set_joint_target_velocities(action)
        self.pr.step()

        # get gripper position and check if its broken, if broken, reinitialize the environment and finish the episode
        ax, ay, az = self.gripper.get_position()
        if math.isnan(ax): # capture the broken gripper cases during exploration
            print('Gripper position is nan.')
            self.reinit()
            done = True # end of episode

        # get target position and calculate the distance between it and the gripper
        tx, ty, tz = self.target.get_position()
        offset = 0.08 # augmented reward: offset of target position above the target object
        sqr_distance = (ax - tx) ** 2 + (ay - ty) ** 2 + (az - (tz + offset)) ** 2  # squared distance between the gripper and the target object
        
        # close the gripper if close enough to the object and the object is detected with the proximity sensor
        if sqr_distance < 0.1 and self.proximity_sensor.is_detected(self.target) == True:
            # make sure the gripper is open before grasping
            self.gripper.actuate(1.0, velocity=0.5) # 1, open gripper
            self.pr.step() # step simulation, after each movement
            self.gripper.actuate(0.0, velocity=0.5) # 0, close gripper, velocity ensures to close with in one frame
            self.pr.step() # step simulation, after each movement

            if self._is_holding():
                reward += self.reward_offset # extra reward for grasping the object
                done = True # end of episode
                hold_flag = True # object is grasped
            else:
                self.gripper.actuate(1.0, velocity=0.5)
                self.pr.step()
        elif np.sum(self.gripper.get_open_amount()) < 1.5: # if gripper is closed (not fully open) due to collision or else, open it; get_open_amount() return list of gripper joint values
            self.gripper.actuate(1.0, velocity=0.5)
            self.pr.step()
        else:
            pass

        # the base reward is negative distance to target
        reward -= np.sqrt(sqr_distance)

        # case when the object fall off the table
        if tz < self.initial_target_positions[2] - self.fall_down_offset:
            done = True # end episode
            reward = -self.reward_offset

        # capture the cases of numerical problems
        if math.isnan(reward): 
            reward = 0.

        return self._get_state(), reward, done, {'finished': hold_flag}

    def shutdown(self):
        """
        Close the simulator
        """
        self.pr.stop()
        self.pr.shutdown()


if __name__ == '__main__':
    # create environment
    env = GraspEnv(headless=False)

    for episode in range(30):
        # initialize the env
        env.reset()
        for step in range(30):
            #  7 dim (+ 1 rotation (z-axis) included, last joint)
            action = np.random.uniform(-2.0, 2.0, 7)
            # perform action
            try:
                env.step(action)
            except KeyboardInterrupt:
                print("Shut down!")
                env.shutdown()