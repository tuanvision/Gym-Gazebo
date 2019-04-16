import gym
import rospy
import roslaunch
import time
import math
import numpy as np
import cv2

from gym import utils, spaces
from gym_gazebo.envs import gazebo_env
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge, CvBridgeError


from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import GetModelState, GetPhysicsProperties, SetModelState
from geometry_msgs.msg import Pose
from sensor_msgs.msg import LaserScan

from gym.utils import seeding

import skimage.measure

class GazeboTurtlebotMazeColorEnv(gazebo_env.GazeboEnv):

    def __init__(self):
        # Launch the simulation with the given launchfile name
        gazebo_env.GazeboEnv.__init__(self, "MazeColor.launch")
        self.vel_pub = rospy.Publisher('/mobile_base/commands/velocity', Twist, queue_size=5)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)

        self.name_model = 'mobile_base'
        self.name_target = 'Target'
        self.name_hint = 'Hint'


        self.robot_pos = [[0, 0], [1.5 , 2], [4.5, 0]]
        self.robot_direction = [0, -3, 5]
        self.target_pos = [[-0.25, -2], [6.5, 1.75], [8.5, 1.5]]
        # self.death_pos = [[2, -2], [3, 0], [-3, 0], [2, 2]]
        self.death_pos = [[-3, 0], [-3, 0], [-3, 0]]
        self.hint_pos = []
        hint_target = [[1, 0], [1.9, -1.9], [3.2, -1.9], [3.2, -3.7], [-0.25, -3.75]]
        self.hint_pos.append(hint_target)
        hint_target = [[1.5, 0], [4.75, 0], [4.75, -3.75], [8, -3.75], [8, 0], [6.5, 0.5]]
        self.hint_pos.append(hint_target)
        hint_target = [[1, 0], [3.2, 1.9], [5, 3.2], [8.5, 3.2]]
        self.hint_pos.append(hint_target)

        self.set_model = rospy.ServiceProxy('gazebo/set_model_state', SetModelState)

        rospy.wait_for_service('gazebo/set_model_state')

        # ===== SEED TO SET RANDOM TARGET AND HINT =====
        # self.num_target = np.random.randint(4)
        self.num_target = 0 # set default to 0

        self.num_hint = len(self.hint_pos[self.num_target])
        self.hint_check = np.zeros(shape=(self.num_hint,))
        self.setTarget()


        #get model state service
        self.model_state = rospy.ServiceProxy('gazebo/get_model_state', GetModelState)
        self.physics_properties = rospy.ServiceProxy('gazebo/get_physics_properties', GetPhysicsProperties)
        self.turtlebot_state = self.model_state(self.name_model, "world")
        self.action_space = spaces.Discrete(21)
        self.reward_range = (-np.inf, np.inf)
        self.count_loop = [0] * 50

        self.channel = 3
        self.width = 64
        self.height = 64
        self.num_action = 5
        self.num_state = [[64, 64, 3], 100, 2]

        self._seed()

    def setTarget(self):
        state = ModelState()
        state.model_name = self.name_target
        state.reference_frame = "world"
        state.pose.position.x = self.target_pos[self.num_target][0]
        state.pose.position.y = self.target_pos[self.num_target][1]
        self.set_model(state)
        # time.sleep(0.5)

        for i in range(self.num_hint):
            state.model_name = self.name_hint + str(i)
            state.pose.position.x = self.hint_pos[self.num_target][i][0]
            state.pose.position.y = self.hint_pos[self.num_target][i][1]
            self.set_model(state)
            # time.sleep(0.5)

    def calculate_observation(self, data, pos):
        min_range = 0.21
        min_target_range = 0.5
        done = False
        for i, item in enumerate(data.ranges):
            if (min_range > data.ranges[i] > 0):
                done = True
        # print ([pos.x, pos.y, self.target_pos[self.num_target][0], self.target_pos[self.num_target][1]])
        if (self.calculate_distance(pos.x, pos.y, self.target_pos[self.num_target][0], self.target_pos[self.num_target][1]) < min_target_range):
            done = True

        # If robot chooses the wrong path, die anyway.
        pos = self.model_state(self.name_model, "world").pose.position
        # if self.calculate_distance(0, 0, pos.x, pos.y) > 4:
        #     done = True
        #     return done


        death_radius = 1
        for death_index in range(len(self.death_pos)):
            if death_index == self.num_target: continue
            x_death, y_death = self.death_pos[death_index]
            if self.calculate_distance(pos.x, pos.y, x_death, y_death) < death_radius:
                done = True
                return done
        return done

    def calculate_distance(self, x, y, a, b):
        return math.sqrt((x-a)*(x-a) + (y-b)*(y-b))

    def calculate_reward(self, done, pos):
        min_hint_range = 0.5
        min_target_range = 0.5
        reward = 1
        if (done):
            if (self.calculate_distance(pos.x, pos.y, self.target_pos[self.num_target][0], self.target_pos[self.num_target][1]) < min_target_range):
                reward = 200
            else:
                reward = -300
        else:
            for i in range(self.num_hint):
                if (self.calculate_distance(pos.x, pos.y, self.hint_pos[self.num_target][i][0], self.hint_pos[self.num_target][i][1]) < min_hint_range):
                    if self.hint_check[i] == 1:
                        reward = 1
                    else:
                        self.hint_check[i] = 1
                        reward = 50
        return reward

    def get_image(self):
        image_data = None
        cv_image = None
        n = 0
        while image_data is None:
            try:
                image_data = rospy.wait_for_message('/camera/rgb/image_raw', Image, timeout=5)
                h = image_data.height
                w = image_data.width
                cv_image = CvBridge().imgmsg_to_cv2(image_data, "bgr8")
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

            except:
                n += 1
                if n == 10:
                    print "Camera error"
                    state = []
                    done = True
                    return state  
        cv_image = cv2.resize(cv_image, (self.height, self.width))
        # cv_image = cv_image.reshape(self.width, self.height, self.channel) 

        return cv_image


    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except rospy.ServiceException, e:
            print ("/gazebo/unpause_physics service call failed")

        # # ===== ACTION: 21 normal actions + 2 advanced actions 
        # # 21 actions
        # vel_cmd = Twist()
        # if action != -1 and action != 21:
        #     vel_cmd.linear.x = 0.3
        #     max_ang_speed = 0.3
        #     ang_vel = (action-10)*max_ang_speed*0.1 #from (-0.33 to + 0.33)
        #     vel_cmd.angular.z = ang_vel

        # # add 2 more actions
        # if action == -1 or action == 21:
        #     vel_cmd.linear.x = 0
        #     max_ang_speed = 0.2
        #     ang_vel = (action-10)*max_ang_speed*0.1 #from (-0.33 to + 0.33)
        #     vel_cmd.angular.z = ang_vel


        # ===== ACTION: 21 normal actions =====
        # action_value, is_laser_action = action
        # vel_cmd = Twist()
        # if is_laser_action:
        #     max_ang_speed = 1.6
        #     ang_vel = (action_value-10)*max_ang_speed*0.1 #from (-0.33 to + 0.33)
        #     vel_cmd.linear.x = 0.2
        #     vel_cmd.angular.z = ang_vel
        # else:
        #     vel_cmd.linear.x = 0.6
        #     max_ang_speed = 0.3
        #     ang_vel = (action_value-10)*max_ang_speed*0.1 #from (-0.33 to + 0.33)
        #     vel_cmd.angular.z = ang_vel
        # self.vel_pub.publish(vel_cmd)

        # ===== ACTION FOR DQN =====
        if action < 0:
            vel_cmd = Twist()
            vel_cmd.linear.x = 0
            vel_cmd.angular.z = 4.0
            self.vel_pub.publish(vel_cmd)
        else:
            max_ang_speed = 1.2
            ang_vel = (action - 5)*max_ang_speed*0.1 #from (-0.33 to + 0.33)

            vel_cmd = Twist()
            vel_cmd.linear.x = 0.2
            vel_cmd.angular.z = ang_vel
            self.vel_pub.publish(vel_cmd)

        # get laser data
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/scan', LaserScan, timeout=5)
            except:
                pass
        
        # get position of robot
        pos = self.model_state(self.name_model, "world").pose.position
        # check terminal state
        done = self.calculate_observation(data, pos)

        # get image, laser data
        image = self.get_image()
        # cv2.imshow('image',image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # image = self.detect_hint(image)
        laser = [min(x, 3) for x in data.ranges]

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            #resp_pause = pause.call()
            self.pause()
        except rospy.ServiceException, e:
            print ("/gazebo/pause_physics service call failed")

        # get reward
        reward = self.calculate_reward(done, pos)

        # ===== RETURN OBSERVATION =====
        # return [image, np.asarray(laser), np.asarray(pos)], reward, done, {}
        return [image, np.asarray(laser)], reward, done, {} # return raw scan data



    def _reset(self):
        self.hint_check = np.zeros(shape=(self.num_hint,))

        rospy.wait_for_service('gazebo/set_model_state')
        # random target after each death
        # self.num_target = np.random.randint(3)
        # self.num_hint = len(self.hint_pos[self.num_target])

        # ===== SEED TO SET RANDOM TARGET AND HINT =====
        # self.num_target = np.random.randint(4)
        # # self.num_target = 0 # set default to 0
        # self.num_hint = len(self.hint_pos[self.num_target])
        # self.setTarget()

        
        state = ModelState()
        state.model_name = self.name_model
        state.reference_frame = "world"
        state.pose= self.turtlebot_state.pose
        state.twist = self.turtlebot_state.twist
        self.set_model(state)

        # # ====== SET RANDOM ROBOT POSITION =====
        # num_robot = np.random.randint(3)
        # state.pose.position.x = self.robot_pos[num_robot][0]
        # state.pose.position.y = self.robot_pos[num_robot][1]

        
        # Unpause simulation to make observation
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except rospy.ServiceException, e:
            print ("/gazebo/unpause_physics service call failed")

        #read laser data
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/scan', LaserScan, timeout=5)
            except:
                pass
        pos = self.model_state(self.name_model, "world").pose.position
        done = self.calculate_observation(data, pos)

        # image, laser = self.discretize_observation(data)
        # change here
        image = self.get_image()
        laser = [min(x, 3) for x in data.ranges]

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            #resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        # ===== RETURN OBSERVATION =====
        return [image, np.asarray(laser)] # return raw scan data
        # return [image, np.asarray(laser), np.asarray(pos)]
