#
#!/opt/homebrew/anaconda3/envs/ros2/bin python
# Copyright 2019 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Authors: Ryan Shim, Gilbert, Tomas
# Modified by: Seunghyeop
# Description: This code has been modified to train the Turtlebot3 Waffle_pi model.

import math
import numpy
import sys
import copy
import math
from numpy.core.numeric import Infinity
import cv2
from matplotlib import cm

from geometry_msgs.msg import Pose, Twist
from rosgraph_msgs.msg import Clock
from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs.msg import LaserScan
from turtlebot3_msgs.srv import DrlFbeStep, Goal, RingGoal

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data, QoSReliabilityPolicy

from . import reward_fbe as rw
from ..common import fbe_utilities as util
from ..common.fbe_settings import ENABLE_BACKWARD, EPISODE_TIMEOUT_SECONDS, ENABLE_MOTOR_NOISE, UNKNOWN, SUCCESS, COLLISION_WALL, COLLISION_OBSTACLE, TIMEOUT, TUMBLE, \
                                TOPIC_SCAN, TOPIC_VELO, TOPIC_ODOM, ARENA_LENGTH, ARENA_WIDTH, MAX_NUMBER_OBSTACLES, OBSTACLE_RADIUS, LIDAR_DISTANCE_CAP, \
                                    SPEED_LINEAR_MAX, SPEED_ANGULAR_MAX, THRESHOLD_COLLISION, THREHSOLD_GOAL, ENABLE_DYNAMIC_GOALS

NUM_SCAN_SAMPLES = util.get_scan_count()
LINEAR = 0
ANGULAR = 1
MAX_GOAL_DISTANCE = math.sqrt(ARENA_LENGTH**2 + ARENA_WIDTH**2)
class DRLEnvironment(Node):
    def __init__(self):
        super().__init__('drl_environment')
        with open('/tmp/drlnav_current_stage.txt', 'r') as f:
            self.stage = int(f.read())
        print(f"running on stage: {self.stage}")
        self.episode_timeout = EPISODE_TIMEOUT_SECONDS

        self.scan_topic = TOPIC_SCAN
        self.velo_topic = TOPIC_VELO
        self.odom_topic = TOPIC_ODOM
        self.goal_topic = 'goal_pose'

        self.goal_x, self.goal_y = 0.0, 0.0
        self.robot_x, self.robot_y = 0.0, 0.0
        self.robot_x_prev, self.robot_y_prev = 0.0, 0.0
        # self.robot_heading = 0.0
        self.total_distance = 0.0
        self.robot_tilt = 0.0
        self.map = 0

        self.done = False
        self.succeed = UNKNOWN
        self.episode_deadline = Infinity
        self.reset_deadline = False
        self.clock_msgs_skipped = 0

        self.obstacle_distances = [Infinity] * MAX_NUMBER_OBSTACLES

        self.new_goal = False
        self.goal_angle = 0.0

        self.previous_exploration_rate = 0.0
        self.exploration_rate = 0.0

        self.scan_ranges = [LIDAR_DISTANCE_CAP] * NUM_SCAN_SAMPLES
        self.obstacle_distance = LIDAR_DISTANCE_CAP

        self.difficulty_radius = 1
        self.local_step = 0
        self.time_sec = 0

        self.map_message_received = False

        """************************************************************
        ** Initialise ROS publishers and subscribers
        ************************************************************"""
        qos = QoSProfile(depth=10)
        qos_clock = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        # publishers
        self.cmd_vel_pub = self.create_publisher(Twist, self.velo_topic, qos)
        # subscribers
        self.goal_pose_sub = self.create_subscription(Pose, self.goal_topic, self.goal_pose_callback, qos)
        self.odom_sub = self.create_subscription(Odometry, self.odom_topic, self.odom_callback, qos)
        self.scan_sub = self.create_subscription(LaserScan, self.scan_topic, self.scan_callback, qos_profile=qos_profile_sensor_data)
        self.clock_sub = self.create_subscription(Clock, '/clock', self.clock_callback, qos_profile=qos_clock)
        self.obstacle_odom_sub = self.create_subscription(Odometry, 'obstacle/odom', self.obstacle_odom_callback, qos)
        self.map_subscriber = self.create_subscription(OccupancyGrid, 'map', self.map_callback, qos)
        # clients
        self.task_succeed_client = self.create_client(RingGoal, 'task_succeed')
        self.task_fail_client = self.create_client(RingGoal, 'task_fail')
        # servers
        self.step_comm_server = self.create_service(DrlFbeStep, 'step_comm', self.step_comm_callback)
        self.goal_comm_server = self.create_service(Goal, 'goal_comm', self.goal_comm_callback)

    """*******************************************************************************
    ** Callback functions and relevant functions
    *******************************************************************************"""

    def goal_pose_callback(self, msg):
        self.goal_x = msg.position.x
        self.goal_y = msg.position.y
        self.new_goal = True
        print(f"new episode!!")

    def goal_comm_callback(self, request, response):
        response.new_goal = self.new_goal
        return response

    def obstacle_odom_callback(self, msg):
        if 'obstacle' in msg.child_frame_id:
            robot_pos = msg.pose.pose.position
            obstacle_id = int(msg.child_frame_id[-1]) - 1
            diff_x = self.robot_x - robot_pos.x
            diff_y = self.robot_y - robot_pos.y
            self.obstacle_distances[obstacle_id] = math.sqrt(diff_y**2 + diff_x**2)
        else:
            print("ERROR: received odom was not from obstacle!")

    def odom_callback(self, msg):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        # _, _, self.robot_heading = util.euler_from_quaternion(msg.pose.pose.orientation)
        self.robot_tilt = msg.pose.pose.orientation.y

    def scan_callback(self, msg):
        if len(msg.ranges) != NUM_SCAN_SAMPLES:
            print(f"more or less scans than expected! check model.sdf, got: {len(msg.ranges)}, expected: {NUM_SCAN_SAMPLES}")
        # normalize laser values
        self.obstacle_distance = 1
        for i in range(NUM_SCAN_SAMPLES):
                self.scan_ranges[i] = numpy.clip(float(msg.ranges[i]) / LIDAR_DISTANCE_CAP, 0, 1)
                if self.scan_ranges[i] < self.obstacle_distance:
                    self.obstacle_distance = self.scan_ranges[i]
        self.obstacle_distance *= LIDAR_DISTANCE_CAP

    def map_callback(self, msg):
        self.map = msg
        self.map_message_received = True

    def clock_callback(self, msg):
        self.time_sec = msg.clock.sec
        if not self.reset_deadline:
            return
        self.clock_msgs_skipped += 1
        if self.clock_msgs_skipped <= 10: # Wait a few message for simulation to reset clock
            return
        episode_time = self.episode_timeout
        if ENABLE_DYNAMIC_GOALS:
            episode_time = numpy.clip(episode_time * self.difficulty_radius, 10, 50)
        self.episode_deadline = self.time_sec + episode_time
        self.reset_deadline = False
        self.clock_msgs_skipped = 0

    def stop_reset_robot(self, success):
        self.cmd_vel_pub.publish(Twist()) # stop robot
        self.episode_deadline = Infinity
        self.done = True
        req = RingGoal.Request()
        req.robot_pose_x = self.robot_x
        req.robot_pose_y = self.robot_y
        req.radius = numpy.clip(self.difficulty_radius, 0.5, 4)
        if success:
            self.difficulty_radius *= 1.01
            while not self.task_succeed_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().info('success service not available, waiting again...')
            self.task_succeed_client.call_async(req)
        else:
            self.difficulty_radius *= 0.99
            while not self.task_fail_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().info('fail service not available, waiting again...')
            self.task_fail_client.call_async(req)

    def get_map_size(self):
        return self.map.info.height * self.map.info.width# * (self.map.info.resolution ** 2)
        # return 216 * 217

    def get_explored_size(self):
        explored = 0
        for cell in self.map.data:
            if cell != -1:
                explored += 1

        return explored

    def get_exploration_rate(self):
        if not self.map_message_received or self.map == 0:
            return 0
        return self.get_explored_size()# / self.get_map_size()

    def resize_map(self, map_arr):
        ## TODO ##
        ## check order of map width and height ##
        data = numpy.array(map_arr, dtype=numpy.int8).reshape((self.map.info.width, self.map.info.height))
        resized_data = cv2.resize(data, (128, 128), interpolation=cv2.INTER_NEAREST)
        resized_data_list = resized_data.flatten().tolist()

        return [float(i) for i in resized_data_list]

    def frontierB(self):
        map_arr = numpy.zeros_like(self.map.data)
        for i in range(self.map.info.width):
            for j in range(self.map.info.height):
                if self.map.data[i + self.map.info.width * j] == 0.0:
                    if i > 0 and self.map.data[(i - 1) + self.map.info.width * j] < 0:
                        map_arr[i + self.map.info.width * j] = 254
                    elif i < self.map.info.width - 1 and self.map.data[(i + 1) + self.map.info.width * j] < 0:
                        map_arr[i + self.map.info.width * j] = 254
                    elif j > 0 and self.map.data[i + self.map.info.width * (j - 1)] < 0:
                        map_arr[i + self.map.info.width * j] = 254
                    elif j < self.map.info.height - 1 and self.map.data[i + self.map.info.width * (j + 1)] < 0:
                        map_arr[i + self.map.info.width * j] = 254
        return map_arr

    def worldToMapValidated(self, wx: float, wy: float):
        if wx < self.map.info.origin.position.x or wy < self.map.info.origin.position.y:
            return (None, None)
        mx = int((wx - self.map.info.origin.position.x) // self.map.info.resolution)
        my = int((wy - self.map.info.origin.position.y) // self.map.info.resolution)
        if mx < self.map.info.width and my < self.map.info.height:
            return (mx, my)
        return (None, None)

    def get_map_state(self):
        resized_map = self.resize_map(self.map.data)

        frontier_mark = self.resize_map(self.frontierB())

        robot_coords = []
        for i in range(int(self.robot_x - THRESHOLD_COLLISION) * 100, int(self.robot_x + THRESHOLD_COLLISION) * 100):
            for j in range(int(self.robot_y - THRESHOLD_COLLISION) * 100, int(self.robot_y + THRESHOLD_COLLISION) * 100):
                robot_coords.append(self.worldToMapValidated(i / 100, j / 100))
        robot_coords = list(set(robot_coords))
        temp_robot_pos = numpy.zeros(self.map.info.width * self.map.info.height)
        for coord in robot_coords:
            temp_robot_pos[coord[0] + (self.map.info.width * coord[1])] = 254

        # r_x, r_y = self.worldToMapValidated(self.robot_x, self.robot_y)
        # temp_robot_pos[r_x + (self.map.info.width * r_y)] = 254

        robot_pos = self.resize_map(temp_robot_pos)


        return resized_map + frontier_mark + robot_pos

    def get_state(self, action_linear_previous, action_angular_previous):
        # state = copy.deepcopy(self.scan_ranges)                                             # range: [ 0, 1]
        state = copy.deepcopy(((numpy.exp((numpy.ones(len(self.scan_ranges)) - self.scan_ranges) * 4) - 1) / (numpy.exp(numpy.ones(len(self.scan_ranges)) * 4) - 1)).tolist())             # range: [ 0, 1]
        state.append(float(max(self.scan_ranges)))
        state.append(float(self.scan_ranges.index(max(self.scan_ranges)) / NUM_SCAN_SAMPLES))
        state.append(float(action_linear_previous))                                         # range: [-1, 1]
        state.append(float(action_angular_previous))                                        # range: [-1, 1]
        self.previous_exploration_rate = self.exploration_rate
        self.exploration_rate = self.get_exploration_rate()
        state.append(float(self.exploration_rate))

        if not self.map_message_received or self.map == 0:
            map_state = numpy.zeros(128 * 128 *3).fill(-1).tolist()
        else:
            map_state = self.get_map_state()

        self.local_step += 1

        if self.local_step <= 80: # Grace period to wait for simulation reset
            return map_state, state
        # Success
        if self.exploration_rate / (112 * 118) > 0.95:
            self.succeed = SUCCESS
        # Collision
        elif self.obstacle_distance < THRESHOLD_COLLISION:
            dynamic_collision = False
            for obstacle_distance in self.obstacle_distances:
                if obstacle_distance < (THRESHOLD_COLLISION + OBSTACLE_RADIUS + 0.05):
                    dynamic_collision = True
            if dynamic_collision:
                self.succeed = COLLISION_OBSTACLE
            else:
                self.succeed = COLLISION_WALL
        # Timeout
        elif self.time_sec >= self.episode_deadline:
            self.succeed = TIMEOUT
        # Tumble
        elif self.robot_tilt > 0.06 or self.robot_tilt < -0.06:
            self.succeed = TUMBLE
        if self.succeed is not UNKNOWN:
            self.stop_reset_robot(self.succeed == SUCCESS)
        return map_state, state

    def initalize_episode(self, response):
        response.map, response.state = self.get_state(0, 0)
        response.reward = 0.0
        response.done = False
        response.distance_traveled = 0.0
        rw.reward_initalize(self.exploration_rate)
        return response

    def step_comm_callback(self, request, response):
        if len(request.action) == 0:
            return self.initalize_episode(response)

        if ENABLE_MOTOR_NOISE:
            request.action[LINEAR] += numpy.clip(numpy.random.normal(0, 0.05), -0.1, 0.1)
            request.action[ANGULAR] += numpy.clip(numpy.random.normal(0, 0.05), -0.1, 0.1)

        # Un-normalize actions
        if ENABLE_BACKWARD:
            action_linear = request.action[LINEAR] * SPEED_LINEAR_MAX
        else:
            action_linear = (request.action[LINEAR] + 1) / 2 * SPEED_LINEAR_MAX
        action_angular = request.action[ANGULAR] * SPEED_ANGULAR_MAX

        # Publish action cmd
        twist = Twist()
        twist.linear.x = action_linear
        twist.angular.z = action_angular
        self.cmd_vel_pub.publish(twist)

        # Prepare repsonse
        response.map, response.state = self.get_state(request.previous_action[LINEAR], request.previous_action[ANGULAR])
        response.reward = rw.get_reward(float(self.exploration_rate - self.previous_exploration_rate), self.succeed, action_linear, action_angular)
        response.done = self.done
        response.success = self.succeed
        response.distance_traveled = 0.0
        if self.done:
            response.distance_traveled = self.total_distance
            # Reset variables
            self.succeed = UNKNOWN
            self.total_distance = 0.0
            self.local_step = 0
            self.done = False
            self.reset_deadline = True
        if self.local_step % 200 == 0:
            print(f"Rtot: {response.reward:<8.2f} ME: {self.exploration_rate / (328 * 635):<8.2f}\t", end='')
            print(f"MinD: {self.obstacle_distance:<8.2f}Alin: {request.action[LINEAR]:<7.1f}Aturn: {request.action[ANGULAR]:<7.1f}")
        return response

def main(args=sys.argv[1:]):
    rclpy.init(args=args)
    if len(args) == 0:
        drl_environment = DRLEnvironment()
    else:
        rclpy.shutdown()
        quit("ERROR: wrong number of arguments!")
    rclpy.spin(drl_environment)
    drl_environment.destroy()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
