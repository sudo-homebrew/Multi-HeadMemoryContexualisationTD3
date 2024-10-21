from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile
from datetime import datetime
import time
import rclpy
import math
import numpy as np
import csv
import os
from turtlebot3_msgs.srv import RingGoal
from geometry_msgs.msg import Pose

EXPERIMENT_COUNT = 101


class SensorSub(Node):
    def __init__(self):
        super().__init__('SensorSub')

        """************************************************************
        ** Initialise variables
        ************************************************************"""

        self.last_pose_x = 0.0
        self.last_pose_y = 0.0
        self.last_pose_theta = 0.0

        self.current_pose_x = 0.0
        self.current_pose_y = 0.0
        self.current_pose_theta = 0.0

        self.odom_message_received = False

        """************************************************************
        ** Initialise ROS publishers and subscribers
        ************************************************************"""
        qos = QoSProfile(depth=10)

        # Initialise subscribers
        self.odom_sub = self.create_subscription(
            Odometry,
            'odom',
            self.odom_callback,
            qos)

    """*******************************************************************************
    ** Callback functions and relevant functions
    *******************************************************************************"""

    def odom_callback(self, msg):
        self.last_pose_x = self.current_pose_x
        self.last_pose_y = self.current_pose_y
        self.last_post_theta = self.current_pose_theta

        self.current_pose_x = msg.pose.pose.position.x
        self.current_pose_y = msg.pose.pose.position.y
        _, _, self.current_pose_theta = self.euler_from_quaternion(msg.pose.pose.orientation)

        self.odom_message_received = True


    """*******************************************************************************
    ** Below should be replaced when porting for ROS 2 Python tf_conversions is done.
    *******************************************************************************"""
    def euler_from_quaternion(self, quat):
        """
        Converts quaternion (w in last place) to euler roll, pitch, yaw
        quat = [x, y, z, w]
        """
        x = quat.x
        y = quat.y
        z = quat.z
        w = quat.w

        sinr_cosp = 2 * (w*x + y*z)
        cosr_cosp = 1 - 2*(x*x + y*y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (w*y - z*x)
        pitch = np.arcsin(sinp)

        siny_cosp = 2 * (w*z + x*y)
        cosy_cosp = 1 - 2 * (y*y + z*z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    def distance_from_last_step(self):
        return math.sqrt((self.current_pose_x - self.last_pose_x) ** 2 + (self.current_pose_y - self.last_pose_y) ** 2)

    def get_current_robot_world_pose(self):
        return [float(self.current_pose_x), float(self.current_pose_y)]

    def finish_waiting(self):
        self.odom_message_received = False


class Experiment(Node):
    def __init__(self):
        super().__init__('DRL_Experiment')

        self.sub = SensorSub()

        self.distance_travled = 0

        self.current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.result_directory = os.path.join(os.path.expanduser("~/turtlebot3_drlnav"), "DRL_Experiment_results", self.current_time)
        self.loop_cnt = 0
        self.episode_cnt = 0
        self.succeed_cnt = 0
        self.fail_cnt = 0
        self.init_x, self.init_y = 0.0, 0.0
        self.goal_x, self.goal_y = 0.0, 0.0

        self.run_once = True
        self.is_init_goal = False

        os.makedirs(self.result_directory, exist_ok=True)
        os.chdir(self.result_directory)

        # Subscriber
        qos = QoSProfile(depth=10)
        self.goal_pose_sub = self.create_subscription(Pose, 'goal_pose', self.goal_pose_callback, qos)

        # Initialise servers
        self.task_succeed_server    = self.create_service(RingGoal, 'task_succeed', self.task_succeed_callback)
        self.task_fail_server       = self.create_service(RingGoal, 'task_fail', self.task_fail_callback)


        while (not self.sub.odom_message_received):
            rclpy.spin_once(self.sub)
            print("Waiting for robot state datas... (if persists: reset gazebo world)")
            time.sleep(1.0)
        # while (not self.is_init_goal):
        #     print("Waiting for new goal... (if persists: reset gazebo_goals node)")
        #     time.sleep(1.0)
        self.sub.finish_waiting()

        print("Starting recording experiment!!")

        self.episode_start = time.perf_counter()
        self.step_timer = self.create_timer(
            0.1,  # unit: s
            self.process_step)


    def process_step(self):
        while (not self.sub.odom_message_received):# or (not self.sub.map_message_received):
            rclpy.spin_once(self.sub)
        self.sub.finish_waiting()

        self.loop_cnt += 1

        self.distance_travled += self.sub.distance_from_last_step()
        robot_pose_world = self.sub.get_current_robot_world_pose()

        self.write_to_csv(self.result_directory, "Distance_Data.log", self.distance_travled)
        self.write_to_csv_list(self.result_directory, "World_Robot_Pose_History.log", robot_pose_world)


    def write_to_csv(self, path, file_name, data):
        os.makedirs(os.path.join(path, 'EP_' + str(self.episode_cnt)), exist_ok=True)

        file_path = os.path.join(path, 'EP_' + str(self.episode_cnt), file_name)

        with open(file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([data])

    def write_to_csv_list(self, path, file_name, data, private=True):
        os.makedirs(os.path.join(path, 'EP_' + str(self.episode_cnt)), exist_ok=True)

        if private:
            file_path = os.path.join(path, 'EP_' + str(self.episode_cnt), file_name)
        else:
            file_path = os.path.join(path, file_name)

        with open(file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(data)

    def goal_pose_callback(self, msg):
        self.is_init_goal = True
        self.goal_x = msg.position.x
        self.goal_y = msg.position.y
        print(f'New goals ({self.goal_x}, {self.goal_y})')

    def task_succeed_callback(self, request, response):
        duration = time.perf_counter() - self.episode_start
        self.write_to_csv(self.result_directory, "Outcome.log", 'Succeed')
        self.write_to_csv(self.result_directory, "Duration.log", str(duration) + ' sec')
        self.write_to_csv_list(self.result_directory, "Goals.log", [self.init_x, self.init_y, self.goal_x, self.goal_y], private=False)
        self.init_x, self.init_y = self.goal_x, self.goal_y
        self.episode_cnt += 1
        self.loop_cnt = 0
        self.succeed_cnt += 1
        self.episode_start = time.perf_counter()
        if self.episode_cnt > EXPERIMENT_COUNT:
            self.finish_experiment()
        return response

    def task_fail_callback(self, request, response):
        duration = time.perf_counter() - self.episode_start
        self.write_to_csv(self.result_directory, "Outcome.log", 'Failed')
        self.write_to_csv(self.result_directory, "Duration.log", str(duration) + ' sec')
        self.write_to_csv_list(self.result_directory, "Goals.log", [self.init_x, self.init_y, self.goal_x, self.goal_y], private=False)
        self.init_x, self.init_y = 0.0, 0.0
        self.episode_cnt += 1
        self.loop_cnt = 0
        self.fail_cnt += 1
        self.episode_start = time.perf_counter()
        if self.episode_cnt > EXPERIMENT_COUNT:
            self.finish_experiment()
        return response

    def finish_experiment(self):
        os.makedirs(self.result_directory, exist_ok=True)

        file_path = os.path.join(self.result_directory, 'Outcomes.log')

        with open(file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([self.succeed_cnt, self.fail_cnt])
        print('Experiment Done!!')
        exit(0)


def main():
    rclpy.init(args=None)
    E = Experiment()
    rclpy.spin(E)
    E.destroy_node()
    rclpy.shutdown()
    print('Experiment Done!!')

if __name__ == '__main__':
    main()