from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile
from datetime import datetime
import time
import rclpy
import csv
import os
from geometry_msgs.msg import Pose


class SensorSub(Node):
    def __init__(self):
        super().__init__('SensorSub')

        """************************************************************
        ** Initialise variables
        ************************************************************"""

        self.obstacle_coords = [[0, 0]] * 10

        self.odom_message_received = False

        """************************************************************
        ** Initialise ROS publishers and subscribers
        ************************************************************"""
        qos = QoSProfile(depth=10)

        # Initialise subscribers
        self.obstacle_odom_sub = self.create_subscription(Odometry, 'obstacle/odom', self.obstacle_odom_callback, qos)

    """*******************************************************************************
    ** Callback functions and relevant functions
    *******************************************************************************"""

    def obstacle_odom_callback(self, msg):
        if 'obstacle' in msg.child_frame_id:
            robot_pos = msg.pose.pose.position
            obstacle_id = int(msg.child_frame_id[-1]) - 1

            self.obstacle_coords[obstacle_id] = [robot_pos.x, robot_pos.y]
            self.odom_message_received = True
        else:
            print("ERROR: received odom was not from obstacle!")


    """*******************************************************************************
    ** Below should be replaced when porting for ROS 2 Python tf_conversions is done.
    *******************************************************************************"""
    def get_current_obstacles_world_pose(self):
        return self.obstacle_coords

    def finish_waiting(self):
        self.odom_message_received = False


class Experiment(Node):
    def __init__(self):
        super().__init__('DRL_Experiment')

        self.sub = SensorSub()

        self.distance_travled = 0

        self.current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.result_directory = os.path.join(os.path.expanduser("~/turtlebot3_drlnav"), "DRL_Experiment_results", self.current_time)

        self.run_once = True
        self.is_init_goal = False

        os.makedirs(self.result_directory, exist_ok=True)
        os.chdir(self.result_directory)

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

        obstacles_pose_world = self.sub.get_current_obstacles_world_pose()

        for i, pose in enumerate(obstacles_pose_world):
            self.write_to_csv_list(self.result_directory, f"Obstacles_Pose_History_{i}.log", pose, private=False)


    def write_to_csv(self, path, file_name, data):
        file_path = os.path.join(path, file_name)

        with open(file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([data])

    def write_to_csv_list(self, path, file_name, data, private=True):
        if private:
            file_path = os.path.join(path, 'EP_' + str(0), file_name)
        else:
            file_path = os.path.join(path, file_name)

        with open(file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(data)


def main():
    rclpy.init(args=None)
    E = Experiment()
    rclpy.spin(E)
    E.destroy_node()
    rclpy.shutdown()
    print('Experiment Done!!')

if __name__ == '__main__':
    main()