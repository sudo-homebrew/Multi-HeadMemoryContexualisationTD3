#!/usr/bin/env python3
#


import math
import time
import signal
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import functools
import logging
import rclpy
import tf2_py
import tf2_geometry_msgs
import datetime
import tf2_ros
import random
import subprocess
from std_msgs.msg import Float64MultiArray
from tf2_ros import TransformListener
from tf2_ros.buffer import Buffer
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
#import tf as tranf
#from tf.transformations import euler_from_quaternion, quaternion_from_euler
from pyquaternion import Quaternion
from collections import deque
#from .real_goal import RealGoal
from nav_msgs.msg import OccupancyGrid

from geometry_msgs.msg import Pose
from std_srvs.srv import Empty
from gazebo_msgs.srv import DeleteEntity, SpawnEntity
from turtlebot3_msgs.srv import RingGoal

from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import Point



class SensorSub(Node):
    def __init__(self, pose):
        super().__init__('SensorSub')

        """************************************************************
        ** Initialise variables
        ************************************************************"""
        self.scan_ranges = []
        self.scanMsg = 0
        
        self.goal_pose_x = pose[0]
        self.goal_pose_y = pose[1]
        self.last_pose_x = 0.0
        self.last_pose_y = 0.0
        self.last_pose_theta = 0.0
        
        self.map = 0
        
        self.tf_buffer = Buffer()
        self.transform_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        
        
        
        """************************************************************
        ** Initialise ROS publishers and subscribers
        ************************************************************"""
        qos = QoSProfile(depth=10)

        # Initialise subscribers
        self.scan_sub = self.create_subscription(
            LaserScan,
            'scan',
            self.scan_callback,
            qos_profile=qos_profile_sensor_data)
            
        self.odom_sub = self.create_subscription(
            Odometry,
            'odom',
            self.odom_callback,
            qos)
            
        qos_profile = QoSProfile(depth=10)
        self.map_subscriber = self.create_subscription(
            OccupancyGrid,
            'map',
            self.map_callback,
            qos_profile)

    """*******************************************************************************
    ** Callback functions and relevant functions
    *******************************************************************************"""

    def scan_callback(self, msg):
        self.scan_ranges = msg.ranges # 0 ~ 359 list if 3.5 > = inf
        self.scanMsg = msg
        
    def odom_callback(self, msg):
        self.last_pose_x = msg.pose.pose.position.x
        self.last_pose_y = msg.pose.pose.position.y
        _, _, self.last_pose_theta = self.euler_from_quaternion(msg.pose.pose.orientation)

        goal_distance = math.sqrt(
            (self.goal_pose_x-self.last_pose_x)**2
            + (self.goal_pose_y-self.last_pose_y)**2)

        path_theta = math.atan2(
            self.goal_pose_y-self.last_pose_y,
            self.goal_pose_x-self.last_pose_x)

        goal_angle = path_theta - self.last_pose_theta
        if goal_angle > math.pi:
            goal_angle -= 2 * math.pi

        elif goal_angle < -math.pi:
            goal_angle += 2 * math.pi

        self.goal_distance = goal_distance
        self.goal_angle = goal_angle
        
    def map_callback(self, msg):
        self.map = msg
        

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
        
    def quaternion_from_euler(roll, pitch, yaw):
        """
        Converts Euler angles to quaternion (w, x, y, z) representation.

        Args:
            roll (float): Roll angle in radians.
            pitch (float): Pitch angle in radians.
            yaw (float): Yaw angle in radians.

        Returns:
            Tuple[float, float, float, float]: Quaternion representation of the input Euler angles.
        """
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        w = cy * cp * cr + sy * sp * sr
        x = cy * cp * sr - sy * sp * cr
        y = sy * cp * sr + cy * sp * cr
        z = sy * cp * cr - cy * sp * sr

        return w, x, y, z

def rotateneg(point, angle):
    x, y = point
    xx = x * math.cos(angle) + y * math.sin(angle)
    yy = -x * math.sin(angle) + y * math.cos(angle)
    return xx, yy

def rotatepos(point, angle):
    x, y = point
    xx = x * math.cos(angle) - y * math.sin(angle)
    yy = x * math.sin(angle) + y * math.cos(angle)
    return xx, yy

def calcqxqy(dist, angl, ang):
#    angl = math.radians(angl)
    angle = angl + ang
    if angle > np.pi:
        angle = np.pi - angle
        angle = -np.pi - angle
    if angle < -np.pi:
        angle = -np.pi - angle
        angle = np.pi - angle
    if angle > 0:
        qx, qy = rotatepos([dist, 0], angle)
    else:
        qx, qy = rotateneg([dist, 0], -angle)
    return qx, qy
    



class ImplementEnv(Node):
    def __init__(self, global_goal):
        super().__init__('IMP')
        
        
        self.entity_dir_path = (os.path.dirname(os.path.realpath(__file__))).replace(
            'turtlebot3_drl/turtlebot3_drl/gdae',
            'turtlebot3_simulations/turtlebot3_gazebo/models/turtlebot3_drl_world/goal_box')
        self.entity_path = os.path.join(self.entity_dir_path, 'model.sdf')
        self.entity = open(self.entity_path, 'r').read()
        self.entity_name = 'goal'
        
        
        self.node = []
        self.global_goal_x = global_goal[0]
        self.global_goal_y = global_goal[1]
        self.step_time = 10 # second
        self.threshold = 1 # for detecting gap
        self.reference_point = None
        self.is_in_process = True
        self.poi = [0, 0]
        self.timeout = 10
        
        self.sub = SensorSub(global_goal)
        
        
        self.init_state = False
        self.goal_pose_x = self.global_goal_x
        self.goal_pose_y = self.global_goal_y
        # Gernerating Goal
        qos = QoSProfile(depth=10)

        # Initialise publishers
        self.goal_pose_pub = self.create_publisher(Pose, 'goal_pose', qos)

        # Initialise client
        self.delete_entity_client = self.create_client(DeleteEntity, 'delete_entity')
        self.spawn_entity_client = self.create_client(SpawnEntity, 'spawn_entity')
        
        # Initialise server
        self.task_succeed_server = self.create_service(RingGoal, 'task_succeed', self.task_succeed_callback)
        self.task_fail_server = self.create_service(RingGoal, 'task_fail', self.task_fail_callback)

        
        # Process

        self.cycle()
        
        
        
        
    def cycle(self):
        while True:
            for _ in range(10):
                rclpy.spin_once(self.sub)
            if self.sub.scanMsg == 0:
                continue

            self.step(self.sub.scanMsg.ranges, self.sub.scanMsg.angle_increment, self.sub.last_pose_x, self.sub.last_pose_y, self.sub.last_pose_theta, self.sub.map, self.sub.transform_listener)
            



            
            self.update_goal_pose(self.poi)
            break
        
        print('Candidates : ', self.node)
        print('\n POI1 : ', self.poi)
#        print('\n POI2 : ', [self.goal_pose_x, self.goal_pose_y])
        self.publish_callback()
        
        
#############################################
#        sellistener = tranf.TransformListener()##
#############################################
        
    def step(self, lidar, angle_increment, pos_x, pos_y, pos_twist, map, transform_listener):
        # Check if robot is reached to Gloabl goal enough
        if self.global_euclidean_distance(pos_x, pos_y) < 1:
        # if distance between global goal and robots location is less than 1m
            self.is_in_process = False
            self.poi = [self.global_goal_x, self.global_goal_y]
            return self.poi
            
        # Reverse lidar order so that lidar sensor values allign in anti-clockwise
#        lidar.reverse()
        
        # condition 1. value difference between 2 sequantial lidar sensor values

        is_gap = False
        gap_start = 0
        gap_end = 0
        gap_mid = 0
        
        for index, cur_value in enumerate(lidar):
            if index == 0:
                pre_value = cur_value
                continue
        
            if abs(pre_value - cur_value) > self.threshold or (index + 1 == len(lidar) and is_gap):
                if not is_gap:
                    gap_start = index
                    is_gap = True
#                    print('check point 0\ngap started\nindex = ', index)
                else:
                    gap_end = index
                    is_gap = False
                    
                    gap_mid = int((gap_start + gap_end) / 2)
                    
                    

                    candidate_x, candidate_y = self.calc_lidar_coordinate(angle_increment * gap_mid, lidar[gap_mid], pos_x, pos_y, pos_twist)
                    #######################
                    
                    
                    self.node.append([candidate_x, candidate_y])
                    
                    gap_start = 0
                    gap_end = 0
                    gap_mid = 0
            pre_value = cur_value
               
            
                    
        # condition 2. infinite readings of lidar sensor values

        is_gap = False
        gap_start = 0
        gap_end = 0
        gap_mid = 0
        
        for index, cur_value in enumerate(lidar):
            if cur_value == math.inf and not is_gap:
                gap_start = index
                is_gap = True
                
            if (cur_value != math.inf and is_gap) or (index + 1 == len(lidar) and is_gap):
                gap_end = index
                is_gap = False
                gap_mid = int((gap_start + gap_end) / 2)
                

                candidate_x, candidate_y = self.calc_lidar_coordinate(angle_increment * gap_mid, lidar[gap_mid], pos_x, pos_y, pos_twist)
                
                
                self.node.append([candidate_x, candidate_y])
                
                gap_start = 0
                gap_end = 0
                gap_mid = 0
                
                
        

        
        heuristic_scores = self.heuristic(pos_x, pos_y, map, transform_listener)
        
        try:
            POI = self.node[heuristic_scores.index(min(heuristic_scores))]
        except ValueError:
            POI = [self.global_goal_x, self.global_goal_y]
        
        if POI[0] == math.inf or POI[1] == math.inf:
            POI = [self.global_goal_x, self.global_goal_y]
        
        self.poi = POI
        self.delete_nodes()
        self.poi = POI
        
        return POI
        
##############################################################
    
    ''' Twist must be considered '''
    def calc_lidar_coordinate(self, theta, dist, pos_x, pos_y, pos_twist):
        if dist > 4:
            dist = 4
        theta -= math.pi / 2

        x_cordinate, y_cordinate = calcqxqy(dist, theta, pos_twist)
        
        return x_cordinate + pos_x, y_cordinate + pos_y
    


    def delete_nodes(self):
        self.del_dup_node()
        
        while len(self.node) > 30:
            self.node.pop(0)
        
        
    def del_dup_node(self):
        for index, element in enumerate(self.node):
            for i, e in reversed(list(enumerate(self.node))):
                if self.dist(element[0], element[1], e[0], e[1]) < 1 and index != i:
                    del self.node[index]
                    break

    
    def heuristic(self, pos_x, pos_y, map, transform_listener):
        heuristic_scores = []
        for candidate in self.node:
            ds = self.dist(candidate[0], candidate[1], pos_x, pos_y)
            ged = self.global_euclidean_distance(candidate[0], candidate[1])
            mi = self.map_information(candidate[0], candidate[1], 5, map, transform_listener)
            heuristic_scores.append(ds + ged + mi)
        
        return heuristic_scores
    
    ### l1 = 5, l2 = 10 (Constnt)
    def distance_score(self, candidate_x, candidate_y, pos_x, pos_y, l1 = 5, l2 = 10):
        numerator = math.e ** ((math.sqrt((candidate_x - pos_x) ** 2 + (candidate_y - pos_y) ** 2) / (l2 - l1)) ** 2)
        
        denominator = math.e ** ((l2/(l2-l1)) ** 2)
        
        tem = math.tanh(numerator / denominator) * l2
        
        return tem
        
    def global_euclidean_distance(self, candidate_x, candidate_y):
        return math.sqrt((candidate_x - self.global_goal_x) ** 2 + (candidate_y - self.global_goal_y) ** 2)
        
    def dist(self, x1, y1, x2, y2):
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        
        
        
    def desired_points_callback(self, point, transform_listener): # msg tyep : PointStamped
        try:
            if self.reference_point is None:
                self.reference_point = [0.0, 0.0]  # Initial point of robot

            transform_msg = transform_listener.lookup_transform('map', 'odom')
            transformed_points = self.transform_points(point, transform_msg)
#            self.publish_markers(transformed_points)
        except Exception as e:
            logging.error('Error occurred: {}'.format(str(e)))
#            print('Error occurred: {}'.format(str(e)))
            
        return transformed_point



    def transform_points(self, point, transform_msg):
        point_stamped = PointStamped()
        point_stamped.header.frame_id = 'odom'
        point_stamped.point = point

        transformed_point = do_transform_point(point_stamped, transform_msg)
        return transformed_point.point
        
        
    def map_information(self, candidate_x, candidate_y, k, map, transform_listener):
        try:
            d = map.data
            d[d < 50] = 5 # obstacle
            d[d >= 253] = 1 # free
            d[d >= 50] = 0 # unknown
            m_value = np.array(d).reshape(map.info.height, map.info.width).tolist()
        except Exception as e:
            return 0
        
        
        for w in range(-int(k/2), int(k/2)):
            for h in range(-int(k/2), int(k/2)):
                map_pont_x, map_pont_y = self.desired_points_callback([candidate_x + w, candidate_y + h], transform_listener)
                sum1 = m_value[map_pont_x][map_pont_y]
                
        tem = sum1 / (k**2)
        
        return math.e ** tem
        

    def publish_callback(self):
        if self.init_state is False:
            self.init_state = True
        goal_pose = Pose()
        goal_pose.position.x = self.goal_pose_x
        goal_pose.position.y = self.goal_pose_y
        self.goal_pose_pub.publish(goal_pose)
        self.spawn_entity()
        
        

    def task_succeed_callback(self, request, response):
        self.delete_entity()
        '''
        if [self.global_goal_x, self.global_goal_y] == self.poi:
            print('Global goal reached!!')
            exit()
        else:
            print("POI Reached", self.poi)
            self.cycle()
        '''
        print("POI Reached", self.poi)
        self.cycle()
        
        return response

    def task_fail_callback(self, request, response):
        self.delete_entity()
        
        print("Couldn't reach to POI in time")
        self.cycle()
        return response
        
    def update_goal_pose(self, pose):
        self.goal_pose_x = pose[0]
        self.goal_pose_y = pose[1]
        
        try:
            self.node.remove(self.poi)
        except Exception:
            pass
            
            
    def spawn_entity(self):
        goal_pose = Pose()
        goal_pose.position.x = self.goal_pose_x
        goal_pose.position.y = self.goal_pose_y
        req = SpawnEntity.Request()
        req.name = self.entity_name
        req.xml = self.entity
        req.initial_pose = goal_pose
        while not self.spawn_entity_client.wait_for_service(timeout_sec=1.0):
            pass
#            self.get_logger().info('service not available, waiting again...')
        self.spawn_entity_client.call_async(req)
        
    def delete_entity(self):
        req = DeleteEntity.Request()
        req.name = self.entity_name
        while not self.delete_entity_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.delete_entity_client.call_async(req)

def timeout_handler(num, stack):
    logging.warning('Goal has not been reached within the given timeout.')
#    print('Goal has not been reached within the given timeout.')
    raise Exception("GOALTIMEOUT")
    

class MarkerPublisher(Node):

    def __init__(self, pose, poi):
        super().__init__('marker_publisher')
        self.goal_pose_x = pose[0]
        self.goal_pose_y = pose[1]
        self.marker_pub = self.create_publisher(MarkerArray, 'goal_point', 10)
        self.marker_pub2 = self.create_publisher(MarkerArray, 'poi_pose', 10)
        self.timer = self.create_timer(1.0, self.publish_marker)
        self.poi_pose_x = poi[0]
        self.poi_pose_y = poi[1]

    def publish_marker(self):
        markerArray = MarkerArray()
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose.position.x = self.goal_pose_x
        marker.pose.position.y = self.goal_pose_y
        marker.pose.position.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.01
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0

        markerArray.markers.append(marker)
        self.marker_pub.publish(markerArray)

        markerArray2 = MarkerArray()
        marker2 = Marker()
        marker2.header.frame_id = 'map'
        marker2.type = Marker.CUBE
        marker2.action = Marker.ADD
        marker2.pose.position.x = self.poi_pose_x
        marker2.pose.position.y = self.poi_pose_y
        marker2.pose.position.z = 0.0
        marker2.pose.orientation.w = 1.0
        marker2.scale.x = 0.2
        marker2.scale.y = 0.2
        marker2.scale.z = 0.01
        marker2.color.a = 1.0
        marker2.color.r = 0.0
        marker2.color.g = 1.0
        marker2.color.b = 0.0

        markerArray2.markers.append(marker2)
        self.marker_pub2.publish(markerArray2)


def main(args=sys.argv[1], args1=sys.argv[2]):
    print("Start gdae codes")
    pose = [float(args), float(args1)]
   
    rclpy.init(args=None)
    
    sub = SensorSub(pose)
    Env = ImplementEnv(pose)
    marker_node = rclpy.create_node("multiple_coordinates_marker")

    while Env.is_in_process:
        for _ in range(10):
                rclpy.spin_once(sub)
        if sub.scanMsg == 0:
            continue
        rclpy.spin_once(Env)
        poi = Env.step(sub.scanMsg.ranges, sub.scanMsg.angle_increment, sub.last_pose_x, sub.last_pose_y, sub.last_pose_theta, sub.map, sub.transform_listener)
        marker_publisher = MarkerPublisher(pose, poi)
        rclpy.spin_once(marker_publisher)

        for i in range(0, len(Env.node)):
            globals()["marker{}".format(i)] = Marker()
            globals()["marker{}".format(i)].header.frame_id = "/map"
            globals()["marker{}".format(i)].type = Marker.CUBE
            globals()["marker{}".format(i)].action = Marker.ADD
            globals()["marker{}".format(i)].id = i
            globals()["marker{}".format(i)].pose.position = Point(x=Env.node[i][0], y=Env.node[i][1], z=0.0)
            globals()["marker{}".format(i)].scale.x = 0.2
            globals()["marker{}".format(i)].scale.y = 0.2
            globals()["marker{}".format(i)].scale.z = 0.01
            globals()["marker{}".format(i)].color.a = 1.0
            globals()["marker{}".format(i)].color.r = 0.0
            globals()["marker{}".format(i)].color.g = 0.0
            globals()["marker{}".format(i)].color.b = 1.0
            globals()["marker_pub{}".format(i)] = marker_node.create_publisher(Marker, 'node', 10)
            #for i in range(0, 2):
            globals()["marker_pub{}".format(i)].publish(globals()["marker{}".format(i)])

    Env.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

