from queue import Empty
from turtlebot3_msgs.srv import DrlFbeStep
from turtlebot3_msgs.srv import Goal
from std_srvs.srv import Empty
import os
import time
import rclpy
import torch
import numpy
from ..common.settings import REWARD_FUNCTION, COLLISION_OBSTACLE, COLLISION_WALL, TUMBLE, SUCCESS, TIMEOUT, RESULTS_NUM

import xml.etree.ElementTree as ET

try:
    with open('/tmp/drlnav_current_stage.txt', 'r') as f:
        stage = int(f.read())
except FileNotFoundError:
    print("\033[1m" + "\033[93m" + "Make sure to launch the gazebo simulation node first!" + "\033[0m}")
except ValueError:
    stage = f.read()

def check_gpu():
    print("gpu torch available: ", torch.backends.mps.is_available())
    if (torch.backends.mps.is_available()):
        print("device name: ", "mps")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def step(agent_self, action, previous_action):
    req = DrlFbeStep.Request()
    req.action = action
    req.previous_action = previous_action

    while not agent_self.step_comm_client.wait_for_service(timeout_sec=1.0):
        agent_self.get_logger().info('env step service not available, waiting again...')
    future = agent_self.step_comm_client.call_async(req)

    while rclpy.ok():
        rclpy.spin_once(agent_self)
        if future.done():
            if future.result() is not None:
                res = future.result()
                return res.map, res.state, res.reward, res.done, res.success, res.distance_traveled
            else:
                agent_self.get_logger().error(
                    'Exception while calling service: {0}'.format(future.exception()))
                print("ERROR getting step service response!")

def init_episode(agent_self):
    map, state, _, _, _, _ = step(agent_self, [], [0.0, 0.0])
    return map, state

def get_goal_status(agent_self):
    req = Goal.Request()
    while not agent_self.goal_comm_client.wait_for_service(timeout_sec=1.0):
        agent_self.get_logger().info('new goal service not available, waiting again...')
    future = agent_self.goal_comm_client.call_async(req)

    while rclpy.ok():
        rclpy.spin_once(agent_self)
        if future.done():
            if future.result() is not None:
                res = future.result()
                return res.new_goal
            else:
                agent_self.get_logger().error(
                    'Exception while calling service: {0}'.format(future.exception()))
                print("ERROR getting   service response!")

def wait_new_goal(agent_self):
    while(get_goal_status(agent_self) == False):
        print("Waiting for new goal... (if persists: reset gazebo_goals node)")
        time.sleep(1.0)

def pause_simulation(agent_self, real_robot):
    if real_robot:
        return
    while not agent_self.gazebo_pause.wait_for_service(timeout_sec=1.0):
        agent_self.get_logger().info('pause gazebo service not available, waiting again...')
    future = agent_self.gazebo_pause.call_async(Empty.Request())
    while rclpy.ok():
        rclpy.spin_once(agent_self)
        if future.done():
            return

def unpause_simulation(agent_self, real_robot):
    if real_robot:
        return
    while not agent_self.gazebo_unpause.wait_for_service(timeout_sec=1.0):
        agent_self.get_logger().info('unpause gazebo service not available, waiting again...')
    future = agent_self.gazebo_unpause.call_async(Empty.Request())
    while rclpy.ok():
        rclpy.spin_once(agent_self)
        if future.done():
            return

def translate_outcome(outcome):
    if outcome == SUCCESS:
        return "SUCCESS"
    elif outcome == COLLISION_WALL:
        return "COLL_WALL"
    elif outcome == COLLISION_OBSTACLE:
        return "COLL_OBST"
    elif outcome == TIMEOUT:
        return "TIMEOUT"
    elif outcome == TUMBLE:
        return "TUMBLE"
    else:
        return f"UNKNOWN: {outcome}"

# --- Environment ---

def euler_from_quaternion(quat):
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
    roll = numpy.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w*y - z*x)
    if sinp < -1:
        sinp = -1
    if sinp > 1:
        sinp = 1
    pitch = numpy.arcsin(sinp)

    siny_cosp = 2 * (w*z + x*y)
    cosy_cosp = 1 - 2 * (y*y + z*z)
    yaw = numpy.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw

def get_scan_count():
    tree = ET.parse(os.getenv('DRLNAV_BASE_PATH') + '/src/turtlebot3_simulations/turtlebot3_gazebo/models/turtlebot3_waffle_pi/model.sdf')

    root = tree.getroot()
    for link in root.find('model').findall('link'):
        if link.get('name') == 'base_scan':
            return int(link.find('sensor').find('ray').find('scan').find('horizontal').find('samples').text)

def get_simulation_speed(stage):
#    tree = ET.parse(os.getenv('DRLNAV_BASE_PATH') + '/src/turtlebot3_simulations/turtlebot3_gazebo/worlds/turtlebot3_drl_stage' + str(stage) + '/waffle_pi.model')
#    root = tree.getroot()
    return 10
#    return int(root.find('world').find('physics').find('real_time_factor').text)



#######################################################################

import numpy as np
import heapq , math , random , time
import scipy.interpolate as si


lookahead_distance = 0.24
speed = 0.26
expansion_size = 3
target_error = 0.15
robot_r = 0.27


pathGlobal = 0

def euler_from_quaternion(x,y,z,w):
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)
    return yaw_z

def heuristic(a, b):
    return np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)

def astar(array, start, goal):
    neighbors = [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]
    close_set = set()
    came_from = {}
    gscore = {start:0}
    fscore = {start:heuristic(start, goal)}
    oheap = []
    heapq.heappush(oheap, (fscore[start], start))
    while oheap:
        current = heapq.heappop(oheap)[1]
        if current == goal:
            data = []
            while current in came_from:
                data.append(current)
                current = came_from[current]
            data = data + [start]
            data = data[::-1]
            return data
        close_set.add(current)
        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j
            tentative_g_score = gscore[current] + heuristic(current, neighbor)
            if 0 <= neighbor[0] < array.shape[0]:
                if 0 <= neighbor[1] < array.shape[1]:
                    if array[neighbor[0]][neighbor[1]] == 1:
                        continue
                else:
                    # array bound y walls
                    continue
            else:
                # array bound x walls
                continue
            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                continue
            if  tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1]for i in oheap]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(oheap, (fscore[neighbor], neighbor))
    # If no path to goal was found, return closest path to goal
    if goal not in came_from:
        closest_node = None
        closest_dist = float('inf')
        for node in close_set:
            dist = heuristic(node, goal)
            if dist < closest_dist:
                closest_node = node
                closest_dist = dist
        if closest_node is not None:
            data = []
            while closest_node in came_from:
                data.append(closest_node)
                closest_node = came_from[closest_node]
            data = data + [start]
            data = data[::-1]
            return data
    return False

def bspline_planning(array, sn):
    try:
        array = np.array(array)
        x = array[:, 0]
        y = array[:, 1]
        N = 2
        t = range(len(x))
        x_tup = si.splrep(t, x, k=N)
        y_tup = si.splrep(t, y, k=N)

        x_list = list(x_tup)
        xl = x.tolist()
        x_list[1] = xl + [0.0, 0.0, 0.0, 0.0]

        y_list = list(y_tup)
        yl = y.tolist()
        y_list[1] = yl + [0.0, 0.0, 0.0, 0.0]

        ipl_t = np.linspace(0.0, len(x) - 1, sn)
        rx = si.splev(ipl_t, x_list)
        ry = si.splev(ipl_t, y_list)
        path = [(rx[i],ry[i]) for i in range(len(rx))]
    except:
        path = array
    return path

def pure_pursuit(current_x, current_y, current_heading, path, index):
    global lookahead_distance
    closest_point = None
    v = speed
    for i in range(index,len(path)):
        x = path[i][0]
        y = path[i][1]
        distance = math.hypot(current_x - x, current_y - y)
        if lookahead_distance < distance:
            closest_point = (x, y)
            index = i
            break
    if closest_point is not None:
        target_heading = math.atan2(closest_point[1] - current_y, closest_point[0] - current_x)
        desired_steering_angle = target_heading - current_heading
    else:
        target_heading = math.atan2(path[-1][1] - current_y, path[-1][0] - current_x)
        desired_steering_angle = target_heading - current_heading
        index = len(path)-1
    if desired_steering_angle > math.pi:
        desired_steering_angle -= 2 * math.pi
    elif desired_steering_angle < -math.pi:
        desired_steering_angle += 2 * math.pi
    if desired_steering_angle > math.pi/6 or desired_steering_angle < -math.pi/6:
        sign = 1 if desired_steering_angle > 0 else -1
        desired_steering_angle = sign * math.pi/4
        v = 0.0
    return v,desired_steering_angle,index

def frontierB(matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix[i][j] == 0.0:
                if i > 0 and matrix[i-1][j] < 0:
                    matrix[i][j] = 2
                elif i < len(matrix)-1 and matrix[i+1][j] < 0:
                    matrix[i][j] = 2
                elif j > 0 and matrix[i][j-1] < 0:
                    matrix[i][j] = 2
                elif j < len(matrix[i])-1 and matrix[i][j+1] < 0:
                    matrix[i][j] = 2
    return matrix

def assign_groups(matrix):
    group = 1
    groups = {}
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] == 2:
                group = dfs(matrix, i, j, group, groups)
    return matrix, groups

def dfs(matrix, i, j, group, groups):
    if i < 0 or i >= len(matrix) or j < 0 or j >= len(matrix[0]):
        return group
    if matrix[i][j] != 2:
        return group
    if group in groups:
        groups[group].append((i, j))
    else:
        groups[group] = [(i, j)]
    matrix[i][j] = 0
    dfs(matrix, i + 1, j, group, groups)
    dfs(matrix, i - 1, j, group, groups)
    dfs(matrix, i, j + 1, group, groups)
    dfs(matrix, i, j - 1, group, groups)
    dfs(matrix, i + 1, j + 1, group, groups) # lower right diagonal
    dfs(matrix, i - 1, j - 1, group, groups) # upper left cross
    dfs(matrix, i - 1, j + 1, group, groups) # upper right cross
    dfs(matrix, i + 1, j - 1, group, groups) # lower left diagonal
    return group + 1

def fGroups(groups):
    sorted_groups = sorted(groups.items(), key=lambda x: len(x[1]), reverse=True)
    top_five_groups = [g for g in sorted_groups[:5] if len(g[1]) > 2]    
    return top_five_groups

def calculate_centroid(x_coords, y_coords):
    n = len(x_coords)
    sum_x = sum(x_coords)
    sum_y = sum(y_coords)
    mean_x = sum_x / n
    mean_y = sum_y / n
    centroid = (int(mean_x), int(mean_y))
    return centroid

#This function selects the one from the largest 5 groups that is farther than target_error*2 and closest to the robot.
"""
def findClosestGroup(matrix,groups, current,resolution,originX,originY):
    targetP = None
    distances = []
    paths = []
    min_index = -1
    for i in range(len(groups)):
        middle = calculate_centroid([p[0] for p in groups[i][1]],[p[1] for p in groups[i][1]]) 
        path = astar(matrix, current, middle)
        path = [(p[1]*resolution+originX,p[0]*resolution+originY) for p in path]
        total_distance = pathLength(path)
        distances.append(total_distance)
        paths.append(path)
    for i in range(len(distances)):
        if distances[i] > target_error*3:
            if min_index == -1 or distances[i] < distances[min_index]:
                min_index = i
    if min_index != -1:
        targetP = paths[min_index]
    else: #gruplar target_error*2 uzaklıktan daha yakınsa random bir noktayı hedef olarak seçer. Bu robotun bazı durumlardan kurtulmasını sağlar.
        index = random.randint(0,len(groups)-1)
        target = groups[index][1]
        target = target[random.randint(0,len(target)-1)]
        path = astar(matrix, current, target)
        targetP = [(p[1]*resolution+originX,p[0]*resolution+originY) for p in path]
    return targetP
"""
def findClosestGroup(matrix,groups, current,resolution,originX,originY):
    targetP = None
    distances = []
    paths = []
    score = []
    max_score = -1 #max score index
    for i in range(len(groups)):
        middle = calculate_centroid([p[0] for p in groups[i][1]],[p[1] for p in groups[i][1]]) 
        path = astar(matrix, current, middle)
        path = [(p[1]*resolution+originX,p[0]*resolution+originY) for p in path]
        total_distance = pathLength(path)
        distances.append(total_distance)
        paths.append(path)
        print(middle)
    for i in range(len(distances)):
        if distances[i] == 0:
            score.append(0)
        else:
            score.append(len(groups[i][1])/distances[i])
    for i in range(len(distances)):
        if distances[i] > target_error*3:
            if max_score == -1 or score[i] > score[max_score]:
                max_score = i
    if max_score != -1:
        targetP = paths[max_score]
    else: #If the groups are closer than target_error*2 distance, it selects a random point as the target. This allows the robot to escape some situations.
        index = random.randint(0,len(groups)-1)
        target = groups[index][1]
        target = target[random.randint(0,len(target)-1)]
        path = astar(matrix, current, target)
        targetP = [(p[1]*resolution+originX,p[0]*resolution+originY) for p in path]
    return targetP

def pathLength(path):
    for i in range(len(path)):
        path[i] = (path[i][0],path[i][1])
        points = np.array(path)
    differences = np.diff(points, axis=0)
    distances = np.hypot(differences[:,0], differences[:,1])
    total_distance = np.sum(distances)
    return total_distance

def costmap(data,width,height,resolution):
    data = np.array(data).reshape(height,width)
    wall = np.where(data == 100)
    for i in range(-expansion_size,expansion_size+1):
        for j in range(-expansion_size,expansion_size+1):
            if i  == 0 and j == 0:
                continue
            x = wall[0]+i
            y = wall[1]+j
            x = np.clip(x,0,height-1)
            y = np.clip(y,0,width-1)
            data[x,y] = 100
    data = data*resolution
    return data

def exploration(data,width,height,resolution,column,row,originX,originY):
        global pathGlobal #Global day counter
        data = costmap(data,width,height,resolution) #expand the barriers
        data[row][column] = 0 #Robot Current Location
        data[data > 5] = 1 # Those with a score of 0 are a go-to place, those with a score of 100 are a definite obstacle.
        data = frontierB(data) #Find frontier points
        data,groups = assign_groups(data) #Group frontier points
        groups = fGroups(groups) #Sort the groups from smallest to largest. Get the 5 biggest groups
        if len(groups) == 0: #If there is no group, the survey is completed
            path = -1
        else: #If there is a group, find the closest group
            data[data < 0] = 1 #-0.05 ones are unknown location. Mark it as ungoable. 0 = can go, 1 = can't go.
            path = findClosestGroup(data,groups,(row,column),resolution,originX,originY) #Find the closest group
            if path != None: #If there is a path, fix it with BSpline
                path = bspline_planning(path,len(path)*5)
            else:
                path = -1
        pathGlobal = path
        return

def localControl(scan):
    v = None
    w = None
    for i in range(60):
        if scan[i] < robot_r:
            v = 0.2
            w = -math.pi/4 
            break
    if v == None:
        for i in range(300,360):
            if scan[i] < robot_r:
                v = 0.2
                w = math.pi/4
                break
    return v,w