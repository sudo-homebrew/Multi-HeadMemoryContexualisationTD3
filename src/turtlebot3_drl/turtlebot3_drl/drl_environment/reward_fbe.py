# Modified by: Seunghyeop
# Description: This code has been modified to train the Turtlebot3 Waffle_pi model.

from ..common.fbe_settings import REWARD_FUNCTION, COLLISION_OBSTACLE, COLLISION_WALL, TUMBLE, SUCCESS, TIMEOUT, RESULTS_NUM, THRESHOLD_COLLISION, STEP_TIME

goal_dist_initial = 0

reward_function_internal = None

def get_reward(exploration_amount, succeed, action_linear, action_angular):
    return reward_function_internal(exploration_amount, succeed, action_linear, action_angular)

def get_reward_A(exploration_amount, succeed, action_linear, action_angular):
        # [-4, 0]
        r_vangular = -1 * (action_angular**2)

        # [-2 * (2.2^2), 0]
        r_vlinear = -1 * (((0.26 - action_linear) * 10) ** 2)

        # reward = r_yaw + r_distance + r_obstacle + r_vlinear + r_vangular - 1
        reward = r_vlinear + r_vangular - 1 + exploration_amount * 1


        if succeed == SUCCESS:
            reward += 2500
        elif succeed == COLLISION_OBSTACLE or succeed == COLLISION_WALL:
            reward -= 2000
        return float(reward)


# Define your own reward function by defining a new function: 'get_reward_X'
# Replace X with your reward function name and configure it in settings.py

def reward_initalize(init_distance_to_goal):
    global goal_dist_initial
    goal_dist_initial = init_distance_to_goal

function_name = "get_reward_" + REWARD_FUNCTION
reward_function_internal = globals()[function_name]
if reward_function_internal == None:
    quit(f"Error: reward function {function_name} does not exist")
