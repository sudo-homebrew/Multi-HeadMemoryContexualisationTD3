# Multi Head Memory Contextualising TD3

## Introduction
Multi Head Memory Contextualising TD3 is a novel DRL algorithm for training LiDAR-base mobile robot for indoor self-navigation
Parer link : UNDERREVIEW

- **Author:**
  1. Seunghyeop Nam(Personal Web: http://117.16.137.18, Linked In: https://www.linkedin.com/in/seunghyeop-nam-6b3201320)
  2. Tuan Anh Nguyen(https://www.linkedin.com/in/tuan-anh-nguyen-9711551b/)
- **Development:** Seunghyeop Nam
- **Thanks to:** Tomasvr (https://github.com/tomasvr/turtlebot3_drlnav)

### Upload Date: 21.Oct.2024


#### Simulation Videos
- Real World Scenario : https://youtu.be/_2VKcgxamzk


# Turtlebot3 Environment Installation on MacOS(ARM64 M processor)

![Logo](img/gazebo.png)

# Turtlebot3 Environment Installation on MacOS(ARM64 M processor)





This tutorial is for install Turtlebot3 environment with ROS2(Humble) on MacOS Apple silicon (ARM64 architecture)



# Be sure your OSX version is up to date!

Later than MacOS Sonoma 14.4 highly recommended



copy and paste on your terminal down below



First, install anaconda3 on your Mac in link below.





and create environment you are gonna use

```zsh
conda create -n <your_virtual_env_name>
source activate <your_virtual_env_name>

# this adds the conda-forge channel to the new created environment configuration 
conda config --env --add channels conda-forge
# and the robostack channel
conda config --env --add channels robostack-staging
# remove the defaults channel just in case, this might return an error if it is not in the list which is ok
conda config --env --remove channels defaults
```



Next, install ROS2 humble

```zsh
install ros-humble-desktop
```

restart anaconda environment

```zsh
conda deactivate
source activate <your_virtual_env_name>
```



install ROS2 tools for local development

```zsh
conda install compilers cmake pkg-config make ninja colcon-common-extensions catkin_tools
```



You can test by running rviz2

```zsh
conda activate <your_virtual_env_name>
rviz2
```





For next step we need to install gazebo classic which is gazebo11

You can install gazebo with single line with curl commend

```zsh
curl -ssL http://get.gazebosim.org | sh
```



or using brew



Installing brew (incase you don’t have brew)

You can skeep this step if you have brew already.

```zsh
ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
```



install gazebo11 with brew

(This process might take several minutes)

```zsh
brew tap osrf/simulation
brew install gazebo11
```



run gazebo

```zsh
gazebo

# In case you have a world file
gazebo /path/to/your/world
```



install conda packages below to cooperate gazebo with ROS2 (humble)

```zsh
conda install "robostack-staging::ros-humble-gazebo-*"
conda install "robostack-staging::ros-humble-cartographer-*"
conda install "robostack-staging::ros-humble-turtlebot3*"
conda install "robostack-staging::ros-humble-turtlebot3-*"
conda install robostack-staging::ros-humble-cartographer-ros
conda install robostack-staging::ros-humble-navigation2
conda install robostack-staging::ros-humble-nav2-bringup
conda install robostack-staging::ros-humble-dynamixel-sdk
conda install robostack-staging::ros-humble-slam-toolbox
conda install robostack-staging::ros-humble-nav2-costmap-2d
conda install robostack-staging::ros-humble-nav2-msgs

conda install robostack-staging::ros-humble-sdformat-urdf
conda install robostack-staging::ros-humble-robot-state-publisher
conda install robostack-staging::ros-humble-sdformat-test-files

## Cyclone DDS is known as faster than fastRTPS.
## Get it if you need.
conda install robostack-staging::ros-humble-rmw-cyclonedds-cpp
echo 'export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp' >> ~/.zshrc
```



# FATAL ERROR IN DISTRIBUTED PACKAGES!

There is fatal error in sdformat_urdf package. Dynamically linked shared object libraries in Ubuntu uses .so format and .dylib in MacOS ARM64. But you might have libsdformat_urdf_plugin.so in your sdformat_urdf package.

You can compile libsdformat_urdf_plugin.dylib by your self like below

1. Get source code from. (Don’t forget to change brench to Humble distribution)

1. Change directory to sdformat_urdf-humble/sdformat_urdf and build project with colcon.

```zsh
cd sdformat_urdf-humble/sdformat_urdf
colcon build --symlink-install --cmake-args -DPython3_FIND_VIRTUALENV="ONLY" -DCMAKE_INSTALL_NAME_DIR=$(pwd)/install/lib  -DPYTHON_EXECUTABLE=/opt/homebrew/anaconda3/envs/<your_virtual_env_name>/bin/python -Wno-dev
```

1. The libsdformat_urdf_plugin.dylib is at path below

```zsh
sdformat_urdf-humble/sdformat_urdf/build/sdformat_urdf/libsdformat_urdf_plugin.dylib
```

1. find where is libsdformat_urdf_plugin.so which you want to replace with libsdformat_urdf_plugin.dylib

```zsh
find /opt/homebrew/anaconda3 -name "libsdformat_urdf_plugin.so"
```

1. replace found libsdformat_urdf_plugin.so files with libsdformat_urdf_plugin.dylib you just build.



Or you could just simply download libsdformat_urdf_plugin.dylib file below and replace  libsdformat_urdf_plugin.so (Go to step 4)





Set ROS2 environment for your mac

```zsh
echo 'export ROS_DOMAIN_ID=30 #TURTLEBOT3' >> ~/.zshrc
```



## Testing Environment

Open 2 different terminal



Terminal 1

```zsh
source activate <your_virtual_env_name>
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py
```



Terminal 2

```zsh
source activate <your_virtual_env_name>
ros2 run turtlebot3_teleop teleop_keyboard
```



If gazebo world launches with turtlebot3 in it and you can control by keyboard, then you are ready to go!



Beside

If you want to create new ROS2 packages make project directory first

```zsh
mkdir -p ~/<your_package_name>/src
cd ~/<your_package_name>/src
```



And then you create your own ROS2 package

```zsh
# For CMAKE(C++) package
ros2 pkg create --build-type ament_cmake <your_package_name>

# For Python package
ros2 pkg create --build-type ament_python <your_package_name>
```



Finally when you build your own package, unfortunately normal case in MacOS you are using virtual environment made with Anacona3. Most of case colcon doesn’t recognise your virtual environment Python execution but detects Anaconda3’s, brew’s or  sometimes MacOS system default Python.

There for you have to designate which CMake and Python to use when you are building your project like below

```zsh
colcon build --symlink-install --cmake-args -DPython3_FIND_VIRTUALENV="ONLY" -DCMAKE_INSTALL_NAME_DIR=$(pwd)/install/lib  -DPYTHON_EXECUTABLE=/opt/homebrew/anaconda3/envs/<your_virtual_env_name>/bin/python -Wno-dev
```

# Download and build the code

Now it's time to download the repository to the actual code.

First, Open a terminal in the desired location for the new workspace. Clone the repository using:
```
git clone https://github.com/sudo-homebrew/fh-ANGUS.git
```

`cd` into the directory and make sure you are on the main branch
```zsh
cd turtlebot3_drlnav
git checkout main
```

Then initialise rosdep by running
```zsh
rosdep update
```

Now we can use rosdep to install all ROS packages needed by our repository
```zsh
rosdep install -i --from-path src --rosdistro humble -y
```


Next, it's time to actually build the repository code!
```zsh
colcon build --symlink-install --cmake-args -DPython3_FIND_VIRTUALENV="ONLY" -DCMAKE_INSTALL_NAME_DIR=$(pwd)/install/lib  -DPYTHON_EXECUTABLE=/opt/homebrew/anaconda3/envs/<your_virtual_env_name>/bin/python -Wno-dev
```

After colcon has finished building source the repository
```zsh
source install/setup.zsh
```

The last thing we need to do before running the code is add a few lines to our `~/.zshrc` file so that they are automatically executed whenever we open a new terminal. Add the following lines at the end of your `~/.zshrc` file and **replace ~/path/to/turtlebot3_drlnav/repo with the path where you cloned the repository. (e.g. ~/turtlebot3_drlnav)**
```zsh
# ROS2 domain id for network communication, machines with the same ID will receive each others' messages
export ROS_DOMAIN_ID=1

# Fill in the path to where you cloned the turtlebot3_drlnav repo
WORKSPACE_DIR=~/path/to/turtlebot3_drlnav
export DRLNAV_BASE_PATH=$WORKSPACE_DIR

# Source the workspace
source $WORKSPACE_DIR/install/setup.zsh

# Allow gazebo to find our turtlebot3 models
export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:$WORKSPACE_DIR/src/turtlebot3_simulations/turtlebot3_gazebo/models

# Select which turtlebot3 model we will be using (default: burger, waffle, waffle_pi)
export TURTLEBOT3_MODEL=waffle_pi

# Allow Gazebo to find the plugin for moving the obstacles
export GAZEBO_PLUGIN_PATH=$GAZEBO_PLUGIN_PATH:$WORKSPACE_DIR/src/turtlebot3_simulations/turtlebot3_gazebo/models/turtlebot3_drl_world/obstacle_plugin/lib
```

For more detailed instructions on ros workspaces check [this guide](https://automaticaddison.com/how-to-create-a-workspace-ros-2-humble-fitzroy/).

**Note: Always make sure to first run ```source install/setup.zsh``` or open a fresh terminal after building with `colcon build`.**


# Training and Model Execution

![System Diagram](img/system_architecture.jpg)
![Neural Network Architecture](img/Neural_Network_Architecture.jpg)

## Steps to Execute and Train the DRL Agent

After successfully setting up the environment, you’re ready to begin training the agent. Follow the steps below to start training:

1. **Launch the Simulation**: Open four separate terminal windows. In the first terminal, execute the following command to start the simulation environment:
   ```zsh
   ros2 launch turtlebot3_gazebo turtlebot3_drl_stage<n>.launch.py
   ```
   This will initiate the simulation GUI where the robot and moving obstacles will appear (loading might take some time).

   **Important:** Run this simulation command before starting other processes to ensure parameters are correctly set.

2. **Initialise Goals**: In the second terminal, set up the goal system:
   ```zsh
   ros2 run turtlebot3_drl goal_manager
   ```

3. **Launch the Environment Node**: Use the third terminal to initiate the environment:
   ```zsh
   ros2 run turtlebot3_drl environment
   ```

4. **Run the Training Agent**: Finally, in the fourth terminal, launch the desired Deep Reinforcement Learning (DRL) algorithm. For example:
   - For Deep Deterministic Policy Gradient (DDPG):
     ```zsh
     ros2 run turtlebot3_drl train_agent ddpg
     ```
   - For Twin Delayed DDPG (TD3):
     ```zsh
     ros2 run turtlebot3_drl train_agent td3
     ```
   - For Deep Q-Network (DQN):
     ```zsh
     ros2 run turtlebot3_drl train_agent dqn
     ```

   The agent should start moving and interact with the environment as the training progresses. Feedback and logs will be printed on the terminal during this process.

**Note**: Initially, the training performance graphs (if enabled) will be empty but will populate after a specific number of episodes, as defined in `settings.py`.

## Saving and Resuming Training Progress

The model automatically saves its current state at regular intervals as defined in the configuration. To resume training or test a pre-trained model:

1. **Testing a Model**:
   ```zsh
   ros2 run turtlebot3_drl test_agent ddpg "model_name" 500
   ```
   Replace `"model_name"` with the appropriate file name and `500` with the episode number you want to test.

2. **Continuing Training**:
   ```zsh
   ros2 run turtlebot3_drl train_agent ddpg "model_name" 500
   ```
   This command continues training from the specified episode and model checkpoint.

## Example Models and Environment Switching

Example models for DDPG and TD3 are provided. To test them:
1. Open four terminals as before.
2. Launch the simulation:
   ```zsh
   ros2 launch turtlebot3_drl example_stage.launch.py
   ```
3. Set up goals, environment, and run the test agent:
   ```zsh
   ros2 run turtlebot3_drl goal_manager
   ros2 run turtlebot3_drl environment_node
   ros2 run turtlebot3_drl test_agent ddpg "example_ddpg_model" 1000
   ```

To switch between environments, modify the stage number in the launch command:
```zsh
ros2 launch turtlebot3_drl stage_change.launch.py --stage 5
```

## Optional Adjustments and Configurations

### Customising Rewards and Settings

1. **Rewards**: Modify the `reward.py` file to implement custom reward functions. Enable them in `settings.py` by specifying the reward strategy.
2. **Parameter Tuning**: Adjust hyperparameters for training and performance in `settings.py`.

### Motion in Reverse

Allow backward movement by enabling `ENABLE_BACKWARD` in the settings file.

### Frame Stacking for Obstacle Prediction

Frame stacking considers multiple laser scan frames to predict obstacle movements. Configure this in `settings.py` by enabling `ENABLE_STACKING` and setting parameters like `STACK_DEPTH`.

