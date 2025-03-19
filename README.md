## Overview

This README file provides instructions on how to set up VMware, install ROS with a catkin workspace, and configure the F1-10th autonomous vehicle simulator. Follow the steps carefully to ensure a smooth setup and execution of our reinforcement learning (RL) model for autonomous vehicle speed optimization.

## Step 1: VMware and ROS Setup

1. **Install VMware:**
   Follow the detailed instructions on the Drones Lab Robotics Algorithms resources page to install VMware:  
   [VMware Setup Instructions](https://droneslab.github.io/RoboticsAlgorithms/resources/)

2. **Install ROS:**
   Once VMware is set up, continue with the ROS installation on your virtual machine using the same link:  
   [ROS Setup Instructions](https://droneslab.github.io/RoboticsAlgorithms/resources/)

## Step 2: Create a Catkin Workspace

1. **Create a New Catkin Workspace:**
   Follow the tutorial on the ROS wiki to create a new catkin workspace:  
   [Creating a Catkin Workspace](https://wiki.ros.org/catkin/Tutorials/CreatingPackage)

2. **Copy Our Code:**
   After building the catkin workspace, navigate to the `src` folder within your catkin workspace directory. Copy our provided code into this `src` folder.

3. **Build the Workspace:**
   Open a terminal, navigate to the root of your catkin workspace, and run the following command to build the workspace:
   ```sh
   catkin_make

## Step 3: Run the Code

1. **Source the Setup File:**
  Before running the ROS nodes, source the setup file by executing the following command in your terminal:
   ```sh
   source devel/setup.bash

2. **Run the ROS Node:**
Use the `rosrun` command to run the specific file you need. For detailed instructions on understanding and running ROS nodes, refer to the following tutorial:
[Understanding ROS Nodes](https://wiki.ros.org/ROS/Tutorials/UnderstandingNodes)

   Example
   ```sh
   rosrun DL_Project run.py

## Step 4: Setup the F1-10th Autonomous Vehicle Simulator

1. **Follow Simulator Setup Instructions:**
Use the link below to access clear instructions on setting up, running, and using the Drones Lab F1-10th simulator:
[F1-10th Simulator Setup Instructions](https://liberating-dash-9ac.notion.site/F1Tenth-Simulator-Setup-5a013b6a723b490cbf6881a42a94d63f)

## Conclusion
By following the steps outlined above, you will set up the necessary environment to run our RL model for autonomous vehicle speed optimization. Ensure you follow each step carefully and refer to the provided links for detailed instructions.
