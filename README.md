# **Dynamic Speed Optimization for Autonomous Vehicles using Reinforcement Learning (RL)**

## **Overview**
This repository provides a step-by-step guide to setting up a **VMware-based ROS environment** and configuring the **F1-10th autonomous vehicle simulator** to train a reinforcement learning (RL) model for **autonomous vehicle speed optimization**. 
![ra](https://github.com/user-attachments/assets/4273e197-a467-4181-94fd-13c35e6e3c14)

## **Installation & Setup**

### **Step 1: VMware and ROS Installation**
1. **Install VMware**  
   Follow the detailed installation guide available on the **Drones Lab Robotics Algorithms resources page**:  
   📌 [VMware Setup Instructions](https://droneslab.github.io/RoboticsAlgorithms/resources/)

2. **Install ROS (Robot Operating System)**  
   After setting up VMware, proceed with ROS installation on your virtual machine by following:  
   📌 [ROS Setup Instructions](https://droneslab.github.io/RoboticsAlgorithms/resources/)

### **Step 2: Catkin Workspace Setup**
1. **Create a Catkin Workspace**  
   Set up a ROS catkin workspace to manage packages. Follow this tutorial:  
   📌 [Creating a Catkin Workspace](https://wiki.ros.org/catkin/Tutorials/CreatingPackage)

2. **Clone and Copy the Project Code**  
   Navigate to your `catkin_ws/src/` directory and copy the project files into it:
   ```sh
   cd ~/catkin_ws/src
   git clone <repository_link>
   ```

3. **Build the Workspace**  
   Navigate to the root of your workspace and compile:
   ```sh
   cd ~/catkin_ws
   catkin_make
   ```

### **Step 3: Running the Code**
1. **Source the ROS Setup File**  
   Before executing the ROS nodes, run:
   ```sh
   source devel/setup.bash
   ```

2. **Run the ROS Node**  
   Use the `rosrun` command to start the RL model:
   ```sh
   rosrun DL_Project run.py
   ```
   For more details on ROS nodes:  
   📌 [Understanding ROS Nodes](https://wiki.ros.org/ROS/Tutorials/UnderstandingNodes)

---

## **F1-10th Autonomous Vehicle Simulator Setup**
To install and configure the F1-10th autonomous vehicle simulator, follow the step-by-step instructions provided here:  
📌 [F1-10th Simulator Setup Instructions](https://liberating-dash-9ac.notion.site/F1Tenth-Simulator-Setup-5a013b6a723b490cbf6881a42a94d63f)

### **Simulator Features**
- **Unity-based realistic simulation**
- **ROS Integration** for real-time control and data processing
- **Lidar & Odometry sensor emulation** for navigation and obstacle detection
- **Multiple Race Tracks** for model testing under various conditions

---

## **Project Architecture**
This project integrates **Deep Q-Network (DQN) reinforcement learning** with a **ROS-powered simulation** to optimize vehicle speed dynamically.  

### **Key Components**
1. **Simulator** – Provides a controlled environment with realistic physics, lidar data, and environmental effects.
2. **Reinforcement Learning Model (DQN)** – Learns an optimal speed control policy using lidar sensor data.
3. **Neural Network (NN)** – Processes sensor observations and predicts optimal acceleration/deceleration.
4. **Reward System** – Encourages safe navigation while optimizing lap times.
5. **PID Steering Control** – Ensures the RL model focuses solely on speed optimization, with an independent PID controller managing steering.

---

## **Performance Evaluation**
This project went through multiple iterations to refine the RL model. **Key improvements** included:
- **Curriculum Training** – First trained the model for speed, then introduced safety constraints.
- **Improved Reward Structuring** – Balanced speed incentives with penalties for collisions.
- **Neural Network Refinements** – Experimented with CNNs and dense networks to enhance learning.
- **Early Stopping Mechanism** – Prevented the model from overfitting to unsafe driving behaviors.

### **Final Results**
- **Lap Completion Time**: RL model completed 2 laps in **36 minutes**, compared to a PID controller's **32 minutes**.
- **Adaptability**: The RL model dynamically adjusted speed based on real-time conditions.
- **Safety Improvements**: Crashes reduced significantly after fine-tuning reward mechanisms.

---

## **Limitations & Future Work**
🔸 **Long Training Time** – Training took approximately **23 hours** due to computational complexity.  
🔸 **Steering Optimization** – Future work should explore **Actor-Critic (A2C)** or **DDPG algorithms** for better control.  
🔸 **Diverse Scenarios** – Extend training to include **more complex environments** and **real-world driving conditions**.  

---

## **Contributors**
- **Elton Roque Lemos** (*SUNY Buffalo*)  
- **Yamini Ramesh** (*SUNY Buffalo*)  

Special thanks to **Professor Changyou Chen** for guidance and mentorship.

---

## **References**
📖 [Research Paper on RL-Based Speed Optimization](link_to_paper)
