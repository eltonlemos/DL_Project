#! /usr/bin/env python3

import rospy
from std_msgs.msg import Float32
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDrive
from math import sin
import time
import numpy as np
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Pose
import torch
import torch.nn as nn
import torch.optim as optim
import random
import pickle
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1081, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 3)  # Output layer with 3 Q-values

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)  # No activation function on output layer
        return x


class Car:
    def __init__(self) -> None:
        # self.tl.waitForTransform("/car_1", "/world_frame", rospy.Time(),rospy.rostime.Duration(1,0))
        self.vision_thresh=500
        self.tn= time.time()
        self.max_speed= 7
        self.max_acc=1.3
        self.time_start = time.time()
        self.has_begun=False
        self.Kd = 3.5
        self.pub = rospy.Publisher("car_1/command", AckermannDrive, queue_size=1)
        self.reset_pub = rospy.Publisher('/car_1/reset', Pose, queue_size=1)
        self.pose_msg = Pose()
        self.pose_msg.position.x = 0.0
        self.pose_msg.position.y = 0.0
        self.pose_msg.position.z = 0.0
        self.pose_msg.orientation.x = 0.0
        self.pose_msg.orientation.y = 0.0
        self.pose_msg.orientation.z = 0.0
        self.pose_msg.orientation.w = 1.0
        self.is_goal= False
        self.min_ld = 1
        self.max_ld = 7
        self.speed = 0
        self.K_dd = 0.5
        self.l_d=False
        self.steering_angle=False
        self.laps=0
        self.end=False
        self.gap=0
        self.at=0.05
        self.avg_speed=[]
        self.flag = True
        self.speed_const=0.1

        self.wheelbase=0.324
        self.observation_space = 1080
        self.action_space = [0, 1, 2]
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')    
        self.gamma=0.99
        self.alpha=0.4
        self.i=1
        self.epsilon=1
        self.Q = Net()
        self.Qphi = Net()
        self.Q = self.Q.to(self.device)
        self.Qphi = self.Qphi.to(self.device)
        self.D = []
        self.len_d = 10000
        self.C = 5
        self.replay_interval = 2
        self.batch_size = 256
        self.loss_fn = nn.MSELoss()
        self.lr=0.0001
        self.optimizer = optim.Adam(self.Q.parameters(), self.lr)
        self.terminated = False
        self.truncated = False
        self.reward = None


    def step(self,obs):
        """Takes a step in the environment by choosing an action randomly.

        Args:
            obs: The current observation.

        Returns:
            The action to take.
        """
        s =torch.FloatTensor(obs)
        s=s.to(self.device)

        if random.uniform(0,1)>=self.epsilon:
            a=torch.argmax(self.Q.forward(s)).tolist()
        else:
            a= np.random.choice(self.action_space)

        return a
    
    def learn(self):
        if len(self.D) < self.batch_size:
            return
        minibatch = random.sample(self.D, self.batch_size)
        states, actions, rewards, states_next, terminated, truncated = zip(*minibatch)

        states = torch.FloatTensor(states)
        states=states.to(self.device)
        actions = torch.LongTensor(actions)
        actions = actions.to(self.device)
        rewards = torch.FloatTensor(rewards)
        rewards=rewards.to(self.device)
        states_next = torch.FloatTensor(states_next)
        states_next=states_next.to(self.device)
        
        terminated = torch.FloatTensor(terminated)
        terminated=terminated.to(self.device)
        truncated = torch.FloatTensor(truncated)
        truncated=truncated.to(self.device)
        # print(terminated)
        # print(truncated)

        pred = self.Q(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        target_next = self.Qphi(states_next).detach().max(1)[0]
        target = rewards + (self.gamma * target_next * (1 - terminated) * (1 - truncated))

        loss = self.loss_fn(pred, target)
        # print(loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def gap_follow(self,data):
        self.angle_increment = data.angle_increment
        self.distances = np.array(data.ranges)
        self.get_gap()

        self.drive()
        if self.reward is None:
            self.reward = -1


    def drive(self):
        msg=AckermannDrive()
        # self.get_velocity()
        if self.end:
            while True:
                    # print(f"race over")
                    p=np.array(self.avg_speed).mean()
                    print(f"final goal is reached at {self.final_time} with avg_speed: {p}")

                    self.pub.publish(msg)
        self.get_steering()
        msg.steering_angle = self.steering_angle
        msg.speed = self.speed
        # msg.speed = 1
        self.pub.publish(msg)

    # def get_velocity(self):
    #     if self.end:
    #         self.speed=0
            
    #     else:    
    #         old_i=self.tn
    #         self.tn = time.time()
    #         desired_speed = min( 6.7 - abs(self.gap) * self.Kd - self.min_dist *2.03, self.max_speed)
    #         if desired_speed >= self.speed:
    #             # self.speed= self.speed + self.max_acc*(float(self.tn-old_i)) - min(self.speed*0.7-2,0)

    #             self.speed= self.speed + self.at 
    #             # print(f"{self.max_acc*(float(self.tn-old_i))}")
    #         else:
    #             self.speed = desired_speed
    #     # print(f"speed: {self.speed}")
    #     # self.avg_speed.append(self.speed)
    #     # self.speed=1


    def get_steering(self):

        self.l_d=min(max(self.speed * self.K_dd,self.min_ld),self.max_ld)
        alpha=self.gap
        # print(f"gap= {self.gap}")
        self.steering_angle = -np.arctan2(2*self.wheelbase*sin(alpha),self.l_d)
        


    def get_gap(self):

        min_index=int((len(self.distances)/2)-self.vision_thresh)
        max_index=int((len(self.distances)/2)+self.vision_thresh)
        sliced=self.distances[min_index:max_index]
        self.min_dist=np.min(sliced,initial= 200,where= (sliced>0))
        # closest=np.argmin(self.distances)
        # safe=-1 if closest<=len(self.distances) else 1
        sliced[sliced>self.min_dist+1.5]=0
        arr=np.zeros(len(sliced))
        arr[sliced==0]=1
        counter=0
        max_counter=0
        gap=0
        for i in range (len(arr)):
            if arr[i]==1:
                counter+=1
            else:
                if max_counter<counter:
                    gap=i-(counter/2)
                    max_counter=counter
                counter=0
        if max_counter<counter:
            gap=len(arr)-(counter/2)
            max_counter=counter
            counter=0
        self.gap= (gap-self.vision_thresh+1)*self.angle_increment 
        

    def setlap(self,data):


        y=data.pose.pose.position.y 
        x=data.pose.pose.position.x
        # if self.is_reset:
        #     self.speed=0
        #     self.reset_car()
        t = time.time()-self.time_start
        if x>2:
            self.has_begun=True
        if t > 120:
            self.truncated = True
            self.reward = -1000

        if self.has_begun:
            if y < 2 and y>-2:
                if x <0.02 and x>-0.02:
                    self.laps+=1
                    if self.laps %2 ==0:
                        self.terminated == True
                        self.reward = 120-int(t)
                        self.laps=0
                    # rate = rospy.Rate(1)
                    # rate.sleep()

    def calculate_speed(self, action):
        if action == 0:
            if self.speed>0:
                self.speed-=self.speed_const
        elif action == 1:
            if self.speed< self.max_speed:
                self.speed+=self.speed_const
        else:
            pass


        # print(f"time: {t:.2f}")
        
    def collision(self,data):
        self.truncated = True
        self.reward = -2000
        # self.speed = 0
        print(f"There was a crash")
                    

    def listener(self):

        rospy.Subscriber("/car_1/scan", LaserScan, self.gap_follow, queue_size= 1)
        rospy.Subscriber("/car_1/odom", Odometry, self.setlap, queue_size= 1)
        rospy.Subscriber("/obstacleCount", Float32, self.collision, queue_size= 1)
        # rospy.spin()

    def reset_car(self):

        self.has_begun=False
        self.reset_pub.publish(self.pose_msg)
        rospy.sleep(3)
        self.time_start = time.time()
        self.has_begun=False
        self.reward = None
        # 
    

def race_train():

    agent = Car()
    
    reward_per_episode=np.empty((1,0))
    epsilon_values=np.empty((1,0))
    episodes=100
    agent.i=(0.01/1)**(1/episodes)
    agent.listener()

    count=0
    for i in range(episodes):
        agent.speed=0
        agent.reset_car()
        # time.sleep(4)
        obs = agent.distances.copy()
        obs= np.append(obs,agent.speed)
        # print(f"{obs.shape}")
        agent.terminated, agent.truncated = False, False
        terminated, truncated = False, False
        cumulative_reward=0
        s = obs
        
        while not (terminated or truncated):
            count+=1
            
            if agent.reward is None:
                continue

            action = agent.step(obs)
            agent.calculate_speed(action)
            
            # time.sleep(0.1)
            terminated, truncated = agent.terminated, agent.truncated
            reward = agent.reward
            agent.reward = None
            obs = agent.distances.copy()
            obs= np.append(obs,agent.speed)
            print(f"{agent.speed} , {reward}")

            # obs, reward, terminated, truncated, info = env.step(action)
            
            if len(agent.D)<=(agent.len_d-1):
                agent.D.append([s,action,reward,obs,terminated,truncated])
            else:
                agent.D[random.randint(0,agent.len_d-1)]=[s, action, reward, obs, terminated, truncated]
            if count % agent.replay_interval == 0:
                count=0
                agent.learn()
            s = obs
            cumulative_reward+=reward
        print(f"Episode number: {i} reward: ({cumulative_reward})")
        reward_per_episode = np.append(reward_per_episode,cumulative_reward)
        epsilon_values = np.append(epsilon_values,agent.epsilon)
        agent.epsilon = agent.epsilon* agent.i
        if i % agent.C == 0:
            agent.Qphi.load_state_dict(agent.Q.state_dict())
    return reward_per_episode, epsilon_values, agent


if __name__ == "__main__":
    """
    1. Init a ros Node
    2. Declare a rospy subscriber object to subscribe to topic ros_basics/pub (Use the appropriate message type)
       Use the decalred function subscriber_callback as the callback function for the subscriber
    3. Loop using rospy.spin()
    """
    rospy.init_node('car_controller', anonymous=True)
    

    rpe, ev, agent = race_train()

    plt.figure()
    plt.plot(rpe)
    plt.xlabel('Episode', fontsize=20)
    plt.ylabel('Cumulative Reward', fontsize=20)
    plt.title('Cumulative Reward Per Episode (Training)', fontsize=24)
    plt.xticks(fontsize=18)
    plt.yticks([0,np.mean(rpe), 470, 10], fontsize=18)
    # plt.ylim(ymin=-20, ymax=20)
    # plt.xlim(xmin=0, xmax=1000)
    plt.grid()
    plt.show()

    plt.figure()
    plt.plot(ev, linewidth=4)
    plt.xlabel('Episode', fontsize=20)
    plt.ylabel('Epsilon Value', fontsize=20)
    plt.title('Epsilon Decay', fontsize=24)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.ylim(ymin=0, ymax=1)
    plt.xlim(xmin=0, xmax=1000)
    plt.grid()
    plt.show()
    with open('race_car.pickle', 'wb') as f:
        pickle.dump(agent.Q, f)

    df = pd.DataFrame({
    'Column1': rpe,
    'Column2': ev
    })

    df.to_csv('output.csv', index=False)