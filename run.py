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
        self.fc1 = nn.Linear(1080, 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 80)  

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x) 
        return x


class Car:
    def __init__(self) -> None:
        # self.tl.waitForTransform("/car_1", "/world_frame", rospy.Time(),rospy.rostime.Duration(1,0))
        self.vision_thresh=400
        # self.tn= time.time()
        self.max_speed= 7
        self.max_acc=0.3
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
        self.K_dd = 0.4
        self.l_d=False
        self.steering_angle=False
        self.laps=0
        self.end=False
        self.gap=0
        self.at=0.04
        self.avg_speed=[]
        self.flag = True
        self.speed_const=0.1
        self.min_index=int(540-self.vision_thresh)
        self.max_index=int(540+self.vision_thresh)
        self.t = 100
        self.wheelbase=0.324

        self.observation_space = 1080
        self.action_space = [i for i in range(80)]
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')    
        self.gamma=0.8
        self.alpha=0.3
        self.i=1
        self.epsilon=1
        self.Q = Net()
        self.Qphi = Net()
        self.Q = self.Q.to(self.device)
        self.Qphi = self.Qphi.to(self.device)
        self.D = []
        self.len_d = 10000
        self.C = 5
        self.replay_interval = 10
        self.batch_size = 8192
        self.loss_fn = nn.MSELoss()
        self.lr=0.0001
        self.optimizer = optim.Adam(self.Q.parameters(), self.lr)
        self.terminated = False
        self.truncated = False
        self.reward = None
        self.obs = None


    def step(self,obs):
        """Takes a step in the environment by choosing an action randomly.

        Args:
            obs: The current observation.

        Returns:
            The action to take.
        """
        # s = torch.FloatTensor(obs)
        s = obs.to(self.device)

        if random.uniform(0,1)>=self.epsilon:
            a=torch.argmax(self.Q.forward(s)).tolist()
        else:
            a= np.random.choice(self.action_space, replace = True)

        return a
    
    def learn(self):
        if len(self.D) < self.batch_size:
            return
        minibatch = random.sample(self.D, self.batch_size)
        states, actions, rewards, states_next_0, terminated, truncated = zip(*minibatch)

        states = torch.stack(states)
        states=states.to(self.device)
        actions = torch.LongTensor(actions)
        actions = actions.to(self.device)
        rewards = torch.FloatTensor(rewards)
        rewards=rewards.to(self.device)
        states_next_0 = torch.stack(states_next_0)
        states_next_0=states_next_0.to(self.device)
        # states_next_1 = torch.stack(states_next_1)
        # states_next_1=states_next_1.to(self.device)
        # states_next_2 = torch.stack(states_next_2)
        # states_next_2=states_next_2.to(self.device)
        # states_next_3 = torch.stack(states_next_3)
        # states_next_3=states_next_3.to(self.device)
        # states_next_4 = torch.stack(states_next_4)
        # states_next_4=states_next_4.to(self.device)
        
        terminated = torch.FloatTensor(terminated)
        terminated=terminated.to(self.device)
        truncated = torch.FloatTensor(truncated)
        truncated=truncated.to(self.device)
        # print(terminated)
        # print(truncated)

        pred = self.Q(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        target_next_0 = self.Qphi(states_next_0).detach().max(1)[0]
        # target_next_1 = self.Qphi(states_next_1).detach().max(1)[0]
        # target_next_2 = self.Qphi(states_next_2).detach().max(1)[0]
        # target_next_3 = self.Qphi(states_next_3).detach().max(1)[0]
        # target_next_4 = self.Qphi(states_next_4).detach().max(1)[0]
        # target = rewards + (self.gamma * (target_next_0 + self.gamma * (target_next_1 + self.gamma * (target_next_2 + self.gamma * (target_next_3 + self.gamma * target_next_4 )))) * (1 - terminated) * (1 - truncated))
        target = rewards + self.gamma * target_next_0
        loss = self.loss_fn(pred, target)
        # print(loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def gap_follow(self,data):
        self.angle_increment = data.angle_increment
        self.distances = np.array(data.ranges)
        self.obs = self.distances.copy()
        self.get_gap()

        self.drive()
        if self.reward is None:
            if self.speed>0:
                self.reward = self.speed
            else:
                self.reward = 0


    def drive(self):
        msg=AckermannDrive()
        # self.get_velocity()
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
        # print(f"speed: {self.speed}")
        # self.avg_speed.append(self.speed)
        # self.speed=1


    def get_steering(self):

        self.l_d=min(max(self.speed * self.K_dd,self.min_ld),self.max_ld)
        alpha=self.gap
        # print(f"gap= {self.gap}")
        self.steering_angle = -np.arctan2(2*self.wheelbase*sin(alpha),self.l_d)
        


    def get_gap(self):


        sliced=self.distances[self.min_index:self.max_index]
        self.min_dist=np.min(sliced,initial= 200,where= (sliced>0))
        # closest=np.argmin(self.distances)
        # safe=-1 if closest<=len(self.distances) else 1
        sliced[sliced>self.min_dist+1.2]=0
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

        if self.flag:
            y=data.pose.pose.position.y 
            x=data.pose.pose.position.x
            # if self.is_reset:
            #     self.speed=0
            #     self.reset_car()
            t = time.time()-self.time_start
            if x>2:
                self.has_begun=True
            if t > 60:
                self.truncated = True
                self.reward = -500

            if self.has_begun:
                if y < 2 and y>-2:
                    if x <0.2 and x>0:
                        self.laps+=1
                        self.reward = 240*self.laps - 4*int(t)
                        print(t)
                        self.has_begun = False
                        if self.laps == 2:
                            self.terminated = True
                            self.laps=0
                            self.t = t
                            print(f"completed lap {self.laps}")
                        # rate = rospy.Rate(1)
                        # rate.sleep()

    def calculate_speed(self, action):
        # if action == 0:
        #     if self.speed>0.5:
        #         self.speed-=self.speed_const
        # elif action == 1:
        #     if self.speed< self.max_speed:
        #         self.speed+=self.speed_const
        # else:
        #     pass
        desired_speed= action*self.speed_const
        # desired_speed = 7
        if desired_speed >= self.speed:
            self.speed= self.speed + self.max_acc
        else:
            self.speed = desired_speed
        # print(f"{desired_speed}, {self.speed}")
            # self.speed= self.speed + self.at 
        
        # print(f"time: {t:.2f}")
        
    def collision(self,data):
        self.collision_counter+=1
        if self.collision_counter %3 ==0:
            self.truncated = True
            self.reward = -50* abs(self.speed) - 50
        else:
            self.reward = -30* abs(self.speed) - 50
        # self.speed = 0
        print(f"There was a crash")
                    

    def listener(self):

        rospy.Subscriber("/car_1/scan", LaserScan, self.gap_follow, queue_size= 1)
        rospy.Subscriber("/car_1/odom", Odometry, self.setlap, queue_size= 1)
        rospy.Subscriber("/obstacleCount", Float32, self.collision, queue_size= 1)
        # rospy.spin()

    def reset_car(self):
        self.flag = False
        self.has_begun=False
        self.collision_counter = 0
        self.reset_pub.publish(self.pose_msg)
        rospy.sleep(6)
        self.time_start = time.time()
        self.has_begun=False
        self.laps = 0
        self.reward = None
        self.flag= True
        # 
    

def race_train():

    agent = Car()
    
    
    reward_per_episode=np.empty((1,0))
    epsilon_values=np.empty((1,0))
    episodes = 1000
    agent.i=(0.01/1)**(1/episodes)
    agent.listener()
    t = []

    count=0
    for i in range(episodes):
        agent.speed=0
        old_col=0
        agent.reset_car()
        # time.sleep(4)
        obs = agent.distances.copy()
        # print(obs)
        # obs = obs[agent.min_index:agent.max_index]
        # print(obs)
        # obs= np.append(obs,agent.speed)
        # print(f"{obs.shape}")
        agent.terminated, agent.truncated = False, False
        terminated, truncated = False, False
        cumulative_reward=0
        s = torch.FloatTensor(obs)
        
        while not (terminated or truncated):
            count+=1

            action = agent.step(s)
            agent.calculate_speed(action)
            if old_col != agent.collision_counter:
                old_col +=1
            else:
                agent.reward = None
            while agent.reward is None:
                pass
            # time.sleep(0.1)
            terminated, truncated = agent.terminated, agent.truncated
            reward = agent.reward
            obs = torch.FloatTensor(agent.obs)
            # agent.reward = None
            # obs_count=0
            # obs = []
            # while obs_count<5:
            #     if agent.obs is not None:
            #         obs.append(agent.obs)
            #         agent.obs = None
            #         obs_count+=1
            reward = agent.reward
            # obs = obs[agent.min_index:agent.max_index]
            # obs= np.append(obs,agent.speed)
            # obs = torch.FloatTensor(obs)
            # print(f"{action*agent.speed_const} , {reward}")

            # obs, reward, terminated, truncated, info = env.step(action)
            
            if len(agent.D)<=(agent.len_d-1):
                agent.D.append([s,action,reward,obs,terminated,truncated])
            else:
                agent.D.pop(0)
                agent.D.append([s, action, reward, obs, terminated, truncated])
            # if count % agent.replay_interval == 0:
            #     count=0
            #     # print(f"leaarning: {time.time()}")
            #     agent.learn()
                # print(f"learnt {time.time()}")
            s = obs
            cumulative_reward+=reward
        agent.learn()
        print(f"Episode number: {i} reward: ({cumulative_reward})")
        reward_per_episode = np.append(reward_per_episode,cumulative_reward)
        epsilon_values = np.append(epsilon_values,agent.epsilon)
        agent.epsilon = agent.epsilon* agent.i
        if i % agent.C == 0:
            agent.Qphi.load_state_dict(agent.Q.state_dict())
        if (agent.t<27) and (agent.collision_counter == 0):
            with open('race_car.pickle_'+str(i), 'wb') as f:
                pickle.dump(agent.Q, f)
        t.append(agent.t)
        agent.t = 100
    return reward_per_episode, epsilon_values, t, agent

def race_test(agent):

    # agent = Car()
    
    reward_per_episode=np.empty((1,0))
    epsilon_values=np.empty((1,0))
    episodes=50
    agent.epsilon = 0
    agent.listener()

    count=0
    for i in range(episodes):
        agent.speed=0
        agent.reset_car()
        # time.sleep(4)
        obs = agent.distances.copy()
        # obs= np.append(obs,agent.speed)
        # print(f"{obs.shape}")
        agent.terminated, agent.truncated = False, False
        terminated, truncated = False, False
        cumulative_reward=0
        
        while not (terminated or truncated):
            count+=1
            
            if agent.reward is None:
                continue

            action = agent.step(obs)
            agent.calculate_speed(action)
            if old_col != agent.collision_counter:
                old_col +=1
            else:
                agent.reward = None
            
            time.sleep(0.1)
            terminated, truncated = agent.terminated, agent.truncated
            while agent.reward is not None:
                pass
            reward = agent.reward

            obs = agent.distances.copy()
            # obs= np.append(obs,agent.speed)
            cumulative_reward+=reward
        print(f"Episode number: {i} reward: ({cumulative_reward})")
        reward_per_episode = np.append(reward_per_episode,cumulative_reward)
        epsilon_values = np.append(epsilon_values,agent.epsilon)
        t.append(agent.t)
        agent.t = 100

    return reward_per_episode, epsilon_values, t, agent

if __name__ == "__main__":
    """
    1. Init a ros Node
    2. Declare a rospy subscriber object to subscribe to topic ros_basics/pub (Use the appropriate message type)
       Use the decalred function subscriber_callback as the callback function for the subscriber
    3. Loop using rospy.spin()
    """
    rospy.init_node('car_controller', anonymous=True)
    

    rpe, ev, t, agent = race_train()

    plt.figure()
    plt.plot(t)
    plt.xlabel('Episode', fontsize=20)
    plt.ylabel('Cumulative Reward', fontsize=20)
    plt.title('Cumulative Reward Per Episode (Training)', fontsize=24)
    plt.xticks(fontsize=18)
    plt.yticks([0,np.mean(t), np.max(t)], fontsize=18)
    # plt.ylim(ymin=-20, ymax=20)
    # plt.xlim(xmin=0, xmax=1000)
    plt.grid()
    plt.show()

    plt.figure()
    plt.plot(rpe)
    plt.xlabel('Episode', fontsize=20)
    plt.ylabel('Cumulative Reward', fontsize=20)
    plt.title('Cumulative Reward Per Episode (Training)', fontsize=24)
    plt.xticks(fontsize=18)
    plt.yticks([0,np.mean(rpe), np.max(rpe)], fontsize=18)
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
    'rpe': rpe,
    'ev': ev
    })

    df.to_csv('output.csv', index=False)

    rpe, ev, agent = race_test(agent)

    plt.figure()
    plt.plot(rpe)
    plt.xlabel('Episode', fontsize=20)
    plt.ylabel('Cumulative Reward', fontsize=20)
    plt.title('Cumulative Reward Per Episode (Training)', fontsize=24)
    plt.xticks(fontsize=18)
    plt.yticks([0,np.mean(rpe), np.max(rpe)], fontsize=18)
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
    # with open('race_car.pickle', 'wb') as f:
    #     pickle.dump(agent.Q, f)

    df = pd.DataFrame({
    'Column1': rpe,
    'Column2': ev
    })
    df.to_csv('output2.csv', index=False)

