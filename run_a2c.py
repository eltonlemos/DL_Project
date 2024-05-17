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

class ActorNet(nn.Module):
    def __init__(self, nn_dim, output_dim):
        super().__init__()
        self.num_layers=len(nn_dim)-1
        ptr=0
        while ptr<len(nn_dim)-1:
            setattr(self, 'layer_'+str(ptr),nn.Linear(nn_dim[ptr],nn_dim[ptr+1]))
            
            ptr+=1
        self.relu=nn.ReLU()
        self.output = nn.Linear(nn_dim[ptr], output_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        for layer_num in range(self.num_layers):
            temp=getattr(self,'layer_'+str(layer_num))
            x = temp(x)
            x = self.relu(x)
        x = self.output(x)
        x = self.softmax(x)
        return x
class CriticNet(nn.Module):
    def __init__(self, nn_dim, output_dim):
        super().__init__()
        self.num_layers=len(nn_dim)-1
        ptr=0
        while ptr<len(nn_dim)-1:
            setattr(self, 'layer_'+str(ptr),nn.Linear(nn_dim[ptr],nn_dim[ptr+1]))
            
            ptr+=1
        self.relu=nn.ReLU()
        self.output = nn.Linear(nn_dim[ptr], output_dim)
        

    def forward(self, x):
        for layer_num in range(self.num_layers):
            temp=getattr(self,'layer_'+str(layer_num))
            x = temp(x)
            x = self.relu(x)
        x = self.output(x)
        return x


class Car:
    def __init__(self) -> None:
        # self.tl.waitForTransform("/car_1", "/world_frame", rospy.Time(),rospy.rostime.Duration(1,0))
        self.vision_thresh=400
        # self.tn= time.time()
        self.max_speed= 7
        self.max_acc=0.1
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
        self.action_space = [i for i in range(70)]
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')    
        self.gamma=0.8
        self.alpha=0.3
        self.i=1
        self.epsilon=1
        self.actor = ActorNet([1080,2048,512,256],70)
        self.critic = CriticNet([1080,2048,512,256],1)
        self.actor = self.actor.to(self.device)
        self.critic = self.critic.to(self.device)

        self.loss_fn = nn.MSELoss()
        self.lr=0.001
        self.actor_optimizer = optim.Adam(self.actor.parameters(), self.lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), self.lr)
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
        s =torch.FloatTensor(obs)
        s=s.to(self.device)
        action_probs = self.actor.forward(s)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        value = self.critic.forward(s)
        return action.tolist(), value, log_prob
    
    def get_returns(self, value, rewards, terminateds, truncateds):
      reward = value
      returns = []
      for step in reversed(range(len(rewards))):
          reward = rewards[step] + self.gamma * reward * (1-terminateds[step]) * (1-truncateds[step])
          returns.insert(0, reward)
      return returns

    def gap_follow(self,data):
        self.angle_increment = data.angle_increment
        self.distances = np.array(data.ranges)
        self.obs = self.distances.copy()
        self.get_gap()

        self.drive()
        if self.reward is None:
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
            # print(f"{(time.time()- self.flip_check_t )}")
            if (time.time()- self.flip_check_t ) > 2:
                d = np.linalg.norm(np.array(self.old_p) - np.array([x,y]))
                # print(f"{d}")
                if d<0.1:
                    self.truncated = True
                    self.reward = -1000
                    print(f"Car flipped")
                self.old_p = [x,y]
                self.flip_check_t = time.time()

            if self.has_begun:
                if y < 1.5 and y>-1.5:
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
        else: self.old_p = [data.pose.pose.position.x, data.pose.pose.position.y] 

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
            self.reward = -100* abs(self.speed) - 50
        else:
            self.reward = -60* abs(self.speed) - 50
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
        self.flip_check_t = time.time()
        # 
    

def race_train():

    agent = Car()
    
    
    reward_per_episode=[]
    episodes = 2000
    agent.listener()
    t = []

    count=0
    for i in range(episodes):
        agent.speed=0
        old_col=0
        agent.reset_car()
        # time.sleep(4)
        obs = agent.distances.copy()
        log_probs = []
        values = []
        rewards = []
        truncateds = []
        terminateds = []
        agent.terminated, agent.truncated = False, False
        terminated, truncated = False, False
        cumulative_reward=0
        
        while not (terminated or truncated):
            count+=1

            action, value, log_prob = agent.step(obs)
            values.append(value)
            log_probs.append(log_prob.unsqueeze(0))
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
            obs = agent.obs

            reward = agent.reward
            rewards.append(torch.tensor([reward], dtype=torch.float, device=agent.device))
            terminateds.append(torch.tensor([terminated], dtype=torch.float, device=agent.device))
            truncateds.append(torch.tensor([truncated], dtype=torch.float, device=agent.device))
            
            cumulative_reward+=reward
        s =torch.tensor(obs, dtype=torch.float, device=agent.device)
        value = agent.critic.forward(s)
        returns = agent.get_returns(value, rewards, terminateds, truncateds)

        log_probs = torch.cat(log_probs)
        
        returns = torch.cat(returns).detach()
        values = torch.cat(values)
        advantage = returns - values
        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()
        agent.critic_optimizer.zero_grad()
        critic_loss.backward()
        agent.critic_optimizer.step()

        agent.actor_optimizer.zero_grad()
        actor_loss.backward()
        agent.actor_optimizer.step()
        reward_per_episode.append(cumulative_reward)
        print(f"Episode number: {i} reward: ({cumulative_reward})")
        if (agent.t<27) and (agent.collision_counter == 0):
            with open('race_car.pickle_'+str(i), 'wb') as f:
                pickle.dump(agent.Q, f)
        t.append(agent.t)
        agent.t = 100
    return reward_per_episode, t, agent

def race_test(agent):

    # agent = Car()
    
    reward_per_episode=[]
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

            action, value, log_prob = agent.step(obs)
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
        reward_per_episode.append(cumulative_reward)
        t.append(agent.t)
        agent.t = 100

    return reward_per_episode, t, agent

if __name__ == "__main__":
    """
    1. Init a ros Node
    2. Declare a rospy subscriber object to subscribe to topic ros_basics/pub (Use the appropriate message type)
       Use the decalred function subscriber_callback as the callback function for the subscriber
    3. Loop using rospy.spin()
    """
    rospy.init_node('car_controller', anonymous=True)
    

    rpe, t, agent = race_train()
    with open('race_car_actor.pickle', 'wb') as f:
        pickle.dump(agent.actor, f)
    with open('race_car_critic.pickle', 'wb') as f:
        pickle.dump(agent.critic, f)
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



    df = pd.DataFrame({
    'rpe': rpe,
    't': t
    })

    df.to_csv('output.csv', index=False)

    rpe, agent = race_test(agent)

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

    # with open('race_car.pickle', 'wb') as f:
    #     pickle.dump(agent.Q, f)

    df = pd.DataFrame({
    'Column1': rpe,
    'Column2': t
    })
    df.to_csv('output2.csv', index=False)

