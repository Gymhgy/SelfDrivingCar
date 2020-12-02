import numpy as np
import torch
from ai import Transition, TARGET_UPDATE
import math

width = 700
height = 500

#goal
xGoal = width - 50
yGoal = height - 50

sand = np.zeros((height, width))
sand[0:100,0:100] = 0
sand[400:500,600:700] = 0

actions = [-20, 0, 20]
def rotate(vector, theta):
    #convert to radians
    theta *= np.pi/180
    rot_matrix = np.array([
        [ np.cos(theta), -np.sin(theta) ], 
        [ np.sin(theta),  np.cos(theta) ]
    ])
    return rot_matrix.dot(vector)
def angle(a, b):
    angle = -(180 / math.pi) * math.atan2(
        a[0] * b[1] - a[1] * b[0],
        a[0] * b[0] + a[1] * b[1])
    return angle
class Car:
    
    def __init__(self, brain):
        self.brain = brain
        #position will be represented as an [x,y] numpy array
        self.position = np.array([50,50])
        self.rotation = 0 #default orientation will be facing east
        #velocity will also be represented by a numpy array
        self.velocity = np.array([6,0])
        self.last_distance = 0
        self.iteration_num = 0
        self.rewards = []
    
    def reset(self):
        self.position = np.array([50,50])
        self.rotation = 0
        self.velocity = np.array([6,0])
        self.last_distance = 0
        self.rewards = []
    
    def iterate(self):
        #0, 1, 2
        #we'll have to index it onto actions
        state = self.get_state()
        action = self.brain.select_action(state)
        new_state, reward, finished = self.move(actions[action])
        if finished:
            new_state = None
        reward = torch.tensor([reward]).float()
        self.brain.memory.add(state, action, new_state, reward)
        self.brain.optimize()
        if self.iteration_num % TARGET_UPDATE == 0:
            self.brain.target_net.load_state_dict(self.brain.policy_net.state_dict())
        self.iteration_num += 1
        return finished
    
    def move(self, rotation):
        self.position = self.position + self.velocity
        self.rotation = (self.rotation + rotation)%360

        #Set velocity to 1 if on sand
        if sand[int(self.position[1]),int(self.position[0])] == 1:
            self.velocity = rotate(np.array([4, 0]), self.rotation)
        #Else set velocity to 6
        else:
            self.velocity = rotate(np.array([6, 0]), self.rotation)
        #Remember image is 32x16 
        #16 pixels away means the car is just touching the edge
        #Clamp its position so that it doesn't go off the edge
        self.position = np.array([
            sorted((16,self.position[0],width-16))[1],
            sorted((16,self.position[1],height-16))[1]
        ])
        finished = self.in_goal()
        reward = self.reward()
        self.rewards.append(reward)
        return self.get_state(), reward, finished
    
    def reward(self):
        distance = np.sqrt((self.position[1] - xGoal)**2 + (self.position[0] - yGoal)**2)
        last_distance = self.last_distance
        self.last_distance = distance
        if self.in_goal():
            return 1
        #penalize it for cuddling to the wall
        if not(16<self.position[0]<width-16) or not (16<self.position[1]<height-16):
            return -1
        if sand[int(self.position[1]),int(self.position[0])] == 1:
            return -1
        if distance < last_distance:
            return 0.3
        else:
            return -0.1
        pass
    
    def in_goal(self):
        return 610 < self.position[0] < 700 and 410 < self.position[1] < 500
    
    def get_state(self):
        orient = angle(self.velocity, (xGoal, yGoal))
        return torch.tensor([*self.sensor_info(), orient, -orient]).float().unsqueeze(0)
    
    def sensor_info(self):
        """
        Returns a 3 element tuple representing the density of sand in 3 
        16x16 areas in front, to the left, and to the right of the car.
        """
        sensor1, sensor2, sensor3 = 0,0,0
        #If it is out of bounds, that means it's near the edge so we return 1 for maximum density of sand
        try:
            sensor1_pos = self.position + rotate(np.array([8,0]), (self.rotation-90)%360)
            sensor1 = np.sum(sand[sensor1_pos[1] - 8:sensor1_pos[1] + 8,sensor1_pos[0] - 8:sensor1_pos[0] + 8])/256
        except:
            sensor1 = 1
        try:
            sensor2_pos = self.position + rotate(np.array([8,0]), self.rotation)
            sensor2 = np.sum(sand[sensor2_pos[1] - 8:sensor1_pos[1] + 8,sensor1_pos[0] - 8:sensor2_pos[0] + 8])/256
        except:
            sensor2 = 1
        try:
            sensor3_pos = self.position + rotate(np.array([8,0]), (self.rotation+90)%360)
            sensor3 = np.sum(sand[sensor3_pos[1] - 8:sensor3_pos[1] + 8,sensor1_pos[0] - 8:sensor3_pos[0] + 8])/256
        except:
            sensor3 = 1
        return (sensor1, sensor2, sensor3)