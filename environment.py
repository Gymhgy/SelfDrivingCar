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
    #Pass a brain as input
    def __init__(self, brain):
        #Set this as our brain
        self.brain = brain
        #position will be represented as an [x,y] numpy array
        self.position = np.array([50,50])
        self.rotation = 0 #default orientation will be facing east
        #velocity will also be represented by a numpy array
        self.velocity = np.array([6,0])
        #last_distance refers to the last distance between the car and the goal
        self.last_distance = 0
        #iteration_num is for keeping track how long it has been since the last target update
        self.iteration_num = 0
        
    #reset position, velocity, position, last_distance
    def reset(self):
        self.position = np.array([50,50])
        self.rotation = 0
        self.velocity = np.array([6,0])
        self.last_distance = 0
    
    #represents one state change
    def iterate(self):
        """
        Represents one training loop iteration.
        The main training loop is implemented within the ui.py file
        """
        #Get our current state
        state = self.get_state()
        #Choose an action with the brain
        action = self.brain.select_action(state)
        #Move our car using that action
        #actions is a list [-20, 0, 20]
        #Remember select_action only returns a number 0, 1, or 2
        #We use that number to index onto our list
        #self.move returns the new state we are in, the reward we got,
        #and whether or not we are in the final state (in the goal area)
        new_state, reward, finished = self.move(actions[action])
        #If we are finished (in the goal area)
        #Set the new state to None
        if finished:
            net_state = None
        #Turn our reward into a tensor (array) that only contains one element: the reward
        reward = torch.tensor([reward]).float()
        #Add this transition to our experience buffer
        self.brain.memory.add(state, action, new_state, reward)
        #Update our brain and the neural networks
        self.brain.optimize()
        #Update the target network to match the policy network
        #If we have gone TARGET_UPDATE steps without updating it
        if self.iteration_num % TARGET_UPDATE == 0:
            self.brain.target_net.load_state_dict(self.brain.policy_net.state_dict())
            
        #Add one to iteration_num
        self.iteration_num += 1
        #Return whether or not we are finished
        return finished
    
    #This function will move the car given the rotation
    def move(self, rotation):
        """Given a rotation, rotate the car by that rotation. Also move it by it's velocity."""
        #Update position according to current velocity
        self.position = self.position + self.velocity
        #Update the rotation according to the rotation
        #Clamp it within 0-360 degrees
        self.rotation = (self.rotation + rotation)%360

        #Set velocity to 4 if on sand
        #So the car goes slower on sand
        if sand[int(self.position[1]),int(self.position[0])] == 1:
            self.velocity = rotate(np.array([4, 0]), self.rotation)
        #Else set velocity to 6
        else:
            self.velocity = rotate(np.array([6, 0]), self.rotation)
        #Remember car is 32x16 
        #16 pixels away means the car is just touching the edge
        #Clamp its position so that it doesn't go off the edge
        self.position = np.array([
            sorted((16,self.position[0],width-16))[1],
            sorted((16,self.position[1],height-16))[1]
        ])
        #Return the reward, the new state, and whether we are finished or not
        finished = self.in_goal()
        reward = self.reward()
        return self.get_state(), reward, finished
    
    #This function gives us the reward for our car in his current state
    def reward(self):
        #Get the distance between our goal and us
        distance = np.sqrt((self.position[1] - xGoal)**2 + (self.position[0] - yGoal)**2)
        #Save the last_distance in a local variable
        last_distance = self.last_distance
        #Update the instance last_distance variable to the current distance
        self.last_distance = distance
        #If we have reached the target
        #Return maximum reward
        if self.in_goal():
            return 1
        #penalize it for cuddling to the wall
        #Give it -1
        if not(16<self.position[0]<width-16) or not (16<self.position[1]<height-16):
            return -1
        #If it's on sand, give it -1
        if sand[int(self.position[1]),int(self.position[0])] == 1:
            return -1
        #If we're closer to the goal than before, give it +0.3
        if distance < last_distance:
            return 0.3
        #Else give it a living penalty
        else:
            return -0.1
        pass
    
    def in_goal(self):
        """Returns whether the car is in the goal area or not"""
        return 610 < self.position[0] < 700 and 410 < self.position[1] < 500
    
    
    def get_state(self):
        """Get the current state of the car: [sensor1, sensor2, sensor3, orientation, -orientation]"""
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