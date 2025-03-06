# import numpy, torch, gym, stable-baselines3

import numpy as np
import torch
import gym
import stable_baselines3 as sb3

#Examples we will add more characteristics later
class Applicant:
    def __init__(self, quality, age, gender):
        self.quality = quality
        self.age = age
        self.gender = gender
        self.accepted = 0
        self.scholarship = 0.0
        self.totalAttributes = 5
        self.xValues = []
        
        self.xValues[0] = quality
        self.xValues[1] = int(self.age > 30)
        self.xValues[2] = int(self.age < 30)
        self.xValues[3] = int(gender == 'male')
        self.xValues[4] = int(gender == 'female')
        #Add other characteristics

    def Pdisappear(self):
        #Add more accurate later v_a(x_a)
        if self.accepted == 0:
            if self.quality < 0.5:
                return 0
            else:
                return self.quality / 8
        
    def PrejectOffer(self):
        #Add more accurate later p_a(x_a)
        if self.accepted == 1:
            return self.quality / 4




class AcceptanceDecisionEnv(gym.Env):
    def __init__(self, applicants=100, betas=[]):
        super(AcceptanceDecisionEnv, self).__init__()

        self.tApplicants = applicants
        self.tEpochs = 10
        self.currentEpoch = 0

        self.maxScholarship = 35000 #F
        self.maxCapacity = 100 #S
        self.marginalCost = 71 #C_s
        self.betas = betas #Beta values for diversity values correspond to xValues of same index
        self.xMeans = [] # Values to compare the characteristic values in objective (x_a bar)

        self.observation_space = spaces.Box(
            low=np.array([0.0] + [0] * (len(betas) - 1)),
            high=np.array([1.0] + [1] * (len(betas) - 1)),
            dtype=np.float32
        )

        self.action_space = spaces.Tuple(
            spaces.Discrete(2),
            spaces.Box(0.0, self.maxScholarship, (1,), dtype=np.float32)
        )

    def reset(self):
        return
    def step(self, action):
        return
    def _generate_state(self):
        return
    def render(self, mode='human'):
        return
    def close(self):
        return




        





    

def main():

    #Model variables and constants

    

    #Establish model and agent


    #Create data structures needed
    accepted = [Applicant]
    acceptedOffer = [Applicant]

    #Establish the random variables that go into the model


    #Link random variables into the model

    
    #Create objective
    objValue = 0
    for i in acceptedOffer:
        objValue += (maxScholarship - i.scholarship)
        characterSum = i.xValues[0] * betas[0]
        for j in range(1, len(i.xValues)):
            characterSum += ((xMeans - i.xValues[j])**2) * betas[j]
        objValue += (1/len(acceptedOffer))* (characterSum)

    totalMarginalCost = 0
    if len(acceptedOffer > maxCapacity):
        totalMarginalCost += marginalCost * (acceptedOffer - maxCapacity)
    
    objValue -= totalMarginalCost





    #Run model


    #Save trained model
    return
