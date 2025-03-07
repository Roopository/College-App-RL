# import numpy, torch, gym, stable-baselines3

import numpy as np
import torch
import gym
import stable_baselines3 as sb3

from gym import spaces
#Examples we will add more characteristics later
class Applicant:
    def __init__(self, quality, age, gender):
        self.quality = quality
        self.age = age
        self.gender = gender


        self.scholarship = 0.0
        self.xValues = [0.0, 0, 0, 0, 0]
        
        self.xValues[0] = quality
        self.xValues[1] = np.float32(self.age > 30)
        self.xValues[2] = np.float32(self.age < 30)
        self.xValues[3] = np.float32(gender == 1)
        self.xValues[4] = np.float32(gender == 0)
        #Add other characteristics

    def Pdisappear(self):
        #Add more accurate later v_a(x_a)
        if self.quality < 0.5:
            return 0
        else:
            if np.random.choice(a=[0, 1], p=[1 - (self.quality / 8), self.quality / 8]) == 1:
                return True
            else:
                return False
        
    def PrejectOffer(self):
        #Add more accutate later p_a(x_a)
        if np.random.choice(a=[0, 1], p=[1 - (self.quality / 8), self.quality**1.5 / 20 + (1 - self.scholarship/35000) * (1/4)]) == 1:
            return True
        else:
            return False

# Takes in the information on the distribution and gives out the list of candidates
# taken in period and the total number of candidates
def applyDist(lamba, periods):
    perList = []
    total = 0
    for per in range(0, periods):
        arrivalsThisPeriod = np.random.poisson(lamba)
        total += arrivalsThisPeriod
        perList.append(arrivalsThisPeriod)
    return perList, total

#For calculating all people's x_a values, takes in 
#INPUT: total applicants, NOT YET: list of data on each distribution 
#OUTPUT: List of all candidates
# in form of [[mean=x, std=y], ..] etc. (dictionary of dictionaries)
#Default distdic = {"quality": {"mean": 0.7, "std": 0.1}, "age": {"mean": 29, "std": 2.5}, "gender": {}}
# Not using that yet maybe for general case needs more overhead so easier to define manually
def individualDist(totalApplicants):
    applicants = [Applicant]
    for i in range(0, totalApplicants):
        tQuality = np.random.normal(0.7, 0.1)
        if tQuality > 1.0:
            tQuality = 1.0
        tAge = np.random.normal(29, 2.5)
        tGender = np.random.choice(a=[0, 1], p=[0.4, 0.6]) #0 is female, 1 is male

        tApplicant = Applicant(quality=tQuality, age=tAge, gender=tGender)
        applicants.append(tApplicant)


    return applicants


class AcceptanceDecisionEnv(gym.Env):
    def __init__(self, betas=[]):
        super(AcceptanceDecisionEnv, self).__init__()

        #Variables
        self.tApplicants = 0
        self.tEpochs = 10
        

        #Timekeeping variables
        self.currentEpoch = 0
        self.checkWL = False
        self.WLIndex = 0


        self.avgAppsPerEpoch = 10
        self.maxScholarship = 35000 #F
        self.maxCapacity = 100 #S
        self.marginalCost = 71 #C_s
        self.betas = betas #Beta values for diversity values correspond to xValues of same index
        self.xMeans = [] # Values to compare the characteristic values in objective (x_a bar)

        self.APPLICANTS = [[Applicant]]
        self.WAITLIST = [Applicant]
        self.ACCEPTED = [Applicant]


        self.state = np.array([0, 0, 0, 0, 0])

        #Setup for counting time periods
        self.pplPerEpoch = []

        self.objValue = 0.0

        self.xMeans = np.array([0, 0, 0, 0, 0])

        

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
        seed = np.random()
        super().reset(seed=seed)
        self.pplPerEpoch, self.tApplicants = applyDist(self.avgAppsPerEpoch, self.tEpochs)
        allApplicants = individualDist(self.tApplicants)

        for ep in self.pplPerEpoch:
            self.APPLICANTS.append(allApplicants[:ep])
            allApplicants = allApplicants[ep:]
         
        self.WAITLIST = []
        self.ACCEPTED = []
        self.currentEpoch = 0
        np.random.shuffle(self.APPLICANTS)
        first = self.APPLICANTS[0]
        self.state = np.array(first.xValues)
        return self.state, {} #Return the first element that the model is going to analyze with step()
    

    #Action is of form [0/1], [0.0-35000]
    def step(self, action):
        #BASED ON STATE WHICH IS AN APPLICANT ADD TO EITHER OF THE LISTS
        decision, scholarship = action
        reward = 0
        terminated = False
        truncated = False

        nextElement = None
        if not self.checkWL:
            if decision == 1:
                self.APPLICANTS[self.currentEpoch][0].scholarship = scholarship
                self.ACCEPTED.append(self.APPLICANTS[self.currentEpoch][0])
                reward = self.calcObjReward()
            else:
                self.WAITLIST.append(self.APPLICANTS[self.currentEpoch][0])

            self.APPLICANTS[self.currentEpoch].pop(0)

            if not self.APPLICANTS[self.currentEpoch]:
                self.checkWL = True
                self.WLIndex = 0
                nextElement = self.WAITLIST[self.WLIndex]
            else:
                nextElement = self.APPLICANTS[self.currentEpoch][0]

        else:
            if decision == 1:
                self.WAITLIST[self.WLIndex].scholarship = scholarship
                self.ACCEPTED.append(self.WAITLIST[self.WLIndex])
                self.WAITLIST.pop(self.WLIndex)
                reward = self.calcObjReward()

            if self.WLIndex < len(self.WAITLIST):
                self.WLIndex += 1
            else:
                self.checkWL = False
                self.currentEpoch += 1
                for i in self.ACCEPTED:
                    if i.PrejectOffer():
                        self.ACCEPTED.pop(i)
                for j in self.WAITLIST:
                    if j.Pdisappear():
                        self.WAITLIST.pop(j)
                if self.currentEpoch == self.tEpochs:
                    reward = self.calcObjReward()
                    self.reset()
                    terminated = True
                
        
        return nextElement, reward, terminated, truncated, {}

        #Calculate new xMeans
    
    def render(self):
        return
    def close(self):
        return

    def calcObjReward(self):
        lastObj = self.objValue
        for i in self.ACCEPTED:
            self.objValue += (self.maxScholarship - i.scholarship)
            characterSum = i.xValues[0] * self.betas[0]
            for j in range(1, len(i.xValues)):
                characterSum += ((self.xMeans[j] - i.xValues[j])**2) * self.betas[j]
            self.objValue += (1/len(self.ACCEPTED)) * (characterSum)

        totalMarginalCost = 0
        if len(self.ACCEPTED) > self.maxCapacity:
            totalMarginalCost += self.marginalCost * (len(self.ACCEPTED) - self.maxCapacity)
        self.objValue -= totalMarginalCost
        return self.objValue - lastObj


def main():

    #Model variables and constants

    #Establish model and agent

    #Create data structures needed

    #Establish the random variables that go into the model

    #Link random variables into the model
    
    #Create objective

    #Run model

    #Save trained model
    return
