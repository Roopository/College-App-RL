# import numpy, torch, gym, stable-baselines3

import numpy as np
import torch
import gymnasium
import stable_baselines3 as sb3

from gymnasium import spaces

#Examples we will add more characteristics later
class Applicant:
    def __init__(self, quality, age, gender):
        self.quality = quality
        self.age = age
        self.gender = gender


        self.scholarship = 0.0
        self.xValues = [0.0, 0, 0, 0, 0]
        
        self.xValues[0] = quality
        self.xValues[1] = np.float64(self.age > 30)
        self.xValues[2] = np.float64(self.age < 30)
        self.xValues[3] = np.float64(gender == 1)
        self.xValues[4] = np.float64(gender == 0)
        #Add other characteristics

    def Pdisappear(self):
        #Add more accurate later v_a(x_a)
        ppp = 0.0
        if self.quality < 0.55:
            ppp = 0.05
        elif self.quality >= 0.55 and self.quality < 0.75:
            ppp = 0.5
        else:
            ppp = 0.8
        pp = 1 - (1 - ppp) ** (1 / 10)
        disappear = np.random.choice(a=[True, False], p=[pp, 1 - pp])
        return disappear
        
    def PrejectOffer(self):
        #Add more accutate later p_a(x_a)
        ppp = 0.0
        if self.quality >= 0.8:
            ppp = (self.quality - 0.8)*(0.75) + (1 - (self.scholarship/35000))*(0.4) + 0.25
        elif self.quality < 0.8 and self.quality > 0.6:
            ppp = (self.quality - 0.6)*(0.5) + (1 - (self.scholarship/35000))*(0.4) + 0.1
        else:
            ppp = .95

        pp = 1 - (1 - ppp) ** (1 / 10)
        leave = np.random.choice(a=[True, False], p=[pp, 1-pp])
        return leave
    
    def printAll(self):
        print("Applicant: ")
        print("Age: ", self.age, "Gender: ", ["Female", "Male"][self.gender], "Quality: ", self.quality)

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
    applicants = []
    for i in range(0, totalApplicants):
        tQuality = np.random.normal(0.7, 0.1)
        if tQuality > 1.0:
            tQuality = 1.0
        tAge = np.random.normal(29, 2.5)
        tGender = np.random.choice(a=[0, 1], p=[0.4, 0.6]) #0 is female, 1 is male

        tApplicant = Applicant(quality=tQuality, age=tAge, gender=tGender)
        applicants.append(tApplicant)

    return applicants


class AcceptanceDecisionEnv(gymnasium.Env):
    def __init__(self, betas=[]):
        super(AcceptanceDecisionEnv, self).__init__()

        self.output = False
        self.test = True
        #Variables
        self.tApplicants = 0
        
        self.steps = 0
        

        #Timekeeping variables
        self.currentEpoch = 0
        self.checkWL = False
        self.WLIndex = 0
        self.objValue = 0.0


        #Constants
        self.tEpochs = 10
        self.avgAppsPerEpoch = 100
        self.maxScholarship = 35000 #F
        self.maxCapacity = 100 #S
        self.marginalCost = 300000 #C_s
        self.betas = betas #Beta values for diversity values correspond to xValues of same index
        self.xMeans = [] # Values to compare the characteristic values in objective (x_a bar)

        #Lists that need to be cleared each time
        self.APPLICANTS = []
        self.WAITLIST = []
        self.ACCEPTED = []
        self.pplPerEpoch = []


        #Change by themselves
        self.state = np.array([0, 0, 0, 0, 0])
        self.xMeans = np.array([0, 0, 0, 0, 0])

        
        #Observations
        # self.observation_space = spaces.Box(
        #     low=np.array([0.0] + [0] * (len(betas) - 1)),
        #     high=np.array([1.0] + [1] * (len(betas) - 1)),
        #     dtype=np.float64
        # )

        #Works by being in order: ACCEPTED size, capacity, current epoch, total epochs, characteristics
        self.observation_space = spaces.Box(
            low=np.array([0.0, self.maxCapacity, 0, self.tEpochs] + [0.0] * len(betas)),
            high=np.array([self.maxCapacity, self.maxCapacity, self.tEpochs, self.tEpochs] + [1.0] * len(betas)),
            dtype=np.float64
        )

        #Actions
        self.action_space = spaces.MultiDiscrete([2, self.maxScholarship])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0

        self.WAITLIST = []
        self.ACCEPTED = []
        self.APPLICANTS = []
        self.pplPerEpoch = []
        
        self.currentEpoch = 0
        self.reward = 0
        self.WLIndex = 0
        self.checkWL = False
        self.objValue = 0.0

        self.pplPerEpoch, self.tApplicants = applyDist(self.avgAppsPerEpoch, self.tEpochs)
        allApplicants = individualDist(self.tApplicants)

        np.random.shuffle(allApplicants)

        for ep in self.pplPerEpoch:
            self.APPLICANTS.append(allApplicants[:ep])
            allApplicants = allApplicants[ep:]
         
        
        if self.APPLICANTS[self.currentEpoch]:
            first = self.APPLICANTS[self.currentEpoch][0]
        else:
            first, _ = self.findNext()
        tt = [len(self.ACCEPTED), self.maxCapacity, self.currentEpoch, self.tEpochs] + first.xValues
        return np.array(tt, dtype=np.float64), {} #Return the first element that the model is going to analyze with step()
    


    def findNext(self):
        nextOb = None
        terminated = False
        while not nextOb:
            if self.checkWL == False:
                if self.currentEpoch < self.tEpochs:
                    if self.APPLICANTS[self.currentEpoch]:
                        nextOb = self.APPLICANTS[self.currentEpoch][0]
                    else: 
                        self.WLIndex = 0
                        self.checkWL = True
                        if self.WAITLIST:
                            nextOb = self.WAITLIST[self.WLIndex]
                        else:
                            _, _ = self.endofEpoch()
                else:
                    terminated = True
                    break
            

        return nextOb, terminated
    

    def endofEpoch(self):
        terminated = False
        reward = 0

        i, j = 0, 0
        while i < len(self.ACCEPTED):
            if self.ACCEPTED[i].PrejectOffer():
                self.ACCEPTED.pop(i)
            else:
                i += 1
        while j < len(self.WAITLIST):
            if self.WAITLIST[j].Pdisappear():
                self.WAITLIST.pop(j)
            else:
                j += 1
        

        self.checkWL = False
        self.currentEpoch += 1
        
        if self.currentEpoch == self.tEpochs:
            print("EPOCH", self.currentEpoch)
            reward = self.calcObjReward()
            terminated = True

        return reward, terminated

    #Used to print the in each step() to see selected applicant
    def printState(self, applicant: Applicant):
        applicant.printAll()
        print(["APPLIED", "WAITLIST"][self.checkWL])
        print("Step: ", self.steps)
        print("Epoch: ", self.currentEpoch)
        return

    def printAction(self, action):
        decision, scholarship = action
        print(["REJECTED", "ACCEPTED " + str(scholarship)][decision])
    
    def printResult(self):
        for acc in self.ACCEPTED:
            acc.printAll()
            print("Scholarship: ", acc.scholarship)
        
        print("TOTAL ACCEPTED: ", len(self.ACCEPTED))
        print("LEFT IN WAITLIST: ", len(self.WAITLIST))

        bezero = 0
        for a in self.APPLICANTS:
            bezero += len(a)
        print("Applicants left: ", bezero)
        _ = self.calcObjReward()
        print("TOTAL ENDING OBJECTIVE:", self.objValue)
        return

    #Action is of form [0/1], [0.0-35000]
    def step(self, action):
        #BASED ON STATE WHICH IS AN APPLICANT ADD TO EITHER OF THE LISTS
        self.steps += 1
        decision, scholarship = action
        reward = 0
        terminated = False
        truncated = False

        if self.output:
            self.printAction(action)

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
                if self.WAITLIST:
                    nextElement = self.WAITLIST[self.WLIndex]
                else:
                    reward, terminated = self.endofEpoch()
                    if not terminated:
                        nextElement, terminated = self.findNext()
            else:
                nextElement = self.APPLICANTS[self.currentEpoch][0]
        else:
            if decision == 1:
                self.WAITLIST[self.WLIndex].scholarship = scholarship
                self.ACCEPTED.append(self.WAITLIST[self.WLIndex])
                self.WAITLIST.pop(self.WLIndex)
                reward = self.calcObjReward()
            else:
                self.WLIndex += 1

            if self.WLIndex < len(self.WAITLIST):
                nextElement = self.WAITLIST[self.WLIndex]
            else:
                reward, terminated = self.endofEpoch()
                if not terminated:
                    nextElement, terminated = self.findNext()


        # if len(self.ACCEPTED) > 110:
        #     terminated = True
        

        if terminated:
            if self.test:
                self.printResult()
            self.reset()
            nextElement = None
        if nextElement is None:
            return np.zeros(9), reward, terminated, truncated, {}
        
        if self.output:
            self.printState(nextElement)

        tt = [len(self.ACCEPTED), self.maxCapacity, self.currentEpoch, self.tEpochs] + nextElement.xValues
        
        return np.array(tt, dtype=np.float64), reward, terminated, truncated, {}

    #Don't need to change because they have no real functionality
    def render(self):
        return
    def close(self):
        return

    # def calcObjReward(self):
    #     lastObj = self.objValue
    #     self.objValue = 0
    #     for i in self.ACCEPTED:
    #         self.objValue += (self.maxScholarship - i.scholarship)
    #         characterSum = i.xValues[0] * self.betas[0]
    #         for j in range(1, len(i.xValues)):
    #             characterSum += ((self.xMeans[j] - i.xValues[j])**2) * self.betas[j]
    #         self.objValue += (1/len(self.ACCEPTED)) * (characterSum)
        
    #     totalMarginalCost = 0
    #     if len(self.ACCEPTED) > self.maxCapacity:
    #         totalMarginalCost += self.marginalCost * (len(self.ACCEPTED) - self.maxCapacity)
    #     self.objValue -= totalMarginalCost
    #     return self.objValue - lastObj
    
    def calcObjReward(self):
        lastObj = self.objValue
        self.objValue = 0
        for i in self.ACCEPTED:
            self.objValue += (self.maxScholarship - i.scholarship)
            characterSum = ((8 * (i.xValues[0] - 0.7))**7) * self.betas[0]
            for j in range(1, len(i.xValues)):
                characterSum += ((10 * (self.xMeans[j] - i.xValues[j]))**2) * self.betas[j]
            self.objValue += (1/len(self.ACCEPTED)) * (characterSum)
        
        totalMarginalCost = 0
        if len(self.ACCEPTED) > self.maxCapacity:
            totalMarginalCost += self.marginalCost * (len(self.ACCEPTED) - self.maxCapacity)
        self.objValue -= totalMarginalCost
        return self.objValue - lastObj



def main():

    env = AcceptanceDecisionEnv(betas=[1000, 50, 50, 50, 50])

    
    #model = sb3.PPO("MlpPolicy", env=env, verbose=1)

    model = sb3.PPO.load("ppo_model", env=env)

    model.learn(total_timesteps=30000)

    model.save("ppo_model")
    
    return

if __name__ == "__main__":
    main()
