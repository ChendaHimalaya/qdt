"""This file is used to generate synthetic data in CPT18 format"""
import random
import numpy as np
from Data_Processing import CPC18_getDist
from tqdm import tqdm
import pandas as pd


def calculate_expectValue(H,pH,L,lot_shape, lot_num):
    dist=CPC18_getDist(H,pH,L,lot_shape,lot_num)
    expected=[dist[i,0]*dist[i,1] for i in range(dist.shape[0])]
    return np.sum(expected)


class DataGenerator():
    def __init__(self,seed=1337):
        random.seed(seed)
        np.random.seed(seed)
        self.prob_set=[0.01,0.05,0.1,0.2,0.25,0.4,0.5,0.6,0.75,0.8,0.9,0.95,0.99,1]
        self.skew_set=[-7,-6,-5,-4,-3,-2,2,3,4,5,6,7,8]
        self.symm_set=[3,5,7,9]

    def draw_EVA(self):
        return np.random.randint(-10,31)

    def draw_option_A(self,evA):
        if np.random.uniform(0,1)<0.4: # if A is a constant options, no uncertainty

            La=evA
            Ha=evA
            pHa=1
            LotNumA=1
            LotShapeA="-"
        else: # A contains some level of uncertainty
            #Find low and high payoffs
            pHa=random.choice(self.prob_set)
            if pHa==1:
                La=evA
                Ha=evA
            else:
                temp=np.random.triangular(-50,evA,120)
                if round(temp)>evA:
                    Ha=round(temp)
                    La=round((evA-Ha*pHa)/(1-pHa))
                if round(temp)<evA:
                    La=round(temp)
                    Ha=round((evA-La*(1-pHa))/pHa)
                else:
                    La=evA
                    Ha=evA
            # Set lotteries for option A
            random_temp=np.random.uniform(0,1)
            if random_temp<0.6:
                LotNumA=1
                LotShapeA="-"
            elif random_temp<0.8:
                temp1=random.choice(self.skew_set)
                if temp1>0: #R skewed distribution of A's lottery
                    LotNumA=temp1
                    LotShapeA="R-skew"
                elif temp1<0: #L skewed distribution of A's lottery
                    LotNumA=-1*temp1
                    LotShapeA="L-skew"
            else: #Symmetrical
                LotShapeA="Symm"
                LotNumA=random.choice(self.symm_set)
        return {"La":La,
                "Ha":Ha,
                "pHa":pHa,
                "LotNumA":LotNumA,
                "LotShapeA":LotShapeA}

    def draw_dev(self):
        sum=0
        for i in range(5):
            sum+=np.random.uniform(-20,20)
        return sum/5

    def draw_option_B(self,evB):

        #Find low and high payoffs
        pHb=random.choice(self.prob_set)
        if pHb==1:
            Lb=evB
            Hb=evB
        else:
            temp=np.random.triangular(-50,evB,120)
            if round(temp)>evB:
                Hb=round(temp)
                Lb=round((evB-Hb*pHb)/(1-pHb))
            if round(temp)<evB:
                Lb=round(temp)
                Hb=round((evB-Lb*(1-pHb))/pHb)
            else:
                Lb=evB
                Hb=evB
        # Set lotteries for option B
        random_temp=np.random.uniform(0,1)
        if random_temp<0.5:
            LotNumB=1
            LotShapeB="-"
        elif random_temp<0.75:
            temp1=random.choice(self.skew_set)
            if temp1>0: #R skewed distribution of B's lottery
                LotNumB=temp1
                LotShapeB="R-skew"
            elif temp1<0: #L skewed distribution of B's lottery
                LotNumB=-1*temp1
                LotShapeB="L-skew"
        else: #Symmetrical
            LotShapeB="Symm"
            LotNumB=random.choice(self.symm_set)
        return {"Lb":Lb,
                "Hb":Hb,
                "pHb":pHb,
                "LotNumB":LotNumB,
                "LotShapeB":LotShapeB}

    def draw_corr(self):
        temp=np.random.uniform(0,1)
        if temp<0.8:
            return 0
        elif temp<0.9:
            return 1
        else: return -1

    def draw_Amb(self):
        if np.random.uniform(0,1)<0.8:
            return 0
        return 1

    def generate_problem(self):
        """Generate one CPT18 format problem,
        returns as a dictionary"""
        evA_temp=self.draw_EVA()
        optionA=self.draw_option_A(evA_temp)
        evA=calculate_expectValue(optionA["Ha"],optionA["pHa"],optionA["La"],optionA["LotShapeA"],optionA["LotNumA"])
        dev=self.draw_dev()
        evB_temp=evA+dev

        if evB_temp<-50: #give up this trial and redo the generation
            return self.generate_problem()
        optionB=self.draw_option_B(evB_temp)
        Corr=self.draw_corr()
        Amb=self.draw_Amb()

        #Check for exceptions


        #there was a positive probability for an outcome larger than 256 or an outcome smaller than -50;
        distA=CPC18_getDist(optionA["Ha"],optionA["pHa"],optionA["La"],optionA["LotShapeA"],optionA["LotNumA"])

        distB=CPC18_getDist(optionB["Hb"],optionB["pHb"],optionB["Lb"],optionB["LotShapeB"],optionB["LotNumB"])

        for item in distA:
            if (item[0]>256 or item[0]<-50) and (item[1]>0):
                return self.generate_problem()
        for item in distB:
            if (item[0]>256 or item[0]<-50) and (item[1]>0):
                return self.generate_problem()

        #options were indistinguishable from participants’ perspectives (i.e., had the same distributions and Amb = 0);
        if Amb==0 and distA.shape==distB.shape:
            if (distA==distB).all():
                return self.generate_problem()

        #Amb = 1, but Option B had only one possible outcome;
        if Amb==1:
            if distB.shape[0]==1:
                return self.generate_problem()

        #at least one option had no variance, but the options were correlated.
        if Corr!=0:
            if optionA["LotShapeA"]=="-" or optionB["LotShapeB"]=="-":
                return self.generate_problem()

        ret=optionA
        ret.update(optionB)
        ret["Corr"]=Corr
        ret["Amb"]=Amb
        return ret

    def batch_gen(self,N,export_path):
        data={"La":[],
              "Ha":[],
              "pHa":[],
              "LotShapeA":[],
              "LotNumA":[],
              "Lb":[],
              "Hb":[],
              "pHb":[],
              "LotShapeB":[],
              "LotNumB":[],
              "Corr":[],
              "Amb":[]}
        for i in tqdm(range(N)):
            problem=self.generate_problem()
            for key in data.keys():
                data[key].append(problem[key])
        df=pd.DataFrame(data=data)
        df.to_csv(export_path)







def main():
    generator=DataGenerator()
    generator.batch_gen(5000,"data/syn_data/5000.csv")

if __name__=="__main__":
    main()





















