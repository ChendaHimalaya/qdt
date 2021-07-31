import numpy as np
from CPC18PF.get_PF_Features import get_PF_Features
import time
import pandas as pd

data=pd.read_csv("data/cpc18/PF_features.csv")




def dummy_attraction(distA, distB,Amb, Corr,util_score):

    return 0


def ambiguity_aversion(c1, Amb):
    return -1*Amb*c1


def QDT_attraction(c1,c2,distA,distB,Amb,Corr,util_score):

    attractionA=ambiguity_aversion(c1,0)
    attractionB=ambiguity_aversion(c1,Amb)
    temp=np.min([util_score,1-util_score])
    temp2=np.tanh(c2*(attractionB-attractionA))
    return temp*temp2


def QDT_attraction_PF_features(gameID):
    features=data[data["GameID"]==gameID]
    return features["pHa"]*features["Ha"]


if __name__=="__main__":

    Data=pd.read_csv("data/syn_data/5000.csv")
    prob=10
    Ha = Data['Ha'][prob]
    pHa = Data['pHa'][prob]
    La = Data['La'][prob]
    LotShapeA = Data['LotShapeA'][prob]
    LotNumA = Data['LotNumA'][prob]
    Hb = Data['Hb'][prob]
    pHb = Data['pHb'][prob]
    Lb = Data['Lb'][prob]
    LotShapeB = Data['LotShapeB'][prob]
    LotNumB = Data['LotNumB'][prob]
    Amb = Data['Amb'][prob]
    Corr = Data['Corr'][prob]
    start=time.time()
    features=get_PF_Features(Ha,pHa,La,LotShapeA,LotNumA,Hb,pHb,Lb,LotShapeB,LotNumB,Amb,Corr)
    print("Calculation cost:{}".format(time.time()-start))

    for i in range(5):
        print(features.iloc[i])






