import sys,os
sys.path.append(os.getcwd())
from src.QDT import QdtClassifier
from src.utility_factors import valueFunction1, weightingFunction, logitFunc, CPT_logit
from src.attraction_factors import dummy_attraction, ambiguity_aversion,QDT_attraction
import numpy as np

class QdtPredicter():
    """One qdt classifier per block"""
    def __init__(self,
                 names,
                 params,
                 num_util_params,
                 ):
        self.clfs=[]

        for i in range(5):
            clf=QdtClassifier("clf1",
                              "data/cpc18/train_block{}_games.csv".format(i+1),
                              names,
                              params[i][:num_util_params],
                              CPT_logit,
                              params[i][num_util_params:],
                              QDT_attraction,
                              #dummy_attraction,
                              )
            self.clfs.append(clf)

    def classify(self,Ha, pHa, La, LotShapeA, LotNumA, Hb, pHb, Lb, LotShapeB, LotNumB, Amb, Corr):
        ret=[]
        for clf in self.clfs:
            ret.append(clf.classify(Ha, pHa, La, LotShapeA, LotNumA, Hb, pHb, Lb, LotShapeB, LotNumB, Amb, Corr))
        return np.array(ret)


def main():
    num_util_param=6 #Number of utility parameters
    param1=[0.82423076,0.70935712,0.94564408,1.06578517,0.89884976,0.30544674,
 0.03596667 ,3.28094825]
    param2=[0.87921886 ,0.74456607 ,0.97137993 ,0.98078196, 0.98707779 ,0.31050745,
 0.04692646 ,2.71401311]
    param3=[0.88313371 ,0.70030474 ,0.98540364 ,0.98057811, 0.9864834 , 0.28892273,
 0.05970801, 1.20828292]
    param4=[0.89381309 ,0.82025816, 0.95580821 ,0.99715025, 0.97985058 ,0.27428034,
 0.02994283 ,1.89892924]
    param5=[0.89283243, 0.74684717 ,0.98413806, 1.00156143 ,0.97659442, 0.28034815,
 0.09734114 ,0.88784065]



    myclf=QdtPredicter("alpha,_lambda,beta,delta,gamma,phi,ambAve,tanhCoe".split(","),
                       [param1,param2,param3,param4,param5],
                       num_util_param)
    print(myclf.classify(10,0.5,0,"-",1,10,0.9,0,"-",1,0,0))

if __name__=="__main__":
    main()
