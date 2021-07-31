import sys,os
sys.path.append(os.getcwd())
import pandas as pd
import numpy as np
import mystic
from typing import Callable
from src.utility_factors import valueFunction1, weightingFunction, logitFunc, CPT_logit
from src.attraction_factors import dummy_attraction, ambiguity_aversion,QDT_attraction, QDT_attraction_PF_features
from src.Data_Processing import CPC18_getDist
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
import pickle


class QdtClassifier:
    def __init__(self,
                 exp_name:str,
                 load_model=None,
                 train_data:str =None, #Path to the training file
                 utility_params_name = None,
                 utility_params = None,
                 utility_function:Callable =None,
                 attraction_params = None,
                 attraction_function:Callable =None,


                 verbose=1,


                 ):
        self.exp_name=exp_name
        if load_model:
            self.load_model(path=load_model)
        else:
            self.raw_data=pd.read_csv(train_data)
            self.train_data=[]
            self.result=[]
            self.util_param_names=utility_params_name
            for index,row in self.raw_data.iterrows():
                data=[row["Ha"],
                      row["pHa"],
                      row["La"],
                      row["LotShapeA"],
                      row["LotNumA"],
                      row["Hb"],
                      row["pHb"],
                      row["Lb"],
                      row["LotShapeB"],
                      row["LotNumB"],
                      row["Amb"],
                      row["Corr"],]
                self.result.append(row["BRate"])
                self.train_data.append(data)
            self.result=np.array(self.result)

            self.utility_params=utility_params
            self.num_util_params=len(utility_params)
            self.utility_function=utility_function
            self.attraction_params=attraction_params
            self.attraction_function=attraction_function
            self.train_count=0

        self.verbose=verbose

    def CallBack(self,params):
        self.train_count+=1
        print("Iteration {}".format(self.train_count))
        for i in range(len(self.util_param_names)):
            print("{}:{}".format(self.util_param_names[i],params[i]))


    def train(self):
        f=open("logs/{}.txt".format(self.exp_name),"a")
        f.write("Training Start\n")
        f.write("Current Parameters are:{}\n".format(str(self.util_param_names)))
        f.write("Initial utility params:{}\n".format(str(self.utility_params)))
        f.write("Initial attraction params:{}\n".format(str(self.attraction_params)))

        result=minimize(self.train_wrapper,
                        np.array(self.utility_params+self.attraction_params),
                        method="Nelder-Mead",
                        tol=1e-6,
                        options={'maxiter': 10 ** 6},
                        callback=self.CallBack,
                        )

        print("Training Done")
        f.write("Final params:{}\n".format(str(result.x)))
        parameters={self.util_param_names[i]:[result.x[i]] for i in range(len(self.util_param_names))}
        #print(parameters)
        df=pd.DataFrame(data=parameters)
        df.to_csv("logs/{}_params.txt".format(self.exp_name))
        self.utility_params=list(result.x[:self.num_util_params])
        self.attraction_params=list(result.x[self.num_util_params:])

    def save_model(self,path=None):
        if not path:
            path="models/"+self.exp_name+".p"
        file=open(path,"wb")
        pickle.dump(self.__dict__,file)

    def load_model(self,path="this"):
        if path=="this":
            path="models/"+self.exp_name+".p"
        file=open(path,'rb')
        self.__dict__=pickle.load(file)


    def classify(self,Ha, pHa, La, LotShapeA, LotNumA, Hb, pHb, Lb, LotShapeB, LotNumB, Amb, Corr):
        distA=CPC18_getDist(Ha,pHa,La,LotShapeA,LotNumA)
        distB=CPC18_getDist(Hb,pHb,Lb,LotShapeB,LotNumB)
        utility_score=self.utility_function(*self.utility_params,distA,distB,Amb,Corr)
        attraction_score=self.attraction_function(*self.attraction_params,distA,distB,Amb,Corr,utility_score)
        return utility_score+attraction_score

    def generate_prediction(self):
        predictions = []
        for problem in self.train_data:
            predictions.append(self.classify(*problem))
        predictions = np.array(predictions)
        df=self.raw_data
        df["predictions"]=pd.Series(predictions)
        df.to_csv("results/{}.csv".format(self.exp_name))

    def classify_with_external_params(self,params,Ha, pHa, La, LotShapeA, LotNumA, Hb, pHb, Lb, LotShapeB, LotNumB, Amb, Corr):
        distA=CPC18_getDist(Ha,pHa,La,LotShapeA,LotNumA)
        distB=CPC18_getDist(Hb,pHb,Lb,LotShapeB,LotNumB)
        utility_score=self.utility_function(*params[:self.num_util_params],distA,distB,Amb,Corr)
        attraction_score=self.attraction_function(*params[self.num_util_params:],distA,distB,Amb,Corr,utility_score)
        return utility_score+attraction_score

    def mse_loss_sum(self,params):
        """The MSE loss over training data of summary
           The parameters are assumed to be last few parameters, e.g. if there are 10 parameters, and len(params)==5, it assumes
           the first five parameters will use the existed value and thus automatically fill with those values
        """

        predictions=[]
        for problem in self.train_data:
            predictions.append(self.classify_with_external_params(params,*problem))
        predictions=np.array(predictions)
        return mean_squared_error(self.result,predictions)

    def train_wrapper(self,params,*args):
        return self.mse_loss_sum(params)




def main():
    # for i in range(1,6):
    #     # clf=QdtClassifier(
    #     #                   "data/cpc18/train_block{}_games.csv".format(i),
    #     #
    #     #                   "alpha,_lambda,beta,delta,gamma,phi".split(","),
    #     #                   [1,1,1,1,1,1],
    #     #                   CPT_logit,
    #     #                   [],
    #     #                   dummy_attraction,
    #     #
    #     #                   )
    #     clf=QdtClassifier(
    #         "cpt-2021Apr-total", None, "data/cpc18/aggregate_games.csv",
    #         "alpha,_lambda,beta,delta,gamma,phi".split(","),
    #                           [1,1,1,1,1,1],
    #                           CPT_logit,
    #                           [],
    #                           dummy_attraction,
    #
    #     )
    #     clf.train()
    #     clf.generate_prediction()
    #     clf.save_model()
    #     # clf=QdtClassifier("cpt-2021Apr-block{}".format(i), load_model="this")
    #     # print(clf.__dict__)

    clf = QdtClassifier(
        "cpt-2021Apr-total", None, "data/cpc18/aggregate_games.csv",
        "alpha,_lambda,beta,delta,gamma,phi".split(","),
        [1, 1, 1, 1, 1, 1],
        CPT_logit,
        [],
        dummy_attraction,

    )
    clf.train()
    clf.generate_prediction()
    clf.save_model()


if __name__=="__main__":
    main()








