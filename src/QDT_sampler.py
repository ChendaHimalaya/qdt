import sys,os
sys.path.append(os.getcwd())
from tqdm import tqdm
import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.kernel_ridge import KernelRidge

from src.Data_Processing import CPC18_getDist
from src.utility_factors import valueFunction1, weightingFunction, logitFunc, CPT_logit
from matplotlib import pyplot as plt



class QdtSampler:
    def __init__(self,
                 exp_name:str,
                 utility_params,
                 training_data,
                 ):
        self.exp_name=exp_name
        try:
            self.load_model()
        except FileNotFoundError:
            self.pf_features = pickle.load(open("data/cpc18/PF_features.p", 'rb'))
            self.utility_params=utility_params
            self.training_data=self.split_training_data(training_data)


    def classify(self,Ha, pHa, La, LotShapeA, LotNumA, Hb, pHb, Lb, LotShapeB, LotNumB, Amb, Corr):
        distA=CPC18_getDist(Ha,pHa,La,LotShapeA,LotNumA)
        distB=CPC18_getDist(Hb,pHb,Lb,LotShapeB,LotNumB)
        return CPT_logit(*self.utility_params,distA,distB,Amb,Corr)

    def calculate_attraction_target(self, row,a=0.1):
        utility_score=self.classify(row["Ha"],
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
                                          row["Corr"])
        if utility_score==0: print("0 utility encountered")


        temp= (row["BRate"]-utility_score)/min(utility_score,1-utility_score)
        if temp>=1:
            temp=0.99999
        if temp<=-1:
            temp=-0.99999

        return (1/a)*np.arctanh(temp)
    def calculate_BRate(self,row,a=0.1):
        utility_score=self.classify(row["Ha"],
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
                                          row["Corr"])
        temp= min(utility_score,1-utility_score)*np.tanh(a*row["predicted_attraction"])
        return utility_score+temp

    def predict_BRate(self,data, y):
        data["predicted_attraction"]=y
        return data.apply(lambda row:self.calculate_BRate(row),axis=1)

    def split_training_data(self,data):
        '''Split the training data according to subject ID and returns a dictionary of splited data'''
        sep_data=dict(tuple(data.groupby('SubjID')))
        ret={}
        for key in tqdm(sep_data.keys()):
            df= sep_data[key]

            prediction=df.apply(lambda row: self.calculate_attraction_target(row),axis=1).to_numpy()


            attraction_features=[self.pf_features[gameid] for gameid in df["GameID"].to_numpy()]

            attraction_features=pd.concat(attraction_features,axis=0)
            attraction_features["block"]=df["block"].to_numpy()



            ret[key]={"train":df, "gameID":df["GameID"].to_numpy(), "target":prediction, "attraction_features":attraction_features}
        return ret

    def train_attraction(self, model_name='linear_regression'):
        ret={}
        scores={}
        model=LinearRegression()
        if model_name == 'linear_regression':
            model=LinearRegression()
        elif model_name == 'ridge_regression':
            model = Ridge()
        elif model_name == 'kernal_ridge':
            model= KernelRidge()
        for key in tqdm(self.training_data):

            X=self.training_data[key]['attraction_features'].to_numpy()
            y=self.training_data[key]['target']
            model=model.fit(X,y)

            ret[key]=[model.coef_, model.intercept_]
            scores[key]=(model.score(X,y))
        self.coef=ret
        self.scores=scores
        return (ret,scores)

    def predict_attraction(self,data:dict,models):
        s=[]
        for key in tqdm(data.keys()):
            X=data[key]['attraction_features'].to_numpy()
            y=models[key].predict(X)

            df=data[key]["train"]
            df["predicted_attraction"]=y
            df["predict_BRate"]=df.apply(lambda row: self.calculate_BRate(row),axis=1)
            s.append(df)
        return pd.concat(s,axis=0)

    def sample(self,data,model_name ='linear_regression'):
        models = []
        for key in self.training_data:
            model = LinearRegression()
            if model_name == 'linear_regression':
                model = LinearRegression()
            elif model_name == 'ridge_regression':
                model = Ridge()
            elif model_name == 'kernal_ridge':
                model = KernelRidge()
            model_coef = self.coef[key]
            model.coef_ = model_coef[0]
            model.intercept_ = model_coef[1]
            models.append(model)
        X=data.iloc[:,12:-3]

        X["block"]=data["block"]
        X=X.to_numpy()
        qdt_prediction=[]
        for model in tqdm(models):
            y=model.predict(X)

            brate=self.predict_BRate(data,y).to_numpy()

            qdt_prediction.append(brate)
        data["qdt_prediction"]=np.mean(qdt_prediction,axis=0)
        return data









    def create_result_summary(self, model_name ='linear_regression'):
        models={}
        for key in self.training_data:
            model = LinearRegression()
            if model_name == 'linear_regression':
                model = LinearRegression()
            elif model_name == 'ridge_regression':
                model = Ridge()
            elif model_name == 'kernal_ridge':
                model = KernelRidge()
            model_coef=self.coef[key]
            model.coef_=model_coef[0]
            model.intercept_=model_coef[1]
            models[key]=model
        df=self.predict_attraction(self.training_data,models=models)
        df.to_csv("results/{}_result.csv".format(self.exp_name))



    def create_samplers(self):
        pass





    def save_model(self, path=None):
        if not path:
            path = "models/" + self.exp_name + ".p"
        file = open(path, "wb")
        pickle.dump(self.__dict__, file)

    def load_model(self, path="this"):
        if path == "this":
            path = "models/" + self.exp_name + ".p"
        file = open(path, 'rb')
        self.__dict__ = pickle.load(file)








def main():
    project_name="2021May-02-ridge-withblock"
    train_data=pd.read_csv("data/cpc18/train_sum.csv")
    sampler=QdtSampler(project_name,[0.88614743, 0.81004461 ,0.95779005 ,1.01166834 ,0.96294392 ,0.2765091 ],train_data)
    # sampler.train_attraction(model_name='ridge_regression')
    # sampler.create_result_summary(model_name='ridge_regression')
    data=pd.read_csv("data/syn_data/5000_syn.csv")
    data=data.iloc[0:1000]
    data=sampler.sample(data,model_name='ridge_regression')
    data.to_csv("results/syn_1000.csv")

    # print(sampler.scores)
    # plt.figure(0)
    # plt.hist(sampler.scores,bins=20)
    # plt.xlabel("R Squared score of training data")
    # plt.savefig("images/{}-scores.png".format(project_name))
    # sampler.save_model()

if __name__=="__main__":
    main()








