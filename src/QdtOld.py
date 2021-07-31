import numpy as np
from numpy import array
from matplotlib import pyplot as plot
from scipy.optimize import minimize
import pandas as pd


def loadData(filepath):
    """filepath: the relative path to the file
    Returns a panda frame and its columns"""
    f=pd.read_excel(io=filepath)
    return f,f.columns




def getPlayer(f,columns,x):
    """Return the dataframe of a specific player with subject ID x"""
    return f[f['Subject'].isin([x])]

def valueFunction1(v,alpha,_lambda=1):
    if v>0:
        return v**alpha
    else:
        return -_lambda*(-v)**alpha

def weightingFunction(p,delta,sigma):
    """The weighting function in prospect theory
    Equals 1 when the probability is 1"""
    #print(np.log(p))
    expo=(-np.log(p))**sigma


    return np.exp(-delta*expo)

def needFunction(vGamble,pGamble,timePressure,neta,framing,x,theta,needDiff):
    """Calculate the attraction factor that corresponds with need factors"""


def calculateUtility(vGamble, pGamble,vSure,alpha,delta,sigma,phi,_lambda=1):
    """Calculates the utility f(x) term for the gamble option in the Quantum Decison Theory framwork
    This one is equiped with a logit choice function"""
    v=valueFunction1(vGamble,alpha,_lambda)
    #print('v:{}'.format(v))
    w=weightingFunction(pGamble,delta,sigma)
    #print('w:{}'.format(w))
    #print("vw:{}".format(v*w-valueFunction1(vSure,alpha)))

    return 1/(1+np.exp(phi*(valueFunction1(vSure,alpha)-v*w)))



def calculateUtility2(vGamble, pGamble,vSure,alpha,delta,sigma,phi,_lambda=1):
    """Calculates the utility f(x) term for the gamble option in the Quantum Decison Theory framwork
    This one is equiped with a logit choice function"""
    #v=valueFunction1(vGamble,alpha)
    v1=valueFunction1(vGamble-vSure,alpha,_lambda)
    v2=valueFunction1(-vSure,alpha,_lambda)
    #print('v:{}'.format(v))
    w1=weightingFunction(pGamble,delta,sigma)
    w2=weightingFunction(1-pGamble,delta,sigma)
    #print('w:{}'.format(w))
    #print("vw:{}".format(v*w-valueFunction1(vSure,alpha)))
    u1=v1*w1+v2*w2

    #return 1/(1+np.exp(phi*(valueFunction1(vSure,alpha)-u1)))
    return 1/(1+np.exp(phi*(0-u1)))



def calculateAttractionAdditive(attractionList, fA,argsA,argsB,a ):
    """Calculate the attraction q(x) term in the Quantum Decision Theory framework"""
    if len(attractionList)==0:
        return 0
    else:
        attractionTermsA=[]
        attractionTermsB=[]
        for i in range(len(attractionList)):
            attractionTermsA.append(attractionList[i](*argsA))
            attractionTermsB.append(attractionList[i](*argsB))
        attractA=np.sum(attractionTermsA)
        attractB=np.sum(attractionTermsB)

        temp=np.min([fA,1-fA])
        temp2=np.tanh(a*(attractA-attractB))
    return temp*temp2


def timePressureTerm(vGamble,pGamble,timePressure,neta,framing,x,need,c):
    expectation=pGamble*vGamble+0
    varGamble=pGamble*(vGamble-expectation)**2+(1-pGamble)*(0-expectation)**2
    stdGamble=varGamble**0.5
    #print(stdGamble**(neta*timePressure))
    return -stdGamble**(neta*timePressure)


def framingEffect(vGamble,pGamble,timePressure,neta,framing,x,need,c):
    """framing 1=positive ,-1=negative"""
    if framing==0:
        framing=-1
    expectation = pGamble * vGamble + 0
    varGamble = pGamble * (vGamble - expectation) ** 2 + (1 - pGamble) * (0 - expectation) ** 2
    stdGamble=varGamble**0.5
    return framing*stdGamble*x
    #return framing*stdGamble**x

def needEffect(vGamble,pGamble,timePressure,neta,framing,x,need,c):
    if need>0:
        normalizedNeed=vGamble/need
    else:
        normalizedNeed=0

    pass



def riskAttraction(vGamble,pGamble,timePressure,neta,framing,x,need,c):
    expectation = pGamble * vGamble + 0
    varGamble = pGamble * (vGamble - expectation) ** 2 + (1 - pGamble) * (0 - expectation) ** 2
    stdGamble = varGamble ** 0.5
    return c*stdGamble



def multiplicativeAttraction(vGamble,pGamble,timePressure,neta,framing,x,need,c):
    """Here we assume time prssure with enhance the framing effect, and an additive risk attraction term """
    expectation = pGamble * vGamble + 0
    varGamble = pGamble * (vGamble - expectation) ** 2 + (1 - pGamble) * (0 - expectation) ** 2
    stdGamble = varGamble ** 0.5
    if framing==0:
        framing=-1

    framingEffect=np.exp(-neta*timePressure)*(framing*(stdGamble**x))+c*stdGamble
    return framingEffect

def multiplicativeAttraction2(vGamble,pGamble,timePressure,neta,framing,x,previous,c,Need,b):

    """Here we assume time prssure with enhance the framing effect, and an additive risk attraction term """
    expectation = pGamble * vGamble + 0
    varGamble = pGamble * (vGamble - expectation) ** 2 + (1 - pGamble) * (0 - expectation) ** 2
    stdGamble = varGamble ** 0.5
    if framing==0:
        framing=-1
    if stdGamble>0:
        temp=1
    else:
        temp=0

    if Need==0:
        NeedEffect=0
    else:
        NeedEffect=0.1*b*(Need-3*varGamble*(1-pGamble))
    #print('NeedEffect:{}'.format(NeedEffect))


    framingEffect=np.exp(-neta*timePressure)*(-1*framing*(stdGamble**x))+c*temp*previous+NeedEffect
    #print('FramingEffect:{}'.format(framingEffect))
    #framingEffect=b*(Need-5*varGamble*(1-pGamble))
    #framingEffect = np.exp(-neta * timePressure) * (framing * (stdGamble ** x))
    #framingEffect = np.exp(-neta * timePressure) * (-1 * framing * (stdGamble ** x))
    return framingEffect
    #return 0
    #return NeedEffect
    #return c*previous*temp
    #return framing*(stdGamble**x) #I changed ** to *
    #return 0

def calculateAttractionMultiplicative(fA,argGamble,argSure,a):
    attractA=multiplicativeAttraction2(*argGamble)
    attractB=multiplicativeAttraction2(*argSure)
    temp = np.min([fA, 1 - fA])
    temp2 = np.tanh(a * (attractA - attractB))


    return temp * temp2
    #return 0

class Dataset2:
    def __init__(self,filepath,gamesPerPlayer,reg1,reg2):
        self.data,self.columnNames=loadData(filepath)
        self.reg1=reg1
        self.reg2=reg2


        self.gpp=gamesPerPlayer
        self.numGameTrain=5*self.gpp
        self.numOfPlayers=self.initializePlayerData()
        self.alphas=[]
        self.sigmas=[]
        self.phis=[]
        self.deltas=[]
        self.count=0
        self.gameSummary = [[] for i in range(10)]
        self.gameSummaryTrain = [[] for i in range(10)]
        self.generalParams=[1,1,1,1]
        self.calculateNeed()
        self.calculatePrevious()
        self.dataBlocks = [[] for i in range(self.numOfPlayers)]
        self.mixData()




    def initializePlayerData(self):
        self.playerData=[]
        self.trainData=[]
        self.testData=[]
        playerId=1
        count=0
        while count<54:
            print(playerId)

            dataTemp=getPlayer(self.data,self.columnNames,playerId)
            if not dataTemp.empty:
                self.playerData.append(dataTemp)
                trainData=dataTemp[dataTemp['Game_Sequence']<=5*self.gpp]
                testData=dataTemp[dataTemp['Game_Sequence']>5*self.gpp]
                self.trainData.append(trainData)
                self.testData.append(testData)
                playerId+=1
                count+=1
            else:
                playerId+=1
                continue
        return count

    def calculateNeed(self):
        """Calculate the amount of score left for one player to meet the minimum required need"""
        for i in range(len(self.trainData)):
            tempData=self.trainData[i]
            need=[]
            for k in range(5):
                accumulated=0
                for j in range(self.gpp):
                    #Iterate each game and calculate respective accumulated points
                    #print(self.gpp*k+j)
                    dataFrame=tempData.iloc[self.gpp*k+j,:]
                    if dataFrame['Need-threshold']==1:
                        needThreshold=0
                    elif dataFrame['Need-threshold']==2:
                        needThreshold=2800
                    else:
                        needThreshold=3600
                    accumulated+=dataFrame['Score']
                    if needThreshold==0:
                        need.append(0)
                    elif needThreshold-accumulated>=0:
                        need.append(needThreshold-accumulated)
                    else:
                        need.append(0)

            self.trainData[i]["Needed"]=need

        for i in range(len(self.testData)):
            tempData=self.testData[i]
            need=[]
            for k in range(1):
                accumulated=0
                for j in range(self.gpp):
                    #Iterate each game and calculate respective accumulated points
                    dataFrame=tempData.iloc[self.gpp*k+j,:]
                    if dataFrame['Need-threshold']==1:
                        needThreshold=0
                    elif dataFrame['Need-threshold']==2:
                        needThreshold=2500
                    else:
                        needThreshold=3500
                    accumulated+=dataFrame['Score']
                    if needThreshold==0:
                        need.append(0)
                    elif needThreshold-accumulated>=0:
                        need.append(needThreshold-accumulated)
                    else:
                        need.append(0)

            self.testData[i]["Needed"]=need


    def mixData(self):
        for i in range(self.numOfPlayers):
            d1=self.trainData[i].copy()
            d2=self.testData[i].copy()
            d=[d1,d2]
            dcon=pd.concat(d)
            #print(dcon)
            R = np.random.RandomState(1337)
            dcon=dcon.sample(frac=1,random_state=R)

            self.dataBlocks[i]=([dcon.iloc[i*self.gpp:(i+1)*self.gpp,:] for i in range(6)])
            print(len(self.dataBlocks))

            self.trainData[i]=dcon.iloc[:5*self.gpp,:]
            self.testData[i]=dcon.iloc[5*self.gpp:,:]

    def calculatePrevious(self):
        for i in range(len(self.trainData)):
            tempData = self.trainData[i]
            result=[0]
            for k in range(1,len(tempData)):
                dataFrame=tempData.iloc[k-1,:]
                if dataFrame['Response']==2:
                    if dataFrame['Score']==0:
                        result.append(-1)
                    else:
                        result.append(1)
                else:
                    result.append(0)
            self.trainData[i]['Previous']=result

        for i in range(len(self.testData)):
            tempData = self.testData[i]
            result=[0]
            for k in range(1,len(tempData)):
                dataFrame=tempData.iloc[k-1,:]
                if dataFrame['Response']==2:
                    if dataFrame['Score']==0:
                        result.append(-1)
                    else:
                        result.append(1)
                else:
                    result.append(0)
            self.testData[i]['Previous']=result






    def predict(self,playerID,alpha,delta,sigma,phi,a,neta,x,c,b):
        count=0
        count2=0
        for i in range(len(self.testData[playerID].index)):
            #print(self.testData[playerID].iloc[i,:])
            tempData=self.testData[playerID].iloc[i,:]
            vGamble=tempData['Amount']
            pGamble=0.01*tempData['PercGamb']
            vSure=tempData['Amount']*tempData['PercSure']*0.01


            #alpha,delta,sigma are still in exploration
            fGamble=calculateUtility(vGamble,pGamble,vSure,alpha,delta,sigma,phi)
            #print(fGamble)
            argGamble = [vGamble, pGamble, tempData['TP'], neta, tempData['Frame'], x, tempData['Previous'], c,tempData['Needed'],b]
            argSure = [vSure, 1, tempData['TP'], neta, tempData['Frame'], x, tempData['Previous'], c,tempData['Needed'],b]
            # qGamble = calculateAttractionAdditive([timePressureTerm, framingEffect], fGamble, argGamble, argSure, a)
            qGamble = calculateAttractionMultiplicative(fGamble, argGamble, argSure, a)
            #print('vGamble:{}, pGamble:{},TP:{},frame:{}, attract score:{}'.format(vGamble,pGamble,tempData['TP'],tempData['Frame'],qGamble))
            uFinal = fGamble + qGamble
            if uFinal > 0.5:

                if tempData['Response'] == 2:
                    count += 1
            else:
                if tempData['Response'] == 1:
                    count += 1
            if tempData['Response'] != 0:
                count2 += 1

        return count / count2



    def predict_utility_ind(self,playerID,alpha,delta,sigma,phi):
        count=0
        count2=0
        for i in range(len(self.testData[playerID].index)):
            #print(self.testData[playerID].iloc[i,:])
            tempData=self.testData[playerID].iloc[i,:]
            vGamble=tempData['Amount']
            pGamble=0.01*tempData['PercGamb']
            vSure=tempData['Amount']*tempData['PercSure']*0.01


            #alpha,delta,sigma are still in exploration
            fGamble=calculateUtility(vGamble,pGamble,vSure,alpha,delta,sigma,phi)

            uFinal = fGamble
            if uFinal > 0.5:

                if tempData['Response'] == 2:
                    count += 1
            else:
                if tempData['Response'] == 1:
                    count += 1
            if tempData['Response'] != 0:
                count2 += 1

        return count / count2






    def summarize(self, playerID, alpha, delta, sigma, phi, a, neta, x, c,b):
        ret = np.array([0 for i in range(10)])
        for i in range(len(self.testData[playerID].index)):
            # print(self.testData[playerID].iloc[i,:])
            tempData = self.testData[playerID].iloc[i, :]
            vGamble = tempData['Amount']
            pGamble = 0.01 * tempData['PercGamb']
            vSure = tempData['Amount'] * tempData['PercSure'] * 0.01

            # alpha,delta,sigma are still in exploration
            fGamble = calculateUtility(vGamble, pGamble, vSure, alpha, delta, sigma, phi)
            # print(fGamble)
            # argGamble = [vGamble, pGamble, tempData['TP'], neta, tempData['Frame'], x, tempData['Needed'], c]
            #             # argSure = [vSure, 1, tempData['TP'], neta, tempData['Frame'], x, tempData['Needed'], c]

            argGamble = [vGamble, pGamble, tempData['TP'], neta, tempData['Frame'], x, tempData['Previous'], c,
                         tempData['Needed'], b]
            argSure = [vSure, 1, tempData['TP'], neta, tempData['Frame'], x, tempData['Previous'], c,
                       tempData['Needed'], b]
            # qGamble = calculateAttractionAdditive([timePressureTerm, framingEffect], fGamble, argGamble, argSure, a)
            qGamble = calculateAttractionMultiplicative(fGamble, argGamble, argSure, a)
            # print('vGamble:{}, pGamble:{},TP:{},frame:{}, attract score:{}'.format(vGamble,pGamble,tempData['TP'],tempData['Frame'],qGamble))
            uFinal = fGamble + qGamble
            ind = uFinal // 0.1
            ind=int(ind)
            if ind == 10:
                ind = 9
            ret[int(ind)] += 1
            self.gameSummary[ind].append((playerID, i))
        return ret


    def summarize_utility_ind(self, playerID, alpha, delta, sigma, phi):
        ret = np.array([0 for i in range(10)])
        for i in range(len(self.testData[playerID].index)):
            # print(self.testData[playerID].iloc[i,:])
            tempData = self.testData[playerID].iloc[i, :]
            vGamble = tempData['Amount']
            pGamble = 0.01 * tempData['PercGamb']
            vSure = tempData['Amount'] * tempData['PercSure'] * 0.01

            # alpha,delta,sigma are still in exploration
            fGamble = calculateUtility(vGamble, pGamble, vSure, alpha, delta, sigma, phi)

            uFinal = fGamble
            ind = uFinal // 0.1
            ind=int(ind)
            if ind == 10:
                ind = 9
            ret[int(ind)] += 1
            self.gameSummary[ind].append((playerID, i))
        return ret

    def summarizeTrain(self, playerID, alpha, delta, sigma, phi, a, neta, x, c,b):
        ret = np.array([0 for i in range(10)])
        for i in range(len(self.trainData[playerID].index)):
            # print(self.testData[playerID].iloc[i,:])
            tempData = self.trainData[playerID].iloc[i, :]
            vGamble = tempData['Amount']
            pGamble = 0.01 * tempData['PercGamb']
            vSure = tempData['Amount'] * tempData['PercSure'] * 0.01

            # alpha,delta,sigma are still in exploration
            fGamble = calculateUtility(vGamble, pGamble, vSure, alpha, delta, sigma, phi)
            # print(fGamble)
            # argGamble = [vGamble, pGamble, tempData['TP'], neta, tempData['Frame'], x, tempData['Needed'], c]
            # argSure = [vSure, 1, tempData['TP'], neta, tempData['Frame'], x, tempData['Needed'], c]

            argGamble = [vGamble, pGamble, tempData['TP'], neta, tempData['Frame'], x, tempData['Previous'], c,
                         tempData['Needed'], b]
            argSure = [vSure, 1, tempData['TP'], neta, tempData['Frame'], x, tempData['Previous'], c,
                       tempData['Needed'], b]
            # qGamble = calculateAttractionAdditive([timePressureTerm, framingEffect], fGamble, argGamble, argSure, a)
            qGamble = calculateAttractionMultiplicative(fGamble, argGamble, argSure, a)
            # print('vGamble:{}, pGamble:{},TP:{},frame:{}, attract score:{}'.format(vGamble,pGamble,tempData['TP'],tempData['Frame'],qGamble))
            uFinal = fGamble + qGamble
            ind = uFinal // 0.1
            if ind == 10:
                ind = 9
            ind=int(ind)
            ret[int(ind)] += 1
            ind=int(ind)
            self.gameSummaryTrain[ind].append((playerID, i))
        return ret

    def summarizeTrain_utility_ind(self, playerID, alpha, delta, sigma, phi):
        ret = np.array([0 for i in range(10)])
        for i in range(len(self.trainData[playerID].index)):
            # print(self.testData[playerID].iloc[i,:])
            tempData = self.trainData[playerID].iloc[i, :]
            vGamble = tempData['Amount']
            pGamble = 0.01 * tempData['PercGamb']
            vSure = tempData['Amount'] * tempData['PercSure'] * 0.01

            # alpha,delta,sigma are still in exploration
            fGamble = calculateUtility(vGamble, pGamble, vSure, alpha, delta, sigma, phi)

            uFinal = fGamble
            ind = uFinal // 0.1
            if ind == 10:
                ind = 9
            ind=int(ind)
            ret[int(ind)] += 1
            ind=int(ind)
            self.gameSummaryTrain[ind].append((playerID, i))
        return ret

    def verification(self):
        """For test Data"""
        ret = []
        for i in range(10):
            data = self.gameSummary[i]
            count = 0
            for game in data:
                tempData = self.testData[game[0]].iloc[game[1], :]
                if tempData['Response'] == 2:
                    count += 1

            if count != 0:
                ret.append(count / len(data))
            else:
                ret.append(0)
        print("Test set verification:{}".format(ret))
        return ret

    def verificationTrain(self):
        """For test Data"""
        ret = []
        for i in range(10):
            data = self.gameSummaryTrain[i]
            count = 0
            for game in data:
                tempData = self.trainData[game[0]].iloc[game[1], :]
                if tempData['Response'] == 2:
                    count += 1
            if count!=0:
                ret.append(count / len(data))
            else:
                ret.append(0)
        print("Train set verification:{}".format(ret))
        return ret



    def errorFunc(self,playerID,alpha,delta,sigma,phi,a,neta,x,c,b):
        count=0
        count2=0
        for i in range(len(self.trainData[playerID].index)):
            tempData=self.trainData[playerID].iloc[i,:]
            vGamble=tempData['Amount']
            pGamble=0.01*tempData['PercGamb']
            vSure=tempData['Amount']*tempData['PercSure']*0.01

            fGamble=calculateUtility(vGamble,pGamble,vSure,alpha,delta,sigma,phi)
            # argGamble = [vGamble, pGamble, tempData['TP'], neta, tempData['Frame'], x, tempData['Needed'], c]
            # argSure = [vSure, 1, tempData['TP'], neta, tempData['Frame'], x, tempData['Needed'], c]

            argGamble = [vGamble, pGamble, tempData['TP'], neta, tempData['Frame'], x, tempData['Previous'], c,
                         tempData['Needed'], b]
            argSure = [vSure, 1, tempData['TP'], neta, tempData['Frame'], x, tempData['Previous'], c,
                       tempData['Needed'], b]
            # qGamble = calculateAttractionAdditive([timePressureTerm, framingEffect], fGamble, argGamble, argSure, a)
            qGamble = calculateAttractionMultiplicative(fGamble, argGamble, argSure, a)
            uFinal = fGamble + qGamble
            if uFinal>0.5:
                count2+=1
                if tempData['Response']==2:
                    count+=1
            else:
                if tempData['Response']==1:
                    count+=1
        print(count2)
        return count

    def indErrorFunc(self,params,playerID):
        params = np.append(params, np.zeros(9 - len(params)))
        alpha, delta, sigma, phi, a, neta, x,c,b=params
        regulize = np.sum(np.abs(neta) + np.abs(x) + np.abs(c) + np.abs(b))

        self.count += 1
        print('Numer of iters:{}'.format(self.count))
        print('PlayerID:{}'.format(playerID))
        ret = 0

        playerData = self.trainData[playerID]
        for j in range(len(self.trainData[playerID].index)):
            tempData = self.trainData[playerID].iloc[j, :]
            vGamble = tempData['Amount']
            pGamble = 0.01 * tempData['PercGamb']
            vSure = tempData['Amount'] * tempData['PercSure'] * 0.01

            fGamble = calculateUtility(vGamble, pGamble, vSure, alpha, delta, sigma, phi)
            # argGamble = [vGamble, pGamble, tempData['TP'], neta, tempData['Frame'], x, tempData['Needed'], c]
            # argSure = [vSure, 1, tempData['TP'], neta, tempData['Frame'], x, tempData['Needed'], c]

            argGamble = [vGamble, pGamble, tempData['TP'], neta, tempData['Frame'], x, tempData['Previous'], c,
                         tempData['Needed'], b]
            argSure = [vSure, 1, tempData['TP'], neta, tempData['Frame'], x, tempData['Previous'], c,
                       tempData['Needed'], b]
            # qGamble = calculateAttractionAdditive([timePressureTerm, framingEffect], fGamble, argGamble, argSure, a)
            qGamble = calculateAttractionMultiplicative(fGamble, argGamble, argSure, a)
            uFinal = fGamble + qGamble
            if tempData['Response'] == 2:
                ret += np.log(uFinal)
            else:
                ret += np.log(1 - uFinal)
        print(-ret)
        return -ret+regulize


    def totalErrorFunc(self,alpha, delta, sigma, phi,a,neta,x,c,b):
        self.count+=1
        print('Numer of iters:{}'.format(self.count))
        print('alpha:{}, delta:{},sigam:{},phi:{}'.format(alpha,delta,sigma,phi))
        ret=0
        for i in range(self.numOfPlayers):
            playerData=self.trainData[i]
            for j in range(len(self.trainData[i].index)):
                tempData=playerData.iloc[j,:]
                if tempData['PercGamb']!=tempData['PercSure']:
                    continue  #Skip all catch trials
                vGamble = tempData['Amount']
                pGamble = 0.01 * tempData['PercGamb']
                vSure = tempData['Amount'] * tempData['PercSure']*0.01
                fGamble=calculateUtility(vGamble,pGamble,vSure,alpha,delta,sigma,phi)
                # argGamble = [vGamble, pGamble, tempData['TP'], neta, tempData['Frame'], x, tempData['Needed'], c]
                # argSure = [vSure, 1, tempData['TP'], neta, tempData['Frame'], x, tempData['Needed'], c]

                argGamble = [vGamble, pGamble, tempData['TP'], neta, tempData['Frame'], x, tempData['Previous'], c,
                             tempData['Needed'], b]
                argSure = [vSure, 1, tempData['TP'], neta, tempData['Frame'], x, tempData['Previous'], c,
                           tempData['Needed'], b]
                # qGamble = calculateAttractionAdditive([timePressureTerm, framingEffect], fGamble, argGamble, argSure, a)
                qGamble = calculateAttractionMultiplicative(fGamble, argGamble, argSure, a)
                uFinal=fGamble+qGamble
                if tempData['Response']==2:
                    ret+=np.log(uFinal)
                else:
                    ret+=np.log(1-uFinal)
        print(-ret)
        print('After regularize:{}'.format(-ret + self.regularizerRational(alpha, delta, sigma, phi)))
        return -ret + self.regularizerRational(alpha, delta, sigma, phi)

    def regularizerRational(self, alpha, delta, sigma, phi):
        """This is the regularizer function for rational coefficients"""
        return self.reg1 * (np.abs(alpha) + np.abs(delta) + np.abs(sigma) + np.abs(phi))

    def regularizerIrational(self, params, Lambda=5 / 19):
        return self.reg2 * np.sum(np.abs(params))



    def totalErrorFunc2(self,alpha, delta, sigma, phi):

        self.count+=1
        print('Numer of iters:{}'.format(self.count))
        print('alpha:{}, delta:{},sigam:{},phi:{}'.format(alpha, delta, sigma, phi))
        ret=0
        for i in range(self.numOfPlayers):
            playerData=self.trainData[i]
            for j in range(len(self.trainData[i].index)):
                tempData=playerData.iloc[j,:]
                # if tempData['PercGamb']!=tempData['PercSure']:
                #     continue  #Skip all catch trials
                vGamble = tempData['Amount']
                pGamble = 0.01 * tempData['PercGamb']
                vSure = tempData['Amount'] * tempData['PercSure']*0.01
                fGamble=calculateUtility(vGamble,pGamble,vSure,alpha,delta,sigma,phi)
                # argGamble=[vGamble,pGamble,tempData['TP'],neta,tempData['Frame'],x]
                # argSure=[vSure,1,tempData['TP'],neta,tempData['Frame'],x]
                # qGamble=calculateAttractionAdditive([timePressureTerm,framingEffect],fGamble,argGamble,argSure,a)
                #print('vGamble:{}, pGamble:{}, Ut score:{}'.format(vGamble, pGamble,  fGamble))
                uFinal=fGamble
                if tempData['Response']==2:
                    ret+=np.log(uFinal)
                else:
                    ret+=np.log(1-uFinal)
        print(-ret)
        print('After regularize:{}'.format(-ret+self.regularizerRational(alpha,delta,sigma,phi)))
        return -ret+self.regularizerRational(alpha,delta,sigma,phi)

    def individual_utility_only_target(self,params,playerID):
        self.count += 1
        alpha, delta, sigma, phi = params
        print('Numer of iters:{}'.format(self.count))
        print('alpha:{}, delta:{},sigam:{},phi:{}'.format(alpha, delta, sigma, phi))
        ret=0

        playerData=self.trainData[playerID]
        for j in range(len(self.trainData[playerID].index)):
            tempData = playerData.iloc[j, :]
            # if tempData['PercGamb'] != tempData['PercSure']:
            #     continue  # Skip all catch trials
            vGamble = tempData['Amount']
            pGamble = 0.01 * tempData['PercGamb']
            vSure = tempData['Amount'] * tempData['PercSure'] * 0.01
            fGamble = calculateUtility(vGamble, pGamble, vSure, alpha, delta, sigma, phi)
            uFinal = fGamble
            if tempData['Response'] == 2:
                ret += np.log(uFinal)
            else:
                ret += np.log(1 - uFinal)
        print(-ret)
        return -ret





    def indErrorFuncAttract(self,params,playerID):
        a, neta, x,c,b=params
        alpha,delta,sigma,phi=self.generalParams

        self.count+=1
        print('Numer of iters:{}'.format(self.count))
        print('PlayerID:{}'.format(playerID))
        ret = 0

        playerData = self.trainData[playerID]
        for j in range(len(self.trainData[playerID].index)):
            tempData = playerData.iloc[j, :]
            if tempData['PercGamb'] != tempData['PercSure']:
                continue  # Skip all catch trials
            vGamble = tempData['Amount']
            pGamble = 0.01 * tempData['PercGamb']
            vSure = tempData['Amount'] * tempData['PercSure'] * 0.01
            fGamble = calculateUtility(vGamble, pGamble, vSure, alpha, delta, sigma, phi)
            # argGamble = [vGamble, pGamble, tempData['TP'], neta, tempData['Frame'], x,tempData['Needed'],c]
            # argSure = [vSure, 1, tempData['TP'], neta, tempData['Frame'], x,tempData['Needed'],c]

            argGamble = [vGamble, pGamble, tempData['TP'], neta, tempData['Frame'], x, tempData['Previous'], c,
                         tempData['Needed'], b]
            argSure = [vSure, 1, tempData['TP'], neta, tempData['Frame'], x, tempData['Previous'], c,
                       tempData['Needed'], b]
            #qGamble = calculateAttractionAdditive([timePressureTerm, framingEffect], fGamble, argGamble, argSure, a)
            qGamble=calculateAttractionMultiplicative(fGamble,argGamble,argSure,a)
            #print('vGamble:{}, pGamble:{},TP:{},frame:{}, attract score:{}'.format(vGamble, pGamble, tempData['TP'],
                                                                                   #tempData['Frame'], qGamble))
            uFinal = fGamble + qGamble
            if tempData['Response'] == 2:
                ret += np.log(uFinal)
            else:
                ret += np.log(1 - uFinal)
        print(-ret)
        print('After regularize:{}'.format(-ret+self.regularizerIrational(params)))
        return -ret+self.regularizerIrational(params)




    def dummy(self,args):
        return self.totalErrorFunc(*args)

    def dummy2(self,args):
        return self.totalErrorFunc2(*args)

    def totalErrorAttrac(self,args):
        a,neta,x=args
        alpha,delta,sigma,phi=self.generalParams
        self.count += 1
        print('Numer of iters:{}'.format(self.count))
        print('alpha:{}, delta:{},sigam:{},phi:{}'.format(alpha, delta, sigma, phi))
        ret = 0
        for i in range(self.numOfPlayers):
            playerData = self.trainData[i]
            for j in range(len(self.trainData[i].index)):
                tempData = playerData.iloc[j, :]
                if tempData['PercGamb'] != tempData['PercSure']:
                    continue  # Skip all catch trials
                vGamble = tempData['Amount']
                pGamble = 0.01 * tempData['PercGamb']
                vSure = tempData['Amount'] * tempData['PercSure'] * 0.01
                fGamble = calculateUtility(vGamble, pGamble, vSure, alpha, delta, sigma, phi)
                argGamble = [vGamble, pGamble, tempData['TP'], neta, tempData['Frame'], x]
                argSure = [vSure, 1, tempData['TP'], neta, tempData['Frame'], x]
                qGamble = calculateAttractionAdditive([timePressureTerm, framingEffect], fGamble, argGamble, argSure, a)
                uFinal = fGamble + qGamble
                if tempData['Response'] == 2:
                    ret += np.log(uFinal)
                else:
                    ret += np.log(1 - uFinal)
        print(-ret)
        return -ret








    def train(self):
        sol=minimize(self.dummy,np.array([0.25215286,  0.90386184, 0.50382084,  0.9815681 ,  0.89839473,
        0.22683087,  0.5]),method='Nelder-Mead',options={'maxiter':2000,'adaptive':True,'disp':True})
        print(sol)
        param=sol.x
        result = []
        trainResult = []
        for i in range(self.numOfPlayers):


            result.append(self.predict(i, *param))
            trainResult.append(self.errorFunc(i, *param) / 640)
        # print(self.generalParams)
        # print(self.indParams)
        print(result)

        print('Average is :{}'.format(np.sum(result) / 19))

        plot.hist(result)
        plot.savefig('Plots/7thTest_agg_mix.png', dpi=240)
        plot.hist(trainResult)
        plot.savefig('Plots/7thTestTrain_agg_mix.png', dpi=240)


        return param

    def trainRational(self):
        print('L_BFGS-B')
        #sol=minimize(self.dummy2,np.array([0.25,0.9,0.5,1]),bounds=[(0,None),(0.5,1.5),(0,1),(0,None)],method='L-BFGS-B',options={'maxiter':2000})
        sol = minimize(self.dummy2, np.array([0.25, 0.9, 0.5, 1]),
                       method='Nelder-Mead', options={'maxiter': 2000})

        print(sol)
        params=sol.x
        self.generalParams=sol.x

    def trainRational2(self):
        """Train the first four parameters together with 4 individual parameters"""
        sol=minimize(self.dummy,np.array([0.25,0.9,0.5,1,1,0.2,0.2,1,0.01]),method='Nelder-Mead',options={'maxiter':4000,'disp':True,'adaptive':True})

        #sol = minimize(self.dummy, np.array([0.25, 0.9, 0.5, 1, 0.8, 0.2, 0.2, 0]), method='L-BFGS-B',bounds=[(0,2),(0.1,2),(0,1),(0,None),(None,None),(None,None),(None,None),(None,None)],
                      # options={'maxiter': 2000, 'disp': True, 'adaptive': True})
        self.generalParams=sol.x[:4]


    def trainIrational(self):
        ret={}
        for i in range(self.numOfPlayers):

            sol=minimize(self.indErrorFuncAttract,np.array([1,0.2,0.2,1,0.01]), args=(i,),method='Nelder-Mead',options={'maxiter':1000,'disp':True})
            ret[i]=sol.x
        self.indParams=ret

        return ret
    def trainUtilityInd(self):
        ret={}
        for i in range(self.numOfPlayers):
            print('Now doing player ID:{}'.format(i))
            sol=minimize(self.individual_utility_only_target,np.array([0.25,1,0.5,1]),args=(i,), tol=1e-6,method='Nelder-Mead',options={'maxiter':1000,'disp':True})
            ret[i]=sol.x
        self.utility_ind=ret

    def trainInd(self):
        ret={}
        for i in range(self.numOfPlayers):
            print('Doing number:{}'.format(i))
            sol = minimize(self.indErrorFunc, x0=np.array([0.25, 1, 0.5, 1, 1,
                                                           0.2, 0.2]), args=(i,), method='Nelder-Mead',
                           tol=1e-6,
                           options={'maxiter': 3000, 'adaptive': True, 'disp': True})
            solution=sol.x
            for j in range(9 - len(solution)):
                solution = np.append(solution, 0)
            ret[i] = solution

        self.indParams = ret


    def predict2_utility_ind(self,filename):
        result = []
        trainResult = []
        count = np.array([0 for i in range(10)])
        countTrain = np.array([0 for i in range(10)])
        for i in range(self.numOfPlayers):
            param=self.utility_ind[i]

            result.append(self.predict_utility_ind(i, *param))
            playerData = self.summarize_utility_ind(i, *param)
            countTrain += self.summarizeTrain_utility_ind(i, *param)
            count += playerData




        with open(filename + '.txt', 'a') as f:
            f.write('General Params: ')
            f.write(str(self.generalParams))
            f.write('\n')
            f.write('Ind-prams')
            f.write(str(self.utility_ind))
            f.write('\n')

            f.write('Result: ')
            f.write(str(result))
            f.write('\n')
            f.write('Average: ')
            f.write(str(np.mean(result)))
            f.write('\n')


        print('Average is :{}'.format(np.sum(result) / self.numOfPlayers))

        plot.hist(result)
        plot.xlabel('Prediction rate')
        plot.ylabel('Count')
        plot.savefig('Plots/test+' + filename + '.png', dpi=240)
        plot.hist(trainResult)
        plot.savefig('Plots/train+' + filename + '.png', dpi=240)
        plot.plot([0.1 * i for i in range(10)], count, '-o')

        plot.plot([0.1 * i for i in range(10)], countTrain, '-o')
        plot.legend(['test set', 'training set'])
        plot.savefig('Plots/count+' + filename + '.png', dpi=240)

        print(self.verification())
        with open(filename + '.txt', 'a') as f:
            f.write('Verification: ')
            f.write(str(self.verification()))
            f.write('\n')
            f.write('Train Verification: ')
            f.write(str(self.verificationTrain()))
            f.write('\n')

        print(self.verificationTrain())







    def predict2(self,filename,ifind=False):
        """Using two step fitting method to predict"""


        result=[]
        trainResult=[]
        count=np.array([0 for i in range(10)])
        countTrain=np.array([0 for i in range(10)])
        for i in range(self.numOfPlayers):
            if not ifind:
                param=[]
                for j in range(4):
                    param.append(self.generalParams[j])
                for j in range(5):
                    param.append(self.indParams[i][j])
            else:
                param=self.indParams[i]

            result.append(self.predict(i,*param))
            playerData=self.summarize(i,*param)
            countTrain+=self.summarizeTrain(i,*param)
            count+=playerData

            trainResult.append(self.errorFunc(i,*param)/520)
        print(self.generalParams)
        print(self.indParams)
        print(result)

        with open(filename + '.txt', 'a') as f:
            f.write('General Params: ')
            f.write(str(self.generalParams))
            f.write('\n')
            f.write('Ind Params: ')
            f.write(str(self.indParams))
            f.write('\n')
            f.write('Result: ')
            f.write(str(result))
            f.write('\n')
            f.write('Average: ')
            f.write(str(np.mean(result)))
            f.write('\n')
            f.write('Train Result: ')
            f.write(str(trainResult))
            f.write('\n')



        print('Average is :{}'.format(np.sum(result) / self.numOfPlayers))

        plot.hist(result)
        plot.xlabel('Prediction rate')
        plot.ylabel('Count')
        plot.savefig('Plots/test+'+filename+'.png', dpi=240)
        plot.hist(trainResult)
        plot.savefig('Plots/train+'+filename+'.png', dpi=240)
        plot.plot([0.1*i for i in range(10)],count,'-o')

        plot.plot([0.1*i for i in range(10)],countTrain,'-o')
        plot.legend(['test set','training set'])
        plot.savefig('Plots/count+'+filename+'.png',dpi=240)

        print(self.verification())
        with open(filename+'.txt','a') as f:
            f.write('Verification: ')
            f.write(str(self.verification()))
            f.write('\n')
            f.write('Train Verification: ')
            f.write(str(self.verificationTrain()))
            f.write('\n')


        print(self.verificationTrain())











    def simulateInd(self,playerId,alpha,delta,sigma,phi,a,neta,x,c):
        testData=self.testData[playerId]
        trainData=self.testData[playerId]
        countTotal=0
        countCorrect=0
        for i in range(len(testData)):
            tempData=testData.iloc[i,:]
            vGamble = tempData['Amount']
            pGamble = 0.01 * tempData['PercGamb']
            vSure = tempData['Amount'] * tempData['PercSure'] * 0.01

            # alpha,delta,sigma are still in exploration
            fGamble = calculateUtility(vGamble, pGamble, vSure, alpha, delta, sigma, phi)
            # print(fGamble)
            # argGamble = [vGamble, pGamble, tempData['TP'], neta, tempData['Frame'], x, tempData['Needed'], c]
            # argSure = [vSure, 1, tempData['TP'], neta, tempData['Frame'], x, tempData['Needed'], c]

            argGamble = [vGamble, pGamble, tempData['TP'], neta, tempData['Frame'], x, tempData['Previous'], c]
            argSure = [vSure, 1, tempData['TP'], neta, tempData['Frame'], x, tempData['Previous'], c]
            # qGamble = calculateAttractionAdditive([timePressureTerm, framingEffect], fGamble, argGamble, argSure, a)
            qGamble = calculateAttractionMultiplicative(fGamble, argGamble, argSure, a)
            # print('vGamble:{}, pGamble:{},TP:{},frame:{}, attract score:{}'.format(vGamble,pGamble,tempData['TP'],tempData['Frame'],qGamble))
            uFinal = fGamble + qGamble
            countTotal+=1
            if np.random.rand()>uFinal:
                #Simulated participant choose sure option
                if tempData['Response']==1:
                    countCorrect+=1


            else:
                #Simulated participant choose to gamble
                if tempData['Response']==2:
                    countCorrect+=1

        for i in range(len(trainData)):
            tempData = trainData.iloc[i, :]
            vGamble = tempData['Amount']
            pGamble = 0.01 * tempData['PercGamb']
            vSure = tempData['Amount'] * tempData['PercSure'] * 0.01

            # alpha,delta,sigma are still in exploration
            fGamble = calculateUtility(vGamble, pGamble, vSure, alpha, delta, sigma, phi)
            # print(fGamble)
            # argGamble = [vGamble, pGamble, tempData['TP'], neta, tempData['Frame'], x, tempData['Needed'], c]
            # argSure = [vSure, 1, tempData['TP'], neta, tempData['Frame'], x, tempData['Needed'], c]

            argGamble = [vGamble, pGamble, tempData['TP'], neta, tempData['Frame'], x, tempData['Previous'], c]
            argSure = [vSure, 1, tempData['TP'], neta, tempData['Frame'], x, tempData['Previous'], c]
            # qGamble = calculateAttractionAdditive([timePressureTerm, framingEffect], fGamble, argGamble, argSure, a)
            qGamble = calculateAttractionMultiplicative(fGamble, argGamble, argSure, a)
            # print('vGamble:{}, pGamble:{},TP:{},frame:{}, attract score:{}'.format(vGamble,pGamble,tempData['TP'],tempData['Frame'],qGamble))
            uFinal = fGamble + qGamble
            countTotal += 1
            if np.random.rand() > uFinal:
                # Simulated participant choose sure option
                if tempData['Response'] == 1:
                    countCorrect += 1


            else:
                # Simulated participant choose to gamble
                if tempData['Response'] == 2:
                    countCorrect += 1

        print('{} correct in total {} questions, playerId {}'.format(countCorrect,countTotal,playerId))
        return countCorrect,countTotal






    def simulate(self,N,filename):
        '''Simulate the response using the fitted parameters for N times and return the summary'''
        ret=[0 for i in range(25)]
        for i in range(N):
            print(i)
            for playerId in range(self.numOfPlayers):
                param = []
                for j in range(4):
                    param.append(self.generalParams[j])
                for j in range(4):
                    param.append(self.indParams[playerId][j])
                countCorrect,countTotal=self.simulateInd(playerId,*param)
                ind=countCorrect/countTotal
                ind=ind//0.04
                if ind==25:
                    ind=24
                ind=int(ind)
                ret[ind]+=1
        print(ret)
        plot.plot([0.04*i for i in range(25)],ret,'-o')
        plot.xlabel('Percent of games that have the same response')
        plot.ylabel('Count of simulations')
        plot.savefig(filename,dpi=240)


    def crossValidation(self,filename):
        for time in range(6):
            print(time)
            self.gameSummary = [[] for i in range(10)]
            self.gameSummaryTrain = [[] for i in range(10)]
            #Form the training set and test set
            for i in range(self.numOfPlayers):
                self.trainData[i]=pd.concat([element for index,element in enumerate(self.dataBlocks[i]) if index!=time])
                self.testData[i]=self.dataBlocks[i][time]
                # print('Train, time{}'.format(time))
                # print(self.trainData[i])
                # print('Test,time{}'.format(time))
                # print(self.testData[i])
            # self.trainRational2()
            # self.count=0
            # self.trainIrational()
            self.trainInd()
            self.count=0
            self.predict2(str(time)+'__'+filename,ifind=True)

    def crossValidation_utility_ind(self,filename):
        for time in range(6):
            print(time)
            self.gameSummary = [[] for i in range(10)]
            self.gameSummaryTrain = [[] for i in range(10)]
            #Form the training set and test set
            for i in range(self.numOfPlayers):
                self.trainData[i]=pd.concat([element for index,element in enumerate(self.dataBlocks[i]) if index!=time])
                self.testData[i]=self.dataBlocks[i][time]
                # print('Train, time{}'.format(time))
                # print(self.trainData[i])
                # print('Test,time{}'.format(time))
                # print(self.testData[i])
            self.trainUtilityInd()
            self.predict2_utility_ind(str(time)+'__'+filename)






































class Dataset1:
    def __init__(self,filepath,gamesPerPlayer,reg1,reg2):
        self.data,self.columnNames=loadData(filepath)
        self.reg1=reg1
        self.reg2=reg2
        self.gpp=gamesPerPlayer
        self.numberOfSessions=self.data.shape[0]/self.gpp
        self.numOfPlayers=self.initializePlayerData()
        self.alphas=[]
        self.sigmas=[]
        self.phis=[]
        self.deltas=[]
        self.count=0
        self.generalParams=[1,1,1,1]
        self.dataBlocks = [[] for i in range(self.numOfPlayers)]
        self.gameSummary=[[] for i in range(10)]
        self.gameSummaryTrain = [[] for i in range(10)]
        self.calculateNeed()
        self.calculatePrevious()
        self.mixData()
        print(self.dataBlocks)


    def regularizerRational(self,alpha,delta,sigma,phi):
        """This is the regularizer function for rational coefficients"""
        return self.reg1*(np.abs(alpha)+np.abs(delta)+np.abs(sigma)+np.abs(phi))

    def regularizerIrational(self,params,Lambda=5/19):
        return self.reg2*np.sum(np.abs(params))


    def summarize(self,playerID,alpha,delta,sigma,phi,a,neta,x,c,b):
        ret=np.array([0 for i in range(10)])
        for i in range(len(self.testData[playerID].index)):
            #print(self.testData[playerID].iloc[i,:])
            tempData=self.testData[playerID].iloc[i,:]
            vGamble=tempData['Amount']
            pGamble=0.01*tempData['PercGamb']
            vSure=tempData['Amount']*tempData['PercSure']*0.01


            #alpha,delta,sigma are still in exploration
            fGamble=calculateUtility(vGamble,pGamble,vSure,alpha,delta,sigma,phi)
            #print(fGamble)
            # argGamble = [vGamble, pGamble, tempData['TP'], neta, tempData['Frame'], x, tempData['Needed'], c]
            #             # argSure = [vSure, 1, tempData['TP'], neta, tempData['Frame'], x, tempData['Needed'], c]

            argGamble = [vGamble, pGamble, tempData['TP'], neta, tempData['Frame'], x, tempData['Previous'], c,
                         tempData['Needed'], b]
            argSure = [vSure, 1, tempData['TP'], neta, tempData['Frame'], x, tempData['Previous'], c,
                       tempData['Needed'], b]
            # qGamble = calculateAttractionAdditive([timePressureTerm, framingEffect], fGamble, argGamble, argSure, a)
            qGamble = calculateAttractionMultiplicative(fGamble, argGamble, argSure, a)
            #print('vGamble:{}, pGamble:{},TP:{},frame:{}, attract score:{}'.format(vGamble,pGamble,tempData['TP'],tempData['Frame'],qGamble))
            uFinal = fGamble + qGamble
            ind=uFinal//0.1
            if ind==10:
                ind=9
            ret[int(ind)]+=1
            ind=int(ind)
            self.gameSummary[ind].append((playerID,i))
        print('Summary:{}'.format(ret))
        return ret

    def summarizeTrain(self,playerID,alpha,delta,sigma,phi,a,neta,x,c,b):
        ret=np.array([0 for i in range(10)])
        for i in range(len(self.trainData[playerID].index)):
            #print(self.testData[playerID].iloc[i,:])
            tempData=self.trainData[playerID].iloc[i,:]
            vGamble=tempData['Amount']
            pGamble=0.01*tempData['PercGamb']
            vSure=tempData['Amount']*tempData['PercSure']*0.01


            #alpha,delta,sigma are still in exploration
            fGamble=calculateUtility(vGamble,pGamble,vSure,alpha,delta,sigma,phi)
            #print(fGamble)
            # argGamble = [vGamble, pGamble, tempData['TP'], neta, tempData['Frame'], x, tempData['Needed'], c]
            # argSure = [vSure, 1, tempData['TP'], neta, tempData['Frame'], x, tempData['Needed'], c]

            argGamble = [vGamble, pGamble, tempData['TP'], neta, tempData['Frame'], x, tempData['Previous'], c,
                         tempData['Needed'], b]
            argSure = [vSure, 1, tempData['TP'], neta, tempData['Frame'], x, tempData['Previous'], c,
                       tempData['Needed'], b]
            # qGamble = calculateAttractionAdditive([timePressureTerm, framingEffect], fGamble, argGamble, argSure, a)
            qGamble = calculateAttractionMultiplicative(fGamble, argGamble, argSure, a)
            #print('vGamble:{}, pGamble:{},TP:{},frame:{}, attract score:{}'.format(vGamble,pGamble,tempData['TP'],tempData['Frame'],qGamble))
            uFinal = fGamble + qGamble
            ind=uFinal//0.1
            if ind==10:
                ind=9
            ret[int(ind)]+=1
            ind=int(ind)
            self.gameSummaryTrain[ind].append((playerID,i))

        print('Summary Train:{}'.format(ret))
        return ret

    def verification(self):
        """For test Data"""
        ret=[]
        for i in range(10):
            data=self.gameSummary[i]
            count=0
            for game in data:
                tempData=self.testData[game[0]].iloc[game[1],:]
                if tempData['Response']==2:
                    count+=1
            if count != 0:
                ret.append(count / len(data))
            else:
                ret.append(0)

        print(ret)
        return ret

    def verificationTrain(self):
        """For test Data"""
        ret = []
        for i in range(10):
            data = self.gameSummaryTrain[i]
            count = 0
            for game in data:
                tempData = self.trainData[game[0]].iloc[game[1], :]
                if tempData['Response'] == 2:
                    count += 1

            if count != 0:
                ret.append(count / len(data))
            else:
                ret.append(0)
        print(ret)
        return ret





    def initializePlayerData(self):
        self.playerData=[]
        self.trainData=[]
        self.testData=[]
        playerId=1
        while True:
            dataTemp=getPlayer(self.data,self.columnNames,playerId)
            if not dataTemp.empty:
                self.playerData.append(dataTemp)
                trainData=dataTemp[dataTemp['Session'].isin([1,2])]
                testData=dataTemp[dataTemp['Session'].isin([3])]
                self.trainData.append(trainData)
                self.testData.append(testData)
                playerId+=1
            else:
                break
        return playerId-1

    def calculateNeed(self):
        """Calculate the amount of score left for one player to meet the minimum required need"""
        for i in range(len(self.trainData)):
            tempData=self.trainData[i]
            need=[]
            for k in range(2):

                for j in range(4):
                    accumulated = 0
                    for x in range(80):
                    #Iterate each game and calculate respective accumulated points
                        dataFrame=tempData.iloc[self.gpp*k+j*80+x,:]
                        if dataFrame['Need-threshold']==1:
                            needThreshold=0
                        elif dataFrame['Need-threshold']==2:
                            needThreshold=2500
                        else:
                            needThreshold=3500
                        accumulated+=dataFrame['score']
                        if needThreshold==0:
                            need.append(0)
                        elif needThreshold-accumulated>=0:
                            need.append(needThreshold-accumulated)
                        else:
                            need.append(0)

            self.trainData[i]["Needed"]=need

        for i in range(len(self.testData)):
            tempData=self.testData[i]
            need=[]
            for k in range(1):

                for j in range(4):
                    accumulated = 0
                    for x in range(80):
                        #Iterate each game and calculate respective accumulated points
                        dataFrame=tempData.iloc[self.gpp*k+j*80+x,:]
                        if dataFrame['Need-threshold']==1:
                            needThreshold=0
                        elif dataFrame['Need-threshold']==2:
                            needThreshold=2500
                        else:
                            needThreshold=3500
                        accumulated+=dataFrame['score']
                        if needThreshold==0:
                            need.append(0)
                        elif needThreshold-accumulated>=0:
                            need.append(needThreshold-accumulated)
                        else:
                            need.append(0)

            self.testData[i]["Needed"]=need


    def calculatePrevious(self):
        for i in range(len(self.trainData)):
            tempData = self.trainData[i]
            result=[0]
            for k in range(1,len(tempData)):
                dataFrame=tempData.iloc[k-1,:]
                if dataFrame['Response']==2:
                    if dataFrame['score']==0:
                        result.append(-1)
                    else:
                        result.append(1)
                else:
                    result.append(0)
            self.trainData[i]['Previous']=result

        for i in range(len(self.testData)):
            tempData = self.testData[i]
            result=[0]
            for k in range(1,len(tempData)):
                dataFrame=tempData.iloc[k-1,:]
                if dataFrame['Response']==2:
                    if dataFrame['score']==0:
                        result.append(-1)
                    else:
                        result.append(1)
                else:
                    result.append(0)
            self.testData[i]['Previous']=result

    def mixData(self):
        for i in range(self.numOfPlayers):
            d1 = self.trainData[i].copy()
            d2 = self.testData[i].copy()
            d = [d1, d2]
            dcon = pd.concat(d)
            # print(dcon)
            R=np.random.RandomState(1337)
            dcon = dcon.sample(frac=1,random_state=R)
            gpp=int(self.gpp/2)

            self.dataBlocks[i] = ([dcon.iloc[i * gpp:(i + 1) * gpp, :] for i in range(6)])
            print(len(self.dataBlocks))












    def predict(self,playerID,alpha,delta,sigma,phi,a,neta,x,c,b):
        count=0
        count2=0

        for i in range(len(self.testData[playerID].index)):
            #print(self.testData[playerID].iloc[i,:])
            tempData=self.testData[playerID].iloc[i,:]
            vGamble=tempData['Amount']
            pGamble=0.01*tempData['PercGamb']
            vSure=tempData['Amount']*tempData['PercSure']*0.01


            #alpha,delta,sigma are still in exploration
            fGamble=calculateUtility(vGamble,pGamble,vSure,alpha,delta,sigma,phi)
            #print(fGamble)
            argGamble = [vGamble, pGamble, tempData['TP'], neta, tempData['Frame'], x, tempData['Previous'], c,tempData['Needed'],b]
            argSure = [vSure, 1, tempData['TP'], neta, tempData['Frame'], x, tempData['Previous'], c,tempData['Needed'],b]
            # qGamble = calculateAttractionAdditive([timePressureTerm, framingEffect], fGamble, argGamble, argSure, a)
            qGamble = calculateAttractionMultiplicative(fGamble, argGamble, argSure, a)
            #print('vGamble:{}, pGamble:{},TP:{},frame:{}, attract score:{}'.format(vGamble,pGamble,tempData['TP'],tempData['Frame'],qGamble))
            uFinal = fGamble + qGamble
            if uFinal > 0.5:

                if tempData['Response'] == 2:
                    count += 1
            else:
                if tempData['Response'] == 1:
                    count += 1
            if tempData['Response']!=0:
                count2+=1

        return count/count2


    def predict_utility_ind(self,playerID,alpha,delta,sigma,phi,_lambda=1):
        count=0
        count2=0
        for i in range(len(self.testData[playerID].index)):
            #print(self.testData[playerID].iloc[i,:])
            tempData=self.testData[playerID].iloc[i,:]
            vGamble=tempData['Amount']
            pGamble=0.01*tempData['PercGamb']
            vSure=tempData['Amount']*tempData['PercSure']*0.01


            #alpha,delta,sigma are still in exploration
            fGamble=calculateUtility2(vGamble,pGamble,vSure,alpha,delta,sigma,phi,_lambda)

            uFinal = fGamble
            if uFinal > 0.5:

                if tempData['Response'] == 2:
                    count += 1
            else:
                if tempData['Response'] == 1:
                    count += 1
            if tempData['Response'] != 0:
                count2 += 1

        return count / count2







    def summarize_utility_ind(self, playerID, alpha, delta, sigma, phi,_lambda=1):
        ret = np.array([0 for i in range(10)])
        for i in range(len(self.testData[playerID].index)):
            # print(self.testData[playerID].iloc[i,:])
            tempData = self.testData[playerID].iloc[i, :]
            vGamble = tempData['Amount']
            pGamble = 0.01 * tempData['PercGamb']
            vSure = tempData['Amount'] * tempData['PercSure'] * 0.01

            # alpha,delta,sigma are still in exploration
            fGamble = calculateUtility2(vGamble, pGamble, vSure, alpha, delta, sigma, phi,_lambda)

            uFinal = fGamble
            ind = uFinal // 0.1
            ind=int(ind)
            if ind == 10:
                ind = 9
            ret[int(ind)] += 1
            self.gameSummary[ind].append((playerID, i))
        return ret




    def summarizeTrain_utility_ind(self, playerID, alpha, delta, sigma, phi,_lambda=1):
        ret = np.array([0 for i in range(10)])
        for i in range(len(self.trainData[playerID].index)):
            # print(self.testData[playerID].iloc[i,:])
            tempData = self.trainData[playerID].iloc[i, :]
            vGamble = tempData['Amount']
            pGamble = 0.01 * tempData['PercGamb']
            vSure = tempData['Amount'] * tempData['PercSure'] * 0.01

            # alpha,delta,sigma are still in exploration
            fGamble = calculateUtility2(vGamble, pGamble, vSure, alpha, delta, sigma, phi,_lambda)

            uFinal = fGamble
            ind = uFinal // 0.1
            if ind == 10:
                ind = 9
            ind=int(ind)
            ret[int(ind)] += 1
            ind=int(ind)
            self.gameSummaryTrain[ind].append((playerID, i))
        return ret

    def individual_utility_only_target(self,params,playerID):
        self.count += 1
        alpha, delta, sigma, phi,_lambda = params
        print('Numer of iters:{}'.format(self.count))
        print('alpha:{}, delta:{},sigam:{},phi:{},lambda:{}'.format(alpha, delta, sigma, phi,_lambda))
        ret=0

        playerData=self.trainData[playerID]
        for j in range(len(self.trainData[playerID].index)):
            tempData = playerData.iloc[j, :]
            # if tempData['PercGamb'] != tempData['PercSure']:
            #     continue  # Skip all catch trials
            vGamble = tempData['Amount']
            pGamble = 0.01 * tempData['PercGamb']
            vSure = tempData['Amount'] * tempData['PercSure'] * 0.01
            fGamble = calculateUtility2(vGamble, pGamble, vSure, alpha, delta, sigma, phi,_lambda=_lambda)
            uFinal = fGamble
            if tempData['Response'] == 2:
                ret += np.log(uFinal)
            elif tempData['Response']==1:
                ret += np.log(1 - uFinal)
        print(-ret)
        return -ret



    def errorFunc(self,playerID,alpha,delta,sigma,phi,a,neta,x,c,b):
        count=0
        count2=0
        for i in range(len(self.trainData[playerID].index)):
            tempData=self.trainData[playerID].iloc[i,:]
            vGamble=tempData['Amount']
            pGamble=0.01*tempData['PercGamb']
            vSure=tempData['Amount']*tempData['PercSure']*0.01

            fGamble=calculateUtility(vGamble,pGamble,vSure,alpha,delta,sigma,phi)
            # argGamble = [vGamble, pGamble, tempData['TP'], neta, tempData['Frame'], x, tempData['Needed'], c]
            # argSure = [vSure, 1, tempData['TP'], neta, tempData['Frame'], x, tempData['Needed'], c]

            argGamble = [vGamble, pGamble, tempData['TP'], neta, tempData['Frame'], x, tempData['Previous'], c,
                         tempData['Needed'], b]
            argSure = [vSure, 1, tempData['TP'], neta, tempData['Frame'], x, tempData['Previous'], c,
                       tempData['Needed'], b]
            # qGamble = calculateAttractionAdditive([timePressureTerm, framingEffect], fGamble, argGamble, argSure, a)
            qGamble = calculateAttractionMultiplicative(fGamble, argGamble, argSure, a)
            uFinal = fGamble + qGamble
            if uFinal>0.5:
                count2+=1
                if tempData['Response']==2:
                    count+=1
            else:
                if tempData['Response']==1:
                    count+=1
        print(count2)
        return count

    def indErrorFunc(self, params, playerID):



        params=np.append(params,np.zeros(9-len(params)))

        alpha, delta, sigma, phi, a, neta, x, c, b = params
        regulize=np.sum(np.abs(neta)+np.abs(x)+np.abs(c)+np.abs(b))
        print(params)
        self.count += 1
        print('Numer of iters:{}'.format(self.count))
        print('PlayerID:{}'.format(playerID))
        ret = 0

        playerData = self.trainData[playerID]


        for j in range(len(self.trainData[playerID].index)):
            tempData = self.trainData[playerID].iloc[j, :]
            vGamble = tempData['Amount']
            pGamble = 0.01 * tempData['PercGamb']
            vSure = tempData['Amount'] * tempData['PercSure'] * 0.01
            # if tempData['PercGamb'] != tempData['PercSure']:
            #     continue  # Skip all catch trials

            fGamble = calculateUtility(vGamble, pGamble, vSure, alpha, delta, sigma, phi)
            # argGamble = [vGamble, pGamble, tempData['TP'], neta, tempData['Frame'], x, tempData['Needed'], c]
            # argSure = [vSure, 1, tempData['TP'], neta, tempData['Frame'], x, tempData['Needed'], c]

            argGamble = [vGamble, pGamble, tempData['TP'], neta, tempData['Frame'], x, tempData['Previous'], c,
                         tempData['Needed'], b]
            argSure = [vSure, 1, tempData['TP'], neta, tempData['Frame'], x, tempData['Previous'], c,
                       tempData['Needed'], b]
            # qGamble = calculateAttractionAdditive([timePressureTerm, framingEffect], fGamble, argGamble, argSure, a)
            qGamble = calculateAttractionMultiplicative(fGamble, argGamble, argSure, a)

            uFinal = fGamble + qGamble
            if tempData['Response'] == 2:
                ret += np.log(uFinal)
            else:
                ret += np.log(1 - uFinal)
        print(-ret)
        print (ret-regulize)
        return -ret+regulize



    def totalErrorFunc(self,alpha, delta, sigma, phi,a,neta,x,c,b):
        self.count+=1
        print('Numer of iters:{}'.format(self.count))
        ret=0
        for i in range(self.numOfPlayers):
            playerData=self.trainData[i]
            for j in range(self.gpp*2):
                tempData=playerData.iloc[j,:]
                if tempData['PercGamb']!=tempData['PercSure']:
                    continue  #Skip all catch trials
                vGamble = tempData['Amount']
                pGamble = 0.01 * tempData['PercGamb']
                vSure = tempData['Amount'] * tempData['PercSure']*0.01
                fGamble=calculateUtility(vGamble,pGamble,vSure,alpha,delta,sigma,phi)
                # argGamble = [vGamble, pGamble, tempData['TP'], neta, tempData['Frame'], x, tempData['Needed'], c]
                # argSure = [vSure, 1, tempData['TP'], neta, tempData['Frame'], x, tempData['Needed'], c]
                argGamble = [vGamble, pGamble, tempData['TP'], neta, tempData['Frame'], x, tempData['Previous'], c,
                             tempData['Needed'], b]
                argSure = [vSure, 1, tempData['TP'], neta, tempData['Frame'], x, tempData['Previous'], c,
                           tempData['Needed'], b]
                # qGamble = calculateAttractionAdditive([timePressureTerm, framingEffect], fGamble, argGamble, argSure, a)
                qGamble = calculateAttractionMultiplicative(fGamble, argGamble, argSure, a)
                uFinal=fGamble+qGamble
                if tempData['Response']==2:
                    ret+=np.log(uFinal)
                else:
                    ret+=np.log(1-uFinal)
        print(-ret)
        print('After regularize:{}'.format(-ret + self.regularizerRational(alpha, delta, sigma, phi)))
        return -ret + self.regularizerRational(alpha, delta, sigma, phi)


    def totalErrorFunc2(self,alpha, delta, sigma, phi):
        self.count+=1
        print('Numer of iters:{}'.format(self.count))
        ret=0
        for i in range(self.numOfPlayers):
            playerData=self.trainData[i]
            for j in range(self.gpp*2):
                tempData=playerData.iloc[j,:]
                if tempData['PercGamb']!=tempData['PercSure']:
                    continue  #Skip all catch trials
                vGamble = tempData['Amount']
                pGamble = 0.01 * tempData['PercGamb']
                vSure = tempData['Amount'] * tempData['PercSure']*0.01
                fGamble=calculateUtility(vGamble,pGamble,vSure,alpha,delta,sigma,phi)
                # argGamble=[vGamble,pGamble,tempData['TP'],neta,tempData['Frame'],x]
                # argSure=[vSure,1,tempData['TP'],neta,tempData['Frame'],x]
                # qGamble=calculateAttractionAdditive([timePressureTerm,framingEffect],fGamble,argGamble,argSure,a)
                #print('vGamble:{}, pGamble:{}, Ut score:{}'.format(vGamble, pGamble,  fGamble))
                uFinal=fGamble
                if tempData['Response']==2:
                    ret+=np.log(uFinal)
                else:
                    ret+=np.log(1-uFinal)
        print(-ret)
        print('After regularize:{}'.format(-ret + self.regularizerRational(alpha, delta, sigma, phi)))
        return -ret + self.regularizerRational(alpha, delta, sigma, phi)


    def indErrorFuncAttract(self,params,playerID):
        a, neta, x,c,b=params
        alpha,delta,sigma,phi=self.generalParams

        self.count += 1
        print('Numer of iters:{}'.format(self.count))
        print('PlayerID:{}'.format(playerID))
        ret = 0

        playerData = self.trainData[playerID]
        for j in range(self.gpp * 2):
            tempData = playerData.iloc[j, :]
            if tempData['PercGamb'] != tempData['PercSure']:
                continue  # Skip all catch trials
            vGamble = tempData['Amount']
            pGamble = 0.01 * tempData['PercGamb']
            vSure = tempData['Amount'] * tempData['PercSure'] * 0.01
            fGamble = calculateUtility(vGamble, pGamble, vSure, alpha, delta, sigma, phi)
            # argGamble = [vGamble, pGamble, tempData['TP'], neta, tempData['Frame'], x,tempData['Needed'],c]
            # argSure = [vSure, 1, tempData['TP'], neta, tempData['Frame'], x,tempData['Needed'],c]

            argGamble = [vGamble, pGamble, tempData['TP'], neta, tempData['Frame'], x, tempData['Previous'], c,
                         tempData['Needed'], b]
            argSure = [vSure, 1, tempData['TP'], neta, tempData['Frame'], x, tempData['Previous'], c,
                       tempData['Needed'], b]
            #qGamble = calculateAttractionAdditive([timePressureTerm, framingEffect], fGamble, argGamble, argSure, a)
            qGamble=calculateAttractionMultiplicative(fGamble,argGamble,argSure,a)
            #print('vGamble:{}, pGamble:{},TP:{},frame:{}, attract score:{}'.format(vGamble, pGamble, tempData['TP'],
                                                                                  # tempData['Frame'], qGamble))
            uFinal = fGamble + qGamble
            if tempData['Response'] == 2:
                ret += np.log(uFinal)
            else:
                ret += np.log(1 - uFinal)
        print(-ret)
        print('After regularize:{}'.format(-ret + self.regularizerIrational(params)))
        return -ret + self.regularizerIrational(params)




    def dummy(self,args):
        return self.totalErrorFunc(*args)

    def dummy2(self,args):
        return self.totalErrorFunc2(*args)

    def totalErrorAttrac(self,args):
        a,neta,x=args
        alpha,delta,sigma,phi=self.generalParams
        self.count += 1
        print('Numer of iters:{}'.format(self.count))
        ret = 0
        for i in range(self.numOfPlayers):
            playerData = self.trainData[i]
            for j in range(self.gpp * 2):
                tempData = playerData.iloc[j, :]
                if tempData['PercGamb'] != tempData['PercSure']:
                    continue  # Skip all catch trials
                vGamble = tempData['Amount']
                pGamble = 0.01 * tempData['PercGamb']
                vSure = tempData['Amount'] * tempData['PercSure'] * 0.01
                fGamble = calculateUtility(vGamble, pGamble, vSure, alpha, delta, sigma, phi)
                argGamble = [vGamble, pGamble, tempData['TP'], neta, tempData['Frame'], x]
                argSure = [vSure, 1, tempData['TP'], neta, tempData['Frame'], x]
                qGamble = calculateAttractionAdditive([timePressureTerm, framingEffect], fGamble, argGamble, argSure, a)
                uFinal = fGamble + qGamble
                if tempData['Response'] == 2:
                    ret += np.log(uFinal)
                else:
                    ret += np.log(1 - uFinal)
        print(-ret)
        return -ret








    def train(self):
        sol=minimize(self.dummy,np.array([0.25215286,  0.90386184, 0.50382084,  0.9815681 ,  0.89839473,
        0.22683087,  0.5]),method='Nelder-Mead',options={'maxiter':2000,'adaptive':True,'disp':True})
        print(sol)
        param=sol.x
        result = []
        trainResult = []
        for i in range(self.numOfPlayers):


            result.append(self.predict(i, *param))
            trainResult.append(self.errorFunc(i, *param) / 640)
        # print(self.generalParams)
        # print(self.indParams)
        print(result)

        print('Average is :{}'.format(np.sum(result) / 19))

        plot.hist(result)
        plot.savefig('Plots/7thTest_agg_mix.png', dpi=240)
        plot.hist(trainResult)
        plot.savefig('Plots/7thTestTrain_agg_mix.png', dpi=240)


        return param

    def trainRational(self):
        #sol=minimize(self.dummy2,np.array([0.25,0.9,0.5,1]),bounds=[(0,None),(0,None),(0,None),(0,None)],method='L-BFGS-B',options={'maxiter':2000})
        sol=minimize(self.dummy2,np.array([0.25,0.9,0.5,1]),method='Nelder-Mead',options={'maxiter': 5000,'disp':True,'adaptive':True})
        print(sol)
        params=sol.x
        self.generalParams=sol.x

    def trainRational2(self):
        """Train the first four parameters together with 4 individual parameters"""
        sol=minimize(self.dummy,np.array([0.25,0.9,0.5,1,0.8,0.2,0.2,1,0.01]),method='Nelder-Mead',options={'maxiter':5000,'disp':True,'adaptive':True})
        self.generalParams=sol.x[:4]


    def trainIrational(self):
        ret={}
        for i in range(self.numOfPlayers):

            sol=minimize(self.indErrorFuncAttract,np.array([0.8,0.2,0.2,0,0.01]), args=(i,),method='Nelder-Mead',options={'maxiter':2000,'disp':True})
            ret[i]=sol.x
        self.indParams=ret

        return ret

    def trainUtilityInd(self):
        ret={}
        for i in range(self.numOfPlayers):
            print('Now doing player ID:{}'.format(i))
            sol=minimize(self.individual_utility_only_target,np.array([0.1,0.5,0.2,0.5,0.5]),args=(i,), tol=1e-3,method='Nelder-Mead',options={'maxiter':3000,'disp':True})
            ret[i]=sol.x
        self.utility_ind=ret


    def trainInd(self):
        ret={}
        for i in range(self.numOfPlayers):
            self.count=0
            print('Doing number:{}'.format(i))
            sol = minimize(self.indErrorFunc, x0=np.array([0.25, 1, 0.5, 1,0.8,1,1,0,0]), args=(i,), method='Nelder-Mead',
                           tol=1e-6,
                           options={'maxiter': 3000, 'adaptive': True, 'disp': True})
            solution=sol.x
            for j in range(9-len(solution)):
                solution=np.append(solution,0)
            ret[i]=solution

        self.indParams=ret



    def predict2(self,filename,ifind=False):
        """Using two step fitting method to predict"""


        result=[]
        trainResult=[]
        count=np.array([0 for i in range(10)])
        countTrain=np.array([0 for i in range(10)])
        for i in range(self.numOfPlayers):
            if not ifind:
                param=[]
                for j in range(4):
                    param.append(self.generalParams[j])
                for j in range(5):
                    param.append(self.indParams[i][j])
            else:
                param=self.indParams[i]

            result.append(self.predict(i,*param))
            playerData=self.summarize(i,*param)
            countTrain+=self.summarizeTrain(i,*param)
            count+=playerData

            trainResult.append(self.errorFunc(i,*param)/520)
        print(self.generalParams)
        print(self.indParams)
        print(result)

        with open(filename + '.txt', 'a') as f:
            f.write('General Params: ')
            f.write(str(self.generalParams))
            f.write('\n')
            f.write('Ind Params: ')
            f.write(str(self.indParams))
            f.write('\n')
            f.write('Result: ')
            f.write(str(result))
            f.write('\n')
            f.write('Average: ')
            f.write(str(np.mean(result)))
            f.write('\n')
            f.write('Train Result: ')
            f.write(str(trainResult))
            f.write('\n')



        print('Average is :{}'.format(np.sum(result) / self.numOfPlayers))

        plot.hist(result)
        plot.xlabel('Prediction rate')
        plot.ylabel('Count')
        plot.savefig('Plots/test+'+filename+'.png', dpi=240)
        plot.hist(trainResult)
        plot.savefig('Plots/train+'+filename+'.png', dpi=240)
        plot.plot([0.1*i for i in range(10)],count,'-o')

        plot.plot([0.1*i for i in range(10)],countTrain,'-o')
        plot.legend(['test set','training set'])
        plot.savefig('Plots/count+'+filename+'.png',dpi=240)

        print(self.verification())
        with open(filename+'.txt','a') as f:
            f.write('Verification: ')
            f.write(str(self.verification()))
            f.write('\n')
            f.write('Train Verification: ')
            f.write(str(self.verificationTrain()))
            f.write('\n')


        print(self.verificationTrain())


    def predict2_utility_ind(self,filename):
        result = []
        trainResult = []
        count = np.array([0 for i in range(10)])
        countTrain = np.array([0 for i in range(10)])
        for i in range(self.numOfPlayers):
            param=self.utility_ind[i]

            result.append(self.predict_utility_ind(i, *param))
            playerData = self.summarize_utility_ind(i, *param)
            countTrain += self.summarizeTrain_utility_ind(i, *param)
            count += playerData




        with open(filename + '.txt', 'a') as f:
            f.write('General Params: ')
            f.write(str(self.generalParams))
            f.write('\n')
            f.write('Ind-prams')
            f.write(str(self.utility_ind))
            f.write('\n')

            f.write('Result: ')
            f.write(str(result))
            f.write('\n')
            f.write('Average: ')
            f.write(str(np.mean(result)))
            f.write('\n')


        print('Average is :{}'.format(np.sum(result) / self.numOfPlayers))

        plot.hist(result)
        plot.xlabel('Prediction rate')
        plot.ylabel('Count')
        plot.savefig('Plots/test+' + filename + '.png', dpi=240)
        plot.hist(trainResult)
        plot.savefig('Plots/train+' + filename + '.png', dpi=240)
        plot.plot([0.1 * i for i in range(10)], count, '-o')

        plot.plot([0.1 * i for i in range(10)], countTrain, '-o')
        plot.legend(['test set', 'training set'])
        plot.savefig('Plots/count+' + filename + '.png', dpi=240)

        print(self.verification())
        with open(filename + '.txt', 'a') as f:
            f.write('Verification: ')
            f.write(str(self.verification()))
            f.write('\n')
            f.write('Train Verification: ')
            f.write(str(self.verificationTrain()))
            f.write('\n')

        print(self.verificationTrain())

    def simulateCPT(self,playerId,alpha,delta,sigma,phi):
        testData=self.testData[playerId]
        trainData=self.trainData[playerId]
        countTotal=0
        countCorrect=0
        for i in range(len(testData)):
            tempData=testData.iloc[i,:]
            vGamble = tempData['Amount']
            pGamble = 0.01 * tempData['PercGamb']
            vSure = tempData['Amount'] * tempData['PercSure'] * 0.01

            # alpha,delta,sigma are still in exploration
            fGamble = calculateUtility(vGamble, pGamble, vSure, alpha, delta, sigma, phi)

            uFinal = fGamble
            #print('f: {},q: {}'.format(fGamble, qGamble))
            if tempData['Response']!=0:
                countTotal+=1
            if np.random.rand()>uFinal:
                #Simulated participant choose sure option
                if tempData['Response']==1:
                    countCorrect+=1


            else:
                #Simulated participant choose to gamble
                if tempData['Response']==2:
                    countCorrect+=1

        for i in range(len(trainData)):
            tempData = trainData.iloc[i, :]
            vGamble = tempData['Amount']
            pGamble = 0.01 * tempData['PercGamb']
            vSure = tempData['Amount'] * tempData['PercSure'] * 0.01

            # alpha,delta,sigma are still in exploration
            fGamble = calculateUtility(vGamble, pGamble, vSure, alpha, delta, sigma, phi)
            # print(fGamble)
            # argGamble = [vGamble, pGamble, tempData['TP'], neta, tempData['Frame'], x, tempData['Needed'], c]
            # argSure = [vSure, 1, tempData['TP'], neta, tempData['Frame'], x, tempData['Needed'], c]


            uFinal = fGamble
            #print('f: {},q: {}'.format(fGamble,qGamble))
            if tempData['Response']!=0:
                countTotal += 1
            if np.random.rand() > uFinal:
                # Simulated participant choose sure option
                if tempData['Response'] == 1:
                    countCorrect += 1


            else:
                # Simulated participant choose to gamble
                if tempData['Response'] == 2:
                    countCorrect += 1

        print('{} correct in total {} questions, playerId {}'.format(countCorrect,countTotal,playerId))
        return countCorrect,countTotal

    def simulateInd(self,playerId,alpha,delta,sigma,phi,a,neta,x,c,b):
        testData=self.testData[playerId]
        trainData=self.trainData[playerId]
        countTotal=0
        countCorrect=0
        for i in range(len(testData)):
            tempData=testData.iloc[i,:]
            vGamble = tempData['Amount']
            pGamble = 0.01 * tempData['PercGamb']
            vSure = tempData['Amount'] * tempData['PercSure'] * 0.01

            # alpha,delta,sigma are still in exploration
            fGamble = calculateUtility(vGamble, pGamble, vSure, alpha, delta, sigma, phi)
            # print(fGamble)
            # argGamble = [vGamble, pGamble, tempData['TP'], neta, tempData['Frame'], x, tempData['Needed'], c]
            # argSure = [vSure, 1, tempData['TP'], neta, tempData['Frame'], x, tempData['Needed'], c]

            argGamble = [vGamble, pGamble, tempData['TP'], neta, tempData['Frame'], x, tempData['Previous'], c,
                         tempData['Needed'], b]
            argSure = [vSure, 1, tempData['TP'], neta, tempData['Frame'], x, tempData['Previous'], c,
                       tempData['Needed'], b]
            # qGamble = calculateAttractionAdditive([timePressureTerm, framingEffect], fGamble, argGamble, argSure, a)
            qGamble = calculateAttractionMultiplicative(fGamble, argGamble, argSure, a)
            # print('vGamble:{}, pGamble:{},TP:{},frame:{}, attract score:{}'.format(vGamble,pGamble,tempData['TP'],tempData['Frame'],qGamble))
            uFinal = fGamble + qGamble
            #print('f: {},q: {}'.format(fGamble, qGamble))
            if tempData['Response']!=0:
                countTotal+=1
            if np.random.rand()>uFinal:
                #Simulated participant choose sure option
                if tempData['Response']==1:
                    countCorrect+=1


            else:
                #Simulated participant choose to gamble
                if tempData['Response']==2:
                    countCorrect+=1

        for i in range(len(trainData)):
            tempData = trainData.iloc[i, :]
            vGamble = tempData['Amount']
            pGamble = 0.01 * tempData['PercGamb']
            vSure = tempData['Amount'] * tempData['PercSure'] * 0.01

            # alpha,delta,sigma are still in exploration
            fGamble = calculateUtility(vGamble, pGamble, vSure, alpha, delta, sigma, phi)
            # print(fGamble)
            # argGamble = [vGamble, pGamble, tempData['TP'], neta, tempData['Frame'], x, tempData['Needed'], c]
            # argSure = [vSure, 1, tempData['TP'], neta, tempData['Frame'], x, tempData['Needed'], c]

            argGamble = [vGamble, pGamble, tempData['TP'], neta, tempData['Frame'], x, tempData['Previous'], c,
                         tempData['Needed'], b]
            argSure = [vSure, 1, tempData['TP'], neta, tempData['Frame'], x, tempData['Previous'], c,
                       tempData['Needed'], b]
            # qGamble = calculateAttractionAdditive([timePressureTerm, framingEffect], fGamble, argGamble, argSure, a)
            qGamble = calculateAttractionMultiplicative(fGamble, argGamble, argSure, a)
            # print('vGamble:{}, pGamble:{},TP:{},frame:{}, attract score:{}'.format(vGamble,pGamble,tempData['TP'],tempData['Frame'],qGamble))
            uFinal = fGamble + qGamble
            #print('f: {},q: {}'.format(fGamble,qGamble))
            if tempData['Response']!=0:
                countTotal += 1
            if np.random.rand() > uFinal:
                # Simulated participant choose sure option
                if tempData['Response'] == 1:
                    countCorrect += 1


            else:
                # Simulated participant choose to gamble
                if tempData['Response'] == 2:
                    countCorrect += 1

        print('{} correct in total {} questions, playerId {}'.format(countCorrect,countTotal,playerId))
        return countCorrect,countTotal






    def simulate(self,N,filename):
        '''Simulate the response using the fitted parameters for N times and return the summary'''
        ret=[0 for i in range(25)]
        ret_CPT=[0 for i in range(25)]
        for i in range(N):
            print(i)
            for playerId in range(self.numOfPlayers):
                # param = []
                # for j in range(4):
                #     param.append(self.generalParams[j])
                # for j in range(4):
                #     param.append(self.indParams[playerId][j])
                # param=self.indParams[playerId]
                # countCorrect,countTotal=self.simulateInd(playerId,*param)
                # ind=countCorrect/countTotal
                # ind=ind//0.04
                # if ind==25:
                #     ind=24
                # ind=int(ind)
                # ret[ind]+=1

                param = self.utility_ind[playerId]
                countCorrect, countTotal = self.simulateCPT(playerId, *param)
                ind = countCorrect / countTotal
                ind = ind // 0.04
                if ind == 25:
                    ind = 24
                ind = int(ind)
                ret_CPT[ind] += 1



        #print(ret)
        # with open('SimulationResult.txt','a') as f:
        #     f.write('CPT Simulation for {} simulations'.format(N))
        #     f.write('\n')
        #     f.write(str(ret_CPT))
        #     f.write('QDT Simulation for {} simulations'.format(N))
        #     f.write('\n')
        #     f.write(str(ret))
        #     f.close()

        with open('SimulationResultData1CPT.txt','a') as f:
            f.write('CPT Simulation for {} simulations'.format(N))
            f.write('\n')
            f.write(str(ret_CPT))


        # plot.plot([0.04*i for i in range(25)],ret,'-o')
        # plot.plot([0.04 * i for i in range(25)], ret_CPT, '-o')
        # plot.xlabel('Percent of games that have the same response')
        # plot.ylabel('Count of players')
        # plot.title('Simulation Results')
        # plot.legend(['CPT with logit choice function','QDT with complete attraction factors'])
        # plot.savefig(filename,dpi=240)







    def crossValidation(self,filename):
        for time in range(6):
            print(time)
            self.gameSummary = [[] for i in range(10)]
            self.gameSummaryTrain = [[] for i in range(10)]
            #Form the training set and test set
            for i in range(self.numOfPlayers):
                # self.trainData[i]=pd.concat([element for index,element in enumerate(self.dataBlocks[i]) if index!=time])
                # self.testData[i]=self.dataBlocks[i][time]
                self.testData[i] = pd.concat(
                    [element for index, element in enumerate(self.dataBlocks[i]) if index != time])
                self.trainData[i]=self.dataBlocks[i][time]

                # print('Train, time{}'.format(time))
                # print(self.trainData[i])
                # print('Test,time{}'.format(time))
                # print(self.testData[i])
            # self.trainRational2()
            # self.count=0
            # self.trainIrational()
            self.trainInd()
            self.count=0
            self.predict2(str(time)+'__'+filename,ifind=True)
            # self.trainUtilityInd()
            # self.predict2_utility_ind(str(time) + '_special__' + filename)

    def crossValidation_utility_ind(self,filename):
        for time in range(6):
            print(time)
            self.gameSummary = [[] for i in range(10)]
            self.gameSummaryTrain = [[] for i in range(10)]
            #Form the training set and test set
            for i in range(self.numOfPlayers):
                self.trainData[i]=pd.concat([element for index,element in enumerate(self.dataBlocks[i]) if index!=time])
                self.testData[i]=self.dataBlocks[i][time]
                # self.testData[i] = pd.concat(
                #     [element for index, element in enumerate(self.dataBlocks[i]) if index != time])
                # self.trainData[i] = self.dataBlocks[i][time]
                # print('Train, time{}'.format(time))
                # print(self.trainData[i])
                # print('Test,time{}'.format(time))
                # print(self.testData[i])
            self.trainUtilityInd()
            self.predict2_utility_ind(str(time)+'__'+filename)
































def main():


    # qdt1=Dataset1("Exp1_DATA.xlsx",320)
    # # # print(qdt1.trainData)
    # # result=[]
    # # trainResult=[]
    # #
    # #
    # #
    # # #print(qdt1.trainData)
    # qdt1.trainRational2()
    # qdt1.trainIrational()
    # qdt1.predict2()
    # for i in range(19):
    #     result.append(qdt1.predict(i))
    #     trainResult.append(qdt1.errorFunc(i,0.85,0.7,1,1)/640)
    #     print(trainResult[i])
    # print('Average is :{}'.format(np.sum(result)/19))
    #print(qdt1.totalErrorFunc(0.5,0.5,0.5,0.5,1,0.05,0.4))
   #params=qdt1.train()
    # params=[0.03148612, 0.85751865, -1.51403294, 1.77121744, 3.31848782,
    #  0.65413045, 0.00475861]
    # print(params)
    # for i in range(19):
    #     result.append(qdt1.predict(i,*params))
    #     trainResult.append(qdt1.errorFunc(i,*params)/640)
    #     print(trainResult[i])
    # print('Average is :{}'.format(np.sum(result)/19))
    #
    #
    #
    #
    # plot.hist(result)
    # plot.savefig('Plots/1stTest.png',dpi=240)
    # plot.hist(trainResult)
    # plot.savefig('Plots/1stTestTrain.png', dpi=240)



    # qdt2=Dataset2("Exp2_DATA.xlsx",104,0,1)
    # qdt2.crossValidation_utility_ind("ReferencePointatExpectedValue_RandomSeed1337_sixfold_fixedPredict_withCatch_CPT_dataset2")
#     # qdt2.crossValidation('Dataset2_Rational1_None_reg{}_reg{}'.format(qdt2.reg1,qdt2.reg2))
    qdt1 = Dataset1("Exp1_DATA.xlsx", 320, 0, 1)
    qdt1.crossValidation_utility_ind("Dataset1_CPT_with_lambda_Randomseed1337_sixfold_tol1e-3_0.5startLambda")
    # qdt1.utility_ind = {0: array([6.13601388e-02, 1.04620214e-01, 5.12644339e-01, 1.45725138e+03]), 1: array([2.32504899e-01, 7.52589940e+00, 2.73386706e-02, 4.67448043e+03]), 2: array([0.18255185, 0.50937495, 0.6597763 , 6.03689051]), 3: array([0.4184124 , 0.52498067, 2.39901567, 1.77285255]), 4: array([0.14012791, 0.12840884, 3.2416526 , 7.3416733 ]), 5: array([1.77884658e-04, 6.34650938e-04, 4.92242877e-01, 4.48243900e+03]), 6: array([9.61304501e-05, 5.39515688e-04, 4.54337002e-01, 1.31934485e+04]), 7: array([ 0.05612542,  0.24403695,  0.56527905, 15.23021082]), 8: array([5.24463681e-03, 2.42724992e-03, 9.06400924e-01, 4.14417715e+02]), 9: array([5.72557551e-05, 4.34953428e-05, 2.56433554e+00, 2.52109786e+04]), 10: array([4.24409250e-01, 7.31603861e-07, 5.54410355e+01, 1.08100860e+00]), 11: array([6.98369174e-04, 2.15020445e-03, 4.60865090e-01, 1.73458981e+03]), 12: array([0.54236624, 0.97651229, 0.49577324, 0.8430122 ]), 13: array([0.98112432, 1.6121074 , 0.6086437 , 0.64354593]), 14: array([1.33334075e+01, 1.33296164e+01, 1.00043926e+00, 9.09181000e-15]), 15: array([0.18416638, 2.36447615, 3.82296886, 1.5643205 ]), 16: array([8.65750505e-05, 3.84559579e-04, 4.57065886e-01, 1.15389240e+04]), 17: array([0.1083976 , 0.27622611, 0.52150767, 9.74485318]), 18: array([0.69373393, 2.27520589, 3.25936457, 0.27908973])}
    # qdt1.indParams = {0: array([ 1.00679983e-01,  1.03245764e-01,  1.08557649e+00,  5.68272983e+02,
    #     2.89022589e+00,  8.11591887e-01,  1.88628682e-05,  7.55029768e-02,
    #    -4.11289098e-05]), 1: array([ 0.37914463,  0.24825622,  2.3356533 ,  2.00880837,  0.60527985,
    #     0.13428088,  0.06935895, -0.06887621, -0.0025363 ]), 2: array([ 7.44246949e-01,  2.00022855e+00,  7.50690801e+00,  1.63764745e-01,
    #     5.33615658e-01, -7.90197163e-11,  1.77128219e-10, -5.14652637e-02,
    #    -6.37808078e-03]), 3: array([8.95686714e-01, 2.76064834e+00, 2.91413235e+00, 2.04445860e-01,
    #    1.73150635e+00, 5.27621686e-01, 6.93018854e-04, 2.13234008e-04,
    #    1.22398453e-03]), 4: array([ 5.31028442e-01,  7.18390922e-01,  3.08134608e+00,  6.46875843e-01,
    #     1.74053323e+00,  2.39151997e-01,  1.32471506e-09, -9.35532656e-03,
    #    -1.46196059e-03]), 5: array([-4.87019889e-02,  5.14642281e-03, -4.51205178e+01, -3.94695048e+00,
    #     7.59840222e-02,  2.60524775e-02,  4.38997232e-01,  1.03987407e+00,
    #    -4.48276660e-03]), 6: array([ 2.49413615e-01,  2.91081580e-01,  1.90257744e+00,  1.26727377e+01,
    #     1.00401588e+00,  1.55302962e-01,  1.39791058e-09, -9.63298751e-02,
    #    -5.20150933e-03]), 7: array([ 7.56669148e-01,  1.07822373e+01,  2.26593859e+01,  1.41047656e-01,
    #     1.26944994e+00,  7.09758280e-01,  9.27083957e-09, -4.19352355e-02,
    #    -1.14551083e-03]), 8: array([4.96361290e-01, 2.95357326e-01, 4.94994517e-01, 1.78170006e+00,
    #    7.08545351e-01, 2.70913389e-01, 9.36327680e-14, 5.99426983e-03,
    #    1.99713282e-03]), 9: array([ 5.19576060e-02,  4.78533892e-02,  1.98946473e+00,  4.25471402e+01,
    #     6.24358164e-01,  2.29590732e-01,  5.29129406e-02, -2.99576796e-01,
    #    -4.17596199e-03]), 10: array([ 6.03755796e-01,  4.93496008e-01,  1.65257373e-02,  8.25178982e-01,
    #     1.25616670e-04, -5.63094198e-10,  4.02749182e-11,  1.93948918e-03,
    #     2.96567654e-04]), 11: array([5.32097897e-01, 5.34470244e-01, 2.13056572e+00, 7.75216179e-01,
    #    6.24008279e-01, 4.18897927e-01, 1.27484012e-10, 1.09631561e-01,
    #    7.08128278e-04]), 12: array([ 3.86349683e-02,  1.98838022e-02,  2.48165652e+00,  5.12322820e+01,
    #     2.48417873e-02, -1.33515835e-01,  8.14610868e-01, -9.25228820e-05,
    #     1.72307450e-02]), 13: array([ 3.72704706e-01,  3.70121424e-01,  1.06995045e+00,  2.71069629e+01,
    #     8.37211225e-01,  4.18740925e-01,  1.16913082e-01, -5.33875899e-03,
    #    -6.09512927e-04]), 14: array([ -0.35237463,   0.05294633,  -2.40322543, -10.68340386,
    #      0.18591404,  -0.09737615,   0.43195224,   0.13266872,
    #      0.01726407]), 15: array([ 0.68793447,  2.90497167,  7.58080427,  0.3414454 ,  0.52554017,
    #     0.97612437,  0.47905917,  0.01278405, -0.01728836]), 16: array([ 2.41857411e-01,  3.09421585e-01,  2.18677245e+00,  4.46077144e+00,
    #     2.87642053e-01, -4.49935889e-02,  5.84450395e-04, -8.17994519e-03,
    #    -1.06311107e-02]), 17: array([-2.93309935e-01,  5.76586787e-02, -1.47350921e+01, -9.04526700e+00,
    #     1.79404139e+01,  2.87885668e+00,  2.22881249e-04, -1.56587706e-02,
    #     1.69493984e-02]), 18: array([ 1.02058418e+00,  4.10135762e+00,  1.28107673e+01,  8.91070983e-02,
    #     4.21325429e-02,  2.11017626e-01,  8.09580697e-01,  1.80567563e-04,
    #    -1.45882961e-02])}
    # # ret1,ret2=qdt1.statistic()
    # # plot.plot([-1+0.1 * i for i in range(20)], ret1, '-o')
    # #
    # # plot.plot([0.05 * i for i in range(20)], ret2, '-o')
    # # plot.ylabel('Number of game trials')
    # # plot.xlabel('Factor values')
    # #
    # # plot.legend(['Attraction factor', 'Utility factor'])
    # # plot.show()
    #
    # CPTret = [0 for i in range(10)]
    # QDTret = [0 for i in range(10)]
    # for i in range(qdt1.numOfPlayers):
    #     paramQDT = qdt1.indParams[i]
    #     paramCPT = qdt1.utility_ind[i]
    #     QDTret += qdt1.summarizeTrain(i, *paramQDT)
    #     QDTret += qdt1.summarize(i, *paramQDT)
    #     CPTret += qdt1.summarizeTrain_utility_ind(i, *paramCPT)
    #     CPTret += qdt1.summarize_utility_ind(i, *paramCPT)
    # plot.plot([0.1 * i for i in range(10)], CPTret, '-o')
    #
    # plot.plot([0.1 * i for i in range(10)], QDTret, '-o')
    # plot.ylabel('Number of game trials')
    # plot.xlabel('Option Probability')
    #
    # plot.legend(['CPT with logit choice function', 'QDT with time, frame,momory and need attraction factor'])
    # plot.show()

    # qdt1 = Dataset1("Exp1_DATA.xlsx", 320, 0, 1)
    # #qdt1.crossValidation("fewshot_Regulize_Randomseed1337_sixfold_fixedPredict_withcatch_Logit_Individual_Dataset1_Time_Frame_start(1,1,0,0)")
    # qdt1.crossValidation_utility_ind("ReferencePointatExpectedValue_Randomseed1337_sixfold_fixedPredict_withCatch_CPT_dataset1[0.1,0.5,0.2,0.5]")
    # qdt1.indParams={0: array([ 1.96594716e-01,  2.05112804e-01,  1.13785245e+00,  1.27377893e+02,
    #     6.96134003e-01,  4.41337455e-01,  2.38849700e-01,  1.34032538e-01,
    #    -2.98781140e-04]), 1: array([-1.45060186e-01,  4.32102640e-01, -9.84560217e+00, -2.32556431e+00,
    #     8.35034266e-01,  4.89132415e-02,  8.03021277e-02, -1.55250577e-01,
    #    -9.55648259e-04]), 2: array([ 4.44842858e-01,  5.49147189e-01,  2.35151851e+00,  1.36978299e+00,
    #     5.16707277e-01,  4.24723474e-02,  2.32328055e-11,  3.24766080e-01,
    #    -2.96861509e-03]), 3: array([9.54703070e-01, 2.61121003e+00, 2.66417112e+00, 1.77834089e-01,
    #    1.96968267e+00, 6.50030846e-01, 9.36213183e-11, 2.16620961e-02,
    #    6.34420027e-04]), 4: array([ 6.13382137e-01,  8.61859838e-01,  3.02337237e+00,  4.58237369e-01,
    #     2.31220479e+00,  2.77972258e-01,  8.97437444e-07,  9.81255948e-03,
    #    -1.05337422e-03]), 5: array([ 6.20469974e-01,  3.89024673e-01,  2.77084260e+00,  7.98715323e-01,
    #     4.01828457e-06,  3.01463705e+00,  3.87750572e+00, -7.30442495e+00,
    #    -1.23285109e-01]), 6: array([ 1.88108239e-01,  2.10184048e-01,  1.75244729e+00,  2.63666159e+01,
    #     6.46097195e-01,  8.95018518e-02,  1.57434712e-07, -1.94592913e-01,
    #    -2.85431435e-03]), 7: array([ 9.10273722e-01,  8.28649786e+00,  6.69267274e+00,  5.96585477e-02,
    #     9.00286144e-01,  4.22849526e-01,  8.76455426e-09, -8.67042234e-02,
    #    -1.00951212e-03]), 8: array([ 4.23428878e-01,  2.31355038e-01,  4.15881377e-01,  2.18008642e+00,
    #     6.58145563e-01,  2.10517838e-01,  9.25264034e-04, -1.20631540e-01,
    #     1.22174310e-03]), 9: array([ 1.13527330e-01,  1.08731589e-01,  1.73832618e+00,  1.94061156e+01,
    #     6.37656919e-01,  1.99370311e-01,  2.75942923e-06, -3.79140115e-01,
    #    -2.57098347e-03]), 10: array([ 6.17850372e-01,  5.12366564e-01,  7.52373058e-02,  7.69091351e-01,
    #     1.94391108e-01,  1.13063985e+00,  9.54910949e-05,  7.28010175e-01,
    #    -2.30417898e-03]), 11: array([ 3.53468771e-01,  3.81163548e-01,  2.17809712e+00,  1.95323862e+00,
    #     6.08227444e-01,  4.10126258e-01,  3.79785588e-13,  4.12310567e-02,
    #    -3.04088498e-05]), 12: array([ 6.54133126e-02,  2.88322463e-02,  2.70714876e+00,  2.61512193e+01,
    #     1.13325048e-02, -2.89105807e-01,  9.22992401e-01, -1.51340416e+01,
    #     5.36677283e-02]), 13: array([ 5.85188272e-01,  5.78605800e-01,  1.16301086e+00,  3.89111363e+00,
    #     1.20984542e+00,  3.90811934e-01,  3.08669056e-02,  2.92071505e-02,
    #    -4.90279771e-04]), 14: array([ 8.27541798e-02,  4.99184531e-02,  2.28759188e+00,  3.73463703e+01,
    #     2.65783527e-03, -1.49809477e-01,  1.49981299e+00,  1.78088830e+01,
    #     7.68128492e-01]), 15: array([ 7.10142245e-01,  2.32044586e+00,  5.98678884e+00,  3.44387855e-01,
    #     6.03578421e-01,  7.89279301e-01,  4.30678120e-01, -2.06394678e-03,
    #    -9.51863177e-03]), 16: array([ 1.95840554e-01,  2.27748955e-01,  2.03651279e+00,  7.15815255e+00,
    #     4.37806901e-01,  2.27366732e-01,  8.94739598e-05, -1.00740511e-01,
    #    -3.12066011e-03]), 17: array([4.55407762e-01, 2.54963306e-01, 2.47777391e+00, 1.73282246e+00,
    #    1.00087414e+00, 9.02473982e-01, 2.76345490e-06, 7.78762558e-02,
    #    1.13517018e-03]), 18: array([ 1.08617482e+00,  5.48745499e+00,  1.52210583e+01,  6.73749163e-02,
    #     1.08172519e-01,  3.43774598e-01,  6.17276859e-01,  7.02990652e-01,
    #    -5.24922461e-03])}
    #
    # #qdt1.utility_ind={0: array([2.45824786e-02, 2.48359792e-02, 1.03397919e+00, 7.62850075e+03]), 1: array([-0.04047522,  1.05687922, -3.49030691, -1.52008703]), 2: array([  0.38473996,   0.38314455,   0.95303418, -48.11280783]), 3: array([7.36518656e-01, 5.35952056e+04, 1.73276733e+01, 2.53981352e-01]), 4: array([0.46971166, 0.56000283, 2.58530128, 1.32419955]), 5: array([ 0.08005754,  0.44386933, -2.38153277, -2.42789177]), 6: array([-2.06623059e+00,  3.07272745e+03,  5.43069058e+01,  1.61421529e+02]), 7: array([ 0.69294395,  0.62797979,  0.54005192, -0.92871544]), 8: array([   0.32814472,    0.33090182,    1.00827679, -233.62296795]), 9: array([-5.23887025e-02,  1.20778456e-02, -4.71066237e+01, -1.26802966e+00]), 10: array([ 1.54179086e+00,  1.61662522e+01, -2.61364022e+01,  2.22930798e-03]), 11: array([ 4.35456297e-01,  1.27352255e-03, -4.82671958e+01, -1.87282126e-01]), 12: array([ -0.11552008,  17.95059303, -15.27234128,  -1.72009707]), 13: array([  0.5343025 ,   0.54186461,   0.93850386, -11.46969697]), 14: array([ -0.53491242,   3.09303656, -23.93233291,  -8.33208493]), 15: array([ 0.45577636,  0.45980309,  1.04562402, 83.47255893]), 16: array([  0.29704894,   0.29315169,   0.84527715, -24.41660861]), 17: array([-0.02675433,  0.39049002, -3.28770924, -2.79751744]), 18: array([ 1.92463766e+00,  2.17946996e+02, -3.19072138e+01, -8.80445377e-04])}
    # temp={0: array([5.70333834e-02, 5.70433377e-02, 1.00056437e+00, 1.71541746e+05,
    #    0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
    #    0.00000000e+00]), 1: array([-2.69832378e-01,  7.70763313e-04, -7.10617408e+01, -4.32935818e+00,
    #     0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
    #     0.00000000e+00]), 2: array([0.52205317, 0.55048231, 2.41645242, 0.98098548, 0.        ,
    #    0.        , 0.        , 0.        , 0.        ]), 3: array([6.93661999e-01, 6.93742817e-01, 1.00020727e+00, 3.41832997e+03,
    #    0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
    #    0.00000000e+00]), 4: array([0.32154311, 0.34822562, 2.61885862, 3.07270267, 0.        ,
    #    0.        , 0.        , 0.        , 0.        ]), 5: array([-1.09943396e-01,  6.00754000e-04, -6.71632151e+01, -5.04024090e+00,
    #     0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
    #     0.00000000e+00]), 6: array([0.31358332, 0.33415099, 1.92057107, 8.49600023, 0.        ,
    #    0.        , 0.        , 0.        , 0.        ]), 7: array([0.43665954, 0.48649095, 2.35222183, 1.42697757, 0.        ,
    #    0.        , 0.        , 0.        , 0.        ]), 8: array([-1.27890971e-04, -8.88539841e-05,  4.46185455e-01, -4.20136732e+04,
    #     0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
    #     0.00000000e+00]), 9: array([ 0.10185645,  0.08523387,  1.8401768 , 21.19217832,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ]), 10: array([0.43561332, 0.3531254 , 0.00951384, 1.97045938, 0.        ,
    #    0.        , 0.        , 0.        , 0.        ]), 11: array([0.26413182, 0.25164747, 2.05406192, 4.0695117 , 0.        ,
    #    0.        , 0.        , 0.        , 0.        ]), 12: array([  -0.51583178,   65.26705512, -185.54113203,   -6.81257148,
    #       0.        ,    0.        ,    0.        ,    0.        ,
    #       0.        ]), 13: array([0.34409527, 0.31920744, 1.42958341, 5.6521893 , 0.        ,
    #    0.        , 0.        , 0.        , 0.        ]), 14: array([-4.32005474e-01,  1.02133463e+04, -7.17517682e+03, -6.42298474e+00,
    #     0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
    #     0.00000000e+00]), 15: array([0.90371258, 3.24304081, 6.41959007, 0.14882108, 0.        ,
    #    0.        , 0.        , 0.        , 0.        ]), 16: array([0.41448221, 0.45161758, 2.49753826, 1.84900927, 0.        ,
    #    0.        , 0.        , 0.        , 0.        ]), 17: array([-1.10580048e-01,  5.33145477e-04, -6.91735619e+01, -3.92734889e+00,
    #     0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
    #     0.00000000e+00]), 18: array([ 1.11256034, 19.03064034, 29.85012585,  0.05869778,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ])}
    # for key in temp.keys():
    #     temp[key]=temp[key][:4]
    # qdt1.utility_ind=temp
    # qdt1.simulate(1000,'simulation_dataset1.png')

    # # print(qdt2.numOfPlayers)
    # qdt2.trainRational2()
    # # # qdt2.generalParams=[ 2.14753350e-01 ,3.19551437e-01 ,-7.88355884e-08 , 1.83747002e+00]
    # qdt2.trainIrational()
    # # # print(qdt2.indParams)
    # # # # # qdt2.generalParams=[ 2.61387791e-01, 4.04423629e-01,-7.51299188e-08 ,1.25835512e+00]
    # #qdt2.indParams={0: array([-4.15220477e-01,  1.31498674e-01,  5.19684351e-09,  1.00581864e-02]), 1: array([-0.18170567, -0.04147311,  0.68967094, -0.35413006]), 2: array([-3.04015791e-01, -6.50724254e-07,  3.49340416e-08,  1.32794571e-02]), 3: array([-0.50061763,  0.10390154,  0.2002814 , -0.01702621]), 4: array([-1.59127537e-01, -2.07350292e-06,  3.62904482e-01,  4.69559724e-03]), 5: array([-0.2432735 ,  0.01487879,  0.42672621,  0.09294259]), 6: array([-7.93078898e-02, -6.57195561e-08,  2.49974683e-01,  7.12959045e-03]), 7: array([-1.69263057e-01,  8.24457620e-02,  5.24902613e-07,  4.94702065e-02]), 8: array([6.35451346e-02, 3.66591978e-06, 2.04007341e-07, 5.14305091e-04]), 9: array([-7.28374748e-06,  1.02765855e-01,  1.97301829e-06, -2.99188273e-03]), 10: array([-1.68128908e-01, -5.71957047e-06,  2.32584621e-07,  5.81909232e-03]), 11: array([-2.52868841e-01, -7.81464789e-09,  3.54906569e-01,  2.39332259e-02]), 12: array([-2.21541182e-01, -1.05052581e-01,  5.99593471e-08, -7.95367733e-02]), 13: array([-4.20133267e-01,  2.12619195e-06,  2.57738808e-05,  2.73814936e-02]), 14: array([-0.13079653, -0.10516242,  0.36363788, -0.04142452]), 15: array([-9.77049023e-02,  7.87358579e-07,  8.81512594e-07,  2.62865608e-03]), 16: array([-2.61608844e-01, -3.04823505e-06,  1.75541162e-05, -6.48110937e-02]), 17: array([-0.48377681,  0.12766405,  0.21924518,  0.01240218]), 18: array([-2.81246059e-01,  1.49411533e-02,  2.36159567e-07, -5.38863475e-02]), 19: array([-2.12672829e-02,  9.26474414e-08,  6.98460061e-09,  3.10098321e-03]), 20: array([-2.18692681e-01, -9.80141719e-06,  1.81218274e-02,  8.11904715e-03]), 21: array([-2.04516557e-01, -1.36607101e-05,  6.79261227e-07, -1.98405126e-02]), 22: array([-3.16431027e-01,  2.15135378e-01,  1.20156444e-07, -4.98642551e-02]), 23: array([-1.92197694e-01,  4.62329611e-03,  2.94674522e-07,  1.39933634e-02]), 24: array([-3.96656692e-01, -3.12739890e-03,  4.76991013e-09,  1.37848605e-02]), 25: array([-2.03849951e-01, -8.51468705e-07,  4.05657432e-08,  6.38624383e-03]), 26: array([1.74098932e-06, 2.91302393e-02, 1.10378913e-05, 1.17045279e-02]), 27: array([-2.38943081e-02, -1.41779000e-06,  2.33148110e-01,  3.88393575e-03]), 28: array([-4.09758600e-01, -1.24907260e-06,  1.89791853e-07,  1.60977538e-02]), 29: array([-0.20783821,  0.04172771,  0.23525419, -0.03882265]), 30: array([-0.42982044,  0.08260816,  0.15096312,  0.04397272]), 31: array([-7.85542480e-01,  5.19316228e-01,  3.87131534e-10, -1.43005542e-02]), 32: array([-2.92362872e-01,  8.55799867e-07,  5.70213089e-07, -1.42175186e-02]), 33: array([-3.34341209e-01, -3.38870813e-08,  2.68776455e-02,  1.42295835e-02]), 34: array([-0.25579182, -0.33876335,  0.1408496 ,  0.070155  ]), 35: array([-1.21680559e-02, -4.54763491e-07,  1.47463038e-07, -3.85029647e-03]), 36: array([-1.08403661e-02,  1.04930380e-06,  9.76221829e-04, -1.51353350e-02]), 37: array([1.26140182e-01, 2.21822404e-08, 1.59459796e-02, 1.73599655e-02]), 38: array([-2.37630460e-01,  5.25817222e-09,  1.25681454e-01,  1.29982868e-02]), 39: array([-2.66254635e-01,  1.02618551e-06,  5.93511744e-03,  4.29878211e-03]), 40: array([-2.26760147e-01, -3.17995640e-07,  6.57303394e-07, -4.45846765e-03]), 41: array([ 1.10280013e-02,  1.15204079e-07,  9.57856966e-07, -1.07568132e-02]), 42: array([-1.06110655e-02,  1.01313979e-08,  2.33282637e-07,  6.19769810e-03]), 43: array([-2.85971688e-04,  1.34924099e-01,  1.27166162e-06,  2.06570294e-03]), 44: array([-0.39885578,  0.1878541 ,  0.10737545, -0.023809  ]), 45: array([-1.60568000e-01, -1.25584768e-06,  4.59607923e-07, -1.71039333e-02]), 46: array([-2.91494570e-01,  2.58473013e-07,  8.55311956e-06, -5.31445673e-02]), 47: array([-1.18952854e-01,  6.85130525e-08,  5.32919325e-07,  8.90707093e-03]), 48: array([-1.24578412e-01, -2.55982021e-06,  1.26487772e-02,  7.73366367e-03]), 49: array([-1.43295203e-01, -8.66654923e-02,  9.44392674e-08,  1.79155103e-02]), 50: array([-1.63670783e-02, -2.88919987e-06,  1.25473733e-03,  9.34602673e-03]), 51: array([-1.79973648e-01,  7.16168166e-06,  1.33987836e-02,  1.77521029e-02]), 52: array([-3.96317206e-01, -2.07477928e-08,  1.54400567e-05,  8.41052652e-03]), 53: array([-1.33649618e-01,  6.20509600e-08,  5.68526839e-06,  8.52168149e-03])}
    # # # #
    # # # #
    # # # #
    # qdt2.predict2('2_Rational1_D2_Time_Frame_Previous_reg{}_reg{}'.format(qdt2.reg1,qdt2.reg2))
    # plot.figure(2)
    # plot.plot([0.1*i for i in range(10)],qdt2.verification(),'-o')
    # #
    # #
    # #plot.plot([0, 0.2727272727272727, 0.38095238095238093, 0.4066390041493776, 0.4838709677419355, 0.5609756097560976, 0.5688342120679555, 0.7142857142857143, 0.8594104308390023, 0.9080459770114943],'r')
    # plot.plot([0.1*i for i in range(10)],[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],'-','r')
    # plot.xlabel('predicted choose proportion using different theories and attraction factors')
    # plot.plot([0.1*i for i in range(10)],[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],'-','r')
    # plot.ylabel('Real choose proportion ')
    # #
    # #
    # plot.title('Choose proportion comparison')
    # #plot.xlabel('QDT predicted choose proportion')
    # plot.legend(['QDT','upper boundary of the best interval','lower boundary of the best interval'])
    # plot.savefig('Plots/2_Rational1_Comparison_Dataset2_Time_Frame_Previous_reg{}_reg{}.png'.format(qdt2.reg1,qdt2.reg2),dpi=240)



    # qdt1 = Dataset1("Exp1_DATA.xlsx", 320, 0, 1)
    # # print(qdt2.numOfPlayers)
    # qdt1.trainRational()
    # # qdt2.generalParams=[ 2.14753350e-01 ,3.19551437e-01 ,-7.88355884e-08 , 1.83747002e+00]
    # #qdt1.trainIrational()
    # # print(qdt2.indParams)
    # # # # qdt2.generalParams=[ 2.61387791e-01, 4.04423629e-01,-7.51299188e-08 ,1.25835512e+00]
    # # # # qdt2.indParams={0: array([-4.15220477e-01,  1.31498674e-01,  5.19684351e-09,  1.00581864e-02]), 1: array([-0.18170567, -0.04147311,  0.68967094, -0.35413006]), 2: array([-3.04015791e-01, -6.50724254e-07,  3.49340416e-08,  1.32794571e-02]), 3: array([-0.50061763,  0.10390154,  0.2002814 , -0.01702621]), 4: array([-1.59127537e-01, -2.07350292e-06,  3.62904482e-01,  4.69559724e-03]), 5: array([-0.2432735 ,  0.01487879,  0.42672621,  0.09294259]), 6: array([-7.93078898e-02, -6.57195561e-08,  2.49974683e-01,  7.12959045e-03]), 7: array([-1.69263057e-01,  8.24457620e-02,  5.24902613e-07,  4.94702065e-02]), 8: array([6.35451346e-02, 3.66591978e-06, 2.04007341e-07, 5.14305091e-04]), 9: array([-7.28374748e-06,  1.02765855e-01,  1.97301829e-06, -2.99188273e-03]), 10: array([-1.68128908e-01, -5.71957047e-06,  2.32584621e-07,  5.81909232e-03]), 11: array([-2.52868841e-01, -7.81464789e-09,  3.54906569e-01,  2.39332259e-02]), 12: array([-2.21541182e-01, -1.05052581e-01,  5.99593471e-08, -7.95367733e-02]), 13: array([-4.20133267e-01,  2.12619195e-06,  2.57738808e-05,  2.73814936e-02]), 14: array([-0.13079653, -0.10516242,  0.36363788, -0.04142452]), 15: array([-9.77049023e-02,  7.87358579e-07,  8.81512594e-07,  2.62865608e-03]), 16: array([-2.61608844e-01, -3.04823505e-06,  1.75541162e-05, -6.48110937e-02]), 17: array([-0.48377681,  0.12766405,  0.21924518,  0.01240218]), 18: array([-2.81246059e-01,  1.49411533e-02,  2.36159567e-07, -5.38863475e-02]), 19: array([-2.12672829e-02,  9.26474414e-08,  6.98460061e-09,  3.10098321e-03]), 20: array([-2.18692681e-01, -9.80141719e-06,  1.81218274e-02,  8.11904715e-03]), 21: array([-2.04516557e-01, -1.36607101e-05,  6.79261227e-07, -1.98405126e-02]), 22: array([-3.16431027e-01,  2.15135378e-01,  1.20156444e-07, -4.98642551e-02]), 23: array([-1.92197694e-01,  4.62329611e-03,  2.94674522e-07,  1.39933634e-02]), 24: array([-3.96656692e-01, -3.12739890e-03,  4.76991013e-09,  1.37848605e-02]), 25: array([-2.03849951e-01, -8.51468705e-07,  4.05657432e-08,  6.38624383e-03]), 26: array([1.74098932e-06, 2.91302393e-02, 1.10378913e-05, 1.17045279e-02]), 27: array([-2.38943081e-02, -1.41779000e-06,  2.33148110e-01,  3.88393575e-03]), 28: array([-4.09758600e-01, -1.24907260e-06,  1.89791853e-07,  1.60977538e-02]), 29: array([-0.20783821,  0.04172771,  0.23525419, -0.03882265]), 30: array([-0.42982044,  0.08260816,  0.15096312,  0.04397272]), 31: array([-7.85542480e-01,  5.19316228e-01,  3.87131534e-10, -1.43005542e-02]), 32: array([-2.92362872e-01,  8.55799867e-07,  5.70213089e-07, -1.42175186e-02]), 33: array([-3.34341209e-01, -3.38870813e-08,  2.68776455e-02,  1.42295835e-02]), 34: array([-0.25579182, -0.33876335,  0.1408496 ,  0.070155  ]), 35: array([-1.21680559e-02, -4.54763491e-07,  1.47463038e-07, -3.85029647e-03]), 36: array([-1.08403661e-02,  1.04930380e-06,  9.76221829e-04, -1.51353350e-02]), 37: array([1.26140182e-01, 2.21822404e-08, 1.59459796e-02, 1.73599655e-02]), 38: array([-2.37630460e-01,  5.25817222e-09,  1.25681454e-01,  1.29982868e-02]), 39: array([-2.66254635e-01,  1.02618551e-06,  5.93511744e-03,  4.29878211e-03]), 40: array([-2.26760147e-01, -3.17995640e-07,  6.57303394e-07, -4.45846765e-03]), 41: array([ 1.10280013e-02,  1.15204079e-07,  9.57856966e-07, -1.07568132e-02]), 42: array([-1.06110655e-02,  1.01313979e-08,  2.33282637e-07,  6.19769810e-03]), 43: array([-2.85971688e-04,  1.34924099e-01,  1.27166162e-06,  2.06570294e-03]), 44: array([-0.39885578,  0.1878541 ,  0.10737545, -0.023809  ]), 45: array([-1.60568000e-01, -1.25584768e-06,  4.59607923e-07, -1.71039333e-02]), 46: array([-2.91494570e-01,  2.58473013e-07,  8.55311956e-06, -5.31445673e-02]), 47: array([-1.18952854e-01,  6.85130525e-08,  5.32919325e-07,  8.90707093e-03]), 48: array([-1.24578412e-01, -2.55982021e-06,  1.26487772e-02,  7.73366367e-03]), 49: array([-1.43295203e-01, -8.66654923e-02,  9.44392674e-08,  1.79155103e-02]), 50: array([-1.63670783e-02, -2.88919987e-06,  1.25473733e-03,  9.34602673e-03]), 51: array([-1.79973648e-01,  7.16168166e-06,  1.33987836e-02,  1.77521029e-02]), 52: array([-3.96317206e-01, -2.07477928e-08,  1.54400567e-05,  8.41052652e-03]), 53: array([-1.33649618e-01,  6.20509600e-08,  5.68526839e-06,  8.52168149e-03])}
    # # #
    # qdt1.indParams={0: array([-3.02462699e-01,  8.56065091e-02,  1.00364962e-07,  9.17585399e-03]), 1: array([-7.28813364e-01,  3.45098041e-01,  1.48617615e-01, -3.05247403e-08]), 2: array([-6.06029720e-01,  3.50119894e-01,  1.74591085e-08, -4.12980705e-04]), 3: array([-6.27266489e-01,  4.55078433e-01,  5.93841384e-09, -8.73707590e-04]), 4: array([-1.38896399e+00,  3.37749497e-01,  2.32233455e-10,  2.47288516e-02]), 5: array([-2.70666145e-01,  4.56792845e-01,  4.98915559e-05,  1.07030882e-07]), 6: array([-2.15139547e-01,  1.26903246e-01,  1.11973447e-01, -5.90107279e-08]), 7: array([-5.48328977e-01,  3.94575944e-01,  9.87561434e-09, -1.98171231e-03]), 8: array([-3.65692667e-01,  2.80582847e-01,  3.29267128e-08,  2.56176066e-03]), 9: array([-8.42219450e-01,  2.62879894e-01,  7.23802285e-09,  8.04778060e-03]), 10: array([-8.92191050e-04,  4.00626817e-05,  1.53967119e-02,  1.88230969e-02]), 11: array([-6.66409091e-01,  6.28096245e-01,  2.19109946e-08,  6.53365451e-04]), 12: array([-0.12823869,  0.11462407,  0.43435027,  0.00080351]), 13: array([-7.48848294e-01,  3.63373111e-01,  1.24449164e-01, -2.77353178e-07]), 14: array([-5.37782407e-02,  1.12847882e-09,  4.03417090e-01,  1.25577637e-03]), 15: array([-9.61956760e-02, -1.28704310e-06,  6.35669922e-07,  2.65276245e-03]), 16: array([-4.53698053e-01,  1.92878060e-01,  2.21141995e-09,  1.24986852e-02]), 17: array([-3.81699334e-01,  7.34845718e-01,  1.11826770e-01, -1.75441236e-08]), 18: array([-5.70717450e-02, -5.91912654e-07,  3.60424082e-01,  2.08805345e-02])}
    # # #
    # # #
    # qdt1.predict2('Fixed_Dataset1_None_reg{}_reg{}'.format(qdt1.reg1, qdt1.reg2))
    # plot.figure(2)
    # plot.plot([0.1*i for i in range(10)],qdt1.verification(),'-o')
    #
    # # plot.plot([0, 0.2727272727272727, 0.38095238095238093, 0.4066390041493776, 0.4838709677419355, 0.5609756097560976, 0.5688342120679555, 0.7142857142857143, 0.8594104308390023, 0.9080459770114943],'r')
    # plot.plot([0.1 * i for i in range(10)], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], '-', 'r')
    # plot.xlabel('predicted choose proportion using different theories and attraction factors')
    # plot.plot([0.1*i for i in range(10)],[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],'-','r')
    # plot.ylabel('Real choose proportion ')
    # #
    # #
    # plot.title('Choose proportion comparison')
    # #plot.xlabel('QDT predicted choose proportion')
    # plot.legend(['QDT','upper boundary of the best interval','lower boundary of the best interval'])
    # plot.savefig('Plots/Fixed_Comparison_Dataset1_None_reg{}_reg{}.png'.format(qdt1.reg1, qdt1.reg2), dpi=240)



    #Simulate Dataset 1
    # qdt1 = Dataset1("Exp1_DATA.xlsx", 320, 5, 1)
    # qdt1.generalParams=[ 0.32553902,  0.32687957 ,-0.00895134 , 1.7692951 ]
    # #qdt1.indParams={0: array([-2.86305055e-01,  7.83024472e-02,  1.45954097e-09,  6.02308839e-03]), 1: array([-7.54898361e-01,  3.30290113e-01,  1.13048218e-01, -6.68801707e-08]), 2: array([-5.90333014e-01,  3.66226719e-01,  7.22661384e-09,  5.13376369e-03]), 3: array([-6.09657761e-01,  4.56602200e-01,  6.34081447e-09, -6.12884824e-03]), 4: array([-1.29007181e+00,  3.37346837e-01,  6.40317524e-09,  2.46445655e-02]), 5: array([-2.45207491e-01,  4.23761854e-01,  5.44480589e-08, -2.91030565e-03]), 6: array([-2.04533528e-01,  1.19807082e-01,  1.06460584e-01, -4.81656283e-08]), 7: array([-5.36894671e-01,  3.99676449e-01,  8.20556615e-10,  9.79417046e-04]), 8: array([-3.93415502e-01,  3.53032697e-01,  1.70915103e-08,  2.96413413e-03]), 9: array([-8.16528595e-01,  2.62523388e-01,  2.40247049e-09,  8.56689388e-03]), 10: array([-1.45679317e-04,  7.94011585e-06,  5.12022007e-02,  8.16800411e-03]), 11: array([-6.38171950e-01,  6.21147723e-01,  1.69463326e-07,  2.47587694e-02]), 12: array([-0.12719159,  0.11234659,  0.420096  ,  0.004938  ]), 13: array([-7.73767151e-01,  3.75472029e-01,  1.08086663e-01,  7.13437357e-08]), 14: array([-5.35736674e-02, -4.79948988e-08,  3.80204061e-01,  4.71525465e-03]), 15: array([-9.52285949e-02,  8.03317233e-03,  3.47365938e-07,  9.40079476e-03]), 16: array([-4.43437040e-01,  2.01397444e-01,  2.35835435e-08,  7.43051725e-03]), 17: array([-0.38728381,  0.71791172,  0.08631366, -0.00766776]), 18: array([-5.61466589e-02,  2.29052299e-07,  3.54123467e-01,  9.33762245e-03])}
    # qdt1.indParams={0: array([-6.44508922e-03,  3.55793935e-05,  2.26651515e-02,  1.76229436e-02]), 1: array([-6.44508922e-03,  3.55793935e-05,  2.26651515e-02,  1.76229436e-02]), 2: array([-6.44508922e-03,  3.55793935e-05,  2.26651515e-02,  1.76229436e-02]), 3: array([-6.44508922e-03,  3.55793935e-05,  2.26651515e-02,  1.76229436e-02]), 4: array([-6.44508922e-03,  3.55793935e-05,  2.26651515e-02,  1.76229436e-02]), 5: array([-6.44508922e-03,  3.55793935e-05,  2.26651515e-02,  1.76229436e-02]), 6: array([-6.44508922e-03,  3.55793935e-05,  2.26651515e-02,  1.76229436e-02]), 7: array([-6.44508922e-03,  3.55793935e-05,  2.26651515e-02,  1.76229436e-02]), 8: array([-6.44508922e-03,  3.55793935e-05,  2.26651515e-02,  1.76229436e-02]), 9: array([-6.44508922e-03,  3.55793935e-05,  2.26651515e-02,  1.76229436e-02]), 10: array([-6.44508922e-03,  3.55793935e-05,  2.26651515e-02,  1.76229436e-02]), 11: array([-6.44508922e-03,  3.55793935e-05,  2.26651515e-02,  1.76229436e-02]), 12: array([-6.44508922e-03,  3.55793935e-05,  2.26651515e-02,  1.76229436e-02]), 13: array([-6.44508922e-03,  3.55793935e-05,  2.26651515e-02,  1.76229436e-02]), 14: array([-6.44508922e-03,  3.55793935e-05,  2.26651515e-02,  1.76229436e-02]), 15: array([-6.44508922e-03,  3.55793935e-05,  2.26651515e-02,  1.76229436e-02]), 16: array([-6.44508922e-03,  3.55793935e-05,  2.26651515e-02,  1.76229436e-02]), 17: array([-6.44508922e-03,  3.55793935e-05,  2.26651515e-02,  1.76229436e-02]), 18: array([-6.44508922e-03,  3.55793935e-05,  2.26651515e-02,  1.76229436e-02])}
    #
    # qdt1.simulate(1000,'Plots/Simulation/Dataset1_None_reg5_1.png')

    #Simulate Dateset 2

    # qdt2 = Dataset2("Exp2_DATA.xlsx", 104, 5, 1)
    # qdt2.generalParams=[ 0.17354242 , 0.2545581 , -0.19949276 , 1.86293286]
    # qdt2.indParams={0: array([-3.79386132e-01,  4.22020339e-01,  4.91026691e-08,  8.98537030e-03]), 1: array([-1.86235228e-01, -3.37806717e-01,  2.03259827e-01,  1.16814851e-06]), 2: array([-1.58314814e-01,  2.64408765e-06,  3.54704355e-03,  1.07726425e-02]), 3: array([-9.31500705e-01,  3.58937220e-01,  1.59860565e-01, -1.59824664e-08]), 4: array([-4.93999211e-01, -2.49104079e-02,  8.87945543e-08,  1.81602736e-02]), 5: array([-5.91825333e-01, -3.89339151e-02,  7.42202079e-07,  1.12330216e-02]), 6: array([-2.32195449e-01,  1.85514002e-01,  4.82657950e-07, -1.39362015e-07]), 7: array([-1.08451504e-01,  8.66615982e-08,  1.55025709e-02,  7.73251894e-03]), 8: array([4.95680093e-02, 1.33810078e-02, 3.79868016e-01, 3.99971715e-09]), 9: array([-1.44428380e-01,  4.50471874e-01,  7.37385004e-08, -6.16301603e-03]), 10: array([-3.28118363e-01,  3.07806349e-01,  2.56258947e-07,  1.27590689e-03]), 11: array([-0.20262041,  0.10332832,  0.47876779, -0.00940076]), 12: array([-6.84558892e-02, -2.61202516e-01,  9.15711443e-02,  3.41442652e-08]), 13: array([-5.18102192e-01,  1.37011125e-01,  7.28871117e-09,  8.59281735e-03]), 14: array([-0.02506745, -0.1448784 ,  0.80355969, -0.00614695]), 15: array([-3.36804278e-02, -5.62815710e-06,  1.39353386e-05,  8.62730473e-03]), 16: array([-4.49680074e-01,  1.19998726e-01,  6.11085396e-09,  1.25174794e-02]), 17: array([-1.56198772,  0.4687666 ,  0.01455832,  0.02018042]), 18: array([-6.66882883e-01,  7.57478075e-01,  1.47526517e-01, -3.46705701e-08]), 19: array([ 3.98899134e-03,  8.30461734e-06,  1.92296593e-02, -4.32886792e-03]), 20: array([-0.2669862 ,  0.42705402,  0.2179764 , -0.00075787]), 21: array([-4.47933455e-02, -2.03769850e-01,  4.20581375e-01,  5.55141818e-08]), 22: array([-4.74930701e-01,  4.83699678e-01,  2.13476624e-08, -6.30580543e-03]), 23: array([-1.84892074e-01,  6.11706225e-06,  8.36853924e-07,  8.73705577e-03]), 24: array([-2.70193155e-01,  4.44442835e-07,  3.73410083e-06, -1.86632632e-03]), 25: array([-2.64999172e-01,  6.73768178e-07,  2.93507156e-05,  1.49166345e-02]), 26: array([-6.55615363e-02, -1.16691823e-01,  1.34521008e-08,  1.72897631e-02]), 27: array([-4.72497630e-02,  7.81704536e-04,  2.00695906e-06,  4.41560509e-03]), 28: array([-2.84177682e-01,  7.71091327e-02,  2.71389238e-07,  2.10226116e-04]), 29: array([-0.07421805,  0.30846971,  0.65589049, -0.00512427]), 30: array([-1.00860922e+00,  6.08343284e-01,  3.96639153e-02, -9.29830497e-08]), 31: array([-3.60442882e-01,  7.72831343e-01,  2.99692002e-01, -1.13163268e-07]), 32: array([-7.29676141e-01,  5.03439994e-01,  9.74455989e-08, -7.02340546e-07]), 33: array([-2.56787057e-01,  1.63962310e-01,  2.14961868e-01,  1.01544916e-07]), 34: array([-0.15648122, -0.39846791,  0.14890356,  0.01792818]), 35: array([-4.53564814e-02, -4.80458520e-01,  1.58983125e-06,  1.88715271e-02]), 36: array([-1.04316999e-01,  2.54025694e-01,  4.04448420e-08,  7.08195888e-03]), 37: array([3.62810571e-01, 5.01153694e-01, 6.74598988e-02, 2.85165154e-07]), 38: array([-1.66887742e-01, -2.33046568e-01,  2.08274373e-07,  1.31613825e-02]), 39: array([-4.71575069e-02, -2.02476294e-01,  4.72695042e-01, -4.62896602e-08]), 40: array([-1.99603186e-01,  1.20488898e-01,  7.06223281e-02, -8.98110142e-08]), 41: array([-3.24117931e-02, -2.47272532e-04,  6.25895485e-07,  3.30293302e-03]), 42: array([-7.09518128e-02,  8.54665480e-06,  1.74322243e-04,  5.31656760e-03]), 43: array([3.20086072e-02, 4.31260809e-06, 2.26631836e-03, 1.08674224e-02]), 44: array([-7.72310253e-01,  6.67386434e-01,  1.65226267e-01, -1.58039450e-08]), 45: array([-9.56509954e-02,  3.39624239e-03,  1.15155771e-06,  1.79455030e-02]), 46: array([-5.12169373e-01,  3.50898007e-01,  4.45959245e-03, -2.66612698e-08]), 47: array([-1.40063324e-01,  1.52513062e-01,  1.88151676e-06,  9.65683757e-03]), 48: array([-1.33215732e-01,  1.96238344e-06,  1.21601978e-05,  1.19914615e-02]), 49: array([-1.16886266e-02,  2.05000857e-06,  1.17921352e-04,  1.30356393e-02]), 50: array([-1.36714643e-01,  4.57903177e-01,  2.61068844e-01, -2.88206171e-08]), 51: array([-2.47307866e-01,  5.72803790e-03,  3.44567690e-07,  6.41177185e-03]), 52: array([-0.25735896,  0.06303629,  0.19262059,  0.0080658 ]), 53: array([-0.00204461,  0.52002983,  1.66225818, -0.01561157])}
    # # qdt2.indParams={0: array([-6.44508922e-03,  3.55793935e-05,  2.26651515e-02,  1.76229436e-02]), 1: array([-6.44508922e-03,  3.55793935e-05,  2.26651515e-02,  1.76229436e-02]), 2: array([-6.44508922e-03,  3.55793935e-05,  2.26651515e-02,  1.76229436e-02]), 3: array([-6.44508922e-03,  3.55793935e-05,  2.26651515e-02,  1.76229436e-02]), 4: array([-6.44508922e-03,  3.55793935e-05,  2.26651515e-02,  1.76229436e-02]), 5: array([-6.44508922e-03,  3.55793935e-05,  2.26651515e-02,  1.76229436e-02]), 6: array([-6.44508922e-03,  3.55793935e-05,  2.26651515e-02,  1.76229436e-02]), 7: array([-6.44508922e-03,  3.55793935e-05,  2.26651515e-02,  1.76229436e-02]), 8: array([-6.44508922e-03,  3.55793935e-05,  2.26651515e-02,  1.76229436e-02]), 9: array([-6.44508922e-03,  3.55793935e-05,  2.26651515e-02,  1.76229436e-02]), 10: array([-6.44508922e-03,  3.55793935e-05,  2.26651515e-02,  1.76229436e-02]), 11: array([-6.44508922e-03,  3.55793935e-05,  2.26651515e-02,  1.76229436e-02]), 12: array([-6.44508922e-03,  3.55793935e-05,  2.26651515e-02,  1.76229436e-02]), 13: array([-6.44508922e-03,  3.55793935e-05,  2.26651515e-02,  1.76229436e-02]), 14: array([-6.44508922e-03,  3.55793935e-05,  2.26651515e-02,  1.76229436e-02]), 15: array([-6.44508922e-03,  3.55793935e-05,  2.26651515e-02,  1.76229436e-02]), 16: array([-6.44508922e-03,  3.55793935e-05,  2.26651515e-02,  1.76229436e-02]), 17: array([-6.44508922e-03,  3.55793935e-05,  2.26651515e-02,  1.76229436e-02]), 18: array([-6.44508922e-03,  3.55793935e-05,  2.26651515e-02,  1.76229436e-02]), 19: array([-6.44508922e-03,  3.55793935e-05,  2.26651515e-02,  1.76229436e-02]), 20: array([-6.44508922e-03,  3.55793935e-05,  2.26651515e-02,  1.76229436e-02]), 21: array([-6.44508922e-03,  3.55793935e-05,  2.26651515e-02,  1.76229436e-02]), 22: array([-6.44508922e-03,  3.55793935e-05,  2.26651515e-02,  1.76229436e-02]), 23: array([-6.44508922e-03,  3.55793935e-05,  2.26651515e-02,  1.76229436e-02]), 24: array([-6.44508922e-03,  3.55793935e-05,  2.26651515e-02,  1.76229436e-02]), 25: array([-6.44508922e-03,  3.55793935e-05,  2.26651515e-02,  1.76229436e-02]), 26: array([-6.44508922e-03,  3.55793935e-05,  2.26651515e-02,  1.76229436e-02]), 27: array([-6.44508922e-03,  3.55793935e-05,  2.26651515e-02,  1.76229436e-02]), 28: array([-6.44508922e-03,  3.55793935e-05,  2.26651515e-02,  1.76229436e-02]), 29: array([-6.44508922e-03,  3.55793935e-05,  2.26651515e-02,  1.76229436e-02]), 30: array([-6.44508922e-03,  3.55793935e-05,  2.26651515e-02,  1.76229436e-02]), 31: array([-6.44508922e-03,  3.55793935e-05,  2.26651515e-02,  1.76229436e-02]), 32: array([-6.44508922e-03,  3.55793935e-05,  2.26651515e-02,  1.76229436e-02]), 33: array([-6.44508922e-03,  3.55793935e-05,  2.26651515e-02,  1.76229436e-02]), 34: array([-6.44508922e-03,  3.55793935e-05,  2.26651515e-02,  1.76229436e-02]), 35: array([-6.44508922e-03,  3.55793935e-05,  2.26651515e-02,  1.76229436e-02]), 36: array([-6.44508922e-03,  3.55793935e-05,  2.26651515e-02,  1.76229436e-02]), 37: array([-6.44508922e-03,  3.55793935e-05,  2.26651515e-02,  1.76229436e-02]), 38: array([-6.44508922e-03,  3.55793935e-05,  2.26651515e-02,  1.76229436e-02]), 39: array([-6.44508922e-03,  3.55793935e-05,  2.26651515e-02,  1.76229436e-02]), 40: array([-6.44508922e-03,  3.55793935e-05,  2.26651515e-02,  1.76229436e-02]), 41: array([-6.44508922e-03,  3.55793935e-05,  2.26651515e-02,  1.76229436e-02]), 42: array([-6.44508922e-03,  3.55793935e-05,  2.26651515e-02,  1.76229436e-02]), 43: array([-6.44508922e-03,  3.55793935e-05,  2.26651515e-02,  1.76229436e-02]), 44: array([-6.44508922e-03,  3.55793935e-05,  2.26651515e-02,  1.76229436e-02]), 45: array([-6.44508922e-03,  3.55793935e-05,  2.26651515e-02,  1.76229436e-02]), 46: array([-6.44508922e-03,  3.55793935e-05,  2.26651515e-02,  1.76229436e-02]), 47: array([-6.44508922e-03,  3.55793935e-05,  2.26651515e-02,  1.76229436e-02]), 48: array([-6.44508922e-03,  3.55793935e-05,  2.26651515e-02,  1.76229436e-02]), 49: array([-6.44508922e-03,  3.55793935e-05,  2.26651515e-02,  1.76229436e-02]), 50: array([-6.44508922e-03,  3.55793935e-05,  2.26651515e-02,  1.76229436e-02]), 51: array([-6.44508922e-03,  3.55793935e-05,  2.26651515e-02,  1.76229436e-02]), 52: array([-6.44508922e-03,  3.55793935e-05,  2.26651515e-02,  1.76229436e-02]), 53: array([-6.44508922e-03,  3.55793935e-05,  2.26651515e-02,  1.76229436e-02])}
    # # qdt2.simulate(2000,'Plots/Simulation/Dataset2_None_reg5_1.png')
    # qdt2.predict2('SecTime_D2_Time_Frame_Previous_reg{}_reg{}'.format(qdt2.reg1,qdt2.reg2))


    # plot.plot([0.04*i for i in range(25)],[0, 0, 0, 0, 0, 0, 0, 1, 44, 633, 3360, 12698, 31713, 34257, 20131, 4877, 279, 7, 0, 0, 0, 0, 0, 0, 0],'-o')
    # plot.plot([0.04*i for i in range(25)],[0, 0, 0, 0, 0, 0, 0, 0, 23, 460, 3124, 9780, 21540, 27903, 24078, 12693, 5829, 2342, 223, 5, 0, 0, 0, 0, 0],'-o')
    # plot.xlabel('Percent of games that have the same answers as in the reality')
    # plot.ylabel('Count of simulated players')
    # plot.legend(['CPT with logit choice function','QDT'],loc=2)
    #
    # plot.title('2000 Simulations, 116000 simulated players')
    # plot.savefig('Plots/Simulation/Dataset2_Comp.png')

#     plot.plot([0.04*i for i in range(25)],[0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 205, 1558, 4114, 9226, 3822, 73, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# ,'-o')
#     plot.plot([0.04*i for i in range(25)],[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 134, 913, 1147, 5152, 8221, 2443, 914, 75, 0, 0, 0, 0, 0, 0, 0],'-o')
#     plot.xlabel('Percent of games that have the same answers as in the reality')
#     plot.ylabel('Count of simulated players')
#     plot.legend(['CPT with logit choice function','QDT'],loc=2)
#
#     plot.title('1000 Simulations, 19000 simulated players')
#     plot.savefig('Plots/Simulation/Dataset1_Comp.png')


def plotsomething():
    # PLot the bar plot
    data=np.array([[1]])







main()

