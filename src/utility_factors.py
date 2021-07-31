import numpy as np
from matplotlib import pyplot as plt
def valueFunction1(v,alpha,_lambda,beta):
    """
    The value function used in prospect theory
    :param v: Value
    :param alpha:
    :param _lambda:
    :return:
    """
    if v>=0:
        return v**alpha
    else:
        return -_lambda*(-v)**beta

def weightingFunction(p,delta,gamma):

    """The weighting function in prospect theory
    Equals 1 when the probability is 1"""
    #print(np.log(p))
    if p==0: return 0
    expo=(-np.log(p))**gamma


    return np.exp(-delta*expo)

def logitFunc(UtilA, UtilB,phi):
    """Returns the probability of picking Options B over Option A"""
    return 1/(1+np.exp(phi*(UtilA-UtilB)))

def CPT_logit(alpha,_lambda,beta,delta,gamma,phi,dist1,dist2,Amb,Corr):


    #Calculate Utility of Option A
    uA=0
    for i in range(len(dist1)): # for each possible outcome
        val=dist1[i][0]
        prob=dist1[i][1]
        uA+=valueFunction1(val,alpha,_lambda,beta)*weightingFunction(prob,delta,gamma)

    uB=0
    for i in range(len(dist2)):
        val=dist2[i][0]
        if Amb==1: #Assume uniform distribution
            prob=1/len(dist2)
        else:
            prob = dist2[i][1]
        uB += valueFunction1(val, alpha, _lambda, beta) * weightingFunction(prob, delta, gamma)

    return logitFunc(uA, uB, phi)





def main():
    def plot_weight():
        plt.figure(0)
        delta = 1.01166834
        gamma = 0.96294392
        x = [0.01 * x for x in range(100)]
        y = [weightingFunction(x[i], delta, gamma) for i in range(100)]
        plt.plot(x, y, '-')
        plt.xlabel("p")
        plt.ylabel("w(p)")
        plt.savefig("images/aggregate_prob_weighting_function.png")

    def plot_value():
        plt.figure(1)
        alpha=0.88614743
        _lambda=0.81004461
        beta=0.95779005
        x=[1*(-100+x) for x in range(200)]
        y=[valueFunction1(x[i],alpha,_lambda,beta) for i in range(len(x))]
        plt.plot(x, y, '-')
        plt.xlabel("x")
        plt.ylabel( "v(x)")
        plt.savefig("images/aggregate_value_function.png")


    def plot_logit():
        plt.figure(2)
        phi=0.2765091
        x=[-10 +0.01*i for i in range(2000)]
        y=[logitFunc(0,0+x[i],phi) for i in range(len(x))]
        plt.plot(x, y, '-')
        plt.xlabel("UA-UB")
        plt.ylabel("Chance of picking option B over option A")
        plt.savefig("images/aggregate_logit_function.png")




    plot_value()
    plot_weight()
    plot_logit()



if __name__=="__main__":
    main()
