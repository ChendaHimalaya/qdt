import pandas as pd
import numpy as np
from scipy import stats
from tqdm import tqdm

TRAINPATH="data/cpc18/All_estimation_raw_data.csv"

#Copied from CPC18-baseline code
def CPC18_getDist(H, pH, L, lot_shape, lot_num):
    # Extract true full distributions of an option in CPC18
    #   input is high outcome (H: int), its probability (pH: double), low outcome
    #   (L: int), the shape of the lottery ('-'/'Symm'/'L-skew'/'R-skew' only), and
    #   the number of outcomes in the lottery (lot_num: int)
    #   output is a matrix (numpy matrix) with first column a list of outcomes (sorted
    #   ascending) and the second column their respective probabilities.

    if lot_shape == '-':
        if pH == 1:
            dist = np.array([H, pH])
            dist.shape = (1, 2)
        else:
            dist = np.array([[L, 1-pH], [H, pH]])

    else:  # H is multi outcome
        # compute H distribution
        high_dist = np.zeros(shape=(lot_num, 2))
        if lot_shape == 'Symm':
            k = lot_num - 1
            for i in range(0, lot_num):
                high_dist[i, 0] = H - k / 2 + i
                high_dist[i, 1] = pH * stats.binom.pmf(i, k, 0.5)

        elif (lot_shape == 'R-skew') or (lot_shape == 'L-skew'):
            if lot_shape == 'R-skew':
                c = -1 - lot_num
                dist_sign = 1
            else:
                c = 1 + lot_num
                dist_sign = -1
            for i in range(1, lot_num+1):
                high_dist[i - 1, 0] = H + c + dist_sign * pow(2, i)
                high_dist[i - 1, 1] = pH / pow(2, i)

            high_dist[lot_num - 1, 1] = high_dist[lot_num - 1, 1] * 2

        # incorporate L into the distribution
        dist = high_dist
        locb = np.where(high_dist[:, 0] == L)
        if all(locb):
            dist[locb, 1] = dist[locb, 1] + (1-pH)
        elif pH < 1:
            dist = np.vstack((dist, [L, 1-pH]))

        dist = dist[np.argsort(dist[:, 0])]

    return dist


def separate_by_block():
    """Separate the data by block number"""

    df=pd.read_csv(TRAINPATH)
    subj_ids=list(set(df["SubjID"].to_list()))
    for i in range(1,6):
        temp=df[df["block"]==i]
        #temp.to_csv("data/cpc18/train_block{}.csv".format(i))
        data={temp.columns[i]:[] for i in range(len(temp.columns))}
        data.pop("Trial")
        data.pop("Button")
        data.pop("Payoff")
        data.pop("B")
        data.pop("RT")
        data.pop("Apay")
        data.pop("Bpay")
        BRate=[]

        for id in tqdm(subj_ids):
            temp1=temp[temp["SubjID"]==id]
            problem_ids=list(set(temp1["GameID"].to_list()))
            for gameid in problem_ids:
                temp2=temp1[temp1["GameID"]==gameid]
                response=temp2["B"]
                BRate.append(np.sum(response)/len(response))
                row1=temp2.iloc[0]
                for key in data.keys():
                    data[key].append(row1[key])
        data["BRate"]=BRate
        sum_data=pd.DataFrame(data=data)
        sum_data.to_csv("data/cpc18/train_block{}_sum.csv".format(i))

def separate_by_gameID():
    """Separate the data by block number"""

    df=pd.read_csv(TRAINPATH)
    Game_ids=list(set(df["GameID"].to_list()))
    for i in range(1,6):
        temp=df[df["block"]==i]
        #temp.to_csv("data/cpc18/train_block{}.csv".format(i))
        data={temp.columns[i]:[] for i in range(len(temp.columns))}
        data.pop("Trial")
        data.pop("Button")
        data.pop("Payoff")
        data.pop("B")
        data.pop("RT")
        data.pop("Apay")
        data.pop("Bpay")
        data.pop("SubjID")
        BRate=[]

        for id in tqdm(Game_ids):
            temp1=temp[temp["GameID"]==id]


            response=temp1["B"]
            BRate.append(np.sum(response)/len(response))
            row1=temp1.iloc[0]
            for key in data.keys():
                data[key].append(row1[key])
        data["BRate"]=BRate
        sum_data=pd.DataFrame(data=data)
        sum_data.to_csv("data/cpc18/train_block{}_games.csv".format(i))


def extract_pf_features():
    data=pd.read_csv("data/cpc18/PF_features210.csv")

    data=data.drop_duplicates(subset=["GameID"])
    data.to_csv("data/cpc18/PF_features.csv")

def generate_PF_feature_dump():
    import pickle
    data = pd.read_csv("data/cpc18/PF_features210.csv")
    dic=dict(tuple(data.groupby("GameID")))
    for key in dic.keys():
        dic[key]=dic[key].iloc[:,17:35].drop_duplicates()
    pickle.dump(dic,open("data/cpc18/PF_features.p",'wb'))








def random_sample_blocks(N):
    """Sample with replacement"""
    pass

def main():
    #separate_by_block()
    #separate_by_gameID()
    #extract_pf_features()
    generate_PF_feature_dump()

if __name__=="__main__":
    main()
