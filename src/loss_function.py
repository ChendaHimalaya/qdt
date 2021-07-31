import numpy as np
import math
from sklearn.metrics import mean_squared_error
def mse_loss(target,
             prediction,
             ):
    return math.sqrt(mean_squared_error(target, prediction))

def prob_prod(response,
              prediction,
              binary=True):
    """

    :param response: list or numpy array that contains real response from the dataset
    :param prediction: predicted probability on different predictions

    :return:
    """
