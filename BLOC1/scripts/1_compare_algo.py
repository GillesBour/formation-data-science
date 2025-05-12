import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold

def compare_algo(y,X):
    """calculate and save algorithms predictions in a panda Dataframe.

    Parameters:
    -----------
    y (pandas Series): the response variable
    X (pandas DataFrame): the explanatory variables

    Returns:
    --------
    pred_comp_df: the prediction results for each algorithm and variables blocks (pandas DataFrame)
    """
    # Initialize the DataFrame to store predictions
    pred_comp_df = pd.DataFrame()

    kf = KFold(n_splits=10, shuffle=True, random_state=0)
    for index_apprentissage, index_test in kf.split(X):
        # Split the data into training and validation sets
        Xapp = X[index_apprentissage,:]
        yapp = y[index_apprentissage]
        Xtest = X[index_test,:]

        # Ordinary Least Squares regression
        reg = smf.ols(yapp, Xapp).fit()


   

    return pred_comp_df