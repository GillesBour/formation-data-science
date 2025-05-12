import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

path="C://Users//cepe-s4-03//Documents//GB//"
euca_path="eucalyptus.txt"

eucalypt=pd.read_csv(path+euca_path,header=0,delimiter=";")

plt.plot(eucalypt["circ"],eucalypt["ht"],"o")

reg=smf.ols('ht~circ', data=eucalypt).fit()
reg.summary()
reg.params
reg.scale
plt.plot(eucalypt['circ'],reg.resid,"o")