import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
eucalypt = pd.read_csv("eucalyptus.txt", header=0, sep=";")
eucalypt.shape
eucalypt.describe()

plt.plot(eucalypt['circ'], eucalypt['ht'], "o")

#get stats param
reg = smf.ols('ht~circ', data=eucalypt).fit()
reg.summary()
reg.params
reg.scale
plt.plot(eucalypt['circ'], reg.resid, "o")
plt.plot(reg.predict(), reg.resid, "o")
plt.plot(np.arange(1,eucalypt.shape[0]+1), reg.resid , "o")


beta1 = []
beta2 = []
rng = np.random.default_rng(seed=123) # fixe la graine du générateur, les tirages seront les mêmes
for k in range(500):
    lines = rng.choice(eucalypt.shape[0], size=10, replace=False)
    euca100 = eucalypt.iloc[lines]
    reg100 = smf.ols('ht~circ', data=euca100).fit()
    beta1.append(reg100.params.iloc[0])
    beta2.append(reg100.params.iloc[1])

plt.hist(beta2, bins=30)
plt.plot(beta1, beta2, "o")

regsqrt = smf.ols('ht~I(np.sqrt(circ))', data=eucalypt).fit()
sel = eucalypt['circ'].argsort()
plt.plot(eucalypt['circ'], eucalypt['ht'], "o", eucalypt['circ'], reg.predict(), "-", eucalypt.circ.iloc[sel], regsqrt.predict()[sel], "-"  )
reg.rsquared
regsqrt.rsquared
