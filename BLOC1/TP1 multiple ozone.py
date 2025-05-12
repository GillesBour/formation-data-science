import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from mpl_toolkits.mplot3d import Axes3D 

ozone = pd.read_csv("ozone.txt", header=0, sep=";")
ozone.shape
ozone.describe()


fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(ozone['T12'], ozone['Vx'], ozone['O3'], marker="o")
ax.set_xlabel('T12')
ax.set_ylabel('Vx')
ax.set_zlabel('O3')
plt.show()


reg = smf.ols('O3~T12+Vx', data=ozone).fit()
reg.summary()

reg = smf.ols('O3~T12+Ne12+Vx', data=ozone).fit()
reg.summary()
