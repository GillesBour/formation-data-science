import pandas as pd
import numpy as np
don = pd.read_csv("C:/Users/cepe-s4-03/Documents/GB/data/ozonecomplet.csv", header=0, sep=";")
don = don.drop(['nomligne', 'Ne', 'Dv'], axis=1)
don.describe()
don.rename(columns={'O3': 'Y'}, inplace=True)
don.describe()
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, LassoCV, ElasticNet, ElasticNetCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold
np.random.seed(1235)
nb=10
tmp = np.arange(don.shape[0])%nb
bloc = np.random.choice(tmp,size=don.shape[0],replace=False)

don["bloc"] = bloc
PREV = pd.DataFrame({'bloc':don['bloc'],'Y':don['Y'],'MCO':0.0,'ridge':0.0,'lasso':0.0,'elas':0.0,'arbre':0.0,'foret':0.0})

for i in np.arange(nb):
    print(i)
    X_train = don[don["bloc"]!=i].drop(["Y","bloc"],axis=1)
    X_test = don[don["bloc"]==i].drop(["Y","bloc"],axis=1)
    Y_train = don[don["bloc"]!=i]["Y"]
#### reg
    reg = LinearRegression()
    reg.fit(X_train,Y_train)
    PREV.loc[PREV.bloc==i,'MCO'] = reg.predict(X_test)
#### pipeline
    kf = KFold(n_splits=10,shuffle=True)
    cr = StandardScaler()
#### lasso
    lassocv = LassoCV(cv=kf)
    pipelassocv = Pipeline(steps=[("cr",cr),("lassocv",lassocv)])
    etape_lassocv = pipelassocv.named_steps["lassocv"]
    pipelassocv.fit(X_train,Y_train)
    PREV.loc[PREV.bloc==i,'lasso'] = pipelassocv.predict(X_test)
##### ridge
    grilleridge = etape_lassocv.alphas_ * 100 
    ridgecv = RidgeCV(cv=kf,alphas=grilleridge)
    piperidgecv = Pipeline(steps=[("cr",cr),("ridgecv",ridgecv)])
    piperidgecv.fit(X_train,Y_train)
    PREV.loc[PREV.bloc==i,'ridge'] = piperidgecv.predict(X_test)
    #####
    grilleelas = etape_lassocv.alphas_ * 2 
    elasticcv = ElasticNetCV(cv=kf,alphas=grilleelas)
    pipeelasticcv = Pipeline(steps=[("cr",cr),("elasticcv",elasticcv)])
    pipeelasticcv.fit(X_train,Y_train)
    PREV.loc[PREV.bloc==i,'elas'] = pipeelasticcv.predict(X_test)
    #### reg
    arbre = DecisionTreeRegressor()
    arbre.fit(X_train,Y_train)
    PREV.loc[PREV.bloc==i,'arbre'] = arbre.predict(X_test)
    #### foret
    foret = RandomForestRegressor(n_estimators=500)
    foret.fit(X_train,Y_train)
    PREV.loc[PREV.bloc==i,'foret'] = foret.predict(X_test)

PREV.head()

Erreur = PREV.copy()
Erreur = Erreur.drop("bloc",axis=1)


def erreur(X, Y):
    return np.mean((X - Y) ** 2)

def apply_erreur(RES):
    return RES.apply(lambda col: erreur(col, RES.iloc[:, 0]), axis=0)

print(apply_erreur(Erreur))

