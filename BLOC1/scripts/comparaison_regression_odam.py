# # Comparaison des méthodes de régression d'apprentissage supervisé

# Importation des modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV, ElasticNetCV, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Chargement des données
ozone = pd.read_csv("C:/Users/cepe-s4-03/Documents/GB/data/ozonecomplet.csv", header=0, sep=";")

# Nettoyage des données
ozone = ozone.drop(["nomligne", "Dv", "Ne"], axis=1)
ozone.rename(columns={"O3": "Y"}, inplace=True)

# Création d'un dataframe pour comparer les résultats de chaque méthode (n blocs)
PREV = pd.DataFrame(
    {
        "Y": ozone["Y"]
    }
)

# # Entraînement des modèles (avec validation croisée) pour chaque méthode
# Moindres carrés ordinaires, lasso, ridge, elasticNet, arbre, forêt

# Initialisation de KFold
kf = KFold(n_splits=5, shuffle=True, random_state=0)

for i, (train_index, test_index) in enumerate(kf.split(ozone)):

    # Séparation des données en train et test
    X_train = ozone.iloc[train_index].drop(["Y"], axis=1)
    X_test = ozone.iloc[test_index].drop(["Y"], axis=1)
    Y_train = ozone.iloc[train_index]["Y"]

    # Mise à jour de la colonne 'bloc' dans PREV
    PREV.loc[test_index, "bloc"] = i

    # Régression linéaire (MCO)
    reg = LinearRegression()
    reg.fit(X_train, Y_train)
    PREV.loc[test_index, "MCO"] = reg.predict(X_test)

    # Définition d'un nouveau kf pour la validation croisée de lambda
    kf_lambda = KFold(n_splits=10, shuffle=True, random_state=0)

    # Lasso avec validation croisée
    lassocv = LassoCV(cv=kf_lambda)
    pipelassocv = Pipeline(steps=[("cr", StandardScaler()), ("lassocv", lassocv)])
    pipelassocv.fit(X_train, Y_train)
    PREV.loc[test_index, "lasso"] = pipelassocv.predict(X_test)

    # Ridge avec validation croisée
    grilleridge = lassocv.alphas_ * 100
    ridgecv = RidgeCV(cv=kf_lambda, alphas=grilleridge)
    piperidgecv = Pipeline(steps=[("cr", StandardScaler()), ("ridgecv", ridgecv)])
    piperidgecv.fit(X_train, Y_train)
    PREV.loc[test_index, "ridge"] = piperidgecv.predict(X_test)

    # ElasticNet avec validation croisée
    grilleelas = lassocv.alphas_ * 2
    elasticcv = ElasticNetCV(cv=kf_lambda, alphas=grilleelas)
    pipeelasticcv = Pipeline(steps=[("cr", StandardScaler()), ("elasticcv", elasticcv)])
    pipeelasticcv.fit(X_train, Y_train)
    PREV.loc[test_index, "elas"] = pipeelasticcv.predict(X_test)

    # Arbre de décision
    arbre = DecisionTreeRegressor()
    arbre.fit(X_train, Y_train)
    PREV.loc[test_index, "arbre"] = arbre.predict(X_test)

    # Forêt aléatoire
    foret = RandomForestRegressor(n_estimators=100)
    foret.fit(X_train, Y_train)
    PREV.loc[test_index, "foret100"] = foret.predict(X_test)

    # Forêt aléatoire
    foret = RandomForestRegressor(n_estimators=500, max_features=0.3)
    foret.fit(X_train, Y_train)
    PREV.loc[test_index, "foret500"] = foret.predict(X_test)

# # Calcul des erreurs pour chaque modèle
Erreur = PREV.copy()
Erreur = Erreur.drop("bloc", axis=1)


def erreur(X, Y):
    return np.mean((X - Y) ** 2)


def apply_erreur(RES):
    return RES.apply(lambda col: erreur(col, RES.iloc[:, 0]), axis=0)


# Affichage des erreurs
erreurs = apply_erreur(Erreur)

# Création de la figure avec deux sous-graphiques
fig, ax = plt.subplots(1, 2, figsize=(15, 6))

# Sous-graphe 1 : Comparaison des prédictions des modèles
for col in ["MCO", "ridge", "lasso", "elas", "arbre", "foret100", "foret500"]:
    ax[0].scatter(PREV["Y"], PREV[col], label=col, alpha=0.6)

ax[0].plot(
    [PREV["Y"].min(), PREV["Y"].max()],
    [PREV["Y"].min(), PREV["Y"].max()],
    "k--",
    lw=2,
    label="Valeurs réelles",
)
ax[0].set_xlabel("Valeurs réelles (Y)")
ax[0].set_ylabel("Prédictions")
ax[0].set_title("Comparaison des prédictions des modèles")
ax[0].legend()

# Sous-graphe 2 : Tableau des erreurs
# Conversion des erreurs en DataFrame pour affichage
erreurs_df = erreurs.reset_index()
erreurs_df.columns = ["Modèle", "Erreur quadratique moyenne"]

# Création du tableau
ax[1].axis("tight")
ax[1].axis("off")
table = ax[1].table(
    cellText=erreurs_df.values,
    colLabels=erreurs_df.columns,
    cellLoc="center",
    loc="center",
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.auto_set_column_width(col=list(range(len(erreurs_df.columns))))

# Affichage de la figure
plt.tight_layout()
plt.show()

# Sauvegarde des résultats dans un fichier CSV
# PREV.to_csv("resultats_predictions.csv", index=False)
# print("Les résultats ont été sauvegardés dans 'resultats_predictions.csv'.")
