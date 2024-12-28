import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import OrdinalEncoder
from lifelines import CoxPHFitter
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt
from lifelines.statistics import pairwise_logrank_test

st.set_page_config(page_title="ND_FA_Biostatistique", page_icon="🎢", layout="centered")
st.title("Modelisation des données")


# Fonction de chargement du jeu de données
@st.cache_data(persist=True)
def chargement():
    donnee = pd.read_excel("Donnnées_Projet_M2SID2023_2024.xlsx")
    return donnee


df = chargement()
portion = df.head(100)
df4 = df.copy()
df4.columns = [
    "AGE",
    "SEXE",
    "Premiers_Signe",
    " Admission_hopital",
    "Hypertension_Arterielle",
    "Diabete",
    "Cardiopathie",
    "hémiplégie",
    "Paralysie_faciale",
    "Aphasie",
    "Hémiparésie",
    "Engagement_Cerebral",
    "Inondation_Ventriculaire",
    "Traitement",
    "Temps_Suivi",
    "Evolution",
]
## Recuperation des variables quantitatives
varQuant = df4.select_dtypes(include="number").columns.tolist()
## Recuperation des variables qualitatives
varQual = df4.select_dtypes(include="object").columns.tolist()
for var in varQuant:
    Q1 = df4[var].quantile(0.25)
    Q3 = df4[var].quantile(0.75)
    IQR = Q3 - Q1
    min = Q1 - 1.5 * IQR
    max = Q3 + 1.5 * IQR
    df4.loc[df4[var] < min, var] = min
    df4.loc[df4[var] > max, var] = max
df5 = df4.copy()
# Encodage des variables qualitative
categories_order = [
    ["Homme", "Femme"],
    ["NON", "OUI"],
    ["NON", "OUI"],
    ["NON", "OUI"],
    ["NON", "OUI"],
    ["NON", "OUI"],
    ["NON", "OUI"],
    ["NON", "OUI"],
    ["NON", "OUI"],
    ["NON", "OUI"],
    ["Thrombolyse", "Chirurgie"],
    ["Deces", "Vivant"],
]
# instanciation
encoder = OrdinalEncoder(categories=categories_order)
df5.loc[:, varQual] = encoder.fit_transform(df5[varQual])
for var in varQual:
    df5[var] = df5[var].apply(lambda x: int(x))
    # if st.button("Normaliser", False):
    # instanciation
    # scaler = RobustScaler()
    # nomvar = df5.drop("Evolution", axis=1).columns.tolist()
    # df5.loc[:, nomvar] = scaler.fit_transform(df5[nomvar])
    # st.text("Statistique des donnormaliser")
    # df5.describe().T
seed = 0
# Train/test Split
y = df5["Evolution"]
x = df5.drop(
    ["Evolution", "AGE", "Hypertension_Arterielle", "Diabete", "Engagement_Cerebral"],
    axis=1,
)
X_train, X_test, Y_train, Y_test = train_test_split(
    x, y, test_size=0.2, random_state=seed
)


# Performance du modèle


def plot_perf(graphes):
    if "Matrice de confusion" in graphes:
        st.subheader("Matrice de confusion")
        ConfusionMatrixDisplay.from_estimator(model, X_test, Y_test)
        st.pyplot()

    if "Courbe de ROC" in graphes:
        st.subheader("Courbe de ROC")
        # fig, ax = plt.subplots()
        RocCurveDisplay.from_estimator(model, X_test, Y_test)
        st.pyplot()


Classification = st.sidebar.selectbox(
    "Classificateur",
    ("Regression Logistique", "SVM", "Random Forest", "CoxPHFitter", "Kaplan-Meier"),
)

# Random Forest
if Classification == "Random Forest":
    st.sidebar.subheader("Hyerparamètres du modèle")
    n_arbre = st.sidebar.number_input("Choisir le nombre d'arbes", 10, 100, step=5)
    profondeur_arbre = st.sidebar.number_input(
        "Profondeur maximale d'un arbe", 1, 20, step=1
    )
    # Performance du modele
    perf_graphe = st.sidebar.multiselect(
        "Choisir un graphique de performance du modèle ML",
        ("Matrice de confusion", "Courbe de ROC"),
    )
    if st.sidebar.button("Execution", key="classify"):
        st.subheader("Résultat du modèle Random Forest")
        # Initialisation du modele
        model = RandomForestClassifier(n_estimators=n_arbre, max_depth=profondeur_arbre)
        # Entrainement du modele
        model.fit(X_train, Y_train)
        # Prediction
        y_pred = model.predict(X_test)
        # Metriques de performance
        accuracy = accuracy_score(Y_test, y_pred)
        precision = precision_score(Y_test, y_pred)
        recall = recall_score(Y_test, y_pred)
        # Affichage des métriques dans l'application
        st.write(f"Accuracy : {round(accuracy,2)}")
        st.write(f"Precision : {round(precision,2)}")
        st.write(f"Recall : {round(recall,2)}")

        # Affichage des graphiques de performance
        plot_perf(perf_graphe)

# Regression Logistique
elif Classification == "Regression Logistique":
    st.sidebar.subheader("Hyerparamètres du modèle")
    hepr_c = st.sidebar.number_input(
        "Choisir le paramétre de régularisation", 0.01, 10.0
    )
    n_max_iter = st.sidebar.number_input(
        "Nombre maximale ditaration", 100, 1000, step=10
    )

    # Performance du modele
    perf_graphe = st.sidebar.multiselect(
        "Choisir un graphique de performance du modèle ML",
        ("Matrice de confusion", "Courbe de ROC"),
    )
    if st.sidebar.button("Execution", key="classify"):
        st.subheader("Résultat du modèle de Regression Logistique")
        # Initialisation du modele
        model = LogisticRegression(C=hepr_c, max_iter=n_max_iter)
        # Entrainement du modele
        model.fit(X_train, Y_train)
        # Prediction
        y_pred = model.predict(X_test)
        # Metriques de performance
        accuracy = accuracy_score(Y_test, y_pred)
        precision = precision_score(Y_test, y_pred)
        recall = recall_score(Y_test, y_pred)
        # Affichage des métriques dans l'application
        st.write(f"Accuracy : {round(accuracy,2)}")
        st.write(f"Precision : {round(precision,2)}")
        st.write(f"Recall : {round(recall,2)}")

        # Affichage des graphiques de performance
        plot_perf(perf_graphe)

# SVM
elif Classification == "SVM":
    st.sidebar.subheader("Hyerparamètres du modèle")
    kernel = st.sidebar.radio(
        "Choisir le Kernel", ("linear", "rbf", "sigmoid", "poly", "precomputed")
    )
    hyp_c = st.sidebar.number_input("Profondeur maximale C", 1, 10, step=1)
    gamma = st.sidebar.radio("Gamma", ("scale", "auto"))
    # Performance du modele
    perf_graphe = st.sidebar.multiselect(
        "Choisir un graphique de performance du modèle ML",
        ("Matrice de confusion", "Courbe de ROC"),
    )
    if st.sidebar.button("Execution", key="classify"):
        st.subheader("Résultat du modèle SVM")

        # Initialisation du modele
        model = SVC(kernel=kernel, C=hyp_c, gamma=gamma)
        # Entrainement du modele
        model.fit(X_train, Y_train)
        # Prediction
        y_pred = model.predict(X_test)
        # Metriques de performance
        accuracy = accuracy_score(Y_test, y_pred)
        precision = precision_score(Y_test, y_pred)
        recall = recall_score(Y_test, y_pred)
        # Affichage des métriques dans l'application
        st.write(f"Accuracy : {round(accuracy,2)}")
        st.write(f"Precision : {round(precision,2)}")
        st.write(f"Recall : {round(recall,2)}")

        # Affichage des métriques dans l'application
        plot_perf(perf_graphe)
elif Classification == "CoxPHFitter":
    st.sidebar.subheader("Hyerparamètres du modèle")
    apha = st.sidebar.number_input("Choisir le nombre alpha", 0.01, 0.1, step=0.01)
    # Performance du modele
    if st.sidebar.button("Execution", key="classify"):
        # Initialisation du modele
        model = CoxPHFitter(alpha=apha)
        # Entrainement du modele
        model.fit(df5, "Temps_Suivi", "Evolution")
        # Prediction
        st.title("Résumé du modèle CoxPHFitter")

        # Convertir le résumé en DataFrame pour un affichage plus propre
        summary_df = model.summary
        summary_df = summary_df.reset_index()
        summary_df = summary_df.rename(columns={"index": "Variable"})

        # Afficher le DataFrame avec Streamlit
        st.dataframe(summary_df)
        # st.write(model.print_summary())
elif Classification == "Kaplan-Meier":

    model = KaplanMeierFitter()
    # Entrainement du modele
    model.fit(df5["Temps_Suivi"], df5["Evolution"])

    # les résultats du test du log-rank
    def print_logrank(col):
        log_rank = pairwise_logrank_test(df5["Temps_Suivi"], df5[col], df5["Evolution"])
        return st.write(log_rank.summary)

    # Fonction d'aide pour tracer des courbes de Kaplan-Meier au niveau des covariables
    def plot_km(col):
        ax = plt.subplot()
        for r in df5[col].unique():
            ix = df5[col] == r
            model.fit(df5["Temps_Suivi"][ix], df5["Evolution"][ix], label=r)
            model.plot(ax=ax)
            plt.ylabel("Probabilité de Survie")
            plt.xlabel("Temps en mois")
            plt.title(f"Courbes de Kaplan-Meier pour {col}")
        st.pyplot()

    st.sidebar.subheader("Hyerparamètres du modèle")
    # apha = st.sidebar.number_input("Choisir le nombre alpha", 0.01, 0.1, step=0.01)
    # Performance du modele
    varQu = df5.drop("AGE", axis=1).columns.tolist()
    v1 = st.sidebar.selectbox("Variable", varQu)
    if st.sidebar.button("Execution", key="classify"):
        # st.title("Résultat du modèle Kaplan-Meier")
        st.header(v1)
        st.subheader("Le résultat du test de log-rank :")
        print_logrank(v1)
        st.subheader("Graphique du Kaplan-Meier")
        r = plot_km(v1)
