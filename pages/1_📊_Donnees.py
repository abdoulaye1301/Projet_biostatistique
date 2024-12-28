# Importation des bubliotheque
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as sc
from sklearn.preprocessing import OrdinalEncoder

st.set_page_config(page_title="ND_FA_Biostatistique", page_icon="🎢", layout="centered")


# Fonction de chargement du jeu de données
@st.cache_data(persist=True)
def chargement():
    donnee = pd.read_excel("Donnnées_Projet_M2SID2023_2024.xlsx")
    return donnee


df = chargement()
portion = df.head(100)
# Sous menu
Sous = st.sidebar.radio("Chargement et visualisation", ("Données", "Statitique"))
if Sous == "Données":
    st.write("Echantillon des 100 premières observations")
    st.write(portion)
elif Sous == "Statitique":
    st.subheader("Stattistique exploratoire du jeu de donnéées")
    # Modification des noms des variables
    df2 = df.copy()
    df2.columns = [
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
    varQuant = df2.select_dtypes(include="number").columns.tolist()
    ## Recuperation des variables qualitatives
    varQual = df2.select_dtypes(include="object").columns.tolist()

    Sous_b = st.sidebar.radio(
        "Exploration", ("Statitique Univariée", "Statitique Buvariée")
    )
    if Sous_b == "Statitique Univariée":
        st.text(
            "!======================== Statitique Univariée ========================!"
        )
        choix = st.sidebar.selectbox("Type statistique", ("Description", "Graphique"))
        # Statistique descriptive
        if choix == "Description":
            descr = st.sidebar.selectbox(
                "Description des variables",
                ("Variables quanitatives", "Variables qualitatives"),
            )
            if descr == "Variables quanitatives":
                st.text("Description des variables quantitatives")
                st.write(df2[varQuant].describe())
            elif descr == "Variables qualitatives":
                st.text("Description des variables qualitatives")
                st.write(df2[varQual].describe())

        # Les graphiques
        elif choix == "Graphique":
            statist = st.sidebar.selectbox(
                "Graphiques", ("Histogramme", "Diagramme en barre", "Boxplot")
            )

            # Visualisation des variables quantitatives
            if statist == "Histogramme":
                st.text("Représantation graphique des variables quantitatives")
                var = st.sidebar.selectbox("Choisire la variable", varQuant)
                colhist = st.sidebar.selectbox(
                    "Couleur de l'Histogramme",
                    (
                        "blue",
                        "red",
                        "green",
                        "orange",
                        "yellow",
                        "purple",
                        "black",
                        "teal",
                        "mustard",
                        "cyan",
                        "gold",
                        "pink",
                    ),
                )
                colde = st.sidebar.selectbox(
                    "Couleur de la densité",
                    (
                        "red",
                        "blue",
                        "green",
                        "orange",
                        "yellow",
                        "purple",
                        "black",
                        "teal",
                        "mustard",
                        "cyan",
                        "gold",
                        "pink",
                    ),
                )

                don, ax = plt.subplots()
                ax = sns.histplot(
                    data=df2,
                    x=var,
                    stat="density",
                    label="Histogramme",
                    color=colhist,
                )
                sns.kdeplot(data=df2, x=var, label="Densité", color=colde)
                plt.title(f"Histogramme de {var}")
                plt.xlabel(var)
                plt.ylabel("Densité")
                st.pyplot(don)

                # Visualisation des variables qualitatives
            elif statist == "Diagramme en barre":
                st.text("Représantation graphique des variables qualitatives")
                var = st.sidebar.selectbox("Choisire la variable", varQual)
                don, ax = plt.subplots()
                ax = sns.countplot(data=df2, x=var, palette="Set2")  # color=colbar)
                plt.title(f"Diagramme en barre de {var}")
                plt.ylabel("Frequence")
                plt.xlabel(var)
                st.pyplot(don)

                # Vsualisation des boxplot après nétoyage
            elif statist == "Boxplot":
                st.text("Observation des outliers")
                # Nettoyage des outliers
                for var in varQuant:
                    Q1 = df2[var].quantile(0.25)
                    Q3 = df2[var].quantile(0.75)
                    IQR = Q3 - Q1
                    min = Q1 - 1.5 * IQR
                    max = Q3 + 1.5 * IQR
                    df2.loc[df2[var] < min, var] = min
                    df2.loc[df2[var] > max, var] = max

                var = st.sidebar.selectbox("Choisire la variable", varQuant)
                colbox = st.sidebar.selectbox(
                    "Couleur de la densité",
                    (
                        "blue",
                        "red",
                        "green",
                        "orange",
                        "yellow",
                        "purple",
                        "black",
                        "teal",
                        "mustard",
                        "cyan",
                        "gold",
                        "pink",
                    ),
                )

                don, ax = plt.subplots()
                ax = sns.boxplot(data=df2, x=var, color=colbox)
                plt.title(f"Boxplot de {var}")
                plt.xlabel(var)
                st.pyplot(don)

    if Sous_b == "Statitique Buvariée":
        st.text(
            "!======================== Statitique Buvariée ========================!"
        )
        df3 = df2.copy()
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
        df3.loc[:, varQual] = encoder.fit_transform(df3[varQual])
        for var in varQual:
            df3[var] = df3[var].apply(lambda x: int(x))

        # Statistique buvariée
        statist1 = st.sidebar.selectbox(
            "Statistiques",
            ("Matrice de correlation", "Test Krouskal-Wllis"),
        )
        if statist1 == "Matrice de correlation":
            st.markdown("Viasualisation graphique de la correlation")
            sns.pairplot(df2)
            st.pyplot()
            st.markdown("Matrice de correlation")
            don, ax = plt.subplots()
            ax = sns.heatmap(df2[varQuant].corr(), annot=True)
            plt.title(f"Matrice de corrélation")
            st.pyplot(don)
        elif statist1 == "Test Krouskal-Wllis":
            st.text("Test Krouskal-Wllis")
            # La liaison des variables qualitatives avec le test de kruskal-wallis
            nom = df3.drop("Evolution", axis=1).columns.tolist()
            results = []
            base = []
            for var in nom:
                base = df2[var].unique()
                group = []
                for x in base:
                    group_x = df2[df2[var] == x]["Evolution"]
                    group.append(group_x)
                corr, p_value = sc.kruskal(*group)
                results.append([var, "Evolution", corr, p_value])
            df_results = pd.DataFrame(
                results, columns=["variable 1", "variable 2", "statistique", "p-value"]
            ).sort_values(by="statistique", ascending=False)
            # Conserver leslignes d'incice impaires
            # df_results = df_results.iloc[1::2]
            st.write(df_results)
