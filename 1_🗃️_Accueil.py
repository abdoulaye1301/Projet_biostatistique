import streamlit as st
import pandas as pd

# from PIL import Image


st.set_page_config(page_title="ND_FA_Biostatistique", page_icon="🎢", layout="centered")
st.sidebar.success("Selectionnez une page")


# Definition de la fonction principale
def main():
    st.title(
        "Application de Machine Learning pour la prédiction des données Biostatistique"
    )
    st.subheader("Auteurs : Abdoulaye NDAO, Malick FAYE")
    st.markdown(
        "**Cette étude consiste à mettre en place un modèle de machine learning ou statistique qui permette de faire un pronostique sur la survenue instantanée de décès après le traitement. Pour la construction de ce modèle, nous allons utiliser les données de patients atteints d’accident cérébral vasculaire (AVC), traités et suivis.Après avoir observé les dimensions de notre base de données, nous avons constaté qu’elle contient 1053 observations réparties sur 16 variables dont 4 variables sont des variables quantitatives et 12 sont des variables qualitatives.**"
    )
    st.text("   ")
    st.text("   ")
    st.image("biostatistique.jpg", use_column_width=True)
    # Ouvrez l'image
    # Image.open("biostatistique.jpg")
    # donne = st.file_uploader("Charge les données", ["xlsx"])
    # df = pd.read_excel("Donnnées_Projet_M2SID2023_2024.xlsx")
    # df = pd.DataFrame(donne)
    # va = st.number_input(
    #   "Le nombre d'observation à afficher", min_value=0, max_value=1050, step=1
    # )
    # if st.button("Afficher"):
    #  st.write(df.head(va))


if __name__ == "__main__":
    main()
