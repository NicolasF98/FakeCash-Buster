import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


#-------------------------# Chargeement du modèle de ML #-------------------------#
@st.cache_resource
def load_model():
    return joblib.load('model_rlo_opt.pkl')

model = load_model()


#-------------------------# Interface de l'app #-------------------------#

# Titre de la page
st.markdown("<h1 style='text-align: center; color: #FF6B6B;'>💸 FakeCash Buster💸</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Détectez de faux billets grâce au Machine Learning !</h3>", unsafe_allow_html=True)

# description sous le titre
st.markdown("""
Nous analysons précisément **les dimensions des billets de banque**, pour déterminer en un clin d’œil si il est authentique.
""")

# Ajout d'une image trouvée sur le web
st.image("https://www.safescan.com/static/images/safescan-cd-1124x720-2017-04/17874/1124x720/revision-2/safescan-cd-1124x720-2017-04.jpg")

# description sous l'image
st.markdown("""
**Notre algorithme** utilise une **regression logistique optimisé par Model-based Feature Selection**,  
ce qui garantit une **détection presque parfaite des faux positifs**.
""")

# Upload du fichier CSV
uploaded_file = st.file_uploader("📂 Sélectionnez un fichier CSV 📂", type=["csv"])


#-------------------------# Lancement de la prédiction #-------------------------#

if uploaded_file:
    st.success("Fichier chargé avec succès !")

    # Lecture du fichier
    data = pd.read_csv(uploaded_file, sep=',')
    data_features = data[['margin_low', 'margin_up', 'length']]
    st.subheader("Aperçu du fichier chargé :")
    st.dataframe(data)

    # Permet e lancer la prediction
    if st.button("🚀 Prédire nos données 🚀"):
        try:
            # Prediction
            predictions = model.predict(data_features)

            # Affichage des resultats
            st.subheader("Résultats des prédictions :")
            data["Prédiction"] = predictions
            data["Prédiction"] = data["Prédiction"].map({0: "Faux", 1: "Vrai"})
            st.dataframe(data)

            # Graphique
            pred_counts = data["Prédiction"].value_counts()

            fig, ax = plt.subplots()
            bars = ax.bar(pred_counts.index, pred_counts.values, color=["#FF6B6B", "#4ECDC4"])

            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height/2, int(height), ha='center', va='bottom', fontsize=12)
            ax.bar(pred_counts.index, pred_counts.values, color=["#FF6B6B", "#4ECDC4"])
            ax.set_ylabel("Nombre de billets")
            ax.set_title("Nombre de billets détectés comme Faux ou Vrai")
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            st.pyplot(fig)

            # Option de telechargement
            csv_result = data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Télécharger les résultat 📥",
                data=csv_result,
                file_name='predictions_nif.csv',
                mime='text/csv'
            )
        # Cas d'erreur
        except Exception as e:
            st.error(f"Erreur au cours de la prédiction : {e}")