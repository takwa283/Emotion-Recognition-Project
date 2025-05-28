import streamlit as st
from PIL import Image
import requests
import io

st.title("Détection d'Émotions - Modèle KDEF")

# ✅ Ajout d'un `key` unique pour éviter l'erreur
upload = st.file_uploader("Chargez une image de visage", type=['png', 'jpeg', 'jpg'], key="file_uploader_unique_1")

if upload:
    # Lire et afficher l'image uploadée
    image = Image.open(upload)
    st.image(image, caption="Image chargée", use_column_width=True)
    
    # Convertir l'image en bytes pour l'envoi à l'API
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="JPEG")
    
    # Envoyer l'image à l'API FastAPI
    files = {"file": img_bytes.getvalue()}
    response = requests.post("http://localhost:8000/predict", files=files)
    
    # Vérifier si la requête est réussie
    if response.status_code == 200:
        resultat = response.json()
        predictions = resultat.get("predictions", {})

        # Afficher les résultats
        st.subheader("Résultat de la prédiction :")
        for emotion, prob in predictions.items():
            st.write(f"**{emotion}** : {prob:.2f}%")

    else:
        st.error("Erreur lors de la prédiction. Vérifiez le serveur API.")
