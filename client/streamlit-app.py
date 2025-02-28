import streamlit as st
import requests
from PIL import Image
import io

# URL de votre API Flask
API_URL_UNET = "https://semantic-seg.azurewebsites.net/predict-mask-unet"
API_URL_SEG = "https://semantic-seg.azurewebsites.net/predict-mask-seg"

# CSS personnalisé pour ajuster les couleurs
custom_css = """
<style>
    .stTitle {
        color: #000000;  /* Couleur de texte noire */
    }
    .stImage > img {
        background-color: #FFFFFF;  /* Couleur de fond blanche */
    }
    .css-1kyxreq {
        background-color: #FFFFFF !important;  /* Couleur de fond pour les conteneurs */
    }
    .css-145kmo2 {
        color: #000000 !important;  /* Couleur de texte pour les boutons */
    }
</style>
"""

# Injecter le CSS personnalisé
st.markdown(custom_css, unsafe_allow_html=True)

# Titre de la page HTML pour l'accessibilité
st.markdown("<title>Dashboard For Semantic Segmentation</title>", unsafe_allow_html=True)

st.title("Dashboard For Semantic Segmentation")

# Create a container for the images
container = st.container()

with container:
    col1, col2 = st.columns(2)

    with col1:
        st.image("assets/combined_image_resized.png", caption="Exemples du dataset")
        st.text("Description de la première image : [Voici 3 exemples des images d'entrainement et leurs masques respectifs]")

    with col2:
        st.image("assets/labels_distribution_resized.png", caption="Distribution des labels")
        st.text("Description de la seconde image : [Le graphique ci-dessus décrit la distribution des labels du train set]")

st.title("Application de Segmentation Sémantique")

# Téléchargement de l'image
uploaded_file = st.file_uploader("Choisissez une image...", type=["jpg", "jpeg", "png"], help="Téléchargez une image au format JPG, JPEG ou PNG.")

if uploaded_file is not None:
    # Afficher l'image téléchargée
    image = Image.open(uploaded_file)
    st.image(image, caption='Image téléchargée', use_column_width=True)
    st.text("Description de l'image téléchargée : [Image envoyée aux modèles pour la segmentation sémantique]")
    
    # Convertir l'image en octets
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='PNG')
    img_bytes = img_bytes.getvalue()
    
    # Envoyer l'image à l'API Flask
    st.write("Envoi de l'image à l'API Flask pour prédiction...")
    response_unet = requests.post(API_URL_UNET, data=img_bytes, headers={"Content-Type": "application/octet-stream"})
    response_seg = requests.post(API_URL_SEG, data=img_bytes, headers={"Content-Type": "application/octet-stream"})
    
    if response_unet.status_code == 200:
        st.write("Prédiction reçue de l'API Flask")
        
        # Charger l'image prédite à partir de la réponse
        predicted_image = Image.open(io.BytesIO(response_unet.content))
        
        # Afficher l'image prédite
        st.image(predicted_image, caption='Image avec Masque Prédit : UNET', use_column_width=True)
        st.text("Description de l'image prédite par UNET : [Masque prédit par U-Net Mini]")
    else:
        st.write("Erreur dans la prédiction. Code de réponse:", response_unet.status_code)

    if response_seg.status_code == 200:
        st.write("Prédiction reçue de l'API Flask")
        
        # Charger l'image prédite à partir de la réponse
        predicted_image = Image.open(io.BytesIO(response_seg.content))
        
        # Afficher l'image prédite
        st.image(predicted_image, caption='Image avec Masque Prédit : SEG_FORMER', use_column_width=True)
        st.text("Description de l'image prédite par SEG_FORMER : [Masque prédit par SegFormer]")
    else:
        st.write("Erreur dans la prédiction. Code de réponse:", response_seg.status_code)
