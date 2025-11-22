import streamlit as st
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Charger le modèle entraîné
model = load_model("model_DL.h5")

# Les mêmes colonnes que pour l'entraînement
features = ['Temperature C', 'Machine Speed RPM', 'Vibration Level mms', 'Energy Consumption kWh']

# Si tu as sauvegardé ton scaler avec joblib ou pickle, charge-le ici
# Exemple : from joblib import load ; scaler = load('scalerX.save')
# Sinon, tu dois recalculer le scaler sur tes données (à éviter en prod)
# -- Ici un scaler recalculé à partir d'un jeu de données d'exemple --
df = pd.read_csv('Manufacturingdataset.csv')
scaler = MinMaxScaler()
scaler.fit(df[features])  # Adapter si tu as un scaler pré-sauvegardé

# Interface Streamlit
st.title("Prédiction du score qualité de production")

# Widgets pour la saisie
temperature = st.number_input("Temperature C")
speed = st.number_input("Machine Speed RPM")
vibration = st.number_input("Vibration Level mms")
energy = st.number_input("Energy Consumption kWh")

# Au clic sur le bouton
if st.button("Prédire"):
    # Mise en forme pour le scaler puis le modèle (séquence de 20 pas)
    input_data = np.array([[temperature, speed, vibration, energy]])
    input_scaled = scaler.transform(input_data)
    seqlen = 20
    # Construire séquence artificielle (répète la ligne pour le test)
    input_seq = np.repeat(input_scaled[np.newaxis, :, :], seqlen, axis=1)
    # Prédiction
    prediction = model.predict(input_seq)
    st.write(f"Production Quality Score prédit : {prediction[0][0]:.2f}")
