from flask import Flask, render_template, request
import numpy as np
import joblib
import pandas as pd

# Charger le modèle pré-entraîné
model = joblib.load('random_forest_model.joblib')

# Créer une instance de l'application Flask
app = Flask(__name__, static_folder='static')

# Définir une route principale "/"
@app.route('/')
def home():
    return render_template('index.html')

# Définir une route pour la prédiction "/predict"
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Extraire les valeurs du formulaire
        nbr_grossesses = int(request.form['nbr_grossesses'])
        glucose = float(request.form['glucose'])
        tension = float(request.form['tension'])
        epaisseur_peau = float(request.form['epaisseur_peau'])
        insulin = float(request.form['insulin'])
        IMC = float(request.form['IMC'])
        Coef_hereditaire = float(request.form['Coef_hereditaire'])
        age = int(request.form['age'])

        # Log des valeurs extraites
        app.logger.debug(f"Features: nbr_grossesses={nbr_grossesses}, glucose={glucose}, tension={tension}, epaisseur_peau={epaisseur_peau}, insulin={insulin}, IMC={IMC}, Coef_hereditaire={Coef_hereditaire}, age={age}")

        # Créer un DataFrame pandas avec les caractéristiques
        features = pd.DataFrame([[nbr_grossesses, glucose, tension, epaisseur_peau, insulin, IMC, Coef_hereditaire, age]],
                               columns=['nbr_grossesses', 'glucose', 'tension', 'epaisseur_peau', 'insulin', 'IMC', 'Coef_hereditaire', 'age'])

        # Faire la prédiction
        prediction = model.predict(features)

        # Log de la prédiction
        app.logger.debug(f"Prediction: {prediction[0]}")

        # Déterminer le message de prédiction
        if prediction[0] == 0:
            prediction_text = "Non diabétique."
        else:
            prediction_text = "Diabétique."

        # Rendre le résultat sur la page HTML avec les données du formulaire
        return render_template('index.html', prediction_text=prediction_text, form_data=request.form)

# Lancer l'application si le script est exécuté directement
if __name__ == "__main__":
    app.run(debug=True)
