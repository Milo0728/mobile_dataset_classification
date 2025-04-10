from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Cargar solo el modelo
modelo = joblib.load("modelo_random_forest.pkl")

# Lista de características esperadas
features = ['Mobile_Weight', 'RAM', 'Front_Camera', 'Back_Camera',
            'Battery_Capacity', 'Screen_Size']

@app.route('/')
def home():
    return render_template('formulario.html')

@app.route('/predecir', methods=['POST'])
def predecir():
    try:
        input_data = [float(request.form[feature]) for feature in features]
        valores_dict = {feature: request.form[feature] for feature in features}

        input_np = np.array(input_data).reshape(1, -1)
        pred = modelo.predict(input_np)[0]

        return render_template('resultado.html', resultado=pred, valores=valores_dict)

    except Exception as e:
        return f"Ocurrió un error: {e}"

if __name__ == '__main__':
    app.run(debug=True)
