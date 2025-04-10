
# 📱 Predicción de Gama de Dispositivos Móviles con Flask + Machine Learning

Este proyecto es una aplicación web que permite predecir la gama de un dispositivo móvil (baja, media o alta) utilizando características físicas como peso, RAM, resolución de cámaras y capacidad de la batería. El modelo de clasificación ha sido entrenado con un algoritmo Random Forest y guardado en un archivo .pkl.

## 🛠 Tecnologías utilizadas
- Flask: Framework web para la interfaz de usuario.
- scikit-learn: Para el entrenamiento y evaluación del modelo de clasificación.
## 📊 Variables utilizadas

| Variable             | Descripción                                                                |
| ----------------- | ------------------------------------------------------------------ |
| Mobile_Weight | Peso del móvil en gramos |
| RAM | Memoria RAM en GB |
| Front_Camera | Resolución de la cámara frontal (MP) |
| Back_Camera | 	Resolución de la cámara trasera (MP) |
| Battery_Capacity | Capacidad de la batería (mAh) |
| Screen_Size | 	Tamaño de la pantalla en pulgadas|

## 🚀 Pasos para ejecutar el proyecto

### **1️⃣ Clonar el repositorio**

```bash
git clone https://github.com/tu_usuario/prediccion-gama-movil.git
cd prediccion-gama-movil
```

### **2️⃣  Instalar las dependencias**
Es necesario tener Python instalado (recomendado Python 3.7 o superior). Luego, instalar las dependencias:
```bash
pip install -r requirements.txt
```

### **3️⃣ Ejecutar la aplicación**

```bash
python app.py
```

## LINK COLAB

```bash
  https://colab.research.google.com/drive/1QyOUCR5grgGWW_uyA5xtj_B0My9C-Iqj?usp=sharing
```


## 🧑‍💻 Créditos

Este proyecto fue desarrollado por:

- Juan Camilo Rodríguez
- Arturo Barona
