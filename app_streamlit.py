import streamlit as st
import pandas as pd
import pickle
import os

# Configuración de la página
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered"
)

# Título y descripción
st.title("🚢 Predicción de Supervivencia: Proyecto Titanic")
st.markdown("""
Esta aplicación utiliza un modelo de **Machine Learning (SVM)** para predecir si un pasajero 
habría sobrevivido al naufragio del Titanic basándose en sus datos personales.
""")

# Carga del modelo de forma robusta
MODEL_PATH = 'notebooks/modelo_titanic.pkl'

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            return pickle.load(f)
    return None

model = load_model()

if model is None:
    st.error(f"No se encontró el archivo '{MODEL_PATH}'. Asegúrate de haberlo exportado desde el Notebook.")
else:
    # Formulario de entrada de datos
    st.sidebar.header("📋 Datos del Pasajero")
    
    with st.form("input_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            pclass = st.selectbox("Clase del Pasajero", [1, 2, 3], help="1 = Primera, 2 = Segunda, 3 = Tercera")
            sex = st.selectbox("Género", ["male", "female"])
            age = st.slider("Edad", 0, 80, 25)
            
        with col2:
            embarked = st.selectbox("Puerto de Embarque", ["S", "C", "Q"], help="S = Southampton, C = Cherbourg, Q = Queenstown")
            fare = st.number_input("Tarifa del billete", min_value=0.0, value=32.0)
            sibsp = st.number_input("Hermanos/Cónyuges a bordo", min_value=0, value=0)
            parch = st.number_input("Padres/Hijos a bordo", min_value=0, value=0)
            
        submit_button = st.form_submit_button("Realizar Predicción")

    if submit_button:
        # Crear DataFrame con los nombres de columnas exactos que espera el modelo
        input_df = pd.DataFrame([{
            "pclass": pclass,
            "sex": sex,
            "age": age,
            "sibsp": sibsp,
            "parch": parch,
            "fare": fare,
            "embarked": embarked
        }])

        # Realizar la predicción
        prediction = model.predict(input_df)[0]
        # Obtener probabilidades (si el modelo lo permite)
        try:
            prob = model.predict_proba(input_df)[0][1]
        except:
            prob = None

        # Mostrar resultados
        st.divider()
        if prediction == 1:
            st.success(f"### 🎉 Resultado: EL PASAJERO SOBREVIVE")
            if prob:
                st.info(f"Probabilidad estimada de supervivencia: **{prob*100:.2f}%**")
        else:
            st.error(f"### ❌ Resultado: EL PASAJERO NO SOBREVIVE")
            if prob:
                st.info(f"Probabilidad estimada de supervivencia: **{(prob)*100:.2f}%**")