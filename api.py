import os
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
import tensorflow.keras.backend as K

# Función de pérdida personalizada
def custom_loss(y_true, y_pred):
    error = K.abs(y_true - y_pred)
    penalty = K.exp(y_true / K.max(y_true))  
    return K.mean(error * penalty)

# Cargar modelo
MODEL_PATH = "nuevos_modelos/modelo_final---10.keras"
model = load_model(MODEL_PATH, custom_objects={"custom_loss": custom_loss})

# Columnas de entrada del modelo
FEATURES = ['qty_trend', 'season', 'is_big_city']

# Cargar el scaler entrenado
SCALER_PATH = "nuevos_modelos/scaler.pkl"
if os.path.exists(SCALER_PATH):
    scaler = joblib.load(SCALER_PATH)
    print(f"Scaler cargado desde {SCALER_PATH}")
else:
    raise FileNotFoundError("No se encontró scaler.pkl. Debes entrenar el modelo y guardarlo.")

# Inicializar FastAPI
app = FastAPI()

# Configurar CORS
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelo de solicitud para `/predict_custom`
class PredictionCustomRequest(BaseModel):
    ciudad: str
    fecha_inicio: str
    fecha_fin: str
    ultimas_semanas: list  

# Función para obtener el lunes de una semana dada
def get_monday_of_week(year_week):
    year, week = map(int, year_week.split('-'))
    return datetime.strptime(f'{year}-W{week}-1', "%Y-W%W-%w").date()

# Función para preparar la secuencia a partir de datos custom
def preparar_secuencia_custom(ultimas_semanas: list):
    if len(ultimas_semanas) != 16:
        return None

    df_custom = pd.DataFrame(ultimas_semanas)
    try:
        input_data = df_custom[FEATURES]
    except KeyError:
        return None
    
    input_data = scaler.transform(input_data)
    input_data = np.expand_dims(input_data, axis=0)
    return input_data

# Endpoint para predecir un rango de semanas usando datos
@app.post("/predict_custom")
def predict_custom(request: PredictionCustomRequest):
    try:
        fecha_inicio = pd.to_datetime(request.fecha_inicio)
        fecha_fin = pd.to_datetime(request.fecha_fin)
        semanas_unicas = sorted(set(pd.date_range(start=fecha_inicio, end=fecha_fin, freq='D').strftime('%Y-%U')))
        
        if not semanas_unicas:
            raise HTTPException(status_code=400, detail="El rango de fechas no generó semanas válidas.")
        
        secuencia = preparar_secuencia_custom(request.ultimas_semanas)
        if secuencia is None:
            raise HTTPException(status_code=400, detail="Debes proporcionar exactamente 16 semanas de datos con las claves: qty_trend, season e is_big_city.")

        predicciones = []
        for week in semanas_unicas:
            fecha_referencia = get_monday_of_week(week)
            
            prediction = model.predict(secuencia)[0][0]
            predicted_qty = np.exp(prediction) - 1

            predicciones.append({
                "semana": week,
                "fecha_referencia": fecha_referencia.strftime('%Y-%m-%d'),
                "ventas_predichas": int(predicted_qty)
            })

            nueva_fila = np.array([[predicted_qty, fecha_referencia.month % 2, 1]])
            nueva_fila = scaler.transform(nueva_fila)
            secuencia = np.append(secuencia[:, 1:, :], np.expand_dims(nueva_fila, axis=0), axis=1)

        return {
            "ciudad": request.ciudad,
            "fecha_inicio": request.fecha_inicio,
            "fecha_fin": request.fecha_fin,
            "predicciones_semanales": predicciones
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
