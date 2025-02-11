from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import MeanAbsoluteError  # Para custom_objects en el load_model
from sklearn.preprocessing import LabelEncoder
from flask_cors import CORS  # Importa CORS

app = Flask(__name__)
CORS(app)
# ======================================================
# CARGA DEL MODELO Y CONFIGURACIÓN DEL LABELENCODER
# ======================================================

# Cargar el modelo LSTM entrenado
model = load_model('model/lstm_model.h5', custom_objects={'mae': MeanAbsoluteError})

# Definir el mapeo fijo tal como se usó en el entrenamiento
cities_entrenamiento = [
    "AMBATO", "AZOGUES", "CAYAMBE", "CHONE", "CUENCA", "ESMERALDAS",
    "GUAYAQUIL", "IBARRA", "LATACUNGA", "LOJA", "MACHALA", "MANTA",
    "MILAGRO", "PAUTE", "PENINSULA DE SANTA ELENA", "PINAS", "PUYO",
    "QUEVEDO", "QUITO", "RIOBAMBA", "STO. DOMINGO", "ZAMORA"
]

le_area = LabelEncoder()
le_area.fit(cities_entrenamiento)

# ======================================================
# Endpoint 1: /predict_manual
# Se espera que el usuario envíe un JSON con la secuencia completa
# de 8 quincenas (historical_data) con las 11 características.
# ======================================================

@app.route('/predict_manual', methods=['POST'])
def predict_manual():
    content = request.get_json()

    # Validar que se envíen los campos requeridos
    areacity = content.get('areacity')
    historical_data = content.get('historical_data')
    if areacity is None or historical_data is None:
        return jsonify({'error': 'Se requieren los campos "areacity" y "historical_data"'}), 400

    # Verificar que se hayan enviado exactamente 8 registros
    if len(historical_data) != 8:
        return jsonify({'error': 'Se requieren exactamente 8 registros en "historical_data"'}), 400

    # Convertir la lista de registros en un DataFrame
    try:
        df_hist = pd.DataFrame(historical_data)
        df_hist['quincena_start'] = pd.to_datetime(df_hist['quincena_start'])
        df_hist = df_hist.sort_values('quincena_start')
    except Exception as e:
        return jsonify({'error': f'Error procesando los datos históricos: {e}'}), 400

    # Definir los campos requeridos para el modelo (en el mismo orden usado en el entrenamiento)
    required_fields = [
        'holiday_flag', 'festivals', 'isWeekend', 'is_big_city', 'is_low_sale',
        'month_sin', 'month_cos', 'quarter_sin', 'quarter_cos', 'quincena_sin', 'quincena_cos'
    ]
    for field in required_fields:
        if field not in df_hist.columns:
            return jsonify({'error': f'El campo requerido "{field}" no se encuentra en historical_data'}), 400

    # Extraer la matriz de características y reestructurarla a forma (1, 8, 11)
    X_feats = df_hist[required_fields].values.reshape(1, 8, 11)

    # Usar el LabelEncoder ya configurado para codificar la ciudad
    try:
        city_code = int(le_area.transform([areacity])[0])
    except Exception as e:
        return jsonify({'error': f'Error al codificar la ciudad: {e}'}), 500

    # Crear la secuencia para el embedding (repetir el código 8 veces)
    X_city_id = np.array([[city_code] * 8])

    # Realizar la predicción (la salida está en log; se aplica np.expm1 para volver a la escala original)
    predicted_sales_log = model.predict([X_feats, X_city_id]).flatten()
    predicted_sales = np.expm1(predicted_sales_log)[0]
    # Redondear la predicción para mostrar un valor entero
    predicted_sales_int = int(round(predicted_sales))

    return jsonify({
        'areacity': areacity,
        'predicted_sales': predicted_sales_int,
        'mensaje': 'La predicción corresponde a la quincena siguiente a la última registrada en historical_data.'
    })



# ======================================================
# Endpoint 2: /predict_csv
# Se utiliza el archivo preprocesado "quincenas_preprocesadas.csv"
# para extraer las últimas 8 quincenas de la ciudad solicitada
# (anteriores a la fecha de corte) y predecir la siguiente quincena.
#
# El JSON de entrada se espera que tenga la siguiente estructura:
#
# {
#    "areacity": "QUITO",
#    "quincena_start": "2025-02-01"
# }
#
# ======================================================

# Cargar el CSV preprocesado una sola vez (al inicio)
try:
    preprocessed_df = pd.read_csv('data/quincenas_preprocesadas.csv')
    preprocessed_df['quincena_start'] = pd.to_datetime(preprocessed_df['quincena_start'])
except Exception as e:
    raise Exception(f"Error al cargar el archivo quincenas_preprocesadas.csv: {e}")

@app.route('/predict_csv', methods=['POST'])
def predict_csv():
    content = request.get_json()

    # Validar campos obligatorios
    areacity = content.get('areacity')
    quincena_start_str = content.get('quincena_start')
    holiday_flag = content.get('holiday_flag')
    festivals = content.get('festivals')
    isWeekend = content.get('isWeekend')
    
    missing_fields = []
    for field in ['areacity', 'quincena_start', 'holiday_flag', 'festivals', 'isWeekend']:
        if content.get(field) is None:
            missing_fields.append(field)
    if missing_fields:
        return jsonify({'error': f"Faltan los siguientes campos: {', '.join(missing_fields)}"}), 400

    # Convertir la fecha de corte a datetime
    try:
        cutoff_date = pd.to_datetime(quincena_start_str)
    except Exception as e:
        return jsonify({'error': f'Formato de fecha inválido en quincena_start: {e}'}), 400

    # Filtrar el CSV para la ciudad indicada y seleccionar registros anteriores a la fecha de corte
    df_city = preprocessed_df[preprocessed_df['areacity'] == areacity].copy()
    df_city = df_city[df_city['quincena_start'] < cutoff_date].sort_values('quincena_start')
    
    # Se necesitan al menos 7 registros históricos para luego agregar el nuevo registro
    if len(df_city) < 7:
        return jsonify({'error': 'No hay suficientes datos históricos para esta ciudad. Se requieren al menos 7 registros.'}), 400

    # Tomar los últimos 7 registros históricos
    df_hist = df_city.tail(7).copy()

    # Construir una nueva fila a partir de los datos de entrada
    new_q_start = pd.to_datetime(quincena_start_str)
    # ¡Asegúrate de que la función get_quincena_start esté definida en el archivo (antes de este endpoint)!
    base_q_start = get_quincena_start(new_q_start)
    
    # Calcular mes y sus funciones cíclicas
    month = new_q_start.month
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    # Calcular quarter y sus funciones cíclicas
    quarter = new_q_start.quarter
    quarter_sin = np.sin(2 * np.pi * quarter / 4)
    quarter_cos = np.cos(2 * np.pi * quarter / 4)
    # Calcular quincena (0 para primera, 1 para segunda)
    quincena_val = 0 if base_q_start.day == 1 else 1
    quincena_sin = np.sin(2 * np.pi * quincena_val / 2)
    quincena_cos = np.cos(2 * np.pi * quincena_val / 2)
    
    # Determinar is_big_city a partir del mapeo (usando el LabelEncoder ya configurado)
    try:
        city_code = int(le_area.transform([areacity])[0])
    except Exception as e:
        return jsonify({'error': f'Error al codificar la ciudad: {e}'}), 500
    is_big_city = 1 if city_code in [6, 18, 20] else 0
    # is_low_sale se setea a 0 (no se conoce la venta futura)
    is_low_sale = 0

    # Construir la nueva fila (con el mismo orden de columnas que en el entrenamiento)
    new_row = {
        'holiday_flag': holiday_flag,
        'festivals': festivals,
        'isWeekend': isWeekend,
        'is_big_city': is_big_city,
        'is_low_sale': is_low_sale,
        'month_sin': month_sin,
        'month_cos': month_cos,
        'quarter_sin': quarter_sin,
        'quarter_cos': quarter_cos,
        'quincena_sin': quincena_sin,
        'quincena_cos': quincena_cos
    }
    df_new = pd.DataFrame([new_row])
    df_new['quincena_start'] = new_q_start

    # Concatenar los 7 registros históricos con la nueva fila para obtener una secuencia de 8 registros
    df_sequence = pd.concat([df_hist, df_new], ignore_index=True)

    # Extraer las 11 características en el orden correcto
    feature_cols = [
        'holiday_flag', 'festivals', 'isWeekend', 'is_big_city', 'is_low_sale',
        'month_sin', 'month_cos', 'quarter_sin', 'quarter_cos', 'quincena_sin', 'quincena_cos'
    ]
    X_feats = df_sequence[feature_cols].values.reshape(1, 8, 11)

    # Crear la secuencia para el embedding: (1, 8) repitiendo el código de la ciudad
    X_city_id = np.array([[city_code] * 8])

    # Realizar la predicción (la salida está en log; se aplica np.expm1 para volver a la escala original)
    predicted_sales_log = model.predict([X_feats, X_city_id]).flatten()
    predicted_sales = np.expm1(predicted_sales_log)[0]
    # Redondear la predicción a entero
    predicted_sales_int = int(round(predicted_sales))

    # Calcular la fecha de la quincena predicha a partir de la última quincena de la secuencia
    last_quincena = pd.to_datetime(df_sequence['quincena_start'].iloc[-1])
    if last_quincena.day == 1:
        predicted_quincena_date = last_quincena.replace(day=16)
    else:
        year = last_quincena.year
        month = last_quincena.month
        if month == 12:
            next_month = 1
            next_year = year + 1
        else:
            next_month = month + 1
            next_year = year
        predicted_quincena_date = pd.to_datetime(f"{next_year}-{next_month:02d}-01")

    return jsonify({
        'areacity': areacity,
        'predicted_sales': predicted_sales_int,
        'predicted_quincena': str(predicted_quincena_date.date()),
        'mensaje': 'La predicción se realizó usando 7 registros históricos del CSV y una nueva fila construida a partir de la información de entrada.'
    })

def get_quincena_start(date):
    """Devuelve la fecha base de la quincena (día 1 o 16)."""
    return pd.to_datetime(f"{date.year}-{date.month:02d}-{1 if date.day <= 15 else 16}")

# ======================================================
# Arranque de la aplicación
# ======================================================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
