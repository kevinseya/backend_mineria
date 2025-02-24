import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import tensorflow as tf
import tensorflow.keras.backend as K
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

#Directorio para guardar gráficas
save_dir = r'C:\Users\Mateo\Desktop\Backend_Mineria\nuevos_modelos\graficas\prueba'
os.makedirs(save_dir, exist_ok=True)

# ========================================================
# (1) CARGA Y PREPROCESAMIENTO DE DATOS
# ========================================================
print("\n📥 Cargando datos...")
df = pd.read_csv('./data/datos_combinados.csv')

# Convertimos la fecha y creamos nuevas features
df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
df['week'] = df['date'].dt.isocalendar().week
df['year_week'] = df['date'].dt.strftime('%Y-%U')

# Ecuador solo tiene Verano (0) e Invierno (1)
df['season'] = df['date'].dt.month.apply(lambda x: 1 if x in [1, 2, 3, 4, 5, 10, 11, 12] else 0)
df.drop(['year', 'month', 'day'], axis=1, inplace=True)

# Crear nueva columna `is_big_city`
big_cities = ['QUITO', 'GUAYAQUIL', 'CUENCA']
df['is_big_city'] = df['areacity'].apply(lambda x: 1 if x in big_cities else 0)

# Incorporamos `es_festivo` y `holiday_flag`
df['es_festivo'] = df['festivals'].fillna(0)
df['holiday_flag'] = df['holiday_flag'].fillna(0)

# Agrupación semanal por ciudad
agg_dict = {
    'qty': 'sum',
    'is_big_city': 'max',
}

df_weekly = df.groupby(['year_week', 'areacity'], as_index=False).agg(agg_dict)

# REAGREGAR `season`
df_season = df.groupby(['year_week', 'areacity'])['season'].agg(lambda x: x.mode()[0]).reset_index()
df_weekly = df_weekly.merge(df_season, on=['year_week', 'areacity'], how='left')

# Eliminamos outliers y aplicamos log()
df_weekly['qty'] = winsorize(df_weekly['qty'], limits=[0, 0.05])
df_weekly['qty_log'] = np.log(df_weekly['qty'] + 1)  # Cambiado a np.log(x + 1)

# Nueva feature: Tendencia de ventas
df_weekly['qty_trend'] = df_weekly.groupby('areacity')['qty'].transform(lambda x: x.rolling(window=4, min_periods=1).mean())

# Normalizar variables
features = ['qty_trend', 'season', 'is_big_city']
scaler = StandardScaler()
df_weekly[features] = scaler.fit_transform(df_weekly[features])

SCALER_PATH = "nuevos_modelos/graficas/prueba/scaler.pkl"
joblib.dump(scaler, SCALER_PATH)
print(f"✅ Scaler guardado en: {SCALER_PATH}")
# Eliminamos variables innecesarias
df_weekly.drop(columns=['year_week'], inplace=True)

# ========================================================
# (2) CREACIÓN DE SECUENCIAS PARA LSTM
# ========================================================
ventana = 16
X_feats, y_all = [], []

for _, subset in df_weekly.groupby('areacity'):
    target_vals = subset['qty_log'].values
    feats = subset[features].values
    
    for i in range(ventana, len(subset)):
        X_feats.append(feats[i-ventana:i])
        y_all.append(target_vals[i])

X_feats = np.array(X_feats)
y_all = np.array(y_all)

# Agregamos ruido gaussiano para mejorar la robustez
X_feats_noisy = X_feats + np.random.normal(loc=0.0, scale=0.005, size=X_feats.shape)

print(f"📊 Total secuencias generadas: {X_feats.shape[0]}")

# ========================================================
# (3) MODELO MEJORADO (Conv1D + LSTM)
# ========================================================
print("\n[⚙️] Construyendo modelo mejorado...")

# Entrada de features exógenas
feats_input = Input(shape=(ventana, len(features)), name='feats_input')

# Conv1D + LSTM
x = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(feats_input)
x = MaxPooling1D(pool_size=2)(x)
x = LSTM(128, return_sequences=True, dropout=0.15)(x)
x = LSTM(32, return_sequences=False, dropout=0.15)(x)

fusion = Dense(64, activation='relu')(x)
fusion = BatchNormalization()(fusion)
fusion = Dropout(0.25)(fusion)

output = Dense(1, activation='linear')(fusion)

# ========================================================
# (4) ENTRENAMIENTO DEL MODELO
# ========================================================
print("\n[🚀] Entrenando modelo...")

def custom_loss(y_true, y_pred):
    error = K.abs(y_true - y_pred)
    penalty = K.exp(y_true / K.max(y_true))  # Penaliza más valores grandes
    return K.mean(error * penalty)

model = Model(inputs=[feats_input], outputs=output)
model.compile(optimizer=Adam(learning_rate=0.0002), loss=custom_loss)

history = model.fit(
    x=[X_feats], y=y_all,
    epochs=100,
    batch_size=16,
    validation_split=0.15,
    verbose=1,
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)
    ]
)

print("\n✅ Entrenamiento Completado.")

# ========================================================
# (5) EVALUACIÓN Y GRÁFICAS
# ========================================================
print("\n[📊] Evaluando modelo...")

y_pred_log = model.predict([X_feats]).flatten()
y_pred = np.exp(y_pred_log) - 1  # Cambiado para corresponder con np.log(x + 1)
y_real = np.exp(y_all) - 1

# 📊 Métricas de evaluación
import math
mse = np.mean((y_real - y_pred) ** 2)
rmse = math.sqrt(mse)
mae = np.mean(np.abs(y_real - y_pred))

def weighted_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    w = y_true / (y_true.sum() + 1e-10)
    rel_error = np.abs(y_true - y_pred) / (y_true + 1e-10)
    return np.sum(w * rel_error) * 100

wmape = weighted_mape(y_real, y_pred)
precision = 100 - wmape

print(f"\n===== MÉTRICAS DEL MODELO MEJORADO =====")
print(f"📌 RMSE        : {rmse:.2f}")
print(f"📌 MAE         : {mae:.2f}")
print(f"📌 wMAPE       : {wmape:.2f}%")
print(f"✅ PRECISIÓN   : {precision:.2f}% (100 - wMAPE)")

# 📈 Guardado de Gráficas
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
plt.xlabel("Épocas")
plt.ylabel("Loss")
plt.title("Curva de Pérdida LSTM Mejorado")
plt.legend()
plt.grid(True, linestyle="--")
plt.savefig(os.path.join(save_dir, 'training_loss_mejorado_10--10.png'))
plt.show()

# 📈 Comparación de Predicciones vs. Valores Reales
plt.figure(figsize=(10, 5))
sns.scatterplot(x=y_real, y=y_pred, alpha=0.5, color="blue")
plt.plot([min(y_real), max(y_real)], [min(y_real), max(y_real)], color="red", linestyle="--")
plt.xlabel("Ventas Reales")
plt.ylabel("Predicciones")
plt.title("Comparación de Predicciones vs. Valores Reales (Mejorado)")
plt.savefig(os.path.join(save_dir, 'comparacion_mejorado_10--10.png'))
plt.show()

# ========================================================
# (5.5) GUARDADO DE MODELO
# ========================================================
model_save_path = r'C:\Users\Mateo\Desktop\Backend_Mineria\nuevos_modelos\graficas\prueba\modelo_final---10.keras'
model.save(model_save_path)
print(f"✅ Modelo guardado en: {model_save_path}")

# ========================================================
# (6) CONJUNTO TEST CON DATOS REALES
# ========================================================

# 📅 Recalcular year_week si es necesario
if 'year_week' not in df_weekly.columns:
    df['date'] = pd.to_datetime(df['date'])
    df['year_week'] = df['date'].dt.strftime('%Y-%U')
    df_weekly = df.groupby(['year_week', 'areacity'], as_index=False).agg(agg_dict)
    df_season = df.groupby(['year_week', 'areacity'])['season'].agg(lambda x: x.mode()[0]).reset_index()
    df_weekly = df_weekly.merge(df_season, on=['year_week', 'areacity'], how='left')

# 📅 Convertir year_week a datetime
df_weekly['date'] = pd.to_datetime(df_weekly['year_week'] + '-1', format='%Y-%U-%w')
df_weekly = df_weekly.sort_values(by=['date', 'areacity'])

# 🔄 Separar conjunto de test
test_size = 30
df_test = df_weekly.groupby('areacity').tail(test_size).copy()
df_train = df_weekly.drop(df_test.index)

# 🔥 Aplicar transformaciones
df_test['qty_log'] = np.log(df_test['qty'] + 1)  # Cambiado a np.log(x + 1)
df_train['qty_log'] = np.log(df_train['qty'] + 1)

df_test['qty_trend'] = df_test.groupby('areacity')['qty'].transform(lambda x: x.rolling(window=4, min_periods=1).mean())
df_train['qty_trend'] = df_train.groupby('areacity')['qty'].transform(lambda x: x.rolling(window=4, min_periods=1).mean())

df_test[features] = scaler.transform(df_test[features])

print(f"📊 Tamaño del conjunto de entrenamiento: {df_train.shape[0]}")
print(f"📊 Tamaño del conjunto de test: {df_test.shape[0]}")

# 🔥 Crear secuencias para test
X_test_feats, y_test_real = [], []

for _, subset in df_test.groupby('areacity'):
    target_vals = subset['qty_log'].values
    feats = subset[features].values
    
    if len(subset) > ventana:
        for i in range(ventana, len(subset)):
            X_test_feats.append(feats[i-ventana:i])
            y_test_real.append(target_vals[i])

X_test_feats = np.array(X_test_feats)
y_test_real = np.array(y_test_real)

print(f"📊 Total secuencias generadas para test: {X_test_feats.shape[0]}")

# 🔮 Predicciones en test
y_test_pred_log = model.predict([X_test_feats]).flatten()
y_test_pred = np.exp(y_test_pred_log) - 1  # Cambiado para corresponder con np.log(x + 1)
y_test_real = np.exp(y_test_real) - 1

# 📊 Métricas en test
mse_test = np.mean((y_test_real - y_test_pred) ** 2)
rmse_test = np.sqrt(mse_test)
mae_test = np.mean(np.abs(y_test_real - y_test_pred))
wmape_test = weighted_mape(y_test_real, y_test_pred)
precision_test = 100 - wmape_test

print("\n===== MÉTRICAS DEL MODELO EN TEST =====")
print(f"📌 RMSE        : {rmse_test:.2f}")
print(f"📌 MAE         : {mae_test:.2f}")
print(f"📌 wMAPE       : {wmape_test:.2f}%")
print(f"✅ PRECISIÓN   : {precision_test:.2f}% (100 - wMAPE)")

# 📊 Gráficas finales
plt.figure(figsize=(10, 5))
sns.scatterplot(x=y_test_real, y=y_test_pred, alpha=0.5, color="blue")
plt.plot([min(y_test_real), max(y_test_real)], [min(y_test_real), max(y_test_real)], color="red", linestyle="--")
plt.xlabel("Ventas Reales")
plt.ylabel("Predicciones")
plt.title("Comparación de Predicciones vs. Valores Reales en Test")
plt.grid(True, linestyle="--")
plt.savefig(os.path.join(save_dir, 'comparacion_test--10.png'))
plt.show()

# 📈 Serie de Tiempo
plt.figure(figsize=(12, 6))
plt.plot(range(len(y_test_real)), y_test_real, label="Ventas Reales", color="blue", marker="o")
plt.plot(range(len(y_test_pred)), y_test_pred, label="Predicciones", color="orange", linestyle="--", marker="s")
plt.xlabel("Tiempo (Semanas)")
plt.ylabel("Cantidad Vendida")
plt.title("Evolución de Ventas Reales vs. Predicciones en Test")
plt.legend()
plt.grid(True, linestyle="--")
plt.savefig(os.path.join(save_dir, 'series_tiempo_test---10.png'))
plt.show()

# 📊 Histograma de Errores en Test
errores = y_test_real - y_test_pred
plt.figure(figsize=(10, 5))
sns.histplot(errores, bins=30, kde=True, color="purple")
plt.axvline(x=0, color="red", linestyle="--")
plt.xlabel("Error (Real - Predicho)")
plt.ylabel("Frecuencia")
plt.title("Distribución de Errores en Test")
plt.grid(True, linestyle="--")
plt.savefig(os.path.join(save_dir, 'histograma_errores---10.png'))
plt.show()
