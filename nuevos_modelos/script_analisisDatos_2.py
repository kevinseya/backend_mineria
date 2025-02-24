import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats.mstats import winsorize

# 📥 Carga de datos
df = pd.read_csv('./data/datos_combinados.csv')

# 📅 Convertir fecha y agregar estación del año
df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
df['week'] = df['date'].dt.isocalendar().week
df['year_week'] = df['date'].dt.strftime('%Y-%U')

# 🌦️ Ecuador solo tiene Verano (1) e Invierno (0)
df['season'] = df['date'].dt.month.apply(lambda x: 1 if x in [1, 2, 3, 4, 5, 10, 11, 12] else 0)

# 🔥 Eliminar columnas innecesarias
df.drop(['year', 'month', 'day'], axis=1, inplace=True)

# 🏙️ Agrupación semanal por ciudad
agg_dict = {
    'qty': 'sum',
    'temperature_2m': 'mean',
    'ipc': 'mean',
    'tasa_empleo': 'mean',
    'salario_nominal': 'mean',
    'salario_real': 'mean',
    'festivals': 'max',
    'holiday_flag': 'max'
}

df_weekly = df.groupby(['year_week', 'areacity'], as_index=False).agg(agg_dict)

# 📌 Reagregar `season`
df_season = df.groupby(['year_week', 'areacity'])['season'].agg(lambda x: x.mode()[0]).reset_index()
df_weekly = df_weekly.merge(df_season, on=['year_week', 'areacity'], how='left')

# 🎯 Nueva transformación de qty
df_weekly['qty'] = winsorize(df_weekly['qty'], limits=[0, 0.05])  # Reducir outliers
df_weekly['qty_log'] = np.log1p(df_weekly['qty'])  # Mejor transformación

# 📈 Nueva feature: Tendencia de ventas (media móvil de 4 semanas)
df_weekly['qty_trend'] = df_weekly.groupby('areacity')['qty'].transform(lambda x: x.rolling(window=4, min_periods=1).mean())

# 🚀 Filtrar ciudades con al menos 50 registros
city_counts = df_weekly['areacity'].value_counts()
valid_cities = city_counts[city_counts >= 50].index
df_weekly = df_weekly[df_weekly['areacity'].isin(valid_cities)]

# 📢 Imprimir las ciudades que quedaron
print("\n✅ Ciudades con al menos 50 semanas registradas:")
for city in valid_cities:
    print(f"- {city} ({city_counts[city]} semanas)")

# 📊 Normalizar variables
features = ['festivals', 'holiday_flag', 'temperature_2m', 'ipc', 'tasa_empleo', 'salario_nominal', 'salario_real', 'qty_trend', 'season']
df_weekly['areacity'] = df_weekly['areacity'].astype('category').cat.codes  # Convertir a números

# ❌ Verificar que todas las columnas existen antes de calcular la correlación
missing_features = [col for col in features if col not in df_weekly.columns]
if missing_features:
    print(f"⚠️ Advertencia: Las siguientes columnas no están en df_weekly y serán omitidas: {missing_features}")
    features = [col for col in features if col in df_weekly.columns]

# 🔥 Matriz de correlación mejorada con ajuste de etiquetas
plt.figure(figsize=(12, 8))
corr_matrix = df_weekly[features + ['areacity']].corr()

sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, square=True)

# Rotar las etiquetas para que sean más legibles
plt.xticks(rotation=45, ha='right')  # Rotar los nombres de las columnas en X
plt.yticks(rotation=0)  # Mantener las etiquetas en Y sin rotación

plt.title("Matriz de Correlación Mejorada (Incluyendo Festivos)")
plt.tight_layout()  # Ajustar automáticamente los márgenes para mejor visualización
plt.savefig(r'C:\Users\Mateo\Desktop\Backend_Mineria\nuevos_modelos\graficas\correlacion_mejorada.png')
plt.show()

print("\n✅ Datos mejorados. Revisa la nueva matriz de correlación.")
