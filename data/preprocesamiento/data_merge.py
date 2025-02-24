import pandas as pd

# Definir las rutas de los archivos CSV
ruta_venta_historica = r"C:\Users\Mateo\Desktop\Backend_Mineria\data\pura\data_ventas.csv"
ruta_temperaturas = r"C:\Users\Mateo\Desktop\Backend_Mineria\data\pura\temperaturas_diarias_actualizado.csv"
ruta_ipc = r"C:\Users\Mateo\Desktop\Backend_Mineria\data\pura\tasa_empleo_salarios_ipc.csv"

# Leer los CSV y optimizar los tipos de datos
df_venta_historica = pd.read_csv(ruta_venta_historica, usecols=['item', 'area','areacity', 'year', 'month', 'day', 'qty'])
df_venta_historica['year'] = df_venta_historica['year'].astype('int16')
df_venta_historica['month'] = df_venta_historica['month'].astype('int8')
df_venta_historica['day'] = df_venta_historica['day'].astype('int8')

df_temperaturas = pd.read_csv(ruta_temperaturas, usecols=['areacity', 'year', 'month', 'day', 'temperature_2m', 
                                                           'climateCondition', 'holidays','festivals','isWeekend'])
df_temperaturas['year'] = df_temperaturas['year'].astype('int16')
df_temperaturas['month'] = df_temperaturas['month'].astype('int8')
df_temperaturas['day'] = df_temperaturas['day'].astype('int8')

df_ipc = pd.read_csv(ruta_ipc, usecols=['year', 'month', 'tasa_empleo', 'salario_nominal', 'salario_real', 'ipc'])
df_ipc['year'] = df_ipc['year'].astype('int16')
df_ipc['month'] = df_ipc['month'].astype('int8')

# Inicializar df_final antes del ciclo
df_final = pd.DataFrame()

# Dividir df_venta_historica por año y hacer merge en partes
for year in df_venta_historica['year'].unique():
    df_venta_year = df_venta_historica[df_venta_historica['year'] == year]
    df_temperaturas_year = df_temperaturas[df_temperaturas['year'] == year]
    df_ipc_year = df_ipc[df_ipc['year'] == year]

    # Merge de las partes del año con la data de temperaturas (según area y fecha)
    df_merged = pd.merge(df_venta_year, df_temperaturas_year, 
                         left_on=['areacity', 'year', 'month', 'day'], 
                         right_on=['areacity', 'year', 'month', 'day'], how='left')

    # Merge con la data de IPC (según fecha)
    df_merged = pd.merge(df_merged, df_ipc_year, on=['year', 'month'], how='left')

    # Concatenar el resultado al DataFrame final
    df_final = pd.concat([df_final, df_merged])

# Guardar el archivo final combinado
df_final.to_csv(r'C:\Users\Mateo\Desktop\Backend_Mineria\data\datos_combinados.csv', index=False)

print("Proceso completado. El archivo combinado se ha guardado como 'datos_combinados.csv'.")
