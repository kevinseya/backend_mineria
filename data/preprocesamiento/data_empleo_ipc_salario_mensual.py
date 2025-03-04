import csv

# Diccionario con las tasas de empleo por trimestre
tasa_empleo_trimestral = {
    "2021": {1: 40.1, 2: 40.1, 3: 38.3, 4: 41.8},
    "2022": {1: 41.2, 2: 42.0, 3: 42.6, 4: 44.4},
    "2023": {1: 44.3, 2: 46.4, 3: 45.7, 4: 45.5},
    "2024": {1: 44.8, 2: 44.7, 3: 44.3, 4: 43.7}
}

# Lista con los salarios nominales y reales
datos_salarios = [
    ["2021", 1, 466.67, 111.97], ["2021", 2, 466.67, 111.88], ["2021", 3, 466.67, 111.68],
    ["2021", 4, 466.67, 111.29], ["2021", 5, 466.67, 111.20], ["2021", 6, 466.67, 111.40],
    ["2021", 7, 466.67, 110.81], ["2021", 8, 466.67, 110.69], ["2021", 9, 466.67, 110.67],
    ["2021", 10, 466.67, 110.44], ["2021", 11, 466.67, 110.04], ["2021", 12, 466.67, 109.97],
    ["2022", 1, 495.66, 115.96], ["2022", 2, 495.66, 115.69], ["2022", 3, 495.66, 115.56],
    ["2022", 4, 495.66, 114.88], ["2022", 5, 495.66, 114.25], ["2022", 6, 495.66, 113.51],
    ["2022", 7, 495.66, 113.33], ["2022", 8, 495.66, 113.30], ["2022", 9, 495.66, 112.89],
    ["2022", 10, 495.66, 112.76], ["2022", 11, 495.66, 112.77], ["2022", 12, 495.66, 112.59],
    ["2023", 1, 524.83, 119.07], ["2023", 2, 524.83, 119.05], ["2023", 3, 524.83, 118.97],
    ["2023", 4, 524.83, 118.74], ["2023", 5, 524.83, 118.64], ["2023", 6, 524.83, 118.19],
    ["2023", 7, 524.83, 117.56], ["2023", 8, 524.83, 116.97], ["2023", 9, 524.83, 116.93],
    ["2023", 10, 524.83, 117.13], ["2023", 11, 524.83, 117.60], ["2023", 12, 524.83, 117.63],
    ["2024", 1, 536.60, 120.12], ["2024", 2, 536.60, 120.01], ["2024", 3, 536.60, 119.66],
    ["2024", 4, 536.60, 118.16], ["2024", 5, 536.60, 118.30], ["2024", 6, 536.60, 119.44],
    ["2024", 7, 536.60, 118.33], ["2024", 8, 536.60, 118.08], ["2024", 9, 536.60, 117.87],
    ["2024", 10, 536.60, 118.15], ["2024", 11, 536.60, 118.46],["2024", 12, 536.60, 118.46]
]

# Lista con el IPC
datos_ipc = [
    ["2021", 1, 104.35], ["2021", 2, 104.44], ["2021", 3, 104.63],
    ["2021", 4, 104.99], ["2021", 5, 105.08], ["2021", 6, 104.89],
    ["2021", 7, 105.45], ["2021", 8, 105.57], ["2021", 9, 105.58],
    ["2021", 10, 100.8], ["2021", 11, 106.18], ["2021", 12, 106.26],
    ["2022", 1, 107.02], ["2022", 2, 107.27], ["2022", 3, 107.39],
    ["2022", 4, 108.03], ["2022", 5, 109.34], ["2022", 6, 109.51],
    ["2022", 7, 109.54], ["2022", 8, 109.93], ["2022", 9, 110.06],
    ["2022", 10, 109.93], ["2022", 11, 110.05], ["2022", 12, 110.23],
    ["2023", 1, 110.36], ["2023", 2, 110.38], ["2023", 3, 110.45],
    ["2023", 4, 110.67], ["2023", 5, 110.77], ["2023", 6, 111.18],
    ["2023", 7, 111.78], ["2023", 8, 112.34], ["2023", 9, 112.39],
    ["2023", 10, 112.19], ["2023", 11, 111.74], ["2023", 12, 111.72],
    ["2024", 1, 111.86], ["2024", 2, 111.96], ["2024", 3, 112.28],
    ["2024", 4, 113.71], ["2024", 5, 113.58], ["2024", 6, 112.49],
    ["2024", 7, 113.54], ["2024", 8, 113.79], ["2024", 9, 113.99],
    ["2024", 10, 113.72], ["2024", 11, 113.42], ["2024", 12, 112.31]
]

# Crear el archivo CSV con los datos mensuales
def crear_csv_tasa_empleo(archivo_csv):
    with open(archivo_csv, mode='w', newline='', encoding='utf-8') as archivo:
        writer = csv.writer(archivo)

        # Escribir el encabezado
        writer.writerow(["item", "year", "month", "tasa_empleo", "salario_nominal", "salario_real", "ipc"])

        item = 1  # Contador para el identificador único (item)
        for year, trimestres in tasa_empleo_trimestral.items():
            for trimestre, tasa in trimestres.items():
                for month in range((trimestre - 1) * 3 + 1, trimestre * 3 + 1):
                    # Obtener los datos de salarios e IPC para el mes actual
                    salario = next((d for d in datos_salarios if int(d[0]) == int(year) and d[1] == month), None)
                    salario_nominal = salario[2] if salario else None
                    salario_real = salario[3] if salario else None
                    ipc = next((d[2] for d in datos_ipc if int(d[0]) == int(year) and d[1] == month), None)
                    writer.writerow([f"{item}", year, month, tasa, salario_nominal, salario_real, ipc])
                    item += 1

# Nombre del archivo CSV
nombre_archivo = "tasa_empleo_salarios_ipc.csv"
crear_csv_tasa_empleo(nombre_archivo)

print(f"Archivo '{nombre_archivo}' creado exitosamente.")
