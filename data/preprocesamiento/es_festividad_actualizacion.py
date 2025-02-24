import pandas as pd
from datetime import datetime, timedelta

# Diccionario de festividades
festividades = {
    'AMBATO': [
        ('Febrero', 'Carnaval de Ambato'),
    ],
    'AZOGUES': [
        ('1 de enero', 'Fiesta de la Virgen de la Nube'),
        ('16 de abril', 'Cantonización de Azogues')
    ],
    'CAYAMBE': [
        ('21-23 de junio', 'Fiestas del Sol - Inti Raymi'),
        ('29 de junio', 'Fiestas de San Pedro y San Pablo')
    ],
    'CHONE': [
        ('7 de agosto', 'Fiestas de San Cayetano'),
        ('24 de julio', 'Cantonización de Chone')
    ],
    'CUENCA': [
        ('1-3 de noviembre', 'Independencia de Cuenca'),
        ('2 de mayo', 'Festival de las Cruces')
    ],
    'ESMERALDAS': [
        ('5 de agosto', 'Independencia de Esmeraldas'),
        ('Junio', 'Festival Internacional de Música y Danza Afro'),
        ('5-7 de enero', 'Gobierno de mujeres en el Onzole')
    ],
    'GUAYAQUIL': [
        ('9-12 de octubre', 'Independencia de Guayaquil'),
        ('25 de julio', 'Fiestas de fundación de Guayaquil'),
        ('Octubre', 'Festival de Artes al Aire Libre - FAAL')
    ],
    'IBARRA': [
        ('Abril', 'Festival del Retorno'),
        ('16 de julio', 'Celebración de la Virgen del Carmen'),
        ('24-28 de septiembre', 'Fiestas de Ibarra'),
        ('29 de junio', 'San Pedro y San Pablo'),
        ('7 de octubre', 'Cacería del Zorro')
    ],
    'LATACUNGA': [
        ('Noviembre', 'Fiesta de la Mama Negra'),
        ('11 de noviembre', 'Independencia de Latacunga'),
        ('24-27 de septiembre', 'Fiesta de la Virgen de las Mercedes')
    ],
    'LOJA': [
        ('18 de noviembre', 'Independencia de Loja'),
        ('Noviembre', 'Festival Internacional de Artes Vivas'),
        ('15 de agosto', 'Virgen del Cisne'),
        ('5-8 de septiembre', 'Fiesta de la Virgen del Cisne')
    ],
    'MACHALA': [
        ('Septiembre', 'Feria Mundial del Banano'),
        ('25 de junio', 'Cantonización de Machala'),
        ('20-26 de septiembre', 'Feria Mundial del Banano')
    ],
    'MANTA': [
        ('29 de junio', 'San Pedro y San Pablo'),
        ('Octubre', 'Fiestas del Comercio'),
        ('4 de noviembre', 'Cantonización de Manta')
    ],
    'MILAGRO': [
        ('17 de septiembre', 'Fiesta de San Francisco de Milagro'),
        ('Octubre', 'Festival de la Piña')
    ],
    'PAUTE': [
        ('Marzo', 'Festival del Dulce y la Manzana')
    ],
    'PENINSULA DE SANTA ELENA': [
        ('Junio', 'Festival de la Ballena Jorobada'),
        ('18 de agosto', 'Fiestas de La Península')
    ],
    'PINAS': [
        ('8 de noviembre', 'Cantonización de Piñas'),
        ('Julio', 'Festival de la Orquídea')
    ],
    'PUYO': [
        ('12 de mayo', 'Fundación de Puyo'),
        ('Febrero', 'Festival de la Canela'),
        ('11-14 de mayo', 'Feria agrícola ganadera e industrial')
    ],
    'QUEVEDO': [
        ('7 de octubre', 'Cantonización de Quevedo'),
        ('Octubre', 'Festival del río Vinces')
    ],
    'QUITO': [
        ('9-11 de agosto', 'Velada Libertaria'),
        ('1-6 de diciembre', 'Fiestas de Quito'),
        ('23-24 de septiembre', 'Fiesta de la Virgen de las Mercedes')
    ],
    'RIOBAMBA': [
        ('21 de abril', 'Independencia de Riobamba'),
        ('19-21 de abril', 'Feria agrícola ganadera y artesanal')
    ],
    'STO. DOMINGO': [
        ('2-4 de julio', 'Cantonización de Santo Domingo'),
        ('14 de abril', 'Kasama comunidad tsáchila')
    ],
    'ZAMORA': [
        ('16 de julio', 'Fiesta de la Virgen del Carmen'),
        ('10 de noviembre', 'Cantonización de Zamora'),
        ('12 de febrero', 'Día del Oriente')
    ]
}

# Función para analizar rangos de fechas
def parse_date_range(date_str):
    month_days = {
        'Enero': 31, 'Febrero': 28, 'Marzo': 31, 'Abril': 30, 'Mayo': 31, 'Junio': 30,
        'Julio': 31, 'Agosto': 31, 'Septiembre': 30, 'Octubre': 31, 'Noviembre': 30, 'Diciembre': 31
    }
    month_names = {name: i + 1 for i, name in enumerate(month_days.keys())}

    try:
        if date_str.capitalize() in month_names:  # Mes completo
            month = month_names[date_str.capitalize()]
            days_in_month = month_days[date_str.capitalize()]
            return (1, month), (days_in_month, month)
        if '-' in date_str:  # Rango de días
            day_range, month = date_str.split(' de ')
            start_day, end_day = map(int, day_range.split('-'))
            month = month_names[month.capitalize()]
            return (start_day, month), (end_day, month)
        if 'de' in date_str:  # Día específico
            day, month = date_str.split(' de ')
            day = int(day)
            month = month_names[month.capitalize()]
            return (day, month), (day, month)
    except Exception as e:
        print(f"Error al analizar la fecha: {date_str}. Detalle: {e}")
    return None

# Función para verificar si una fecha está en rango
def is_date_in_range(day, month, start_date, end_date):
    start_day, start_month = start_date
    end_day, end_month = end_date
    if start_month == end_month:  # Rango en un solo mes
        return start_month == month and start_day <= day <= end_day
    elif start_month < end_month:  # Rango entre diferentes meses
        return (start_month == month and start_day <= day) or (end_month == month and day <= end_day) or (start_month < month < end_month)
    return False

# Función para verificar si un día es fin de semana
def is_weekend(day, month, year):
    try:
        # Crear una fecha con el año, mes y día
        date = datetime(year, month, day)
        # Si el día es sábado (5) o domingo (6), retorna True
        return 1 if date.weekday() >= 5 else 0
    except Exception as e:
        print(f"Error al verificar si es fin de semana: {e}")
        return 0

# Función para actualizar las festividades en el DataFrame
def update_festivities(df):
    df['festivals'] = 0  # Inicializa la columna 'festivals' con 0
    df['isWeekend'] = 0  # Inicializa la columna 'isWeekend' con 0
    for index, row in df.iterrows():
        city = row['areacity']
        day, month, year = row['day'], row['month'], row['year']
        # Verificar si es fin de semana
        df.at[index, 'isWeekend'] = is_weekend(day, month, year)

        if city in festividades:
            for date_str, _ in festividades[city]:
                date_range = parse_date_range(date_str)
                if date_range:
                    start_date, end_date = date_range
                    if is_date_in_range(day, month, start_date, end_date):
                        df.at[index, 'festivals'] = 1
                        break  # Detiene la búsqueda si encuentra una festividad
    return df

# Leer el archivo CSV y actualizar
try:
    df = pd.read_csv(r'C:\Users\Mateo\Desktop\Backend_Mineria\data\pura\temperaturas_diarias_ecuador.csv')

    # Asegúrate de que las columnas 'day' y 'month' sean enteros
    df['day'] = df['day'].astype(int)
    df['month'] = df['month'].astype(int)
    df['year'] = df['year'].astype(int)

    # Actualizar las festividades
    df = update_festivities(df)

    # Guardar el CSV actualizado
    output_path = r'C:\Users\Mateo\Desktop\Backend_Mineria\data\pura\temperaturas_diarias_actualizado.csv'
    df.to_csv(output_path, index=False)
    print(f"Proceso completado. El archivo actualizado se guardó en: {output_path}")
except Exception as e:
    print(f"Error durante el proceso: {e}")