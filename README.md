# Backend para Predicción con Flask

Este repositorio contiene el backend de una aplicación basada en Flask que utiliza un modelo preentrenado para realizar predicciones. El modelo se encuentra en formato `.h5` y es cargado por la aplicación Flask para hacer inferencias sobre los datos proporcionados.

## Requisitos

Antes de ejecutar el proyecto, asegúrate de tener los siguientes requisitos:

- Python 3.9+
- Docker (para la ejecución del contenedor)
- pip

### Dependencias
- FastAPI
- CORS
- TensorFlow (para el modelo)
- NumPy
- Pandas
- Seaborn
- Joblib
- Matplotlib

## Instalación

1. Clona este repositorio:

    ```bash
    git clone (https://github.com/kevinseya/backend_mineria.git)
    ```

2. Crea un entorno virtual (opcional, pero recomendado):

    ```bash
    python -m venv venv
    source venv/bin/activate  # En Windows usa venv\Scripts\activate
    ```

3. Instala las dependencias desde `requirements.txt`:

    ```bash
    pip install -r requirements.txt
    ```

4. O también puedes crear un contenedor Docker (si usas Docker):

    ```bash
    docker build -t backend_mineria .
    ```

## Estructura del Proyecto

###data: 
- Se encuentra los archivos .csv que fueron utilizados para para generar el .csv final que consumira el modelo para entrenarse.
- Así como también, los scripts .py que fueron utilizados para unir la data extraida de la API Forecast, asi como tambien de los diferentes sitios web para datos económicos
  
###nuevos_modelos:
- Las graficas de resultados del mejor modelo escogido.
- Script del modelo mejorado LSTM.
- Script del análisis de correlación de las variables unidas en el .csv
- Modelo entrenado en formato .keras
- Normalizador de datos para la entrada de datos a entrenar en modelo en formato .pkl
  
###diagrama:
- Diagrama de flujo de trabajo que hace modelo, integrado con la interfaz y con agente llm para un chatbot que se conecta con la base de datos original
  
###api:
- Api que expone endpoints para el consumo de la interfaz hacia el modelo y su respectiva predicción.

###Dockerfile:
- Dockerfile para la dockerización del proyecto, para su posible despliegue.
  
###Otros:
- Mas archivos importantes para el proyecto

## Docker

### Construir y ejecutar el contenedor Docker

Si prefieres ejecutar el proyecto dentro de un contenedor Docker, puedes hacerlo de la siguiente manera:

1. Construir la imagen de Docker:

    ```bash
    sudo docker build -t backend_mineria .
    ```

2. Ejecutar el contenedor, mapeando el puerto 8000 del contenedor al puerto 5000 de la máquina local:

    ```bash
    sudo docker run -p 8000:8000 backend_mineria
    ```

### Exposición del puerto

El servidor FastAPI está configurado para escuchar en el puerto `8000`. Para que el servidor sea accesible fuera del contenedor, asegúrate de que esté configurado para escuchar en **0.0.0.0** en lugar de **127.0.0.1** (localhost). Esto se configura en el archivo `api.py`:

```python
app.run(host="0.0.0.0", port=8000)
```
