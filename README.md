# Backend para Predicción con Flask

Este repositorio contiene el backend de una aplicación basada en Flask que utiliza un modelo preentrenado para realizar predicciones. El modelo se encuentra en formato `.h5` y es cargado por la aplicación Flask para hacer inferencias sobre los datos proporcionados.

## Requisitos

Antes de ejecutar el proyecto, asegúrate de tener los siguientes requisitos:

- Python 3.9+
- Docker (para la ejecución del contenedor)
- pip

### Dependencias
- Flask
- Flask-CORS
- TensorFlow (para el modelo)
- NumPy
- Pandas

## Instalación

1. Clona este repositorio:

    ```bash
    git clone https://tu-repositorio.git
    cd nombre-del-repositorio
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

/
├── app/
│   ├── data/
│   │   └── quincenas_preprocesadas.csv
│   ├── model/
│   │   └── lstm_model.h5
│   ├── server.py
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── README.md

- **data/quincenas_preprocesadas.csv**: Contiene los datos de entrada preprocesados para realizar las predicciones.
- **model/lstm_model.h5**: El modelo de predicción entrenado en formato `.h5`.
- **server.py**: El servidor Flask que maneja las solicitudes y hace las predicciones utilizando el modelo.
- **requirements.txt**: Contiene las dependencias necesarias para ejecutar el servidor Flask.

## Docker

### Construir y ejecutar el contenedor Docker

Si prefieres ejecutar el proyecto dentro de un contenedor Docker, puedes hacerlo de la siguiente manera:

1. Construir la imagen de Docker:

    ```bash
    sudo docker build -t backend_mineria .
    ```

2. Ejecutar el contenedor, mapeando el puerto 5000 del contenedor al puerto 5000 de la máquina local:

    ```bash
    sudo docker run -p 5000:5000 backend_mineria
    ```

### Exposición del puerto

El servidor Flask está configurado para escuchar en el puerto `5000`. Para que el servidor sea accesible fuera del contenedor, asegúrate de que esté configurado para escuchar en **0.0.0.0** en lugar de **127.0.0.1** (localhost). Esto se configura en el archivo `server.py`:

```python
app.run(host="0.0.0.0", port=5000)
```
