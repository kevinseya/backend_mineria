# Utilizar una imagen base de Python
FROM python:3.9.12-slim

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar el archivo requirements.txt al contenedor
COPY requirements.txt /app/

# Crear un entorno virtual dentro del contenedor
RUN python -m venv /env

# Activar el entorno virtual y luego instalar las dependencias
RUN /env/bin/pip install --upgrade pip
RUN /env/bin/pip install -r requirements.txt


# Copiar el archivo de datos preprocesados en la carpeta data
COPY data/quincenas_preprocesadas.csv /app/data/quincenas_preprocesadas.csv

# Copiar el archivo del modelo en la carpeta model
COPY model/lstm_model.h5 /app/model/lstm_model.h5

# Copiar el archivo server.py
COPY server.py /app/server.py

# Exponer el puerto 5000 para que el contenedor acepte conexiones externas
EXPOSE 5000

# Establecer el comando para ejecutar el servidor
CMD ["/env/bin/python", "server.py"]
