# Utilizar una imagen base de Python
FROM python:3.9.12-slim

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar el archivo requirements.txt al contenedor
COPY requirements.txt /app/

# Crear un entorno virtual dentro del contenedor
RUN python -m venv /env

# Activar el entorno virtual e instalar las dependencias
RUN /env/bin/pip install --upgrade pip
RUN /env/bin/pip install -r requirements.txt

# Copiar el archivo del modelo en la carpeta nuevos_modelos
COPY nuevos_modelos/modelo_final---10.keras /app/nuevos_modelos/modelo_final---10.keras

# Copiar el archivo scaler para la normalizacion de datos
COPY nuevos_modelos/scaler.pkl /app/nuevos_modelos/scaler.pkl


# Copiar el archivo api.py
COPY api.py /app/api.py

# Exponer el puerto 8000 para que el contenedor acepte conexiones externas
EXPOSE 8000

# Actualizar el PATH para que se use el Python del entorno virtual
ENV PATH="/env/bin:$PATH"

# Establecer el comando para ejecutar el servidor
CMD ["python", "api.py"]
