import os
import sys

# preprocesar imagenes
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

# optimizadores para el algoritmo
from tensorflow.python.keras import optimizers

# redes neuronales sequenciales
from tensorflow.python.keras.models import Sequential

# Dropout:
# Flatten:
# Dense:
# Activation:
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation

# Para hacer nuestras capas en las cuales vamos a hacer convulsiones
# Convolution2D:
# MaxPooling2D:
from tensorflow.python.keras.layers import Convolution2D, MaxPooling2D

# matar entrenamientos de keras que se esten haciendo en algun otro lugar en nuestra computadora
from tensorflow.python.keras import backend as K


K.clear_session()  # Cerrar Sesiones de Keras

data_entrenamiento = "./data/entrenamiento"
data_validacion = "./data/validacion"

# ------ Parametros ------
"""
    numero de veces que vamos a iterar en nuestro set 
    de datos durante todo el entrenamiento
"""
epocas = 20

"""
    Cambiamos el tamaño de las imaganes
"""
altura, longitud = 100, 100

"""
    Numero de imaganes que vamos a mandar a nuestra computadora
    en cada uno de los pasos
"""
batch_size = 32

"""
    Numero de veces que se procesa la informacion en cada una
    de las epocas
"""
pasos = 1000

"""
    Al final de cada una de las epocas se van a correr 200 pasos
    con nuestro set de datos de validacion, para ver que aprende
    nuestro algoritmo
"""
pasos_validacion = 200

"""
    Numero de flitros en cada convolusion    
"""
filtrosConv1 = 32
filtrosConv2 = 64

"""
    Tamaño de filtro por cada convolusion
"""
tamano_filtro1 = (3, 3)
tamano_filtro2 = (2, 2)

"""
    Tamano de filtro en nuestro pool
"""
tamano_pool = (2, 2)

"""
    Gato, perro, gorila
"""
clases = 3

"""
    Learning rate = Que tan grande van a ser los ajustes que va a hacer
    nuestra red neuronal para acercarse a una solucion
    normalmente es un numero pequeño
"""
lr = 0.0005

# ------ Pre Procesamiento de Imagenes ------
"""
    rescale = las imagenes tienen valored siempre de 0 al 255,
        entonces las reescalamos para que sus valores esten desde el 0 al 1,
        para que sea mas eficiente
    shear_range = genera imagenes inclinadas, para que la red neuronal aprenda
        que no siempre un perro esta parado
    zoom_range = la misma logica de arriba pero con zoom
    horizontal_flip = la misma logica de arriba pero invierte la imagen
"""
entrenamiento_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True
)
"""
    Para la validacion solo pasamos las imagenes tal y como son...
"""
validacion_datagen = ImageDataGenerator(
    rescale=1./255
)

imagen_entrenamiento = entrenamiento_datagen.flow_from_directory(
    data_entrenamiento,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode="categorical"
)
imagen_validacion = validacion_datagen.flow_from_directory(
    data_validacion,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode="categorical"
)

# ------ Crear red convolusional ------
cnn = Sequential()

"""
    activation = funcion de activacion relu
    el input_shape solo va a ser necesario en la primera capa
"""

# Crear primera capa
cnn.add(Convolution2D(
    filtrosConv1,
    tamano_filtro1,
    padding="same",
    input_shape=(altura, longitud, 3),
    activation="relu"
))

cnn.add(MaxPooling2D(
    pool_size=tamano_pool
))

cnn.add(Convolution2D(
    filtrosConv2,
    tamano_filtro2,
    padding="same",
    activation="relu"
))

cnn.add(MaxPooling2D(
    pool_size=tamano_pool
))

"""
    La imagen la hacemos plana
"""
cnn.add(Flatten())

"""
    Despues de aplanar la informacion,
    le mandamos la informacion a una capa normal
    con 256 neuronas, con funcion de activacion relu
"""

cnn.add(Dense(
    256,
    activation="relu"
))

"""
    Durante el entrenamiento le apagamos el 50% de las neuronas cada paso,
    esto se hace para evitar sobreajustar, ya que si todo el tiempo todas
    las neuronas estan activadas puede que nuestra red neuronal aprenda
    en especifico un camino para clasificar algo en especifico.
    
    Si apagamos el 50% hacemos que tome caminos alternos. De esta manera
    se hace un modelo que se adapta mejor a informacion nueva
"""

cnn.add(Dropout(
    0.5
))

"""
    softmax = nos va a decir el porcentaje 
    ejemplo: 20% perro, 30% gato, 50% gorila
"""
cnn.add(Dense(
    clases,
    activation="softmax"
))

cnn.compile(
    loss="categorical_crossentropy",
    optimizer=optimizers.Adam(lr=lr),
    metrics=["accuracy"]
)

cnn.fit(
    imagen_entrenamiento,
    steps_per_epoch=pasos,
    epochs=epocas,
    validation_data=imagen_validacion,
    validation_steps=pasos_validacion
)

dir = "./modelo/"
if not os.path.exists(dir):
    os.mkdir(dir)
cnn.save("./modelo/modelo.h5")
cnn.save_weights("./modelo/pesos.h5")
