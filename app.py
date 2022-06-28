import tensorflow as tf
import os
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns


categorias = []
labels= []
imagenes = []


categorias = os.listdir('./Dataset/entrenamiento/')


x =0
for directorio in categorias:
    for imagen in os.listdir('./Dataset/entrenamiento/'+directorio):
        img = Image.open('./Dataset/entrenamiento/'+directorio+'/'+imagen).convert('RGB').resize((100,100))
        img = np.asanyarray(img)
        imagenes.append(img)
        labels.append(x)

    x +=1

imagenes = np.asanyarray(imagenes)

labels = np.asanyarray(labels)

imagenes = imagenes/255

# plt.figure(figsize=(10,10))

# for i, (imagen, etiqueta) in enumerate(imagenes.take(25)):
#   imagen = imagen.numpy().reshape((28,28))
#   plt.subplot(5,5,i+1)
#   plt.xticks([])
#   plt.yticks([])
#   plt.grid(False)
#   plt.imshow(imagen, cmap=plt.cm.binary)
#   plt.xlabel(clases[etiqueta])

# plt.show()  


#creamos el modelo
modelo = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32,(3,3), input_shape=(100,100,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=100, activation = 'relu'),
    tf.keras.layers.Dense(5, activation='softmax')

    
])


modelo.compile(
    optimizer = 'adam',
    loss = tf.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

historial = modelo.fit(imagenes,labels, epochs = 60)

plt.xlabel('# Iteraciones')
plt.ylabel('magnitud de perdida')
plt.plot(historial.history['loss'], label='Error', color='blue')
plt.show()


target_dir = './modelo/'
if not os.path.exists(target_dir):
  os.mkdir(target_dir)
modelo.save('./modelo/modelo.h5')
modelo.save_weights('./modelo/pesos.h5')
