from re import T
from keras.models import load_model
import matplotlib.pyplot as plt
from PIL import Image
import os
import numpy as np
import seaborn as sns
import tensorflow as tf
from tkinter import Tk, Label, StringVar, Button



def predict():
  predecir = []
  labels = []
  validar =[]

  predecirImagenes = []
  labelsImagenes = []
  validarImagenes =[]

  predecir =os.listdir('./Dataset/validacion/')

  modelo = './modelo/modelo.h5'
  pesos_modelo = './modelo/pesos.h5'
  historial = load_model(modelo)
  historial.load_weights(pesos_modelo)


  x=0
  for directorio in predecir:
      for imagen in os.listdir('./Dataset/validacion/'+directorio):
        img = Image.open('./Dataset/validacion/'+directorio+'/'+imagen).convert('RGB').resize((100,100))
        img = np.asanyarray(img)
        validar.append(img)
        labels.append(x)
      x+=1

  validar = np.asanyarray(validar)

  print(validar.shape)
  labels = np.asanyarray(labels)

  plt.figure()
  for i in validar:
    plt.imshow(i)
    plt.colorbar()
    plt.title('imagen que se va a predecir')
    plt.show()


  predicciones = historial.predict(validar)
  yPredict = np.argmax(predicciones, axis=1)

  prediccionaxis = []
  for j in predicciones:
    prediccionaxis.append(predecir[np.argmax(j)])

  plt.figure()
  contador = 0
  for k in validar:
    plt.imshow(k)
    plt.colorbar()
    plt.title('prediccion: '+prediccionaxis[contador])
    plt.show()
    contador = contador +1


  #aqui voy a cargar todo para la matriz de confucion
  predecirImagenes =os.listdir('./Dataset/entrenamiento/')

  x=0
  for directorio in predecirImagenes:
      for imagen in os.listdir('./Dataset/entrenamiento/'+directorio):
          img = Image.open('./Dataset/entrenamiento/'+directorio+'/'+imagen).convert('RGB').resize((100,100))
          img = np.asanyarray(img)
          validarImagenes.append(img)
          labelsImagenes.append(x)
      x+=1


  validarImagenes = np.asanyarray(validarImagenes)

  labelsImagenes = np.asanyarray(labelsImagenes)

  prediccionesimagenes = historial.predict(validarImagenes)
  yPredictimagen = np.argmax(prediccionesimagenes, axis=1)



  confusion_mtx = tf.math.confusion_matrix(labelsImagenes, yPredictimagen)
  plt.figure(figsize=(10, 5))
  sns.heatmap(confusion_mtx,
              xticklabels=predecirImagenes,
              yticklabels=predecirImagenes,
              annot=True, fmt='g')
  plt.xlabel('Prediction')
  plt.ylabel('Label')
  plt.show()


def ejecutar_comandos():
  os.system('cls')
  os.system('python app.py')
  resultadoX.set('se termino de cargar el modelo')

ventana = Tk()
ventana.geometry('400x490')
ventana.title('Redes neuronales')

resultadoX = StringVar()


decoractionCuadro = Label(ventana, bg='#00C1C1')
decoractionCuadro.place(x=0, y=0, width=400, height=44)

label = Label(ventana, text='REDES NEURONALES', fg='#000000')
label.place(x=110, y=90)
label.config(font="Inter 12 bold")

botonX = Button(ventana, text='CARGAR MODELO DE RED', command=ejecutar_comandos)
botonX.place(x=100, y=180, width=205, height=30)
botonX.config(font='Inter 8 bold', bg='#00C1C1', fg='white', bd=0)

label_resultadox = Label(ventana, textvariable=resultadoX)
label_resultadox.place(x=100, y=210)
label_resultadox.config(font='Inter 10',fg='red')




boton = Button(ventana, text='Iniciar Entrenamiento', command=predict)
boton.place(x=100, y=300, width=205, height=30)
boton.config(font='Inter 13', bg='#00C1C1', fg='white', bd=0)

ventana.mainloop()